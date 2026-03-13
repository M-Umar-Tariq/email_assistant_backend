import base64
import email as email_lib
import json
import re
import time
import traceback
import uuid
from datetime import datetime, timezone, timedelta
from email.header import decode_header as decode_email_header

from bson import ObjectId
from pymongo.errors import DuplicateKeyError

from api.utils.email_body import clean_email_body, collapse_long_urls_in_html
from api.utils.attachment_text import extract_attachments_from_message
from django.conf import settings
from imapclient import IMAPClient

from database.db import mailboxes_col, email_metadata_col, email_attachments_col, follow_ups_col, get_qdrant, ensure_qdrant_collection
from api.utils.encryption import encrypt, decrypt
from api.utils.chunking import chunk_text
from api.utils.embedding import embed_texts
from api.utils.classify import classify_emails_batch
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue


# ── CRUD ─────────────────────────────────────────────────────────────────────

def _imap_friendly_error(err_msg: str) -> str:
    if "LOGIN" in err_msg or "BAD" in err_msg or "AUTHENTICATIONFAILED" in err_msg.upper():
        return (
            "Connection failed: invalid email or password. "
            "For Gmail, use an App Password (Google Account → Security → App passwords). "
            "For Outlook, enable IMAP and use your password or app password."
        )
    return f"Connection failed: {err_msg}"


def verify_imap_connection(imap_host: str, imap_port: int, imap_secure: bool, username: str, password: str) -> None:
    """Verify IMAP credentials by connecting and selecting INBOX. Raises ValueError on failure."""
    try:
        with IMAPClient(imap_host, port=imap_port, ssl=imap_secure) as client:
            client.login(username, password)
            client.select_folder("INBOX", readonly=True)
    except Exception as e:
        raise ValueError(_imap_friendly_error(str(e).strip()))


def create_mailbox(user_id: str, data: dict) -> dict:
    verify_imap_connection(
        data["imap_host"],
        data.get("imap_port", 993),
        data.get("imap_secure", True),
        data["username"],
        data["password"],
    )
    doc = {
        "user_id": user_id,
        "name": data["name"],
        "email": data["email"],
        "color": data.get("color", "#0ea5e9"),
        "imap_host": data["imap_host"],
        "imap_port": data.get("imap_port", 993),
        "imap_secure": data.get("imap_secure", True),
        "smtp_host": data["smtp_host"],
        "smtp_port": data.get("smtp_port", 587),
        "smtp_secure": data.get("smtp_secure", True),
        "username": data["username"],
        "encrypted_password": encrypt(data["password"]),
        "last_sync_at": None,
        "sync_status": "pending",
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    result = mailboxes_col().insert_one(doc)
    doc["_id"] = result.inserted_id
    return _serialize(doc)


def list_mailboxes(user_id: str) -> list[dict]:
    docs = mailboxes_col().find({"user_id": user_id}).sort("created_at", -1)
    result = []
    for d in docs:
        out = _serialize(d)
        now = datetime.now(timezone.utc)
        base_q = {
            "user_id": user_id,
            "mailbox_id": str(d["_id"]),
            "archived": False,
            "trashed": False,
            "$or": [{"snoozed_until": None}, {"snoozed_until": {"$lte": now}}],
        }
        out["total_emails"] = email_metadata_col().count_documents(base_q)
        out["unread"] = email_metadata_col().count_documents({**base_q, "read": False})
        result.append(out)
    return result


def get_mailbox(user_id: str, mailbox_id: str) -> dict | None:
    doc = mailboxes_col().find_one({"_id": ObjectId(mailbox_id), "user_id": user_id})
    if not doc:
        return None
    out = _serialize(doc)
    now = datetime.now(timezone.utc)
    base_q = {
        "user_id": user_id,
        "mailbox_id": mailbox_id,
        "archived": False,
        "trashed": False,
        "$or": [{"snoozed_until": None}, {"snoozed_until": {"$lte": now}}],
    }
    out["total_emails"] = email_metadata_col().count_documents(base_q)
    out["unread"] = email_metadata_col().count_documents({**base_q, "read": False})
    return out


def update_mailbox(user_id: str, mailbox_id: str, data: dict) -> dict | None:
    data["updated_at"] = datetime.now(timezone.utc)
    mailboxes_col().update_one(
        {"_id": ObjectId(mailbox_id), "user_id": user_id}, {"$set": data}
    )
    return get_mailbox(user_id, mailbox_id)


def delete_mailbox(user_id: str, mailbox_id: str) -> bool:
    result = mailboxes_col().delete_one({"_id": ObjectId(mailbox_id), "user_id": user_id})
    if result.deleted_count:
        # Delete follow-ups that reference emails from this mailbox (before we delete metadata)
        email_ids = list(email_metadata_col().distinct("_id", {"user_id": user_id, "mailbox_id": mailbox_id}))
        if email_ids:
            email_id_strs = [str(eid) for eid in email_ids]
            follow_ups_col().delete_many({"user_id": user_id, "email_id": {"$in": email_id_strs}})
        email_metadata_col().delete_many({"user_id": user_id, "mailbox_id": mailbox_id})
        _delete_qdrant_chunks_for_mailbox(user_id, mailbox_id)
    return result.deleted_count > 0


# ── IMAP sync ────────────────────────────────────────────────────────────────

def sync_mailbox(
    user_id: str,
    mailbox_id: str,
    *,
    initial_sync: str | None = None,
    limit: int | None = None,
) -> dict:
    """
    Fetch emails from IMAP, then store them.
    - initial_sync "only_new": set last_sync_at to now, do not fetch (only new emails later).
    - initial_sync "last_n" with limit: fetch only the newest `limit` emails.
    - initial_sync "all" or None: full sync (or incremental if last_sync_at set).
    """
    mb = mailboxes_col().find_one({"_id": ObjectId(mailbox_id), "user_id": user_id})
    if not mb:
        raise ValueError("Mailbox not found")

    # Only new: no IMAP fetch, just set last_sync_at so future syncs get only new mail
    if initial_sync == "only_new":
        mailboxes_col().update_one(
            {"_id": mb["_id"]},
            {"$set": {"last_sync_at": datetime.now(timezone.utc), "sync_status": "synced"}},
        )
        total = email_metadata_col().count_documents({"user_id": user_id, "mailbox_id": mailbox_id})
        return {"synced": 0, "total": total, "total_fetched": 0}

    # Atomic lock with stale detection
    now = datetime.now(timezone.utc)
    stale_cutoff = now - timedelta(minutes=15)
    locked = mailboxes_col().find_one_and_update(
        {
            "_id": mb["_id"],
            "$or": [
                {"sync_status": {"$ne": "syncing"}},
                {"sync_started_at": {"$exists": False}},
                {"sync_started_at": {"$lte": stale_cutoff}},
            ],
        },
        {"$set": {"sync_status": "syncing", "sync_started_at": now}},
    )
    if not locked:
        total = email_metadata_col().count_documents({"user_id": user_id, "mailbox_id": mailbox_id})
        return {"synced": 0, "total": total, "total_fetched": 0, "skipped_reason": "already_syncing"}

    password = decrypt(mb["encrypted_password"])

    IMAP_BATCH = 10  # fetch N emails at a time from IMAP for fast first-result

    # 1) Connect to IMAP & get message IDs (fast — no email data downloaded yet)
    try:
        imap_client = IMAPClient(mb["imap_host"], port=mb["imap_port"], ssl=mb["imap_secure"])
        imap_client.login(mb["username"], password)
        imap_client.select_folder("INBOX", readonly=True)
        since = mb.get("last_sync_at")
        if initial_sync in ("last_n", "all"):
            criteria = ["ALL"]
        else:
            criteria = ["SINCE", since.strftime("%d-%b-%Y")] if since else ["ALL"]
        msg_ids = imap_client.search(criteria)
        if not msg_ids:
            imap_client.logout()
            total_after = email_metadata_col().count_documents({"user_id": user_id, "mailbox_id": mailbox_id})
            _finish_sync(mb["_id"], 0)
            return {"synced": 0, "total": total_after, "total_fetched": 0}
        if initial_sync == "last_n" and limit is not None and limit > 0:
            msg_ids = msg_ids[-limit:]
        # Process newest emails first (IMAP returns UIDs oldest-first by default)
        msg_ids = list(reversed(msg_ids))
        total_fetched = len(msg_ids)
    except Exception as e:
        mailboxes_col().update_one(
            {"_id": mb["_id"]}, {"$set": {"sync_status": "error", "sync_started_at": None}}
        )
        err_msg = str(e).strip()
        if "LOGIN" in err_msg or "BAD" in err_msg or "AUTHENTICATIONFAILED" in err_msg.upper():
            raise RuntimeError(
                "IMAP login failed. Check that the username is your full email and the password is correct. "
                "For Gmail, use an App Password (not your normal password). For Outlook, enable IMAP and use your password or app password."
            )
        raise RuntimeError(f"IMAP failed: {e}")

    # 2) Ensure Qdrant collection + indexes
    try:
        ensure_qdrant_collection()
    except Exception as qdrant_err:
        raise RuntimeError(f"Vector database unavailable — please try again later. ({qdrant_err})")

    # 3) Stream in batches: IMAP fetch → parse → classify → store
    synced_count = 0
    thread_replies_added = 0
    skipped = 0
    flags_updated = 0
    qdrant = get_qdrant()

    try:
        for batch_start in range(0, len(msg_ids), IMAP_BATCH):
            if _is_sync_cancelled(mb["_id"]):
                print(f"[SYNC] Cancelled by user after {synced_count} emails")
                break

            batch_ids = msg_ids[batch_start : batch_start + IMAP_BATCH]

            # 3a) Fetch this batch from IMAP
            try:
                fetched_batch = list(imap_client.fetch(batch_ids, ["RFC822", "FLAGS"]).items())
            except Exception:
                traceback.print_exc()
                continue

            # 3b) First pass: extract message IDs from raw emails
            raw_emails = []
            for uid, data in fetched_batch:
                raw = data.get(b"RFC822", b"")
                if not raw:
                    continue
                msg = email_lib.message_from_bytes(raw)
                mid_raw = (msg.get("Message-ID") or "").strip()
                mid = mid_raw if mid_raw else f"<uid-{uid}>"
                raw_emails.append((uid, data, msg, mid))

            if not raw_emails:
                continue

            # Batch-check which message_ids already exist (as primary or as thread replies)
            batch_mids = [item[3] for item in raw_emails]
            existing_docs = list(email_metadata_col().find(
                {"user_id": user_id, "mailbox_id": mailbox_id, "message_id": {"$in": batch_mids}},
                {"message_id": 1, "read": 1, "starred": 1, "replied_at": 1, "date": 1},
            ))
            existing_map = {doc["message_id"]: doc for doc in existing_docs}
            # Also skip message_ids already stored as thread replies on another email
            thread_parent_docs = list(email_metadata_col().find(
                {"user_id": user_id, "mailbox_id": mailbox_id, "thread_message_ids": {"$in": batch_mids}},
                {"thread_message_ids": 1},
            ))
            thread_known_mids: set = set()
            for td in thread_parent_docs:
                for tmid in td.get("thread_message_ids", []):
                    if tmid in batch_mids:
                        thread_known_mids.add(tmid)
            batch_num = batch_start // IMAP_BATCH + 1
            truly_new = len(raw_emails) - len(existing_map) - len(thread_known_mids)
            print(f"[SYNC] Batch {batch_num}: {len(raw_emails)} msgs, {len(existing_map)} existing, {len(thread_known_mids)} thread-known, {max(truly_new, 0)} new")

            # 3c) Parse new emails, update flags on existing ones
            parsed_batch = []
            for uid, data, msg, mid in raw_emails:
                if mid in thread_known_mids:
                    continue
                existing = existing_map.get(mid)
                if existing:
                    flags = data.get(b"FLAGS", [])
                    flag_updates = {}
                    is_read = b"\\Seen" in flags
                    is_starred = b"\\Flagged" in flags
                    is_replied = b"\\Answered" in flags
                    if existing.get("read") != is_read:
                        flag_updates["read"] = is_read
                    if existing.get("starred") != is_starred:
                        flag_updates["starred"] = is_starred
                    if is_replied and not existing.get("replied_at"):
                        flag_updates["replied_at"] = existing.get("date", datetime.now(timezone.utc)).isoformat() if isinstance(existing.get("date"), datetime) else datetime.now(timezone.utc).isoformat()
                        flag_updates["reply_count"] = existing.get("reply_count", 0) + 1
                    if flag_updates:
                        email_metadata_col().update_one({"_id": existing["_id"]}, {"$set": flag_updates})
                        flags_updated += 1

                    _backfill_attachments(existing, msg, user_id, mailbox_id)
                    continue

                body_plain = _extract_body(msg)
                body = clean_email_body(body_plain)
                body_html = _extract_body_html(msg)
                if body_html:
                    cid_map = _get_inline_image_cids(msg)
                    body_html = _replace_cid_in_html(body_html, cid_map)
                    body_html = collapse_long_urls_in_html(body_html)
                subject = _decode_header_value(msg.get("Subject", ""))
                from_str = _decode_header_value(msg.get("From", ""))
                to_str = _decode_header_value(msg.get("To", ""))
                date_str = msg.get("Date", "")

                from_name, from_email_addr = _parse_address(from_str)
                to_list = [{"name": n, "email": e} for n, e in [_parse_address(a.strip()) for a in to_str.split(",") if a.strip()]]
                parsed_date = _parse_date(date_str)
                if parsed_date and not parsed_date.tzinfo:
                    parsed_date = parsed_date.replace(tzinfo=timezone.utc)

                # IMAP SINCE is date-level only; filter at time-level so
                # "only new" mailboxes never pull in older same-day emails.
                since_aware = since.replace(tzinfo=timezone.utc) if since and not since.tzinfo else since
                if since_aware and parsed_date and initial_sync not in ("last_n", "all") and parsed_date < since_aware:
                    skipped += 1
                    continue

                preview = (body[:300] if body else "") or "(no preview)"

                # Thread detection: check In-Reply-To / References
                in_reply_to = (msg.get("In-Reply-To") or "").strip()
                references = (msg.get("References") or "").strip()
                parent_mid = in_reply_to or (references.split()[-1] if references else "")

                if parent_mid:
                    parent = email_metadata_col().find_one(
                        {"user_id": user_id, "mailbox_id": mailbox_id, "message_id": parent_mid}
                    )
                    if not parent:
                        ref_list = references.split() if references else []
                        for ref_mid in reversed(ref_list):
                            if ref_mid == parent_mid:
                                continue
                            parent = email_metadata_col().find_one(
                                {"user_id": user_id, "mailbox_id": mailbox_id, "message_id": ref_mid}
                            )
                            if parent:
                                break

                    if parent:
                        # Skip if this is our own sent reply (already stored in sent_replies)
                        mb_email = mb.get("email", "") or mb.get("username", "")
                        if from_email_addr and mb_email and from_email_addr.lower() == mb_email.lower():
                            email_metadata_col().update_one(
                                {"_id": parent["_id"]},
                                {"$addToSet": {"thread_message_ids": mid}},
                            )
                            continue

                        thread_reply_doc = {
                            "message_id": mid,
                            "from_name": from_name,
                            "from_email": from_email_addr,
                            "to": to_list,
                            "subject": subject,
                            "body": body or "",
                            "body_html": body_html or "",
                            "date": parsed_date.isoformat() if parsed_date else datetime.now(timezone.utc).isoformat(),
                            "preview": preview,
                        }
                        update_ops: dict = {
                            "$push": {"thread_replies": thread_reply_doc},
                            "$addToSet": {"thread_message_ids": mid},
                        }
                        # Bump parent date and mark unread so new reply surfaces in inbox
                        reply_dt = parsed_date or datetime.now(timezone.utc)
                        if reply_dt and not reply_dt.tzinfo:
                            reply_dt = reply_dt.replace(tzinfo=timezone.utc)
                        parent_dt = parent.get("date")
                        if parent_dt and not parent_dt.tzinfo:
                            parent_dt = parent_dt.replace(tzinfo=timezone.utc)
                        set_fields: dict = {"read": False}
                        if not parent_dt or (reply_dt and reply_dt > parent_dt):
                            set_fields["date"] = reply_dt
                        update_ops["$set"] = set_fields
                        email_metadata_col().update_one(
                            {"_id": parent["_id"]}, update_ops,
                        )
                        thread_replies_added += 1
                        print(f"[SYNC] Thread reply from {from_name or from_email_addr} added to parent '{parent.get('_id')}'")
                        continue

                email_id = str(uuid.uuid4())
                chunks = chunk_text(body) if body else [subject or "(empty)"]
                total_chunks = len(chunks)

                attachments = extract_attachments_from_message(msg, include_binary=True)
                attachment_text_parts = [
                    f"[{a['filename']}]\n{a['text']}"
                    for a in attachments if a["text"]
                ]
                attachment_text = "\n\n---\n\n".join(attachment_text_parts)
                attachment_meta = [
                    {"filename": a["filename"], "content_type": a["content_type"],
                     "size": a["size"], "has_text": bool(a["text"])}
                    for a in attachments
                ]
                attachment_binaries = [
                    {"index": idx, "filename": a["filename"], "content_type": a["content_type"],
                     "size": a["size"], "data_b64": a.get("data_b64", "")}
                    for idx, a in enumerate(attachments) if a.get("data_b64")
                ]

                parsed_batch.append({
                    "email_id": email_id,
                    "mid": mid,
                    "subject": subject,
                    "from_name": from_name,
                    "from_email": from_email_addr,
                    "to": to_list,
                    "date": parsed_date,
                    "preview": preview,
                    "body": body,
                    "body_html": body_html,
                    "read": b"\\Seen" in data.get(b"FLAGS", []),
                    "starred": b"\\Flagged" in data.get(b"FLAGS", []),
                    "replied": b"\\Answered" in data.get(b"FLAGS", []),
                    "has_attachment": _has_attachment(msg),
                    "attachments": json.dumps(attachment_meta) if attachment_meta else "[]",
                    "attachment_text": attachment_text,
                    "attachment_binaries": attachment_binaries,
                    "chunks": chunks,
                    "total_chunks": total_chunks,
                })

            if not parsed_batch:
                continue

            # 3c) AI classification — only for today's emails to save LLM calls
            start_of_today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
            today_emails_for_classify = [
                {"subject": e["subject"], "from_name": e["from_name"], "from_email": e["from_email"], "preview": e["preview"]}
                for e in parsed_batch
                if e["date"] and e["date"] >= start_of_today
            ]
            today_classifications = classify_emails_batch(today_emails_for_classify) if today_emails_for_classify else []

            # Build a map: index into today_classifications for today's emails
            classify_iter = iter(today_classifications)
            ai_classifications = []
            for e in parsed_batch:
                if e["date"] and e["date"] >= start_of_today:
                    ai_classifications.append(next(classify_iter))
                else:
                    ai_classifications.append({"priority": "medium", "category": None})

            # 3d) Store this batch: MongoDB (state) + Qdrant (content)
            for email_data, ai_cls in zip(parsed_batch, ai_classifications):
                try:
                    ai_priority = ai_cls.get("priority", "medium")
                    ai_category = ai_cls.get("category")
                    date_iso = email_data["date"].isoformat() if email_data["date"] else ""
                    to_json = json.dumps(email_data["to"]) if email_data["to"] else "[]"

                    meta_doc = {
                        "_id": email_data["email_id"],
                        "user_id": user_id,
                        "mailbox_id": mailbox_id,
                        "message_id": email_data["mid"],
                        "subject": email_data.get("subject", ""),
                        "from_name": email_data.get("from_name", ""),
                        "from_email": email_data.get("from_email", ""),
                        "date": email_data["date"],
                        "read": email_data["read"],
                        "starred": email_data["starred"],
                        "replied_at": email_data["date"].isoformat() if email_data["replied"] and email_data["date"] else None,
                        "reply_count": 1 if email_data["replied"] else 0,
                        "labels": [],
                        "snoozed_until": None,
                        "archived": False,
                        "trashed": False,
                        "created_at": datetime.now(timezone.utc),
                    }
                    try:
                        email_metadata_col().insert_one(meta_doc)
                    except DuplicateKeyError:
                        skipped += 1
                        continue

                    raw_html = email_data["body_html"] or ""
                    safe_html = raw_html[:524288] if len(raw_html) > 524288 else raw_html

                    texts_to_embed = [f"{email_data['subject']}\n\n{c}" for c in email_data["chunks"]]
                    embeddings = embed_texts(texts_to_embed)
                    points = []
                    for idx, (chunk_txt, emb) in enumerate(zip(email_data["chunks"], embeddings)):
                        payload = {
                            "email_id": email_data["email_id"],
                            "user_id": user_id,
                            "mailbox_id": mailbox_id,
                            "chunk_index": idx,
                            "total_chunks": email_data["total_chunks"],
                            "body_chunk": chunk_txt,
                            "subject": email_data["subject"],
                            "from_name": email_data["from_name"],
                            "from_email": email_data["from_email"],
                            "to": to_json,
                            "date": date_iso,
                            "preview": email_data["preview"],
                            "has_attachment": email_data["has_attachment"],
                            "priority": ai_priority,
                            "category": ai_category or "",
                        }
                        if idx == 0:
                            payload["body_html"] = safe_html
                            payload["attachments"] = email_data.get("attachments", "[]")
                            if email_data.get("attachment_text"):
                                payload["attachment_text"] = email_data["attachment_text"][:32000]
                        points.append(PointStruct(id=str(uuid.uuid4()), vector=emb, payload=payload))

                    for attempt in range(2):
                        try:
                            qdrant.upsert(
                                collection_name=settings.QDRANT_COLLECTION_EMAIL_CHUNKS,
                                points=points,
                            )
                            break
                        except Exception as upsert_err:
                            if attempt == 0:
                                time.sleep(1)
                            else:
                                raise upsert_err

                    # Store attachment binary data in MongoDB for download
                    for att_bin in email_data.get("attachment_binaries", []):
                        try:
                            email_attachments_col().insert_one({
                                "email_id": email_data["email_id"],
                                "user_id": user_id,
                                "mailbox_id": mailbox_id,
                                "index": att_bin["index"],
                                "filename": att_bin["filename"],
                                "content_type": att_bin["content_type"],
                                "size": att_bin["size"],
                                "data_b64": att_bin["data_b64"],
                            })
                        except Exception:
                            pass

                    synced_count += 1
                except Exception as email_err:
                    skipped += 1
                    print(f"[SYNC] Skipped email '{email_data.get('subject', '?')}': {email_err}")
                    traceback.print_exc()

        total_after = email_metadata_col().count_documents({"user_id": user_id, "mailbox_id": mailbox_id})
        _finish_sync(mb["_id"], synced_count)
        return {"synced": synced_count, "thread_replies_added": thread_replies_added, "skipped": skipped, "flags_updated": flags_updated, "total": total_after, "total_fetched": total_fetched}
    except Exception as e:
        traceback.print_exc()
        mailboxes_col().update_one(
            {"_id": mb["_id"]}, {"$set": {"sync_status": "error", "sync_started_at": None}}
        )
        raise RuntimeError(f"Sync failed: {e}")
    finally:
        try:
            imap_client.logout()
        except Exception:
            pass


# ── Helpers ──────────────────────────────────────────────────────────────────

def _backfill_attachments(existing_meta: dict, msg, user_id: str, mailbox_id: str):
    """If an existing email has attachments but no attachment data in Qdrant, extract and store it."""
    if not _has_attachment(msg):
        return
    email_id = str(existing_meta["_id"])
    try:
        # Check if already backfilled
        already_stored = email_attachments_col().find_one({"email_id": email_id, "user_id": user_id})
        if already_stored:
            return

        qdrant = get_qdrant()
        results = qdrant.scroll(
            collection_name=settings.QDRANT_COLLECTION_EMAIL_CHUNKS,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="email_id", match=MatchValue(value=email_id)),
                    FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                    FieldCondition(key="chunk_index", match=MatchValue(value=0)),
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        points = results[0] if results else []
        if not points:
            return
        point = points[0]

        attachments = extract_attachments_from_message(msg, include_binary=True)
        if not attachments:
            return

        attachment_text_parts = [
            f"[{a['filename']}]\n{a['text']}"
            for a in attachments if a["text"]
        ]
        attachment_text = "\n\n---\n\n".join(attachment_text_parts)
        attachment_meta = [
            {"filename": a["filename"], "content_type": a["content_type"],
             "size": a["size"], "has_text": bool(a["text"])}
            for a in attachments
        ]

        # Update Qdrant payload with text metadata
        existing_attachments = point.payload.get("attachments")
        if not existing_attachments or existing_attachments == "[]":
            qdrant.set_payload(
                collection_name=settings.QDRANT_COLLECTION_EMAIL_CHUNKS,
                payload={
                    "attachments": json.dumps(attachment_meta),
                    "attachment_text": attachment_text[:32000] if attachment_text else "",
                },
                points=[point.id],
            )

        # Store binary data in MongoDB for download
        for idx, a in enumerate(attachments):
            if a.get("data_b64"):
                try:
                    email_attachments_col().insert_one({
                        "email_id": email_id,
                        "user_id": user_id,
                        "mailbox_id": mailbox_id,
                        "index": idx,
                        "filename": a["filename"],
                        "content_type": a["content_type"],
                        "size": a["size"],
                        "data_b64": a["data_b64"],
                    })
                except Exception:
                    pass
        print(f"[SYNC] Backfilled attachments for email '{email_id}': {[a['filename'] for a in attachment_meta]}")
    except Exception as e:
        print(f"[SYNC] Attachment backfill failed for '{email_id}': {e}")


def stop_sync(user_id: str, mailbox_id: str) -> dict:
    """Cancel an ongoing sync by resetting sync_status."""
    result = mailboxes_col().update_one(
        {"_id": ObjectId(mailbox_id), "user_id": user_id, "sync_status": "syncing"},
        {"$set": {"sync_status": "cancelled", "sync_started_at": None}},
    )
    return {"stopped": result.modified_count > 0}


def _is_sync_cancelled(mb_id) -> bool:
    doc = mailboxes_col().find_one({"_id": mb_id}, {"sync_status": 1})
    return doc.get("sync_status") == "cancelled" if doc else True


def _finish_sync(mb_id, count):
    mailboxes_col().update_one(
        {"_id": mb_id},
        {"$set": {"last_sync_at": datetime.now(timezone.utc), "sync_status": "synced", "sync_started_at": None}},
    )


def _extract_body(msg) -> str:
    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            if ct == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    return payload.decode("utf-8", errors="replace")
        for part in msg.walk():
            ct = part.get_content_type()
            if ct == "text/html":
                payload = part.get_payload(decode=True)
                if payload:
                    return payload.decode("utf-8", errors="replace")
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            return payload.decode("utf-8", errors="replace")
    return ""


def _extract_body_html(msg) -> str | None:
    """Return the HTML part of the message if present, else None (for same-as-server display)."""
    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            if ct == "text/html":
                payload = part.get_payload(decode=True)
                if payload:
                    return payload.decode("utf-8", errors="replace")
    elif (msg.get_content_type() or "").startswith("text/html"):
        payload = msg.get_payload(decode=True)
        if payload:
            return payload.decode("utf-8", errors="replace")
    return None


def _get_inline_image_cids(msg) -> dict[str, str]:
    """Build a map of Content-ID (normalized, no angle brackets) -> data URL for inline image parts."""
    cid_to_data: dict[str, str] = {}
    if not msg.is_multipart():
        return cid_to_data
    for part in msg.walk():
        ct = (part.get_content_type() or "").lower()
        if not ct.startswith("image/"):
            continue
        cid = part.get("Content-ID")
        if not cid:
            continue
        cid_clean = cid.strip().strip("<>").strip()
        if not cid_clean:
            continue
        payload = part.get_payload(decode=True)
        if not payload:
            continue
        try:
            b64 = base64.b64encode(payload).decode("ascii")
            data_url = f"data:{ct};base64,{b64}"
            cid_to_data[cid_clean] = data_url
            # Also allow matching with angle brackets in HTML (some clients use cid: <...>)
            cid_to_data[f"<{cid_clean}>"] = data_url
        except Exception:
            continue
    return cid_to_data


def _replace_cid_in_html(html: str, cid_map: dict[str, str]) -> str:
    """Replace src=\"cid:...\" in HTML with the corresponding data URL from cid_map."""
    if not cid_map:
        return html

    def repl(m: re.Match) -> str:
        cid_val = (m.group(1) or "").strip().strip("<>").strip()
        data = cid_map.get(cid_val) or cid_map.get(f"<{cid_val}>")
        if data:
            return f'src="{data}"'
        return m.group(0)

    # Match src="cid:value" or src='cid:value'
    return re.sub(r'src\s*=\s*["\']cid:([^"\']+)["\']', repl, html, flags=re.I)


def _has_attachment(msg) -> bool:
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_disposition() == "attachment":
                return True
    return False


def _decode_header_value(raw: str) -> str:
    parts = decode_email_header(raw)
    decoded = []
    for byt, charset in parts:
        if isinstance(byt, bytes):
            decoded.append(byt.decode(charset or "utf-8", errors="replace"))
        else:
            decoded.append(byt)
    return " ".join(decoded)


def _parse_address(addr_str: str) -> tuple[str, str]:
    if "<" in addr_str and ">" in addr_str:
        name = addr_str.split("<")[0].strip().strip('"')
        email_addr = addr_str.split("<")[1].split(">")[0].strip()
        return name, email_addr
    return "", addr_str.strip()


def _parse_date(date_str: str) -> datetime:
    from email.utils import parsedate_to_datetime
    try:
        return parsedate_to_datetime(date_str)
    except Exception:
        return datetime.now(timezone.utc)


def _delete_qdrant_chunks_for_mailbox(user_id: str, mailbox_id: str):
    try:
        qdrant = get_qdrant()
        qdrant.delete(
            collection_name=settings.QDRANT_COLLECTION_EMAIL_CHUNKS,
            points_selector=Filter(
                must=[
                    FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                    FieldCondition(key="mailbox_id", match=MatchValue(value=mailbox_id)),
                ]
            ),
        )
    except Exception:
        pass


def set_email_read_on_imap(user_id: str, mailbox_id: str, message_id: str, read: bool) -> bool:
    """Sync read/unread to the mailbox via IMAP \\Seen flag."""
    mb = mailboxes_col().find_one({"_id": ObjectId(mailbox_id), "user_id": user_id})
    if not mb:
        return False
    try:
        password = decrypt(mb["encrypted_password"])
        with IMAPClient(mb["imap_host"], port=mb["imap_port"], ssl=mb["imap_secure"]) as client:
            client.login(mb["username"], password)
            client.select_folder("INBOX")
            uids = _find_uids(client, message_id)
            if not uids:
                return False
            if read:
                client.add_flags(uids, [b"\\Seen"])
            else:
                client.remove_flags(uids, [b"\\Seen"])
            return True
    except Exception:
        return False


def _find_uids(client, message_id: str) -> list:
    """Resolve a message_id to IMAP UIDs."""
    stripped = message_id.strip("<>")
    if stripped.startswith("uid-") and stripped[4:].isdigit():
        return [int(stripped[4:])]
    return client.search(["HEADER", "Message-ID", message_id])


def set_email_starred_on_imap(user_id: str, mailbox_id: str, message_id: str, starred: bool) -> bool:
    """Sync starred status to the remote mailbox via IMAP \\Flagged flag."""
    mb = mailboxes_col().find_one({"_id": ObjectId(mailbox_id), "user_id": user_id})
    if not mb:
        return False
    try:
        password = decrypt(mb["encrypted_password"])
        with IMAPClient(mb["imap_host"], port=mb["imap_port"], ssl=mb["imap_secure"]) as client:
            client.login(mb["username"], password)
            client.select_folder("INBOX")
            uids = _find_uids(client, message_id)
            if not uids:
                return False
            if starred:
                client.add_flags(uids, [b"\\Flagged"])
            else:
                client.remove_flags(uids, [b"\\Flagged"])
            return True
    except Exception:
        return False


def _move_email_on_imap(user_id: str, mailbox_id: str, message_id: str, dest_folder: str) -> bool:
    """Move an email to a different IMAP folder (Archive, Trash, Spam, etc.)."""
    mb = mailboxes_col().find_one({"_id": ObjectId(mailbox_id), "user_id": user_id})
    if not mb:
        print(f"[IMAP MOVE] Mailbox not found: {mailbox_id}")
        return False
    try:
        password = decrypt(mb["encrypted_password"])
        with IMAPClient(mb["imap_host"], port=mb["imap_port"], ssl=mb["imap_secure"]) as client:
            client.login(mb["username"], password)
            client.select_folder("INBOX")
            uids = _find_uids(client, message_id)
            if not uids:
                print(f"[IMAP MOVE] No UIDs found for message_id={message_id}")
                return False
            folders = [f[2] for f in client.list_folders()]
            print(f"[IMAP MOVE] Available folders: {folders}")
            target = None
            for candidate in dest_folder.split("|"):
                candidate_lower = candidate.lower().strip()
                for f in folders:
                    f_lower = f.lower()
                    if (f_lower == candidate_lower
                            or f_lower.endswith("/" + candidate_lower)
                            or f_lower.endswith("." + candidate_lower)):
                        target = f
                        break
                if target:
                    break
            if not target:
                print(f"[IMAP MOVE] No matching folder for '{dest_folder}' among {folders}")
                return False
            print(f"[IMAP MOVE] Moving UIDs {uids} to '{target}'")
            client.move(uids, target)
            return True
    except Exception as e:
        print(f"[IMAP MOVE] Error: {e}")
        traceback.print_exc()
        return False


def archive_email_on_imap(user_id: str, mailbox_id: str, message_id: str) -> bool:
    return _move_email_on_imap(user_id, mailbox_id, message_id, "[Gmail]/All Mail|Archive|All Mail|INBOX.Archive")


def trash_email_on_imap(user_id: str, mailbox_id: str, message_id: str) -> bool:
    return _move_email_on_imap(user_id, mailbox_id, message_id, "[Gmail]/Trash|Trash|INBOX.Trash|Deleted Items|Deleted")


def spam_email_on_imap(user_id: str, mailbox_id: str, message_id: str) -> bool:
    return _move_email_on_imap(user_id, mailbox_id, message_id, "[Gmail]/Spam|Spam|Junk|INBOX.Spam|INBOX.Junk|Junk Email")


def _serialize(doc: dict) -> dict:
    return {
        "id": str(doc["_id"]),
        "name": doc["name"],
        "email": doc["email"],
        "color": doc.get("color", ""),
        "imap_host": doc["imap_host"],
        "imap_port": doc["imap_port"],
        "smtp_host": doc["smtp_host"],
        "smtp_port": doc["smtp_port"],
        "username": doc["username"],
        "last_sync_at": doc.get("last_sync_at"),
        "sync_status": doc.get("sync_status", "pending"),
        "created_at": doc.get("created_at"),
    }
