import smtplib
import uuid
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import make_msgid

from bson import ObjectId
from django.conf import settings
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

from database.db import (
    email_metadata_col,
    mailboxes_col,
    follow_ups_col,
    email_attachments_col,
    meetings_col,
    get_qdrant,
)
from api.utils.encryption import decrypt
from api.utils.email_body import clean_email_body
from api.utils.qdrant_helpers import get_email_content, get_emails_content_batch, get_email_ids_by_sender, scroll_all_chunk0
from api.controllers.mailboxes import services as mailbox_services

_INBOX_PRESET_KEYS = frozenset({
    "today",
    "today_unread",
    "today_replied",
    "today_unreplied",
    "total_unread",
    "total_replied",
    "total_unreplied",
})


def _utc_start_of_day(dt: datetime) -> datetime:
    return dt.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)


def _today_received_filter(now: datetime) -> dict:
    """Emails received on the current UTC calendar day (matches list + stats)."""
    start = _utc_start_of_day(now)
    return {"$or": [
        {"original_date": {"$gte": start}},
        {"original_date": None, "date": {"$gte": start}},
    ]}


def _not_replied_filter() -> dict:
    return {"$or": [{"replied_at": None}, {"replied_at": {"$exists": False}}]}


def _apply_inbox_preset_to_query(query: dict, preset: str, now: datetime) -> None:
    """Mutate inbox query with $and / read filters (same semantics as email_stats)."""
    ands: list = list(query.get("$and", []))
    today_f = _today_received_filter(now)
    if preset == "today":
        ands.append(today_f)
    elif preset == "today_unread":
        ands.append(today_f)
        query["read"] = False
    elif preset == "today_replied":
        ands.append(today_f)
        ands.append({"replied_at": {"$ne": None}})
    elif preset == "today_unreplied":
        ands.append(today_f)
        ands.append(_not_replied_filter())
    elif preset == "total_unread":
        query["read"] = False
    elif preset == "total_replied":
        ands.append({"replied_at": {"$ne": None}})
    elif preset == "total_unreplied":
        ands.append(_not_replied_filter())
    query["$and"] = ands


# ── List ─────────────────────────────────────────────────────────────────────

def email_stats(user_id: str, mailbox_id: str | None = None) -> dict:
    """Return email counts directly from MongoDB — no Qdrant fetch, no limit cap."""
    base_q: dict = {"user_id": user_id, "archived": False, "trashed": False}
    if mailbox_id:
        base_q["mailbox_id"] = mailbox_id
    now = datetime.now(timezone.utc)
    snooze_filter = {"$or": [{"snoozed_until": None}, {"snoozed_until": {"$lte": now}}]}

    start_of_today = _utc_start_of_day(now)
    today_date_filter = {"$or": [
        {"original_date": {"$gte": start_of_today}},
        {"original_date": None, "date": {"$gte": start_of_today}},
    ]}
    today_q = {"$and": [base_q, snooze_filter, today_date_filter]}
    all_q = {**base_q, **snooze_filter}

    col = email_metadata_col()
    grand_total = col.count_documents(all_q)
    total_unread = col.count_documents({**all_q, "read": False})
    total_replied = col.count_documents({**all_q, "replied_at": {"$ne": None}})
    total_unreplied = grand_total - total_replied

    today_total = col.count_documents(today_q)
    today_unread = col.count_documents({**today_q, "read": False})
    today_replied = col.count_documents({**today_q, "replied_at": {"$ne": None}})
    today_unreplied = today_total - today_replied

    # Total number of reply actions (sum of reply_count); 2 replies on same email = 2
    total_replies_sent_cursor = col.aggregate([
        {"$match": all_q},
        {"$group": {"_id": None, "n": {"$sum": {"$ifNull": ["$reply_count", 0]}}}},
    ])
    total_replies_sent = (next(total_replies_sent_cursor, None) or {}).get("n", 0)
    today_replies_sent_cursor = col.aggregate([
        {"$match": today_q},
        {"$group": {"_id": None, "n": {"$sum": {"$ifNull": ["$reply_count", 0]}}}},
    ])
    today_replies_sent = (next(today_replies_sent_cursor, None) or {}).get("n", 0)

    return {
        "grand_total": grand_total,
        "total_unread": total_unread,
        "total_replied": total_replied,
        "total_unreplied": total_unreplied,
        "today_total": today_total,
        "today_unread": today_unread,
        "today_replied": today_replied,
        "today_unreplied": today_unreplied,
        "total_replies_sent": total_replies_sent,
        "today_replies_sent": today_replies_sent,
    }


def unique_senders(user_id: str, mailbox_id: str | None = None) -> dict:
    """Inbox (non-archived, non-trashed) ke unique senders ka count aur list.
    from_email lives in Qdrant payloads, not in MongoDB, so we use Qdrant + inbox IDs from MongoDB."""
    base_q: dict = {"user_id": user_id, "archived": False, "trashed": False}
    now = datetime.now(timezone.utc)
    base_q["$or"] = [{"snoozed_until": None}, {"snoozed_until": {"$lte": now}}]
    if mailbox_id:
        base_q["mailbox_id"] = mailbox_id

    col = email_metadata_col()
    inbox_ids = {str(d["_id"]) for d in col.find(base_q, {"_id": 1})}
    if not inbox_ids:
        return {"unique_senders_count": 0, "senders": []}

    # from_email is in Qdrant; get chunk0 content for user/mailbox and filter to inbox
    all_content = scroll_all_chunk0(user_id, mailbox_id=mailbox_id)
    inbox_contents = [c for c in all_content if c.get("email_id") in inbox_ids]

    # Aggregate by sender (case-insensitive); track latest email date per sender
    sender_map: dict = {}  # key: lower_email, value: {from_email, from_name, count, last_date}
    for c in inbox_contents:
        fe = (c.get("from_email") or "").strip()
        fn = (c.get("from_name") or "").strip()
        key = fe.lower() if fe else "__unknown__"
        if key not in sender_map:
            sender_map[key] = {"from_email": fe or "(unknown)", "from_name": fn or fe or "(unknown)", "count": 0, "last_date": None}
        sender_map[key]["count"] += 1
        date_val = c.get("date") or ""
        if date_val:
            current = sender_map[key]["last_date"]
            sender_map[key]["last_date"] = max(current or "", date_val) if current else date_val

    senders = sorted(sender_map.values(), key=lambda x: -x["count"])
    return {
        "unique_senders_count": len(senders),
        "senders": senders,
    }


def folder_counts(user_id: str, mailbox_id: str | None = None) -> dict:
    """Return email counts per virtual folder."""
    col = email_metadata_col()
    base: dict = {"user_id": user_id}
    if mailbox_id:
        base["mailbox_id"] = mailbox_id
    now = datetime.now(timezone.utc)

    inbox_q = {**base, "archived": False, "trashed": False,
               "$or": [{"snoozed_until": None}, {"snoozed_until": {"$lte": now}}]}
    sent_q = {**base, "is_sent": True}
    trash_q = {**base, "trashed": True, "$or": [{"spam": {"$ne": True}}, {"spam": {"$exists": False}}]}
    spam_q = {**base, "spam": True}
    snoozed_q = {**base, "snoozed_until": {"$gt": now}, "trashed": False}
    archive_q = {**base, "archived": True, "trashed": False}
    star_q = {**base, "starred": True, "trashed": False}

    return {
        "inbox": col.count_documents(inbox_q),
        "sent": col.count_documents(sent_q),
        "trash": col.count_documents(trash_q),
        "archive": col.count_documents(archive_q),
        "star": col.count_documents(star_q),
        "spam": col.count_documents(spam_q),
        "snoozed": col.count_documents(snoozed_q),
    }


def list_emails(
    user_id: str,
    mailbox_id: str | None = None,
    category: str | None = None,
    unread_only: bool = False,
    from_email: str | None = None,
    subject: str | None = None,
    keywords: str | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    label: str | None = None,
    folder: str | None = None,
    limit: int = 50,
    offset: int = 0,
    inbox_preset: str | None = None,
) -> list[dict]:
    now = datetime.now(timezone.utc)

    if folder == "trash":
        query: dict = {"user_id": user_id, "trashed": True,
                       "$or": [{"spam": {"$ne": True}}, {"spam": {"$exists": False}}]}
    elif folder == "spam":
        query = {"user_id": user_id, "spam": True}
    elif folder == "sent":
        query = {"user_id": user_id, "is_sent": True}
    elif folder == "archive":
        query = {"user_id": user_id, "archived": True, "trashed": False}
    elif folder == "star":
        query = {"user_id": user_id, "starred": True, "trashed": False}
    elif folder == "snoozed":
        query = {"user_id": user_id, "trashed": False, "snoozed_until": {"$gt": now}}
    else:
        query = {"user_id": user_id, "archived": False, "trashed": False,
                 "$or": [{"is_sent": {"$ne": True}}, {"is_sent": {"$exists": False}}]}
        query["$and"] = [
            {"$or": query.pop("$or")},
            {"$or": [{"snoozed_until": None}, {"snoozed_until": {"$lte": now}}]},
        ]

    if mailbox_id:
        query["mailbox_id"] = mailbox_id

    allow_inbox_preset = folder in (None, "", "inbox")
    preset_key = (inbox_preset or "").strip()
    if preset_key in _INBOX_PRESET_KEYS and allow_inbox_preset:
        _apply_inbox_preset_to_query(query, preset_key, now)
    elif unread_only:
        query["read"] = False

    if label:
        query["labels"] = label

    if subject and subject.strip():
        query["subject"] = {"$regex": subject.strip(), "$options": "i"}

    if keywords and keywords.strip():
        kw_or = {
            "$or": [
                {"subject": {"$regex": keywords.strip(), "$options": "i"}},
                {"preview": {"$regex": keywords.strip(), "$options": "i"}},
            ]
        }
        existing_and = list(query.get("$and", []))
        existing_and.append(kw_or)
        query["$and"] = existing_and

    if date_from or date_to:
        date_range: dict = {}
        if date_from:
            date_range["$gte"] = date_from
        if date_to:
            date_range["$lte"] = date_to
        query["date"] = date_range

    # Pre-filter email IDs from Qdrant when filtering by category or sender
    id_filter: list[str] | None = None
    if category:
        qdrant_emails = scroll_all_chunk0(user_id, mailbox_id=mailbox_id, category=category)
        id_filter = [e["email_id"] for e in qdrant_emails if e.get("email_id")]
    sender_email = (from_email or "").strip()
    if sender_email:
        sender_ids = get_email_ids_by_sender(user_id, sender_email, mailbox_id)
        id_filter = sender_ids if id_filter is None else [eid for eid in id_filter if eid in set(sender_ids)]
    if id_filter is not None:
        if not id_filter:
            return []
        query["_id"] = {"$in": id_filter}

    cursor = (
        email_metadata_col()
        .find(query)
        .sort("date", -1)
        .skip(offset)
        .limit(limit)
    )
    mongo_docs = list(cursor)
    if not mongo_docs:
        return []

    email_ids = [str(d["_id"]) for d in mongo_docs]
    content_map = get_emails_content_batch(email_ids, user_id)

    results = []
    for doc in mongo_docs:
        eid = str(doc["_id"])
        content = content_map.get(eid, {})
        results.append(_merge_email(doc, content))

    return results


# ── Detail (body from Qdrant) ────────────────────────────────────────────────

def get_email(user_id: str, email_id: str) -> dict | None:
    meta = email_metadata_col().find_one({"_id": email_id, "user_id": user_id})
    if not meta:
        return None

    try:
        content = get_email_content(email_id, user_id) or {}
    except Exception:
        content = {}

    result = _merge_email(meta, content)

    body_html = content.get("body_html", "")
    if body_html:
        result["body"] = body_html
        result["body_is_html"] = True
    else:
        try:
            body = _reassemble_body(email_id, user_id)
            result["body"] = clean_email_body(body) if body else body or ""
        except Exception:
            result["body"] = ""
        result["body_is_html"] = False

    # Fallback for compose-sent emails (body stored in MongoDB, not Qdrant)
    if not result.get("body") and meta.get("sent_body"):
        result["body"] = meta["sent_body"]
        result["body_is_html"] = False

    result["thread_id"] = meta.get("thread_id")
    result["total_chunks"] = content.get("total_chunks", 0)
    result["sent_replies"] = meta.get("sent_replies", [])
    result["thread_replies"] = meta.get("thread_replies", [])
    return result


# ── Update metadata ──────────────────────────────────────────────────────────

def _sync_read_to_imap(user_id: str, meta: dict, read: bool) -> None:
    mailbox_services.set_email_read_on_imap(
        user_id, meta["mailbox_id"], meta["message_id"], read
    )
    for tmid in meta.get("thread_message_ids", []):
        mailbox_services.set_email_read_on_imap(
            user_id, meta["mailbox_id"], tmid, read
        )


def mark_all_inbox_read(user_id: str, mailbox_id: str | None = None) -> dict:
    """Mark every unread message in the inbox as read (MongoDB + IMAP per message)."""
    query = _inbox_scope_query(user_id, mailbox_id)
    query["read"] = False

    col = email_metadata_col()
    marked = 0
    failed = 0
    for meta in col.find(query):
        try:
            col.update_one(
                {"_id": meta["_id"], "user_id": user_id},
                {"$set": {"read": True}},
            )
            _sync_read_to_imap(user_id, meta, True)
            marked += 1
        except Exception:
            failed += 1
    return {"marked": marked, "failed": failed}


def _inbox_scope_query(user_id: str, mailbox_id: str | None) -> dict:
    """Same inbox scope as list_emails (default folder): non-archived, non-trash, not sent-only, snooze not hiding."""
    now = datetime.now(timezone.utc)
    query: dict = {
        "user_id": user_id,
        "archived": False,
        "trashed": False,
        "$or": [{"is_sent": {"$ne": True}}, {"is_sent": {"$exists": False}}],
    }
    query["$and"] = [
        {"$or": query.pop("$or")},
        {"$or": [{"snoozed_until": None}, {"snoozed_until": {"$lte": now}}]},
    ]
    if mailbox_id:
        query["mailbox_id"] = mailbox_id
    return query


def mark_all_inbox_unread(user_id: str, mailbox_id: str | None = None) -> dict:
    """Mark every currently-read inbox message as unread (MongoDB + IMAP per message)."""
    q = _inbox_scope_query(user_id, mailbox_id)
    q["read"] = True

    col = email_metadata_col()
    marked = 0
    failed = 0
    for meta in col.find(q):
        try:
            col.update_one(
                {"_id": meta["_id"], "user_id": user_id},
                {"$set": {"read": False}},
            )
            _sync_read_to_imap(user_id, meta, False)
            marked += 1
        except Exception:
            failed += 1
    return {"marked": marked, "failed": failed}


def update_email(user_id: str, email_id: str, data: dict) -> dict | None:
    meta = email_metadata_col().find_one({"_id": email_id, "user_id": user_id})
    if not meta:
        return None
    email_metadata_col().update_one(
        {"_id": email_id, "user_id": user_id}, {"$set": data}
    )
    if "read" in data:
        _sync_read_to_imap(user_id, meta, data["read"])
    if "starred" in data:
        mailbox_services.set_email_starred_on_imap(
            user_id, meta["mailbox_id"], meta["message_id"], data["starred"]
        )
    return get_email(user_id, email_id)


def snooze_email(user_id: str, email_id: str, hours: int) -> dict | None:
    until = datetime.now(timezone.utc) + timedelta(hours=hours)
    email_metadata_col().update_one(
        {"_id": email_id, "user_id": user_id},
        {"$set": {"snoozed_until": until}},
    )
    return get_email(user_id, email_id)


def archive_email(user_id: str, email_id: str) -> bool:
    meta = email_metadata_col().find_one({"_id": email_id, "user_id": user_id})
    if not meta:
        return False
    r = email_metadata_col().update_one(
        {"_id": email_id, "user_id": user_id}, {"$set": {"archived": True}}
    )
    if r.modified_count > 0:
        mailbox_services.archive_email_on_imap(
            user_id, meta["mailbox_id"], meta["message_id"]
        )
    return r.modified_count > 0


def trash_email(user_id: str, email_id: str) -> bool:
    meta = email_metadata_col().find_one({"_id": email_id, "user_id": user_id})
    if not meta:
        return False
    r = email_metadata_col().update_one(
        {"_id": email_id, "user_id": user_id}, {"$set": {"trashed": True}}
    )
    if r.modified_count > 0:
        mailbox_services.trash_email_on_imap(
            user_id, meta["mailbox_id"], meta["message_id"]
        )
    return r.modified_count > 0


def move_email_to_inbox(user_id: str, email_id: str) -> bool:
    meta = email_metadata_col().find_one({"_id": email_id, "user_id": user_id})
    if not meta:
        return False
    r = email_metadata_col().update_one(
        {"_id": email_id, "user_id": user_id},
        {"$set": {"trashed": False, "archived": False, "spam": False}},
    )
    if r.modified_count > 0:
        mailbox_services.move_to_inbox_on_imap(
            user_id, meta["mailbox_id"], meta["message_id"]
        )
    return r.modified_count > 0


def spam_email(user_id: str, email_id: str) -> bool:
    meta = email_metadata_col().find_one({"_id": email_id, "user_id": user_id})
    if not meta:
        return False
    r = email_metadata_col().update_one(
        {"_id": email_id, "user_id": user_id}, {"$set": {"trashed": True, "spam": True}}
    )
    if r.modified_count > 0:
        mailbox_services.spam_email_on_imap(
            user_id, meta["mailbox_id"], meta["message_id"]
        )
    return r.modified_count > 0


def delete_email(user_id: str, email_id: str) -> bool:
    """Permanently delete an email and related local records."""
    meta = email_metadata_col().find_one({"_id": email_id, "user_id": user_id})
    if not meta:
        return False

    mailbox_services.trash_email_on_imap(
        user_id, meta["mailbox_id"], meta["message_id"]
    )

    email_metadata_col().delete_one({"_id": email_id, "user_id": user_id})
    email_attachments_col().delete_many({"email_id": email_id, "user_id": user_id})
    follow_ups_col().delete_many({"email_id": email_id, "user_id": user_id})
    meetings_col().delete_many({"email_id": email_id, "user_id": user_id})

    try:
        get_qdrant().delete(
            collection_name=settings.QDRANT_COLLECTION_EMAIL_CHUNKS,
            points_selector=Filter(
                must=[
                    FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                    FieldCondition(key="email_id", match=MatchValue(value=email_id)),
                ]
            ),
        )
    except Exception:
        pass
    return True


# ── Bulk actions (one request → one DB call + one IMAP session per mailbox) ──
#
# These replace the old pattern of issuing N individual HTTP requests from
# the frontend (each doing its own Mongo lookup + IMAP login/select/search).
# A bulk archive of 50 emails now becomes:
#   * 1 metadata batch find_one  →  `find({_id: {$in: [...]}})`
#   * 1 Mongo `update_many`
#   * 1 IMAP session per mailbox (usually 1), with 1 `move`
# Instead of 50 × (login + select + search + move) round-trips.


def _group_metas_by_mailbox(metas: list[dict]) -> dict[str, list[str]]:
    """From a list of email metadata docs, return {mailbox_id: [message_id...]}."""
    by_mailbox: dict[str, list[str]] = {}
    for m in metas:
        mb_id = str(m.get("mailbox_id") or "")
        mid = m.get("message_id")
        if not mb_id or not mid:
            continue
        by_mailbox.setdefault(mb_id, []).append(mid)
        for tmid in m.get("thread_message_ids", []) or []:
            if tmid:
                by_mailbox[mb_id].append(tmid)
    return by_mailbox


def _fetch_metas_for_bulk(user_id: str, email_ids: list[str]) -> tuple[list[dict], list[str]]:
    """Returns (matching_metas, failed_ids) for a bulk operation."""
    if not email_ids:
        return [], []
    metas = list(email_metadata_col().find(
        {"_id": {"$in": email_ids}, "user_id": user_id}
    ))
    found = {m["_id"] for m in metas}
    failed = [eid for eid in email_ids if eid not in found]
    return metas, failed


def bulk_update_emails(user_id: str, email_ids: list[str], data: dict) -> dict:
    """Bulk patch `read` and/or `starred` across many emails.

    `data` accepts the same keys as `update_email` (`read`, `starred`).
    """
    metas, failed = _fetch_metas_for_bulk(user_id, email_ids)
    if not metas:
        return {"processed": 0, "failed": failed}

    update_fields: dict = {}
    if "read" in data:
        update_fields["read"] = bool(data["read"])
    if "starred" in data:
        update_fields["starred"] = bool(data["starred"])
    if not update_fields:
        return {"processed": 0, "failed": email_ids}

    ids = [m["_id"] for m in metas]
    email_metadata_col().update_many(
        {"_id": {"$in": ids}, "user_id": user_id},
        {"$set": update_fields},
    )

    by_mailbox = _group_metas_by_mailbox(metas)
    for mb_id, mids in by_mailbox.items():
        if "read" in update_fields:
            mailbox_services.bulk_set_flag_on_imap(
                user_id, mb_id, mids, b"\\Seen", update_fields["read"]
            )
        if "starred" in update_fields:
            mailbox_services.bulk_set_flag_on_imap(
                user_id, mb_id, mids, b"\\Flagged", update_fields["starred"]
            )
    return {"processed": len(ids), "failed": failed}


def bulk_archive_emails(user_id: str, email_ids: list[str]) -> dict:
    metas, failed = _fetch_metas_for_bulk(user_id, email_ids)
    if not metas:
        return {"processed": 0, "failed": failed}
    ids = [m["_id"] for m in metas]
    email_metadata_col().update_many(
        {"_id": {"$in": ids}, "user_id": user_id},
        {"$set": {"archived": True}},
    )
    by_mailbox = _group_metas_by_mailbox(metas)
    for mb_id, mids in by_mailbox.items():
        mailbox_services.bulk_archive_on_imap(user_id, mb_id, mids)
    return {"processed": len(ids), "failed": failed}


def bulk_trash_emails(user_id: str, email_ids: list[str]) -> dict:
    metas, failed = _fetch_metas_for_bulk(user_id, email_ids)
    if not metas:
        return {"processed": 0, "failed": failed}
    ids = [m["_id"] for m in metas]
    email_metadata_col().update_many(
        {"_id": {"$in": ids}, "user_id": user_id},
        {"$set": {"trashed": True}},
    )
    by_mailbox = _group_metas_by_mailbox(metas)
    for mb_id, mids in by_mailbox.items():
        mailbox_services.bulk_trash_on_imap(user_id, mb_id, mids)
    return {"processed": len(ids), "failed": failed}


def bulk_spam_emails(user_id: str, email_ids: list[str]) -> dict:
    metas, failed = _fetch_metas_for_bulk(user_id, email_ids)
    if not metas:
        return {"processed": 0, "failed": failed}
    ids = [m["_id"] for m in metas]
    email_metadata_col().update_many(
        {"_id": {"$in": ids}, "user_id": user_id},
        {"$set": {"trashed": True, "spam": True}},
    )
    by_mailbox = _group_metas_by_mailbox(metas)
    for mb_id, mids in by_mailbox.items():
        mailbox_services.bulk_spam_on_imap(user_id, mb_id, mids)
    return {"processed": len(ids), "failed": failed}


def bulk_move_to_inbox_emails(user_id: str, email_ids: list[str]) -> dict:
    metas, failed = _fetch_metas_for_bulk(user_id, email_ids)
    if not metas:
        return {"processed": 0, "failed": failed}
    ids = [m["_id"] for m in metas]
    email_metadata_col().update_many(
        {"_id": {"$in": ids}, "user_id": user_id},
        {"$set": {"trashed": False, "archived": False, "spam": False}},
    )
    by_mailbox = _group_metas_by_mailbox(metas)
    for mb_id, mids in by_mailbox.items():
        mailbox_services.bulk_move_to_inbox_on_imap(user_id, mb_id, mids)
    return {"processed": len(ids), "failed": failed}


def bulk_snooze_emails(user_id: str, email_ids: list[str], hours: int) -> dict:
    """Snooze many emails until `now + hours` in a single Mongo call."""
    if not email_ids:
        return {"processed": 0, "failed": []}
    until = datetime.now(timezone.utc) + timedelta(hours=hours)
    res = email_metadata_col().update_many(
        {"_id": {"$in": email_ids}, "user_id": user_id},
        {"$set": {"snoozed_until": until}},
    )
    matched = res.matched_count
    failed = [] if matched == len(email_ids) else [
        eid for eid in email_ids
        if not email_metadata_col().find_one(
            {"_id": eid, "user_id": user_id}, {"_id": 1}
        )
    ]
    return {"processed": matched, "failed": failed}


def bulk_delete_emails(user_id: str, email_ids: list[str]) -> dict:
    """Permanently delete many emails + related records (single IMAP session
    per mailbox, batched Mongo + Qdrant deletes)."""
    metas, failed = _fetch_metas_for_bulk(user_id, email_ids)
    if not metas:
        return {"processed": 0, "failed": failed}

    by_mailbox = _group_metas_by_mailbox(metas)
    for mb_id, mids in by_mailbox.items():
        mailbox_services.bulk_trash_on_imap(user_id, mb_id, mids)

    ids = [m["_id"] for m in metas]
    email_metadata_col().delete_many({"_id": {"$in": ids}, "user_id": user_id})
    email_attachments_col().delete_many({"email_id": {"$in": ids}, "user_id": user_id})
    follow_ups_col().delete_many({"email_id": {"$in": ids}, "user_id": user_id})
    meetings_col().delete_many({"email_id": {"$in": ids}, "user_id": user_id})

    try:
        get_qdrant().delete(
            collection_name=settings.QDRANT_COLLECTION_EMAIL_CHUNKS,
            points_selector=Filter(
                must=[
                    FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                    FieldCondition(key="email_id", match=MatchAny(any=ids)),
                ]
            ),
        )
    except Exception:
        pass

    return {"processed": len(ids), "failed": failed}


# ── Detected meetings (user must approve each one) ───────────────────────────

def list_emails_with_meetings(
    user_id: str,
    mailbox_id: str | None = None,
    status_filter: str | None = None,
) -> dict:
    """Return received emails that have a detected meeting.

    `status_filter`: None (all), "pending", "added", or "dismissed".
    Result includes per-status counts so the UI can show a pending badge.
    """
    col = email_metadata_col()
    base: dict = {
        "user_id": user_id,
        "archived": False,
        "trashed": False,
        "$or": [{"is_sent": {"$ne": True}}, {"is_sent": {"$exists": False}}],
        "detected_meeting": {"$exists": True, "$ne": None},
    }
    if mailbox_id:
        base["mailbox_id"] = mailbox_id

    pending_count = col.count_documents({**base, "meeting_status": "pending"})
    added_count = col.count_documents({**base, "meeting_status": "added"})
    dismissed_count = col.count_documents({**base, "meeting_status": "dismissed"})
    total_count = col.count_documents(base)

    q = dict(base)
    sf = (status_filter or "").strip().lower()
    if sf in ("pending", "added", "dismissed"):
        q["meeting_status"] = sf

    docs = list(col.find(q).sort("date", -1).limit(200))
    if not docs:
        return {
            "emails": [],
            "total": total_count,
            "pending": pending_count,
            "added": added_count,
            "dismissed": dismissed_count,
        }

    email_ids = [str(d["_id"]) for d in docs]
    content_map = get_emails_content_batch(email_ids, user_id)
    rows = [_merge_email(d, content_map.get(str(d["_id"]), {})) for d in docs]
    return {
        "emails": rows,
        "total": total_count,
        "pending": pending_count,
        "added": added_count,
        "dismissed": dismissed_count,
    }


def add_detected_meeting_to_calendar(user_id: str, email_id: str) -> dict | None:
    """Promote an email's detected meeting to a real calendar event.

    Returns {"meeting": {...}, "email": {...}} on success, or None if the
    email is missing or has no detection to promote.
    """
    from api.controllers.calendar.services import upsert_meeting_from_email

    meta = email_metadata_col().find_one({"_id": email_id, "user_id": user_id})
    if not meta:
        return None
    dm = meta.get("detected_meeting")
    if not dm or not isinstance(dm, dict):
        return None

    payload = {
        "title": dm.get("title") or meta.get("subject") or "Meeting",
        "start": dm.get("start"),
        "end": dm.get("end"),
        "location": dm.get("location"),
        "attendees": dm.get("attendees") or [],
    }
    meeting = upsert_meeting_from_email(
        user_id,
        email_id,
        payload,
        mailbox_id=meta.get("mailbox_id"),
    )
    if not meeting:
        return None

    email_metadata_col().update_one(
        {"_id": email_id, "user_id": user_id},
        {"$set": {"meeting_status": "added", "meeting_id": meeting.get("id")}},
    )
    refreshed = get_email(user_id, email_id)
    return {"meeting": meeting, "email": refreshed}


def dismiss_detected_meeting(user_id: str, email_id: str) -> dict | None:
    """Mark a detected meeting as dismissed so the banner no longer appears."""
    res = email_metadata_col().update_one(
        {
            "_id": email_id,
            "user_id": user_id,
            "detected_meeting": {"$exists": True, "$ne": None},
        },
        {"$set": {"meeting_status": "dismissed"}},
    )
    if not res.matched_count:
        return None
    return get_email(user_id, email_id)


# ── Send / Reply / Forward via SMTP ─────────────────────────────────────────

def send_email(user_id: str, data: dict) -> dict:
    mb = mailboxes_col().find_one({"_id": ObjectId(data["mailbox_id"]), "user_id": user_id})
    if not mb:
        raise ValueError("Mailbox not found")

    msg = MIMEMultipart()
    msg["From"] = mb["email"]
    msg["To"] = ", ".join(data["to"])
    if data.get("cc"):
        msg["Cc"] = ", ".join(data["cc"])
    msg["Subject"] = data["subject"]
    msg["Message-ID"] = _sender_msgid(mb)
    msg.attach(MIMEText(data["body"], "plain"))

    _smtp_send(mb, msg)
    mailbox_services.append_sent_to_imap(user_id, data["mailbox_id"], msg.as_bytes())

    sent_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    raw_body = data.get("body") or ""
    body_preview = raw_body[:300] or "(no preview)"
    to_list = data["to"] if isinstance(data["to"], list) else [data["to"]]
    email_metadata_col().insert_one({
        "_id": sent_id,
        "user_id": user_id,
        "mailbox_id": data["mailbox_id"],
        "message_id": f"<sent-{sent_id}>",
        "thread_id": f"<sent-{sent_id}>",
        "in_reply_to": "",
        "subject": data["subject"],
        "from_name": mb.get("name", ""),
        "from_email": mb["email"],
        "to": [{"name": "", "email": e} for e in to_list],
        "preview": body_preview,
        "sent_body": raw_body,
        "date": now,
        "original_date": now,
        "read": True,
        "starred": False,
        "replied_at": None,
        "reply_count": 0,
        "labels": [],
        "priority": "medium",
        "snoozed_until": None,
        "archived": False,
        "trashed": False,
        "is_sent": True,
        "created_at": now,
    })

    try:
        from api.controllers.settings.services import get_settings as _get_settings
        from api.utils.classify import assign_labels_batch

        _settings = _get_settings(user_id)
        if _settings.get("auto_labeling", True):
            _rules = _settings.get("ai_label_rules") or []
            if _rules:
                sent_labels = assign_labels_batch(
                    [
                        {
                            "subject": data["subject"],
                            "from_name": mb.get("name", ""),
                            "from_email": mb["email"],
                            "preview": body_preview,
                        }
                    ],
                    _rules,
                )
                if sent_labels and sent_labels[0]:
                    email_metadata_col().update_one(
                        {"_id": sent_id, "user_id": user_id},
                        {"$set": {"labels": sent_labels[0]}},
                    )
    except Exception:
        pass

    return {"status": "sent", "to": data["to"], "subject": data["subject"]}


def reply_email(user_id: str, email_id: str, data: dict) -> dict:
    meta = email_metadata_col().find_one({"_id": email_id, "user_id": user_id})
    if not meta:
        raise ValueError("Email not found")

    try:
        content = get_email_content(email_id, user_id) or {}
    except Exception:
        content = {}

    mb = mailboxes_col().find_one({"_id": ObjectId(data["mailbox_id"]), "user_id": user_id})
    if not mb:
        raise ValueError("Mailbox not found")

    raw_to = data.get("to")
    if isinstance(raw_to, str):
        to_list = [raw_to]
    elif isinstance(raw_to, list):
        to_list = list(raw_to)
    else:
        to_list = []
    to_list = [t.strip() for t in to_list if t and str(t).strip()]
    if not to_list:
        fallback = (content.get("from_email") or meta.get("from_email") or "").strip()
        if fallback:
            to_list = [fallback]
    if not to_list:
        raise ValueError(
            "Cannot determine reply recipient — original sender email missing. "
            "Provide an explicit 'to' address."
        )
    to_str = ", ".join(to_list)

    cc_list = data.get("cc") or []
    if isinstance(cc_list, str):
        cc_list = [cc_list]
    cc_list = [c.strip() for c in cc_list if c and str(c).strip()]

    original_subject = content.get("subject", "") or meta.get("subject", "") or ""
    raw_subject = (data.get("subject") or "").strip() or original_subject
    subject = _ensure_prefix(raw_subject, "Re:")

    reply_body = (data.get("body") or "").strip()
    if not reply_body:
        raise ValueError("Reply body is required")
    final_body = _compose_reply_with_quote(reply_body, content, meta)

    parent_mid = meta.get("message_id", "")
    reply_msg_id = _sender_msgid(mb)
    msg = MIMEMultipart()
    msg["From"] = mb["email"]
    msg["To"] = to_str
    if cc_list:
        msg["Cc"] = ", ".join(cc_list)
    msg["Subject"] = subject
    msg["Message-ID"] = reply_msg_id
    if parent_mid:
        msg["In-Reply-To"] = parent_mid
        msg["References"] = parent_mid
    msg.attach(MIMEText(final_body, "plain"))

    _smtp_send(mb, msg)
    mailbox_services.append_sent_to_imap(user_id, data["mailbox_id"], msg.as_bytes())
    now_iso = datetime.now(timezone.utc).isoformat()
    sent_reply_doc = {
        "message_id": reply_msg_id.strip(),
        "body": reply_body,
        "subject": subject,
        "to": to_list,
        "cc": cc_list,
        "from_email": mb.get("email", ""),
        "date": now_iso,
    }
    email_metadata_col().update_one(
        {"_id": email_id, "user_id": user_id},
        {
            "$set": {"replied_at": now_iso},
            "$inc": {"reply_count": 1},
            "$push": {"sent_replies": sent_reply_doc},
            "$addToSet": {"thread_message_ids": reply_msg_id.strip()},
        },
    )
    return {"status": "sent", "to": sent_reply_doc["to"], "subject": subject, "sent_reply": sent_reply_doc}


def forward_email(user_id: str, email_id: str, data: dict) -> dict:
    meta = email_metadata_col().find_one({"_id": email_id, "user_id": user_id})
    if not meta:
        raise ValueError("Email not found")

    content = get_email_content(email_id, user_id) or {}

    mb = mailboxes_col().find_one({"_id": ObjectId(data["mailbox_id"]), "user_id": user_id})
    if not mb:
        raise ValueError("Mailbox not found")

    to_list = data.get("to") or []
    if isinstance(to_list, str):
        to_list = [to_list]
    to_list = [t.strip() for t in to_list if t and str(t).strip()]
    if not to_list:
        raise ValueError("Recipient (to) is required for forward_email")

    cc_list = data.get("cc") or []
    if isinstance(cc_list, str):
        cc_list = [cc_list]
    cc_list = [c.strip() for c in cc_list if c and str(c).strip()]

    original_subject = content.get("subject", "") or meta.get("subject", "") or ""
    raw_subject = (data.get("subject") or "").strip() or original_subject
    fwd_subject = _ensure_prefix(raw_subject, "Fwd:")

    body = _reassemble_body(email_id, user_id) or ""
    intro = (data.get("body") or "").strip()
    header_lines = [
        "--- Forwarded message ---",
        f"From: {content.get('from_name', '')} <{content.get('from_email', '')}>".strip(),
        f"Date: {meta.get('date', '')}",
        f"Subject: {original_subject}",
    ]
    to_field = content.get("to", []) or []
    if to_field:
        header_lines.append(
            "To: " + ", ".join(
                t.get("email", "") if isinstance(t, dict) else str(t)
                for t in to_field
            )
        )
    full_body = (
        (intro + "\n\n" if intro else "")
        + "\n".join(header_lines)
        + "\n\n"
        + body
    )

    msg = MIMEMultipart()
    msg["From"] = mb["email"]
    msg["To"] = ", ".join(to_list)
    if cc_list:
        msg["Cc"] = ", ".join(cc_list)
    msg["Subject"] = fwd_subject
    msg["Message-ID"] = _sender_msgid(mb)
    msg.attach(MIMEText(full_body, "plain"))

    _smtp_send(mb, msg)
    mailbox_services.append_sent_to_imap(user_id, data["mailbox_id"], msg.as_bytes())
    return {"status": "sent", "to": to_list, "cc": cc_list, "subject": fwd_subject}


# ── Delete conversation reply ─────────────────────────────────────────────────

def delete_thread_reply(user_id: str, email_id: str, reply_index: int) -> dict | None:
    """Remove a received thread reply by index."""
    meta = email_metadata_col().find_one({"_id": email_id, "user_id": user_id})
    if not meta:
        return None
    replies = meta.get("thread_replies", [])
    if reply_index < 0 or reply_index >= len(replies):
        return None
    replies.pop(reply_index)
    email_metadata_col().update_one(
        {"_id": email_id, "user_id": user_id},
        {"$set": {"thread_replies": replies}},
    )
    return get_email(user_id, email_id)


def delete_sent_reply(user_id: str, email_id: str, reply_index: int) -> dict | None:
    """Remove a sent reply by index."""
    meta = email_metadata_col().find_one({"_id": email_id, "user_id": user_id})
    if not meta:
        return None
    replies = meta.get("sent_replies", [])
    if reply_index < 0 or reply_index >= len(replies):
        return None
    replies.pop(reply_index)
    email_metadata_col().update_one(
        {"_id": email_id, "user_id": user_id},
        {"$set": {"sent_replies": replies}},
    )
    return get_email(user_id, email_id)


# ── Delete all ───────────────────────────────────────────────────────────────

def delete_all_emails(user_id: str) -> dict:
    """Delete all synced email data (MongoDB) and vectors (Qdrant) for the user.

    Mailboxes and account settings are unchanged. Removes attachments and
    email-derived calendar rows; manual meetings are kept.
    """
    from api.controllers.calendar.services import recompute_conflicts_for_user

    result = email_metadata_col().delete_many({"user_id": user_id})
    deleted_count = result.deleted_count
    follow_ups_col().delete_many({"user_id": user_id})
    email_attachments_col().delete_many({"user_id": user_id})
    meetings_col().delete_many(
        {
            "user_id": user_id,
            "$or": [
                {"source": "email"},
                {"email_id": {"$exists": True, "$nin": [None, ""]}},
            ],
        }
    )
    recompute_conflicts_for_user(user_id)
    try:
        qdrant = get_qdrant()
        qdrant.delete(
            collection_name=settings.QDRANT_COLLECTION_EMAIL_CHUNKS,
            points_selector=Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
            ),
        )
    except Exception:
        pass
    return {"deleted": deleted_count}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _reassemble_body(email_id: str, user_id: str) -> str:
    qdrant = get_qdrant()
    results = qdrant.scroll(
        collection_name=settings.QDRANT_COLLECTION_EMAIL_CHUNKS,
        scroll_filter=Filter(
            must=[
                FieldCondition(key="email_id", match=MatchValue(value=email_id)),
                FieldCondition(key="user_id", match=MatchValue(value=user_id)),
            ]
        ),
        limit=500,
        with_payload=True,
        with_vectors=False,
    )
    points = results[0] if results else []
    sorted_chunks = sorted(points, key=lambda p: p.payload.get("chunk_index", 0))
    return "\n".join(p.payload.get("body_chunk", "") for p in sorted_chunks)


def _ensure_prefix(subject: str, prefix: str) -> str:
    """Prepend ``prefix`` (e.g. 'Re:' / 'Fwd:') only if it isn't already there.

    Case-insensitive and tolerant of extra whitespace so we don't end up with
    'Re: Re: Re: ...' chains when replying to a reply, or 'Fwd: Fwd: ...' when
    forwarding an already-forwarded email.
    """
    subj = (subject or "").strip()
    low = subj.lower()
    p = prefix.strip().lower().rstrip(":")
    if low.startswith(p + ":") or low.startswith(p + " :"):
        return subj
    return f"{prefix.strip()} {subj}".strip()


def _compose_reply_with_quote(reply_body: str, content: dict, meta: dict) -> str:
    """Append a standard quoted block of the original email below the reply."""
    original_body = (content.get("body_chunk") or "").strip()
    if not original_body:
        return reply_body

    from_line = (
        f"{content.get('from_name', '')} <{content.get('from_email', '')}>".strip()
        or content.get("from_email", "")
        or "(unknown sender)"
    )
    date_val = meta.get("original_date") or meta.get("date")
    if isinstance(date_val, datetime):
        date_str = date_val.strftime("%a, %b %d, %Y at %I:%M %p")
    else:
        date_str = str(date_val or "")

    header = f"On {date_str}, {from_line} wrote:".strip()
    # Cap quoted original to keep SMTP payload small — most replies only need
    # context, not the entire thread history.
    quote_src = original_body[:4000]
    quoted = "\n".join(f"> {line}" for line in quote_src.splitlines())
    return f"{reply_body.strip()}\n\n{header}\n{quoted}"


def _smtp_send(mb: dict, msg):
    """Send an email via SMTP and raise on refused recipients.

    smtplib.send_message() returns a dict of recipients that were refused but
    it does NOT raise unless EVERY recipient was refused. We treat any refusal
    as a hard error so the user isn't told "sent" when the real recipient
    silently dropped off.
    """
    password = decrypt(mb["encrypted_password"])
    use_ssl = mb.get("smtp_secure", True)
    port = mb["smtp_port"]

    if use_ssl and port == 465:
        server = smtplib.SMTP_SSL(mb["smtp_host"], port)
    else:
        server = smtplib.SMTP(mb["smtp_host"], port)
        if use_ssl:
            server.starttls()

    try:
        server.login(mb["username"], password)
        refused = server.send_message(msg) or {}
    finally:
        server.quit()

    if refused:
        details = ", ".join(f"{r}: {v}" for r, v in refused.items())
        raise ValueError(f"SMTP refused recipient(s): {details}")


def _sender_msgid(mb: dict) -> str:
    """Generate a Message-ID using the sender's real domain.

    The default ``email.utils.make_msgid()`` uses ``@localhost`` which many
    spam filters (especially Gmail / Outlook / corporate filters) reject or
    flag. Using the authenticated mailbox domain dramatically improves
    deliverability and thread reassembly on the receiving side.
    """
    sender = (mb.get("email") or "").strip()
    domain = sender.split("@", 1)[1] if "@" in sender else "localhost"
    return make_msgid(domain=domain)


def get_attachment(user_id: str, email_id: str, attachment_index: int) -> dict | None:
    """Return attachment binary data for download."""
    meta = email_metadata_col().find_one({"_id": email_id, "user_id": user_id})
    if not meta:
        return None
    att = email_attachments_col().find_one({
        "email_id": email_id,
        "user_id": user_id,
        "index": attachment_index,
    })
    if not att:
        return None
    return {
        "filename": att["filename"],
        "content_type": att["content_type"],
        "data_b64": att["data_b64"],
    }


def diagnose_missing_attachment(user_id: str, email_id: str, attachment_index: int) -> str:
    """Return a human-readable reason why an attachment download failed."""
    meta = email_metadata_col().find_one({"_id": email_id, "user_id": user_id})
    if not meta:
        return "email_not_found"
    any_att = email_attachments_col().find_one({"email_id": email_id, "user_id": user_id})
    if not any_att:
        return "no_binary_data_stored_for_email"
    stored = list(email_attachments_col().find(
        {"email_id": email_id, "user_id": user_id}, {"index": 1, "filename": 1}
    ))
    indices = [s["index"] for s in stored]
    return f"index_{attachment_index}_missing (stored indices: {indices})"


def _serialize_detected_meeting(dm: dict | None) -> dict | None:
    """Turn stored detected_meeting (with datetime objects) into JSON-safe dict."""
    if not dm or not isinstance(dm, dict):
        return None
    start = dm.get("start")
    end = dm.get("end")
    if isinstance(start, datetime):
        start = start.isoformat()
    if isinstance(end, datetime):
        end = end.isoformat()
    if not start or not end:
        return None
    return {
        "title": dm.get("title") or "Meeting",
        "start": start,
        "end": end,
        "location": dm.get("location"),
        "attendees": dm.get("attendees") or [],
    }


def _merge_email(mongo_doc: dict, qdrant_content: dict) -> dict:
    """Merge MongoDB mutable state with Qdrant immutable content."""
    priority = mongo_doc.get("priority") or qdrant_content.get("priority", "medium")
    thread_count = 1 + len(mongo_doc.get("thread_replies", [])) + len(mongo_doc.get("sent_replies", []))
    od = mongo_doc.get("original_date")
    if isinstance(od, datetime):
        od = od.isoformat()
    return {
        "id": str(mongo_doc["_id"]),
        "mailbox_id": mongo_doc.get("mailbox_id", ""),
        "subject": qdrant_content.get("subject") or mongo_doc.get("subject", ""),
        "from_name": qdrant_content.get("from_name") or mongo_doc.get("from_name", ""),
        "from_email": qdrant_content.get("from_email") or mongo_doc.get("from_email", ""),
        "to": qdrant_content.get("to") or mongo_doc.get("to", []),
        "date": mongo_doc.get("date"),
        "original_date": od,
        "preview": qdrant_content.get("preview") or mongo_doc.get("preview", ""),
        "read": mongo_doc.get("read", False),
        "starred": mongo_doc.get("starred", False),
        "labels": mongo_doc.get("labels", []),
        "has_attachment": qdrant_content.get("has_attachment", False),
        "attachments": qdrant_content.get("attachments", []),
        "priority": priority,
        "category": qdrant_content.get("category"),
        "ai_summary": None,
        "sentiment_score": None,
        "replied_at": mongo_doc.get("replied_at"),
        "snoozed_until": mongo_doc.get("snoozed_until"),
        "thread_count": thread_count,
        "detected_meeting": _serialize_detected_meeting(mongo_doc.get("detected_meeting")),
        "meeting_status": mongo_doc.get("meeting_status") or None,
    }
