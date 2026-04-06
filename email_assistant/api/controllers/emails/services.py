import smtplib
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from bson import ObjectId
from django.conf import settings
from qdrant_client.models import Filter, FieldCondition, MatchValue

from database.db import email_metadata_col, mailboxes_col, follow_ups_col, email_attachments_col, get_qdrant
from api.utils.encryption import decrypt
from api.utils.email_body import clean_email_body
from api.utils.qdrant_helpers import get_email_content, get_emails_content_batch, get_email_ids_by_sender, scroll_all_chunk0
from api.controllers.mailboxes import services as mailbox_services


# ── List ─────────────────────────────────────────────────────────────────────

def email_stats(user_id: str) -> dict:
    """Return email counts directly from MongoDB — no Qdrant fetch, no limit cap."""
    base_q: dict = {"user_id": user_id, "archived": False, "trashed": False}
    now = datetime.now(timezone.utc)
    snooze_filter = {"$or": [{"snoozed_until": None}, {"snoozed_until": {"$lte": now}}]}

    start_of_today = now.replace(hour=0, minute=0, second=0, microsecond=0)
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
    label: str | None = None,
    folder: str | None = None,
    limit: int = 50,
    offset: int = 0,
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
        query = {"user_id": user_id, "archived": False, "trashed": False}
        query["$or"] = [
            {"snoozed_until": None},
            {"snoozed_until": {"$lte": now}},
        ]

    if mailbox_id:
        query["mailbox_id"] = mailbox_id
    if unread_only:
        query["read"] = False
    if label:
        query["labels"] = label

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

    result["thread_id"] = meta.get("thread_id")
    result["total_chunks"] = content.get("total_chunks", 0)
    result["sent_replies"] = meta.get("sent_replies", [])
    result["thread_replies"] = meta.get("thread_replies", [])
    return result


# ── Update metadata ──────────────────────────────────────────────────────────

def update_email(user_id: str, email_id: str, data: dict) -> dict | None:
    meta = email_metadata_col().find_one({"_id": email_id, "user_id": user_id})
    if not meta:
        return None
    email_metadata_col().update_one(
        {"_id": email_id, "user_id": user_id}, {"$set": data}
    )
    if "read" in data:
        mailbox_services.set_email_read_on_imap(
            user_id, meta["mailbox_id"], meta["message_id"], data["read"]
        )
        # Also mark all thread reply message_ids as read/unread on IMAP
        for tmid in meta.get("thread_message_ids", []):
            mailbox_services.set_email_read_on_imap(
                user_id, meta["mailbox_id"], tmid, data["read"]
            )
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
    msg.attach(MIMEText(data["body"], "plain"))

    _smtp_send(mb, msg)
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

    to_list = data.get("to") if isinstance(data.get("to"), list) else [content.get("from_email", "")]
    if not to_list and content.get("from_email"):
        to_list = [content["from_email"]]
    to_str = ", ".join(to_list) if to_list else ""
    subject = data.get("subject") or f"Re: {content.get('subject', '')}"

    msg = MIMEMultipart()
    msg["From"] = mb["email"]
    msg["To"] = to_str
    msg["Subject"] = subject
    msg["In-Reply-To"] = meta.get("message_id", "")
    msg.attach(MIMEText(data["body"], "plain"))

    _smtp_send(mb, msg)
    now_iso = datetime.now(timezone.utc).isoformat()
    sent_reply_doc = {
        "body": data["body"],
        "subject": subject,
        "to": to_list or [content.get("from_email", "")],
        "from_email": mb.get("email", ""),
        "date": now_iso,
    }
    email_metadata_col().update_one(
        {"_id": email_id, "user_id": user_id},
        {
            "$set": {"replied_at": now_iso},
            "$inc": {"reply_count": 1},
            "$push": {"sent_replies": sent_reply_doc},
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

    body = _reassemble_body(email_id, user_id)
    full_body = f"{data.get('body', '')}\n\n--- Forwarded message ---\n{body}"

    msg = MIMEMultipart()
    msg["From"] = mb["email"]
    msg["To"] = ", ".join(data["to"])
    msg["Subject"] = f"Fwd: {content.get('subject', '')}"
    msg.attach(MIMEText(full_body, "plain"))

    _smtp_send(mb, msg)
    return {"status": "sent", "to": data["to"], "subject": msg["Subject"]}


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
    """Delete all email state (MongoDB) and content (Qdrant) for the user."""
    result = email_metadata_col().delete_many({"user_id": user_id})
    deleted_count = result.deleted_count
    follow_ups_col().delete_many({"user_id": user_id})
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


def _smtp_send(mb: dict, msg):
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
        server.send_message(msg)
    finally:
        server.quit()


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


def _merge_email(mongo_doc: dict, qdrant_content: dict) -> dict:
    """Merge MongoDB mutable state with Qdrant immutable content."""
    priority = mongo_doc.get("priority") or qdrant_content.get("priority", "medium")
    return {
        "id": str(mongo_doc["_id"]),
        "mailbox_id": mongo_doc.get("mailbox_id", ""),
        "subject": qdrant_content.get("subject", ""),
        "from_name": qdrant_content.get("from_name", ""),
        "from_email": qdrant_content.get("from_email", ""),
        "to": qdrant_content.get("to", []),
        "date": mongo_doc.get("date"),
        "preview": qdrant_content.get("preview", ""),
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
    }
