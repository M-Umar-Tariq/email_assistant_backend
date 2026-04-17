"""
Action executor for AI/voice actions.
Supports read/open/search actions, compose/reply/forward, and mailbox actions.
"""

import re
from datetime import datetime, timezone, timedelta

from database.db import mailboxes_col, follow_ups_col, email_metadata_col
from api.utils.qdrant_helpers import get_email_content, get_email_ids_by_sender
from api.utils.llm import chat


def execute_action(user_id: str, action: dict) -> dict:
    action_type = action.get("type", "")
    executors = {
        "read_emails": _exec_read_emails,
        "open_email": _exec_open_email,
        "open_latest_email": _exec_open_latest_email,
        "search_emails": _exec_search_emails,
        "send_email": _exec_send_email,
        "draft_email": _exec_draft_email,
        "draft_reply": _exec_draft_reply,
        "send_reply": _exec_send_reply,
        "reply_all": _exec_reply_all,
        "send_whatsapp": _exec_send_whatsapp,
        "set_reminder": _exec_set_reminder,
        "forward_email": _exec_forward_email,
        "move_to_trash": _exec_trash_email,
        "trash_email": _exec_trash_email,
        "delete_email": _exec_delete_email,
        "archive_email": _exec_archive_email,
        "mark_read": _exec_mark_read,
        "mark_all_read": _exec_mark_all_read,
        "mark_all_unread": _exec_mark_all_unread,
        "mark_unread": _exec_mark_unread,
        "snooze_email": _exec_snooze_email,
    }
    executor_fn = executors.get(action_type)
    if not executor_fn:
        return _log(user_id, action, "failed", f"Unknown action type: {action_type}")

    try:
        result = executor_fn(user_id, action)
        ok = result.get("marked", 1)
        status = "completed" if ok else "failed"
        log = _log(user_id, action, status, result.get("details", ""))
        for key in ("emails", "email", "draft", "marked", "failed"):
            if key in result:
                log[key] = result[key]
        return log
    except Exception as exc:
        return _log(user_id, action, "failed", str(exc))


# ── Bulk resolution ────────────────────────────────────────────────────────────


def _resolve_email_ids(user_id: str, action: dict) -> list[str]:
    """Resolve an action's target to concrete email IDs.

    Priority:
    1. Explicit ``email_id`` → single-item list.
    2. ``from_email`` → all inbox emails from that sender (via Qdrant).
    3. ``subject`` / ``keywords`` → regex match on MongoDB metadata.

    Returns an empty list when nothing matches.
    """
    single = (action.get("email_id") or "").strip()
    if single:
        col = email_metadata_col()
        if col.find_one({"_id": single, "user_id": user_id}, {"_id": 1}):
            return [single]
        cleaned = single.strip("`'\" ")
        if cleaned and col.find_one({"_id": cleaned, "user_id": user_id}, {"_id": 1}):
            return [cleaned]

    sender = (action.get("from_email") or "").strip()
    subj = (action.get("subject") or "").strip()
    kw = (action.get("keywords") or "").strip()
    mb = action.get("mailbox_id") or None

    candidate_ids: list[str] | None = None

    if sender:
        candidate_ids = get_email_ids_by_sender(user_id, sender, mb)

    query: dict = {"user_id": user_id, "archived": False, "trashed": False}
    if mb:
        query["mailbox_id"] = mb

    ands: list[dict] = []
    if subj:
        ands.append({"subject": {"$regex": re.escape(subj), "$options": "i"}})
    if kw:
        kw_safe = re.escape(kw)
        ands.append({"$or": [
            {"subject": {"$regex": kw_safe, "$options": "i"}},
            {"preview": {"$regex": kw_safe, "$options": "i"}},
        ]})
    if ands:
        query["$and"] = ands

    if candidate_ids is not None:
        if not candidate_ids:
            return []
        query["_id"] = {"$in": candidate_ids}

    if candidate_ids is None and not ands:
        return []

    col = email_metadata_col()
    return [str(d["_id"]) for d in col.find(query, {"_id": 1})]


# Read / Open / Search


def _exec_read_emails(user_id: str, action: dict) -> dict:
    from api.controllers.emails.services import list_emails

    date_from = _parse_iso_dt(action.get("date_from"))
    date_to = _parse_iso_dt(action.get("date_to"))
    items = list_emails(
        user_id=user_id,
        mailbox_id=action.get("mailbox_id"),
        unread_only=bool(action.get("unread_only", False)),
        from_email=action.get("from_email"),
        subject=action.get("subject"),
        keywords=action.get("keywords"),
        date_from=date_from,
        date_to=date_to,
        folder=action.get("folder"),
        limit=int(action.get("limit", 10)),
    )
    return {"details": f"Fetched {len(items)} emails", "emails": items}


def _exec_open_email(user_id: str, action: dict) -> dict:
    from api.controllers.emails.services import get_email

    email_id = (action.get("email_id") or "").strip()
    if not email_id:
        raise ValueError("email_id is required for open_email")
    result = get_email(user_id, email_id)
    if not result:
        raise ValueError("Email not found")
    return {"details": "Opened email", "email": result}


def _exec_open_latest_email(user_id: str, action: dict) -> dict:
    from api.controllers.emails.services import list_emails

    date_from = _parse_iso_dt(action.get("date_from"))
    date_to = _parse_iso_dt(action.get("date_to"))
    items = list_emails(
        user_id=user_id,
        mailbox_id=action.get("mailbox_id"),
        unread_only=bool(action.get("unread_only", False)),
        from_email=action.get("from_email"),
        subject=action.get("subject"),
        keywords=action.get("keywords"),
        date_from=date_from,
        date_to=date_to,
        folder=action.get("folder"),
        limit=1,
    )
    if not items:
        raise ValueError("No matching emails found")
    return {"details": "Opened latest email", "email": items[0]}


def _exec_search_emails(user_id: str, action: dict) -> dict:
    from api.controllers.search.services import search_emails

    query = (action.get("query") or action.get("keywords") or "").strip()
    if not query:
        raise ValueError("query is required for search_emails")
    items = search_emails(
        user_id=user_id,
        query=query,
        mailbox_id=action.get("mailbox_id"),
        limit=int(action.get("limit", 10)),
    )
    return {"details": f"Found {len(items)} emails", "emails": items}


# Send / Draft / Reply


def _exec_draft_email(user_id: str, action: dict) -> dict:
    to_field = action.get("to", [])
    if isinstance(to_field, str):
        to_field = [to_field]
    draft = {
        "to": [t.strip() for t in to_field if t and str(t).strip()],
        "cc": action.get("cc", []),
        "subject": action.get("subject", ""),
        "body": action.get("body", ""),
        "mailbox_id": action.get("mailbox_id") or _default_mailbox_id(user_id),
        "status": "draft",
    }
    return {"details": "Draft prepared", "draft": draft}


def _exec_send_email(user_id: str, action: dict) -> dict:
    from api.controllers.emails.services import send_email as smtp_send

    mailbox_id = action.get("mailbox_id") or _default_mailbox_id(user_id)
    to_field = action.get("to", [])
    if isinstance(to_field, str):
        to_field = [to_field]
    to_field = [t.strip() for t in to_field if t and str(t).strip()]
    if not to_field:
        raise ValueError("Recipient (to) is required for send_email")

    cc_field = action.get("cc", [])
    if isinstance(cc_field, str):
        cc_field = [cc_field]

    data = {
        "mailbox_id": mailbox_id,
        "to": to_field,
        "cc": cc_field,
        "subject": action.get("subject", ""),
        "body": action.get("body", ""),
    }
    smtp_send(user_id, data)
    return {"details": f"Email sent to {', '.join(to_field)}"}


def _exec_draft_reply(user_id: str, action: dict) -> dict:
    from api.controllers.settings.services import get_user_preferences_prompt

    email_id = action.get("email_id", "")
    content = get_email_content(email_id, user_id) if email_id else None
    if not content:
        raise ValueError("Email not found for reply")

    context = (
        f"From: {content.get('from_name', '')} <{content.get('from_email', '')}>\n"
        f"Subject: {content.get('subject', '')}\n\n"
        f"{content.get('body_chunk', '')[:2000]}"
    )
    instructions = action.get("instructions", "Write a helpful reply")
    user_prefs = get_user_preferences_prompt(user_id)
    style_note = f"\n{user_prefs}\nFollow the user's preferred draft style.\n" if user_prefs else ""
    draft = chat(
        system_prompt=(
            "Draft a reply email. Match the formality of the original. Be concise and natural."
            + style_note
        ),
        user_message=f"Original email:\n{context}\n\nInstructions: {instructions}",
        temperature=0.6,
    )
    return {"details": draft, "draft": draft}


def _exec_send_reply(user_id: str, action: dict) -> dict:
    from api.controllers.emails.services import reply_email as do_reply

    email_id = action.get("email_id", "")
    content = get_email_content(email_id, user_id) if email_id else None
    if not content:
        raise ValueError("Email not found for reply")

    body = action.get("body", "").strip()
    if not body:
        from api.controllers.settings.services import get_user_preferences_prompt

        instructions = action.get("instructions", "Write a helpful reply")
        context = (
            f"From: {content.get('from_name', '')} <{content.get('from_email', '')}>\n"
            f"Subject: {content.get('subject', '')}\n\n"
            f"{content.get('body_chunk', '')[:2000]}"
        )
        user_prefs = get_user_preferences_prompt(user_id)
        style_note = f"\n{user_prefs}\nFollow the user's preferred draft style.\n" if user_prefs else ""
        body = chat(
            system_prompt=(
                "Draft a reply email. Match the formality of the original. Be concise and natural. "
                "Return ONLY the email body text (no subject, no metadata)."
                + style_note
            ),
            user_message=f"Original email:\n{context}\n\nInstructions: {instructions}",
            temperature=0.6,
        )

    mailbox_id = action.get("mailbox_id") or _default_mailbox_id(user_id)
    do_reply(user_id, email_id, {"mailbox_id": mailbox_id, "body": body})
    to_addr = content.get("from_email", "")
    return {"details": f"Reply sent to {to_addr}"}


def _exec_reply_all(user_id: str, action: dict) -> dict:
    from api.controllers.emails.services import reply_email as do_reply

    email_id = action.get("email_id", "")
    content = get_email_content(email_id, user_id) if email_id else None
    if not content:
        raise ValueError("Email not found for reply_all")

    mailbox_id = action.get("mailbox_id") or _default_mailbox_id(user_id)
    recipients = []
    seen = set()

    from_email = (content.get("from_email") or "").strip()
    if from_email:
        recipients.append(from_email)
        seen.add(from_email.lower())

    for recipient in content.get("to", []) or []:
        if isinstance(recipient, dict):
            em = (recipient.get("email") or "").strip()
        else:
            em = str(recipient).strip()
        if em and em.lower() not in seen:
            recipients.append(em)
            seen.add(em.lower())

    body = (action.get("body") or "").strip()
    if not body:
        raise ValueError("body is required for reply_all")

    do_reply(user_id, email_id, {"mailbox_id": mailbox_id, "to": recipients, "body": body})
    return {"details": f"Reply all sent to {', '.join(recipients)}"}


# WhatsApp / Reminder


def _exec_send_whatsapp(user_id: str, action: dict) -> dict:
    from django.conf import settings as s

    sid = getattr(s, "TWILIO_ACCOUNT_SID", "")
    token = getattr(s, "TWILIO_AUTH_TOKEN", "")
    from_num = getattr(s, "TWILIO_WHATSAPP_FROM", "")
    if not all([sid, token, from_num]):
        raise ValueError(
            "WhatsApp not configured. Set TWILIO_ACCOUNT_SID, "
            "TWILIO_AUTH_TOKEN, and TWILIO_WHATSAPP_FROM in .env"
        )

    from twilio.rest import Client  # type: ignore
    client = Client(sid, token)

    to_num = (action.get("to") or "").strip()
    if not to_num:
        raise ValueError("Recipient (to) is required for send_whatsapp")
    if not to_num.startswith("whatsapp:"):
        to_num = f"whatsapp:{to_num}"

    body = (action.get("body") or "").strip()
    if not body:
        raise ValueError("Message body is required for send_whatsapp")

    msg = client.messages.create(
        body=body,
        from_=from_num,
        to=to_num,
    )
    return {"details": f"WhatsApp sent to {to_num} (SID: {msg.sid})"}


def _exec_set_reminder(user_id: str, action: dict) -> dict:
    hours = int(action.get("hours", 24))
    if hours < 0:
        hours = 24
    now = datetime.now(timezone.utc)
    due = now + timedelta(hours=hours)
    follow_ups_col().insert_one({
        "user_id": user_id,
        "email_id": action.get("email_id", ""),
        "due_date": due,
        "status": "pending",
        "auto_reminder_sent": False,
        "days_waiting": 0,
        "created_at": now,
        "updated_at": now,
    })
    return {"details": f"Reminder set for {due.strftime('%Y-%m-%d %H:%M UTC')}"}


# Forward / Trash / Archive / Mark / Snooze / Delete


def _exec_forward_email(user_id: str, action: dict) -> dict:
    from api.controllers.emails.services import forward_email

    email_id = (action.get("email_id") or "").strip()
    if not email_id:
        raise ValueError("email_id is required for forward_email")

    mailbox_id = action.get("mailbox_id") or _default_mailbox_id(user_id)
    to_field = action.get("to", [])
    if isinstance(to_field, str):
        to_field = [to_field]
    to_field = [t.strip() for t in to_field if t and str(t).strip()]
    if not to_field:
        raise ValueError("Recipient (to) is required for forward_email")

    forward_email(user_id, email_id, {
        "mailbox_id": mailbox_id,
        "to": to_field,
        "body": action.get("body", ""),
    })
    return {"details": f"Forwarded to {', '.join(to_field)}"}


def _exec_trash_email(user_id: str, action: dict) -> dict:
    from api.controllers.emails.services import trash_email as do_trash

    ids = _resolve_email_ids(user_id, action)
    if not ids:
        raise ValueError("No matching emails found. Provide email_id, from_email, subject, or keywords.")
    done, fail = 0, 0
    for eid in ids:
        if do_trash(user_id, eid):
            done += 1
        else:
            fail += 1
    return {
        "details": f"Moved {done} email(s) to trash" + (f" ({fail} failed)" if fail else ""),
        "marked": done, "failed": fail,
    }


def _exec_archive_email(user_id: str, action: dict) -> dict:
    from api.controllers.emails.services import archive_email as do_archive

    ids = _resolve_email_ids(user_id, action)
    if not ids:
        raise ValueError("No matching emails found. Provide email_id, from_email, subject, or keywords.")
    done, fail = 0, 0
    for eid in ids:
        if do_archive(user_id, eid):
            done += 1
        else:
            fail += 1
    return {
        "details": f"Archived {done} email(s)" + (f" ({fail} failed)" if fail else ""),
        "marked": done, "failed": fail,
    }


def _exec_mark_read(user_id: str, action: dict) -> dict:
    from api.controllers.emails.services import update_email

    ids = _resolve_email_ids(user_id, action)
    if not ids:
        raise ValueError("No matching emails found. Provide email_id, from_email, subject, or keywords.")
    done, fail = 0, 0
    for eid in ids:
        if update_email(user_id, eid, {"read": True}):
            done += 1
        else:
            fail += 1
    return {
        "details": f"Marked {done} email(s) as read" + (f" ({fail} failed)" if fail else ""),
        "marked": done, "failed": fail,
    }


def _exec_mark_all_read(user_id: str, action: dict) -> dict:
    from api.controllers.emails.services import mark_all_inbox_read

    mb = action.get("mailbox_id")
    if mb is not None and isinstance(mb, str) and not mb.strip():
        mb = None
    result = mark_all_inbox_read(user_id, mailbox_id=mb)
    marked = result.get("marked", 0)
    failed = result.get("failed", 0)
    return {
        "details": f"Marked {marked} inbox email(s) as read" + (f" ({failed} failed)" if failed else ""),
        "marked": marked,
        "failed": failed,
    }


def _exec_mark_all_unread(user_id: str, action: dict) -> dict:
    from api.controllers.emails.services import mark_all_inbox_unread

    mb = action.get("mailbox_id")
    if mb is not None and isinstance(mb, str) and not mb.strip():
        mb = None
    result = mark_all_inbox_unread(user_id, mailbox_id=mb)
    marked = result.get("marked", 0)
    failed = result.get("failed", 0)
    return {
        "details": f"Marked {marked} inbox email(s) as unread" + (f" ({failed} failed)" if failed else ""),
        "marked": marked,
        "failed": failed,
    }


def _exec_mark_unread(user_id: str, action: dict) -> dict:
    from api.controllers.emails.services import update_email

    ids = _resolve_email_ids(user_id, action)
    if not ids:
        raise ValueError("No matching emails found. Provide email_id, from_email, subject, or keywords.")
    done, fail = 0, 0
    for eid in ids:
        if update_email(user_id, eid, {"read": False}):
            done += 1
        else:
            fail += 1
    return {
        "details": f"Marked {done} email(s) as unread" + (f" ({fail} failed)" if fail else ""),
        "marked": done, "failed": fail,
    }


def _exec_snooze_email(user_id: str, action: dict) -> dict:
    from api.controllers.emails.services import snooze_email

    ids = _resolve_email_ids(user_id, action)
    if not ids:
        raise ValueError("No matching emails found. Provide email_id, from_email, subject, or keywords.")
    hours = int(action.get("hours", 24))
    if hours < 1:
        raise ValueError("hours must be >= 1 for snooze_email")
    done, fail = 0, 0
    for eid in ids:
        if snooze_email(user_id, eid, hours):
            done += 1
        else:
            fail += 1
    return {
        "details": f"Snoozed {done} email(s) for {hours} hours" + (f" ({fail} failed)" if fail else ""),
        "marked": done, "failed": fail,
    }


def _exec_delete_email(user_id: str, action: dict) -> dict:
    from api.controllers.emails.services import delete_email

    ids = _resolve_email_ids(user_id, action)
    if not ids:
        raise ValueError("No matching emails found. Provide email_id, from_email, subject, or keywords.")
    done, fail = 0, 0
    for eid in ids:
        if delete_email(user_id, eid):
            done += 1
        else:
            fail += 1
    return {
        "details": f"Permanently deleted {done} email(s)" + (f" ({fail} failed)" if fail else ""),
        "marked": done, "failed": fail,
    }


# Helpers


def _default_mailbox_id(user_id: str) -> str:
    mb = mailboxes_col().find_one({"user_id": user_id})
    if not mb:
        raise ValueError("No mailbox found — please add a mailbox first.")
    return str(mb["_id"])


def _log(user_id: str, action: dict, status: str, details: str = "") -> dict:
    target = action.get("to", action.get("target", ""))
    if isinstance(target, list):
        target = ", ".join(target)
    now = datetime.now(timezone.utc)
    return {
        "type": action.get("type", "unknown"),
        "label": action.get("label", ""),
        "description": action.get("description", details),
        "status": status,
        "target": target,
        "execution_details": details,
        "timestamp": now.isoformat(),
    }


def _parse_iso_dt(value):
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
