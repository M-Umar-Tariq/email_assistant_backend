"""
Action executor — performs real actions on behalf of the user.
Supports: send email (SMTP), draft reply (AI), send reply, forward email,
trash/archive/mark read, WhatsApp (Twilio), set reminder.
"""

from datetime import datetime, timezone, timedelta

from database.db import mailboxes_col, follow_ups_col
from api.utils.qdrant_helpers import get_email_content
from api.utils.llm import chat


def execute_action(user_id: str, action: dict) -> dict:
    action_type = action.get("type", "")
    executors = {
        "send_email": _exec_send_email,
        "draft_reply": _exec_draft_reply,
        "send_reply": _exec_send_reply,
        "send_whatsapp": _exec_send_whatsapp,
        "set_reminder": _exec_set_reminder,
        "forward_email": _exec_forward_email,
        "trash_email": _exec_trash_email,
        "archive_email": _exec_archive_email,
        "mark_read": _exec_mark_read,
    }
    executor_fn = executors.get(action_type)
    if not executor_fn:
        return _log(user_id, action, "failed", f"Unknown action type: {action_type}")

    try:
        result = executor_fn(user_id, action)
        return _log(user_id, action, "completed", result.get("details", ""))
    except Exception as exc:
        return _log(user_id, action, "failed", str(exc))


# ── Send email via SMTP ──────────────────────────────────────────────────────

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


# ── AI-generated reply draft ─────────────────────────────────────────────────

def _exec_draft_reply(user_id: str, action: dict) -> dict:
    email_id = action.get("email_id", "")
    content = get_email_content(email_id, user_id) if email_id else None
    if not content:
        raise ValueError("Email not found for reply")

    context = (
        f"From: {content.get('from_name','')} <{content.get('from_email','')}>\n"
        f"Subject: {content.get('subject','')}\n\n"
        f"{content.get('body_chunk','')[:2000]}"
    )
    instructions = action.get("instructions", "Write a helpful reply")
    draft = chat(
        system_prompt=(
            "Draft a reply email. Match the formality of the original. Be concise and natural."
        ),
        user_message=f"Original email:\n{context}\n\nInstructions: {instructions}",
        temperature=0.6,
    )
    return {"details": draft, "draft": draft}


# ── Send reply (actually send the reply email) ─────────────────────────────────

def _exec_send_reply(user_id: str, action: dict) -> dict:
    from api.controllers.emails.services import reply_email as do_reply

    email_id = action.get("email_id", "")
    content = get_email_content(email_id, user_id) if email_id else None
    if not content:
        raise ValueError("Email not found for reply")

    body = action.get("body", "").strip()
    if not body:
        instructions = action.get("instructions", "Write a helpful reply")
        context = (
            f"From: {content.get('from_name','')} <{content.get('from_email','')}>\n"
            f"Subject: {content.get('subject','')}\n\n"
            f"{content.get('body_chunk','')[:2000]}"
        )
        body = chat(
            system_prompt=(
                "Draft a reply email. Match the formality of the original. Be concise and natural. "
                "Return ONLY the email body text (no subject, no metadata)."
            ),
            user_message=f"Original email:\n{context}\n\nInstructions: {instructions}",
            temperature=0.6,
        )

    mailbox_id = action.get("mailbox_id") or _default_mailbox_id(user_id)
    do_reply(user_id, email_id, {"mailbox_id": mailbox_id, "body": body})
    to_addr = content.get("from_email", "")
    return {"details": f"Reply sent to {to_addr}"}


# ── WhatsApp via Twilio ───────────────────────────────────────────────────────

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


# ── Reminder / follow-up ─────────────────────────────────────────────────────

def _exec_set_reminder(user_id: str, action: dict) -> dict:
    hours = int(action.get("hours", 24))
    if hours < 0:
        hours = 24
    due = datetime.now(timezone.utc) + timedelta(hours=hours)
    follow_ups_col().insert_one({
        "user_id": user_id,
        "email_id": action.get("email_id", ""),
        "due_date": due,
        "status": "pending",
        "auto_reminder_sent": False,
        "suggested_action": action.get("note", action.get("description", "")),
        "created_at": datetime.now(timezone.utc),
    })
    return {"details": f"Reminder set for {due.strftime('%Y-%m-%d %H:%M UTC')}"}


# ── Forward email ─────────────────────────────────────────────────────────────

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


# ── Trash / Archive / Mark read ────────────────────────────────────────────────

def _exec_trash_email(user_id: str, action: dict) -> dict:
    from api.controllers.emails.services import trash_email as do_trash

    email_id = (action.get("email_id") or "").strip()
    if not email_id:
        raise ValueError("email_id is required for trash_email")
    ok = do_trash(user_id, email_id)
    if not ok:
        raise ValueError("Email not found or could not be moved to trash")
    return {"details": "Email moved to trash"}


def _exec_archive_email(user_id: str, action: dict) -> dict:
    from api.controllers.emails.services import archive_email as do_archive

    email_id = (action.get("email_id") or "").strip()
    if not email_id:
        raise ValueError("email_id is required for archive_email")
    ok = do_archive(user_id, email_id)
    if not ok:
        raise ValueError("Email not found or could not be archived")
    return {"details": "Email archived"}


def _exec_mark_read(user_id: str, action: dict) -> dict:
    from api.controllers.emails.services import update_email

    email_id = (action.get("email_id") or "").strip()
    if not email_id:
        raise ValueError("email_id is required for mark_read")
    result = update_email(user_id, email_id, {"read": True})
    if not result:
        raise ValueError("Email not found or could not be marked as read")
    return {"details": "Email marked as read"}


# ── Helpers ───────────────────────────────────────────────────────────────────

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
