"""
Action executor for AI/voice actions.
Supports read/open/search actions, compose/reply/forward, and mailbox actions.
"""

import re
from datetime import datetime, timezone, timedelta

from bson import ObjectId

from database.db import mailboxes_col, follow_ups_col, email_metadata_col
from api.utils.qdrant_helpers import get_email_content
from api.utils.llm import chat


def execute_action(user_id: str, action: dict) -> dict:
    action = _normalize_action_mailbox_scope(action)
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
        "star_email": _exec_star_email,
        "unstar_email": _exec_unstar_email,
        "mark_starred": _exec_star_email,
        "mark_unstarred": _exec_unstar_email,
        "mark_all_starred": _exec_mark_all_starred,
        "mark_all_unstarred": _exec_mark_all_unstarred,
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
    1. ``email_ids`` (list) → all of them (deduped, kept in order, existence-checked).
    2. ``email_id`` (single) → single-item list.
    3. Filter combination: ``from_email`` / ``subject`` / ``keywords`` /
       ``read`` / ``date_from`` / ``date_to`` / ``mailbox_id`` / ``limit``.
       At least one filter must be supplied for bulk to return anything.

    Returns an empty list when nothing matches.
    """
    col = email_metadata_col()

    # 1) Explicit list of IDs
    raw_list = action.get("email_ids")
    if isinstance(raw_list, list) and raw_list:
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in raw_list:
            if not item:
                continue
            s = str(item).strip().strip("`'\" ")
            if s and s not in seen:
                seen.add(s)
                cleaned.append(s)
        if cleaned:
            existing = {
                str(d["_id"])
                for d in col.find(
                    {"_id": {"$in": cleaned}, "user_id": user_id},
                    {"_id": 1},
                )
            }
            resolved = [c for c in cleaned if c in existing]
            if resolved:
                return resolved

    # 2) Single explicit ID
    single = (action.get("email_id") or "").strip()
    if single:
        if col.find_one({"_id": single, "user_id": user_id}, {"_id": 1}):
            return [single]
        cleaned_id = single.strip("`'\" ")
        if cleaned_id and col.find_one({"_id": cleaned_id, "user_id": user_id}, {"_id": 1}):
            return [cleaned_id]

    # 3) Filters
    sender = (action.get("from_email") or "").strip()
    subj = (action.get("subject") or "").strip()
    kw = (action.get("keywords") or "").strip()
    label = (action.get("label_name") or action.get("label") or "").strip()
    folder = (action.get("folder") or "").strip().lower()
    mb = action.get("mailbox_id") or None

    read_val = action.get("read")
    read_filter: bool | None
    if isinstance(read_val, bool):
        read_filter = read_val
    elif isinstance(read_val, str) and read_val.strip().lower() in ("true", "false"):
        read_filter = read_val.strip().lower() == "true"
    elif action.get("unread_only"):
        read_filter = False
    else:
        read_filter = None

    date_from = _parse_iso_dt(action.get("date_from"))
    date_to = _parse_iso_dt(action.get("date_to"))

    try:
        limit_val = int(action.get("limit") or 0)
    except (TypeError, ValueError):
        limit_val = 0

    query: dict = {"user_id": user_id}
    if folder == "trash":
        query.update({"trashed": True, "$or": [{"spam": {"$ne": True}}, {"spam": {"$exists": False}}]})
    elif folder == "spam":
        query.update({"spam": True})
    elif folder == "sent":
        query.update({"is_sent": True})
    elif folder == "archive":
        query.update({"archived": True, "trashed": False})
    elif folder == "star":
        query.update({"starred": True, "trashed": False})
    elif folder == "snoozed":
        query.update({"trashed": False, "snoozed_until": {"$gt": datetime.now(timezone.utc)}})
    else:
        # Inbox/default scope.
        query.update({"archived": False, "trashed": False})
    if mb:
        query["mailbox_id"] = mb

    if sender:
        query["from_email"] = {"$regex": f"^{re.escape(sender)}$", "$options": "i"}

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

    if read_filter is not None:
        query["read"] = read_filter

    starred_val = action.get("starred")
    starred_filter: bool | None
    if isinstance(starred_val, bool):
        starred_filter = starred_val
    elif isinstance(starred_val, str) and starred_val.strip().lower() in ("true", "false"):
        starred_filter = starred_val.strip().lower() == "true"
    else:
        starred_filter = None
    if starred_filter is not None:
        query["starred"] = starred_filter

    if label:
        query["labels"] = label

    if date_from or date_to:
        date_range: dict = {}
        if date_from:
            date_range["$gte"] = date_from
        if date_to:
            date_range["$lte"] = date_to
        query["date"] = date_range

    # Require at least one targeting signal before returning bulk matches.
    has_filter = (
        bool(sender)
        or bool(ands)
        or read_filter is not None
        or starred_filter is not None
        or bool(label)
        or bool(folder)
        or date_from is not None
        or date_to is not None
    )
    if not has_filter:
        return []

    cursor = col.find(query, {"_id": 1}).sort("date", -1)
    if limit_val > 0:
        cursor = cursor.limit(limit_val)
    return [str(d["_id"]) for d in cursor]


def _resolve_single_email_id(user_id: str, action: dict) -> str:
    """Return one concrete email_id for an action targeting a single email.

    Priority:
      1. Explicit ``email_id`` if it exists in MongoDB (with trimmed fallback).
      2. First entry of ``email_ids`` list if present and valid.
      3. First (latest-by-date) match from ``_resolve_email_ids`` using
         ``from_email`` / ``subject`` / ``keywords`` / date / read filters.

    Returns "" when nothing matches — caller decides whether to raise.
    """
    raw_id = (action.get("email_id") or "").strip()
    if raw_id:
        col = email_metadata_col()
        if col.find_one({"_id": raw_id, "user_id": user_id}, {"_id": 1}):
            return raw_id
        cleaned = raw_id.strip("`'\" ")
        if cleaned and cleaned != raw_id and col.find_one(
            {"_id": cleaned, "user_id": user_id}, {"_id": 1}
        ):
            return cleaned

    ids = _resolve_email_ids(user_id, {**action, "email_id": ""})
    return ids[0] if ids else ""


def _resolve_reply_email_id(user_id: str, action: dict) -> str:
    """Stricter resolver for send_reply / reply_all.

    To avoid replying to the wrong person, we refuse to resolve a reply purely
    from fuzzy filters like keywords/subject. Acceptable targeting signals:
      1. Exact ``email_id`` that exists.
      2. ``email_ids`` (first valid entry).
      3. ``from_email`` — latest email from that exact sender (optionally
         narrowed by subject / keywords / date for disambiguation).

    Returns "" when nothing matches, and raises ValueError when the caller
    tried to target a reply by keywords alone.
    """
    raw_id = (action.get("email_id") or "").strip()
    if raw_id:
        col = email_metadata_col()
        if col.find_one({"_id": raw_id, "user_id": user_id}, {"_id": 1}):
            return raw_id
        cleaned = raw_id.strip("`'\" ")
        if cleaned and cleaned != raw_id and col.find_one(
            {"_id": cleaned, "user_id": user_id}, {"_id": 1}
        ):
            return cleaned

    ids_list = action.get("email_ids")
    if isinstance(ids_list, list) and ids_list:
        ids = _resolve_email_ids(user_id, {"email_ids": ids_list, "email_id": ""})
        if ids:
            return ids[0]

    sender = (action.get("from_email") or "").strip()
    if not sender:
        raise ValueError(
            "Cannot safely target a reply without email_id or from_email. "
            "Please specify which email to reply to by its sender address or id."
        )

    # Sender is required; subject/keywords/date only narrow further.
    narrowed = {
        "from_email": sender,
        "subject": action.get("subject", ""),
        "keywords": action.get("keywords", ""),
        "date_from": action.get("date_from"),
        "date_to": action.get("date_to"),
        "mailbox_id": action.get("mailbox_id"),
        "email_id": "",
    }
    ids = _resolve_email_ids(user_id, narrowed)
    return ids[0] if ids else ""


def _normalize_addr(addr: str) -> str:
    return (addr or "").strip().lower()


def _guard_reply_recipient(
    content: dict,
    action: dict,
    resolved_email_id: str,
) -> str:
    """Ensure the resolved email actually belongs to the intended sender and,
    when the agent proposes a ``to`` override, that the override still points
    at the original sender (plain reply semantics).

    Returns the final, verified reply recipient (always the original sender).
    Raises ValueError on any mismatch.
    """
    resolved_sender = _normalize_addr(content.get("from_email", ""))
    if not resolved_sender:
        raise ValueError(
            "Cannot determine reply recipient — resolved email has no sender "
            "address. Aborting to avoid sending to the wrong person."
        )

    claimed_sender = _normalize_addr(action.get("from_email", ""))
    if claimed_sender and claimed_sender != resolved_sender:
        raise ValueError(
            f"Safety abort: the email resolved for this reply is from "
            f"'{resolved_sender}', but the action requested '{claimed_sender}'. "
            f"Refusing to send to prevent replying to the wrong person."
        )

    raw_to = action.get("to")
    if isinstance(raw_to, str):
        raw_to = [raw_to]
    if isinstance(raw_to, list):
        override = [_normalize_addr(t) for t in raw_to if t and str(t).strip()]
        if override and resolved_sender not in override:
            raise ValueError(
                f"Safety abort: the reply 'to' override "
                f"({', '.join(override)}) does not include the original "
                f"sender '{resolved_sender}'. If you want to send a new "
                f"message to a different recipient, use send_email instead "
                f"of send_reply."
            )

    return content.get("from_email", "").strip()


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
    from api.controllers.emails.services import get_email, update_email

    email_id = _resolve_single_email_id(user_id, action)
    if not email_id:
        raise ValueError(
            "No matching email found. Provide email_id, from_email, subject, or keywords."
        )
    result = get_email(user_id, email_id)
    if not result:
        raise ValueError("Email not found")

    if not result.get("read"):
        try:
            updated = update_email(user_id, email_id, {"read": True})
            if updated:
                result = updated
        except Exception:
            pass

    return {"details": "Opened email", "email": result}


def _exec_open_latest_email(user_id: str, action: dict) -> dict:
    from api.controllers.emails.services import list_emails, get_email, update_email

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

    top = items[0]
    email_id = str(top.get("id") or top.get("_id") or "")
    if email_id:
        full = get_email(user_id, email_id) or top
        if not full.get("read"):
            try:
                updated = update_email(user_id, email_id, {"read": True})
                if updated:
                    full = updated
            except Exception:
                pass
        return {"details": "Opened latest email", "email": full}

    return {"details": "Opened latest email", "email": top}


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
    cc_field = [c.strip() for c in cc_field if c and str(c).strip()]

    subject = (action.get("subject") or "").strip()
    body = _sanitize_draft_body(action.get("body") or "")
    if not subject:
        raise ValueError("Subject is required for send_email")
    if not body:
        raise ValueError("Body is required for send_email")

    data = {
        "mailbox_id": mailbox_id,
        "to": to_field,
        "cc": cc_field,
        "subject": subject,
        "body": body,
    }
    smtp_send(user_id, data)
    details = f"Email sent to {', '.join(to_field)}"
    if cc_field:
        details += f" (cc: {', '.join(cc_field)})"
    return {"details": details}


def _get_content_with_mongo_fallback(email_id: str, user_id: str) -> dict | None:
    """Try Qdrant first; if not indexed yet, build minimal content from MongoDB."""
    content = get_email_content(email_id, user_id) if email_id else None
    if content:
        return content
    if not email_id:
        return None
    meta = email_metadata_col().find_one({"_id": email_id, "user_id": user_id})
    if not meta:
        return None
    date_val = meta.get("date")
    date_str = date_val.isoformat() if isinstance(date_val, datetime) else str(date_val or "")
    return {
        "email_id": email_id,
        "mailbox_id": meta.get("mailbox_id", ""),
        "subject": meta.get("subject", ""),
        "from_name": meta.get("from_name", ""),
        "from_email": meta.get("from_email", ""),
        "to": meta.get("to", []),
        "date": date_str,
        "preview": meta.get("preview", ""),
        "body_chunk": meta.get("preview", ""),
        "has_attachment": bool(meta.get("has_attachment", False)),
        "priority": meta.get("priority", "medium"),
        "attachment_text": "",
    }


def _exec_draft_reply(user_id: str, action: dict) -> dict:
    from api.controllers.settings.services import get_user_preferences_prompt

    email_id = _resolve_single_email_id(user_id, action)
    content = _get_content_with_mongo_fallback(email_id, user_id)
    if not content:
        raise ValueError(
            "Email not found for reply. Provide email_id, from_email, subject, or keywords."
        )

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
            "Draft a reply email. Match the formality of the original. Be concise and natural. "
            "Return ONLY the email body text (no subject, no metadata, no markdown code fences)."
            + style_note
        ),
        user_message=f"Original email:\n{context}\n\nInstructions: {instructions}",
        temperature=0.6,
    )
    draft = _sanitize_draft_body(draft)
    return {"details": draft, "draft": draft}


def _exec_send_reply(user_id: str, action: dict) -> dict:
    from api.controllers.emails.services import reply_email as do_reply

    email_id = _resolve_reply_email_id(user_id, action)
    content = _get_content_with_mongo_fallback(email_id, user_id)
    if not content:
        raise ValueError(
            "Email not found for reply. Provide an exact email_id or from_email."
        )

    verified_to = _guard_reply_recipient(content, action, email_id)

    body = (action.get("body") or "").strip()
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
                "Return ONLY the email body text (no subject, no metadata, no markdown code fences, "
                "no 'Here is the draft:' preamble)."
                + style_note
            ),
            user_message=f"Original email:\n{context}\n\nInstructions: {instructions}",
            temperature=0.6,
        )
    body = _sanitize_draft_body(body)

    mailbox_id = action.get("mailbox_id") or _default_mailbox_id(user_id)
    # Always reply to the resolved email's original sender. Any 'to' the
    # agent proposed has already been validated by _guard_reply_recipient;
    # we still pin the canonical address here to avoid subtle case / alias
    # drift leaking into the envelope.
    payload: dict = {"mailbox_id": mailbox_id, "body": body, "to": [verified_to]}
    if action.get("subject"):
        payload["subject"] = str(action["subject"]).strip()
    cc = action.get("cc")
    if isinstance(cc, str):
        cc = [cc]
    if isinstance(cc, list) and cc:
        payload["cc"] = [c.strip() for c in cc if c and str(c).strip()]

    do_reply(user_id, email_id, payload)
    return {"details": f"Reply sent to {verified_to}"}


def _exec_reply_all(user_id: str, action: dict) -> dict:
    from api.controllers.emails.services import reply_email as do_reply

    email_id = _resolve_reply_email_id(user_id, action)
    content = _get_content_with_mongo_fallback(email_id, user_id)
    if not content:
        raise ValueError(
            "Email not found for reply_all. Provide an exact email_id or from_email."
        )

    # Validate that the resolved email matches any claimed sender. The 'to'
    # override check is intentionally skipped for reply_all (it expands the
    # recipient list from the thread rather than targeting the sender only).
    claimed_sender = _normalize_addr(action.get("from_email", ""))
    resolved_sender = _normalize_addr(content.get("from_email", ""))
    if claimed_sender and claimed_sender != resolved_sender:
        raise ValueError(
            f"Safety abort: the thread resolved for reply_all is from "
            f"'{resolved_sender}', but the action requested '{claimed_sender}'. "
            f"Refusing to send to avoid replying to the wrong thread."
        )

    mailbox_id = action.get("mailbox_id") or _default_mailbox_id(user_id)
    mb = mailboxes_col().find_one({"_id": ObjectId(mailbox_id), "user_id": user_id})
    own_email = (mb.get("email") or "").strip().lower() if mb else ""

    recipients = []
    seen = set()
    if own_email:
        seen.add(own_email)

    from_email = (content.get("from_email") or "").strip()
    if from_email and from_email.lower() not in seen:
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

    body = _sanitize_draft_body(action.get("body") or "")
    if not body:
        raise ValueError("body is required for reply_all")

    payload: dict = {"mailbox_id": mailbox_id, "to": recipients, "body": body}
    if action.get("subject"):
        payload["subject"] = str(action["subject"]).strip()
    cc = action.get("cc")
    if isinstance(cc, str):
        cc = [cc]
    if isinstance(cc, list) and cc:
        payload["cc"] = [c.strip() for c in cc if c and str(c).strip()]

    do_reply(user_id, email_id, payload)
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
    email_id = _resolve_single_email_id(user_id, action)
    follow_ups_col().insert_one({
        "user_id": user_id,
        "email_id": email_id,
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

    email_id = _resolve_single_email_id(user_id, action)
    if not email_id:
        raise ValueError(
            "No matching email found to forward. Provide email_id, from_email, subject, or keywords."
        )

    mailbox_id = action.get("mailbox_id") or _default_mailbox_id(user_id)
    to_field = action.get("to", [])
    if isinstance(to_field, str):
        to_field = [to_field]
    to_field = [t.strip() for t in to_field if t and str(t).strip()]
    if not to_field:
        raise ValueError("Recipient (to) is required for forward_email")

    cc_field = action.get("cc", [])
    if isinstance(cc_field, str):
        cc_field = [cc_field]
    cc_field = [c.strip() for c in cc_field if c and str(c).strip()]

    forward_email(user_id, email_id, {
        "mailbox_id": mailbox_id,
        "to": to_field,
        "cc": cc_field,
        "subject": (action.get("subject") or "").strip(),
        "body": _sanitize_draft_body(action.get("body") or ""),
    })
    details = f"Forwarded to {', '.join(to_field)}"
    if cc_field:
        details += f" (cc: {', '.join(cc_field)})"
    return {"details": details}


def _exec_trash_email(user_id: str, action: dict) -> dict:
    from api.controllers.emails.services import bulk_trash_emails

    ids = _resolve_email_ids(user_id, action)
    if not ids:
        raise ValueError("No matching emails found. Provide email_id, from_email, subject, or keywords.")
    res = bulk_trash_emails(user_id, ids)
    done = int(res.get("processed", 0) or 0)
    fail = len(res.get("failed", []) or [])
    return {
        "details": f"Moved {done} email(s) to trash" + (f" ({fail} failed)" if fail else ""),
        "marked": done, "failed": fail,
    }


def _exec_archive_email(user_id: str, action: dict) -> dict:
    from api.controllers.emails.services import bulk_archive_emails

    ids = _resolve_email_ids(user_id, action)
    if not ids:
        raise ValueError("No matching emails found. Provide email_id, from_email, subject, or keywords.")
    res = bulk_archive_emails(user_id, ids)
    done = int(res.get("processed", 0) or 0)
    fail = len(res.get("failed", []) or [])
    return {
        "details": f"Archived {done} email(s)" + (f" ({fail} failed)" if fail else ""),
        "marked": done, "failed": fail,
    }


def _exec_mark_read(user_id: str, action: dict) -> dict:
    from api.controllers.emails.services import bulk_update_emails

    ids = _resolve_email_ids(user_id, action)
    if not ids:
        raise ValueError("No matching emails found. Provide email_id, from_email, subject, or keywords.")
    res = bulk_update_emails(user_id, ids, {"read": True})
    done = int(res.get("processed", 0) or 0)
    fail = len(res.get("failed", []) or [])
    return {
        "details": f"Marked {done} email(s) as read" + (f" ({fail} failed)" if fail else ""),
        "marked": done, "failed": fail,
    }


def _exec_mark_all_read(user_id: str, action: dict) -> dict:
    from api.controllers.emails.services import bulk_update_emails

    mb = action.get("mailbox_id")
    if mb is not None and isinstance(mb, str) and not mb.strip():
        mb = None
    ids = _resolve_email_ids(user_id, {
        "folder": "inbox",
        "read": False,
        "mailbox_id": mb,
        "limit": int(action.get("limit") or 0) or None,
    })
    if not ids:
        return {"details": "No unread inbox emails found", "marked": 0, "failed": 0}
    res = bulk_update_emails(user_id, ids, {"read": True})
    marked = int(res.get("processed", 0) or 0)
    failed = len(res.get("failed", []) or [])
    return {
        "details": f"Marked {marked} inbox email(s) as read" + (f" ({failed} failed)" if failed else ""),
        "marked": marked,
        "failed": failed,
    }


def _exec_mark_all_unread(user_id: str, action: dict) -> dict:
    from api.controllers.emails.services import bulk_update_emails

    mb = action.get("mailbox_id")
    if mb is not None and isinstance(mb, str) and not mb.strip():
        mb = None
    ids = _resolve_email_ids(user_id, {
        "folder": "inbox",
        "read": True,
        "mailbox_id": mb,
        "limit": int(action.get("limit") or 0) or None,
    })
    if not ids:
        return {"details": "No read inbox emails found", "marked": 0, "failed": 0}
    res = bulk_update_emails(user_id, ids, {"read": False})
    marked = int(res.get("processed", 0) or 0)
    failed = len(res.get("failed", []) or [])
    return {
        "details": f"Marked {marked} inbox email(s) as unread" + (f" ({failed} failed)" if failed else ""),
        "marked": marked,
        "failed": failed,
    }


def _exec_mark_unread(user_id: str, action: dict) -> dict:
    from api.controllers.emails.services import bulk_update_emails

    ids = _resolve_email_ids(user_id, action)
    if not ids:
        raise ValueError("No matching emails found. Provide email_id, from_email, subject, or keywords.")
    res = bulk_update_emails(user_id, ids, {"read": False})
    done = int(res.get("processed", 0) or 0)
    fail = len(res.get("failed", []) or [])
    return {
        "details": f"Marked {done} email(s) as unread" + (f" ({fail} failed)" if fail else ""),
        "marked": done, "failed": fail,
    }


def _exec_star_email(user_id: str, action: dict) -> dict:
    from api.controllers.emails.services import bulk_update_emails

    ids = _resolve_email_ids(user_id, action)
    if not ids:
        raise ValueError("No matching emails found. Provide email_id, from_email, subject, or keywords.")
    res = bulk_update_emails(user_id, ids, {"starred": True})
    done = int(res.get("processed", 0) or 0)
    fail = len(res.get("failed", []) or [])
    return {
        "details": f"Starred {done} email(s)" + (f" ({fail} failed)" if fail else ""),
        "marked": done, "failed": fail,
    }


def _exec_unstar_email(user_id: str, action: dict) -> dict:
    from api.controllers.emails.services import bulk_update_emails

    ids = _resolve_email_ids(user_id, action)
    if not ids:
        raise ValueError("No matching emails found. Provide email_id, from_email, subject, or keywords.")
    res = bulk_update_emails(user_id, ids, {"starred": False})
    done = int(res.get("processed", 0) or 0)
    fail = len(res.get("failed", []) or [])
    return {
        "details": f"Unstarred {done} email(s)" + (f" ({fail} failed)" if fail else ""),
        "marked": done, "failed": fail,
    }


def _exec_mark_all_starred(user_id: str, action: dict) -> dict:
    from api.controllers.emails.services import bulk_update_emails

    mb = action.get("mailbox_id")
    if mb is not None and isinstance(mb, str) and not mb.strip():
        mb = None
    ids = _resolve_email_ids(user_id, {
        "folder": "inbox",
        "mailbox_id": mb,
        "limit": int(action.get("limit") or 0) or None,
    })
    if not ids:
        return {"details": "No inbox emails found", "marked": 0, "failed": 0}
    res = bulk_update_emails(user_id, ids, {"starred": True})
    marked = int(res.get("processed", 0) or 0)
    failed = len(res.get("failed", []) or [])
    return {
        "details": f"Starred {marked} inbox email(s)" + (f" ({failed} failed)" if failed else ""),
        "marked": marked, "failed": failed,
    }


def _exec_mark_all_unstarred(user_id: str, action: dict) -> dict:
    from api.controllers.emails.services import bulk_update_emails

    mb = action.get("mailbox_id")
    if mb is not None and isinstance(mb, str) and not mb.strip():
        mb = None
    ids = _resolve_email_ids(user_id, {
        "folder": "inbox",
        "starred": True,
        "mailbox_id": mb,
        "limit": int(action.get("limit") or 0) or None,
    })
    if not ids:
        return {"details": "No starred inbox emails found", "marked": 0, "failed": 0}
    res = bulk_update_emails(user_id, ids, {"starred": False})
    marked = int(res.get("processed", 0) or 0)
    failed = len(res.get("failed", []) or [])
    return {
        "details": f"Unstarred {marked} inbox email(s)" + (f" ({failed} failed)" if failed else ""),
        "marked": marked, "failed": failed,
    }


def _exec_snooze_email(user_id: str, action: dict) -> dict:
    from api.controllers.emails.services import bulk_snooze_emails

    ids = _resolve_email_ids(user_id, action)
    if not ids:
        raise ValueError("No matching emails found. Provide email_id, from_email, subject, or keywords.")
    hours = int(action.get("hours", 24))
    if hours < 1:
        raise ValueError("hours must be >= 1 for snooze_email")
    res = bulk_snooze_emails(user_id, ids, hours)
    done = int(res.get("processed", 0) or 0)
    fail = len(res.get("failed", []) or [])
    return {
        "details": f"Snoozed {done} email(s) for {hours} hours" + (f" ({fail} failed)" if fail else ""),
        "marked": done, "failed": fail,
    }


def _exec_delete_email(user_id: str, action: dict) -> dict:
    """Natural-language 'delete' moves messages to trash (same as trash_email)."""
    return _exec_trash_email(user_id, action)


# Helpers


def _default_mailbox_id(user_id: str) -> str:
    mb = mailboxes_col().find_one({"user_id": user_id})
    if not mb:
        raise ValueError("No mailbox found — please add a mailbox first.")
    return str(mb["_id"])


def _normalize_action_mailbox_scope(action: dict) -> dict:
    """Treat synthetic mailbox values as global scope (None)."""
    if not isinstance(action, dict):
        return action
    mb = action.get("mailbox_id")
    if isinstance(mb, str):
        cleaned = mb.strip()
        if not cleaned or cleaned.lower() in {"all", "*", "any"}:
            action["mailbox_id"] = None
    return action


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


_DRAFT_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$", re.MULTILINE)
_DRAFT_PREAMBLE_RE = re.compile(
    r"^(?:here(?:'s| is)?|sure|of\s+course|certainly|below\s+is|please\s+find)"
    r"[^\n]*(?:draft|reply|response|email)[^\n]*[:.\-—]?\s*\n+",
    re.IGNORECASE,
)


def _sanitize_draft_body(text: str) -> str:
    """Strip LLM scaffolding (markdown fences, 'Here is the draft:' preambles,
    stray whitespace) from a drafted email body so the recipient never sees it."""
    if not text:
        return ""
    cleaned = text.strip()
    cleaned = _DRAFT_FENCE_RE.sub("", cleaned).strip()
    cleaned = _DRAFT_PREAMBLE_RE.sub("", cleaned, count=1).strip()
    if cleaned.startswith('"') and cleaned.endswith('"') and len(cleaned) > 2:
        cleaned = cleaned[1:-1].strip()
    return cleaned
