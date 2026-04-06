from datetime import datetime, timezone, timedelta

from database.db import email_metadata_col, mailboxes_col, follow_ups_col, meetings_col
from api.controllers.calendar.services import get_today_meeting_stats
from api.utils.llm import chat, chat_json
from api.utils.qdrant_helpers import get_email_content, get_emails_content_batch


def get_briefing(user_id: str) -> dict:
    now = datetime.now(timezone.utc)
    start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)

    unread_count = email_metadata_col().count_documents({
        "user_id": user_id, "read": False, "archived": False, "trashed": False,
    })

    # Get today's emails from MongoDB (use original_date to avoid
    # thread-bumped old emails appearing as today's)
    today_docs = list(email_metadata_col().find({
        "user_id": user_id,
        "$or": [
            {"original_date": {"$gte": start_of_day}},
            {"original_date": None, "date": {"$gte": start_of_day}},
        ],
        "archived": False,
        "trashed": False,
    }).sort("date", -1).limit(30))

    # Batch-fetch content from Qdrant for all unique IDs
    all_ids = list({str(d["_id"]) for d in today_docs})
    content_map = get_emails_content_batch(all_ids, user_id)

    # Filter high priority from all today's emails (read status doesn't matter)
    high_priority = []
    for doc in today_docs:
        eid = str(doc["_id"])
        content = content_map.get(eid, {})
        if content.get("priority") == "high":
            high_priority.append({"doc": doc, "content": content})

    recent = []
    for doc in today_docs:
        eid = str(doc["_id"])
        content = content_map.get(eid, {})
        recent.append({"doc": doc, "content": content})

    # Only count follow-ups whose email still exists (e.g. not orphaned after mailbox delete)
    valid_email_ids = list(email_metadata_col().distinct("_id", {"user_id": user_id}))
    valid_email_id_strs = [str(eid) for eid in valid_email_ids] if valid_email_ids else []
    fu_query_overdue = {"user_id": user_id, "status": "overdue"}
    fu_query_pending = {"user_id": user_id, "status": "pending"}
    if valid_email_id_strs:
        fu_query_overdue["email_id"] = {"$in": valid_email_id_strs}
        fu_query_pending["email_id"] = {"$in": valid_email_id_strs}
    else:
        # No emails left (e.g. all mailboxes deleted) → no follow-ups
        fu_query_overdue["email_id"] = {"$in": []}
        fu_query_pending["email_id"] = {"$in": []}
    overdue_follow_ups = follow_ups_col().count_documents(fu_query_overdue)
    pending_follow_ups = follow_ups_col().count_documents(fu_query_pending)

    mailboxes = list(mailboxes_col().find({"user_id": user_id}))
    mailbox_summary = [
        {
            "id": str(mb["_id"]),
            "name": mb["name"],
            "email": mb["email"],
            "color": mb.get("color", ""),
            "unread": email_metadata_col().count_documents({
                "user_id": user_id, "mailbox_id": str(mb["_id"]),
                "read": False, "archived": False, "trashed": False,
            }),
            "synced": mb.get("sync_status") == "synced",
            "last_sync": mb.get("last_sync_at"),
            "sync_status": mb.get("sync_status", ""),
        }
        for mb in mailboxes
    ]

    briefing_items = _build_briefing_items(high_priority, recent, user_id)

    mtg_stats = get_today_meeting_stats(user_id)
    meeting_items = _meeting_briefing_items(user_id, start_of_day)

    return {
        "stats": {
            "unread_total": unread_count,
            "high_priority": len(high_priority),
            "overdue_follow_ups": overdue_follow_ups,
            "pending_follow_ups": pending_follow_ups,
            "meetings_today_count": mtg_stats["meetings_today_count"],
            "meetings_today_conflicts": mtg_stats["meetings_today_conflicts"],
            "next_meeting": mtg_stats["next_meeting"],
        },
        "mailboxes": mailbox_summary,
        "items": meeting_items + briefing_items,
    }


def _meeting_briefing_items(user_id: str, start_of_day: datetime) -> list[dict]:
    """Upcoming today's calendar entries for the briefing list."""
    end_of_day = start_of_day + timedelta(days=1)
    docs = list(
        meetings_col()
        .find(
            {
                "user_id": user_id,
                "start": {"$lt": end_of_day},
                "end": {"$gt": start_of_day},
            }
        )
        .sort("start", 1)
        .limit(8)
    )
    out = []
    for m in docs:
        mid = str(m["_id"])
        st = m["start"]
        en = m["end"]
        st_s = st.isoformat() if isinstance(st, datetime) else str(st)
        en_s = en.isoformat() if isinstance(en, datetime) else str(en)
        eid = m.get("email_id")
        desc = f"{st_s} – {en_s}"
        if m.get("conflict"):
            desc += " · Overlapping time — check calendar"
        out.append(
            {
                "id": f"cal-{mid}",
                "type": "meeting",
                "title": m.get("title") or "Meeting",
                "description": desc,
                "priority": "high" if m.get("conflict") else "medium",
                "email_ids": [eid] if eid else [],
                "meeting_id": mid,
            }
        )
    return out


def _build_briefing_items(high_priority: list, recent: list, user_id: str) -> list[dict]:
    items = []

    for entry in high_priority:
        content = entry["content"]
        doc = entry["doc"]
        eid = str(doc["_id"])
        items.append({
            "id": eid,
            "type": "urgent",
            "title": content.get("subject", ""),
            "description": content.get("preview", ""),
            "priority": "high",
            "email_ids": [eid],
        })

    seen_ids = {item["id"] for item in items}

    overdue = list(follow_ups_col().find({"user_id": user_id, "status": "overdue"}).limit(5))
    for fu in overdue:
        if fu["email_id"] not in seen_ids:
            content = get_email_content(fu["email_id"], user_id)
            items.append({
                "id": str(fu["_id"]),
                "type": "followup",
                "title": f"Overdue: {content['subject']}" if content else "Overdue follow-up",
                "description": "",
                "priority": "high",
                "email_ids": [fu["email_id"]],
            })
            seen_ids.add(fu["email_id"])

    for entry in recent:
        content = entry["content"]
        doc = entry["doc"]
        eid = str(doc["_id"])
        if eid not in seen_ids and len(items) < 15:
            items.append({
                "id": eid,
                "type": "info",
                "title": content.get("subject", ""),
                "description": content.get("preview", ""),
                "priority": content.get("priority", "medium"),
                "email_ids": [eid],
            })
            seen_ids.add(eid)

    return items


def generate_ai_briefing(user_id: str) -> list[dict]:
    """Generate per-mailbox AI summaries for today's emails only."""
    now = datetime.now(timezone.utc)
    start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)

    mailboxes = list(mailboxes_col().find({"user_id": user_id}))
    if not mailboxes:
        return []

    today_docs = list(email_metadata_col().find({
        "user_id": user_id,
        "date": {"$gte": start_of_day},
        "archived": False,
        "trashed": False,
    }).sort("date", -1).limit(50))

    if not today_docs:
        return [
            {
                "mailbox_name": mb["name"],
                "mailbox_email": mb["email"],
                "color": mb.get("color", "#0ea5e9"),
                "today_count": 0,
                "summary": "No emails received today.",
            }
            for mb in mailboxes
        ]

    all_ids = [str(d["_id"]) for d in today_docs]
    content_map = get_emails_content_batch(all_ids, user_id)

    mailbox_map = {str(mb["_id"]): mb for mb in mailboxes}
    grouped: dict[str, list[dict]] = {}
    for doc in today_docs:
        mb_id = doc.get("mailbox_id", "")
        if mb_id not in grouped:
            grouped[mb_id] = []
        eid = str(doc["_id"])
        content = content_map.get(eid, {})
        grouped[mb_id].append({
            "subject": content.get("subject", "No subject"),
            "from": content.get("from_name") or content.get("from_email", "Unknown"),
            "preview": (content.get("preview", "") or "")[:120],
            "priority": content.get("priority", "medium"),
            "read": doc.get("read", False),
        })

    sections = []
    mailbox_order = []
    for mb_id, emails in grouped.items():
        mb = mailbox_map.get(mb_id)
        if not mb:
            continue
        mailbox_order.append({
            "id": mb_id,
            "name": mb["name"],
            "email": mb["email"],
            "color": mb.get("color", "#0ea5e9"),
            "count": len(emails),
        })
        email_lines = "\n".join(
            f"  - From: {e['from']} | Subject: {e['subject']} | Priority: {e['priority']} | {'Read' if e['read'] else 'Unread'}"
            for e in emails[:15]
        )
        sections.append(
            f"Mailbox: {mb['name']} ({mb['email']}) — {len(emails)} emails today:\n{email_lines}"
        )

    if not sections:
        return []

    prompt = "\n\n".join(sections)

    from api.controllers.settings.services import get_user_preferences_prompt
    user_prefs = get_user_preferences_prompt(user_id)
    prefs_hint = f"\n{user_prefs}\nPrioritise items the user marked as important.\n" if user_prefs else ""

    llm_result = chat_json(
        system_prompt=(
            "You are a smart email assistant. Analyze today's emails for each mailbox separately. "
            + prefs_hint +
            'Return JSON: {"summaries": [{"mailbox": "exact mailbox name", "summary": "2-3 bullet points"}]}. '
            "Each summary must have 2-3 concise bullet points using the • character. "
            "Mention specific senders and subjects. Be actionable and specific. "
            "If a mailbox only has routine emails, summarize briefly."
        ),
        user_message=prompt,
        temperature=0.4,
    )

    summaries = llm_result.get("summaries", [])
    summary_map = {s.get("mailbox", ""): s.get("summary", "") for s in summaries}

    result = []
    for info in mailbox_order:
        result.append({
            "mailbox_name": info["name"],
            "mailbox_email": info["email"],
            "color": info["color"],
            "today_count": info["count"],
            "summary": summary_map.get(info["name"], "No summary available."),
        })

    for mb in mailboxes:
        mb_id = str(mb["_id"])
        if mb_id not in grouped:
            result.append({
                "mailbox_name": mb["name"],
                "mailbox_email": mb["email"],
                "color": mb.get("color", "#0ea5e9"),
                "today_count": 0,
                "summary": "No emails received today.",
            })

    return result
