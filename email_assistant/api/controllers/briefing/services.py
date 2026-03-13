from datetime import datetime, timezone

from database.db import email_metadata_col, mailboxes_col, follow_ups_col
from api.utils.llm import chat, chat_json
from api.utils.qdrant_helpers import get_email_content, get_emails_content_batch


def get_briefing(user_id: str) -> dict:
    now = datetime.now(timezone.utc)
    start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)

    unread_count = email_metadata_col().count_documents({
        "user_id": user_id, "read": False, "archived": False, "trashed": False,
    })

    # Get today's emails from MongoDB
    today_docs = list(email_metadata_col().find({
        "user_id": user_id,
        "date": {"$gte": start_of_day},
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

    return {
        "stats": {
            "unread_total": unread_count,
            "high_priority": len(high_priority),
            "overdue_follow_ups": overdue_follow_ups,
            "pending_follow_ups": pending_follow_ups,
        },
        "mailboxes": mailbox_summary,
        "items": briefing_items,
    }


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

    llm_result = chat_json(
        system_prompt=(
            "You are a smart email assistant. Analyze today's emails for each mailbox separately. "
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
