from datetime import datetime, timezone

from database.db import user_settings_col, next_user_settings_int_id


def get_settings(user_id: str) -> dict:
    doc = user_settings_col().find_one({"user_id": user_id})
    if not doc:
        doc = _default_settings(user_id)
        doc["id"] = next_user_settings_int_id()
        user_settings_col().insert_one(doc)
    return _serialize(doc)


def update_settings(user_id: str, data: dict) -> dict:
    data["updated_at"] = datetime.now(timezone.utc)
    col = user_settings_col()
    if col.count_documents({"user_id": user_id}, limit=1) == 0:
        base = _default_settings(user_id)
        base["id"] = next_user_settings_int_id()
        base.update(data)
        col.insert_one(base)
        return get_settings(user_id)
    col.update_one({"user_id": user_id}, {"$set": data})
    return get_settings(user_id)


def _default_settings(user_id: str) -> dict:
    return {
        "user_id": user_id,
        "daily_briefing": True,
        "slack_digest": False,
        "critical_alerts": True,
        "ai_suggestions": True,
        "auto_labeling": True,
        "thread_summaries": True,
        "sync_range_months": 12,
        "occupation": "",
        "important_emails_notes": "",
        "draft_style_notes": "",
        "ai_label_rules": [],
        "onboarding_completed": False,
        "updated_at": datetime.now(timezone.utc),
    }


def get_user_preferences_prompt(user_id: str) -> str:
    """Build a plain-text block describing the user's stated preferences.
    Designed to be injected into LLM system prompts."""
    doc = user_settings_col().find_one({"user_id": user_id}) or {}

    parts: list[str] = []

    occ = (doc.get("occupation") or "").strip()
    if occ:
        parts.append(f"Occupation/role: {occ}")

    imp = (doc.get("important_emails_notes") or "").strip()
    if imp:
        parts.append(f"Important emails: {imp}")

    draft = (doc.get("draft_style_notes") or "").strip()
    if draft:
        parts.append(f"Preferred draft style: {draft}")

    rules = doc.get("ai_label_rules")
    if isinstance(rules, list) and rules:
        lines = [f"  - {r.get('name','')}: {r.get('instruction','')}" for r in rules if r.get("name")]
        if lines:
            parts.append("Custom label rules:\n" + "\n".join(lines))

    if not parts:
        return ""

    return "USER PREFERENCES (stated by the user):\n" + "\n".join(parts)


def relabel_all_emails(user_id: str) -> dict:
    """Re-classify ALL of the user's emails using their current ai_label_rules.
    Also refreshes priority. Returns count of updated emails."""
    from database.db import email_metadata_col
    from api.utils.qdrant_helpers import get_emails_content_batch
    from api.utils.classify import assign_labels_batch, classify_emails_batch

    settings = get_settings(user_id)
    label_rules = settings.get("ai_label_rules") or []

    col = email_metadata_col()
    docs = list(col.find({"user_id": user_id, "archived": False, "trashed": False}).sort("date", -1))
    if not docs:
        return {"updated": 0}

    email_ids = [str(d["_id"]) for d in docs]
    content_map = get_emails_content_batch(email_ids, user_id)

    email_summaries = []
    for doc in docs:
        eid = str(doc["_id"])
        c = content_map.get(eid, {})
        email_summaries.append({
            "subject": c.get("subject") or doc.get("subject", ""),
            "from_name": c.get("from_name") or doc.get("from_name", ""),
            "from_email": c.get("from_email") or doc.get("from_email", ""),
            "preview": c.get("preview") or doc.get("preview", ""),
        })

    if label_rules:
        all_labels = assign_labels_batch(email_summaries, label_rules)
    else:
        all_labels = [[] for _ in docs]

    all_priorities = classify_emails_batch(email_summaries, user_id=user_id)

    updated = 0
    for doc, labels, cls in zip(docs, all_labels, all_priorities):
        priority = cls.get("priority", "medium")
        col.update_one(
            {"_id": doc["_id"]},
            {"$set": {"labels": labels, "priority": priority}},
        )
        updated += 1

    return {"updated": updated}


def _serialize(doc: dict) -> dict:
    rules = doc.get("ai_label_rules")
    if not isinstance(rules, list):
        rules = []
    return {
        "user_id": doc["user_id"],
        "daily_briefing": doc.get("daily_briefing", True),
        "slack_digest": doc.get("slack_digest", False),
        "critical_alerts": doc.get("critical_alerts", True),
        "ai_suggestions": doc.get("ai_suggestions", True),
        "auto_labeling": doc.get("auto_labeling", True),
        "thread_summaries": doc.get("thread_summaries", True),
        "sync_range_months": doc.get("sync_range_months", 12),
        "occupation": doc.get("occupation") or "",
        "important_emails_notes": doc.get("important_emails_notes") or "",
        "draft_style_notes": doc.get("draft_style_notes") or "",
        "ai_label_rules": rules,
        "onboarding_completed": bool(doc.get("onboarding_completed")),
    }
