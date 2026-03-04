from datetime import datetime, timezone

from database.db import user_settings_col


def get_settings(user_id: str) -> dict:
    doc = user_settings_col().find_one({"user_id": user_id})
    if not doc:
        doc = _default_settings(user_id)
        user_settings_col().insert_one(doc)
    return _serialize(doc)


def update_settings(user_id: str, data: dict) -> dict:
    data["updated_at"] = datetime.now(timezone.utc)
    user_settings_col().update_one(
        {"user_id": user_id},
        {"$set": data},
        upsert=True,
    )
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
        "updated_at": datetime.now(timezone.utc),
    }


def _serialize(doc: dict) -> dict:
    return {
        "user_id": doc["user_id"],
        "daily_briefing": doc.get("daily_briefing", True),
        "slack_digest": doc.get("slack_digest", False),
        "critical_alerts": doc.get("critical_alerts", True),
        "ai_suggestions": doc.get("ai_suggestions", True),
        "auto_labeling": doc.get("auto_labeling", True),
        "thread_summaries": doc.get("thread_summaries", True),
        "sync_range_months": doc.get("sync_range_months", 12),
    }
