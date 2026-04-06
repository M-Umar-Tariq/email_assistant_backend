from datetime import datetime, timezone

from django.contrib.auth.hashers import make_password, check_password
from bson import ObjectId

from database.db import users_col, user_settings_col, next_user_settings_int_id
from api.admin_auth import user_is_admin


def create_user(email: str, password: str, name: str) -> dict:
    if users_col().find_one({"email": email}):
        raise ValueError("A user with this email already exists")

    doc = {
        "email": email,
        "password_hash": make_password(password),
        "name": name,
        "timezone": "UTC",
        "disabled": False,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    result = users_col().insert_one(doc)
    doc["_id"] = result.inserted_id

    user_settings_col().insert_one({
        "id": next_user_settings_int_id(),
        "user_id": str(result.inserted_id),
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
    })

    return _serialize(doc)


def authenticate(email: str, password: str) -> dict:
    user = users_col().find_one({"email": email})
    if not user or not check_password(password, user["password_hash"]):
        raise ValueError("Invalid email or password")
    if user.get("disabled"):
        raise ValueError("Account disabled")
    return _serialize(user)


def get_user_by_id(user_id: str) -> dict | None:
    user = users_col().find_one({"_id": ObjectId(user_id)})
    return _serialize(user) if user else None


def update_user(user_id: str, data: dict) -> dict | None:
    data["updated_at"] = datetime.now(timezone.utc)
    users_col().update_one({"_id": ObjectId(user_id)}, {"$set": data})
    return get_user_by_id(user_id)


def _serialize(doc: dict) -> dict:
    return {
        "id": str(doc["_id"]),
        "email": doc["email"],
        "name": doc["name"],
        "timezone": doc.get("timezone", "UTC"),
        "created_at": doc["created_at"],
        "is_admin": user_is_admin(doc),
        "disabled": bool(doc.get("disabled")),
    }
