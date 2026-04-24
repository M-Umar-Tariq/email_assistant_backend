from datetime import datetime, timezone, timedelta
from bson import ObjectId
import json

from pymongo import UpdateOne
from database.db import follow_ups_col, email_metadata_col
from api.utils.qdrant_helpers import get_email_content, get_emails_content_batch
from api.utils.llm import chat_json


def create_follow_up(user_id: str, data: dict) -> dict:
    doc = {
        "user_id": user_id,
        "email_id": data["email_id"],
        "due_date": data["due_date"],
        "status": "pending",
        "auto_reminder_sent": False,
        "days_waiting": 0,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    result = follow_ups_col().insert_one(doc)
    doc["_id"] = result.inserted_id
    return _serialize(doc)


def list_follow_ups(user_id: str, status_filter: str | None = None) -> list[dict]:
    query: dict = {"user_id": user_id}
    if status_filter and status_filter != "all":
        query["status"] = status_filter
    docs = list(follow_ups_col().find(query).sort("due_date", 1))
    if not docs:
        return []

    now = datetime.now(timezone.utc)

    # Detect overdue in memory and batch-update MongoDB in one round-trip
    bulk_ops = []
    for doc in docs:
        if doc.get("status") not in ("completed", "snoozed"):
            due = doc.get("due_date")
            if isinstance(due, datetime) and due < now:
                days_waiting = max((now - due).days, 1)
                doc["status"] = "overdue"
                doc["days_waiting"] = days_waiting
                bulk_ops.append(UpdateOne(
                    {"_id": doc["_id"]},
                    {"$set": {"status": "overdue", "days_waiting": days_waiting, "updated_at": now}},
                ))
    if bulk_ops:
        follow_ups_col().bulk_write(bulk_ops, ordered=False)

    # Batch-fetch all email content from Qdrant in one shot
    email_ids = [doc["email_id"] for doc in docs]
    content_map = get_emails_content_batch(email_ids, user_id)

    results = []
    for doc in docs:
        content = content_map.get(doc["email_id"])
        # Skip orphan follow-ups (email was deleted e.g. when mailbox was removed)
        if not content:
            continue
        fu = _serialize(doc)
        fu["email_subject"] = content.get("subject", "")
        fu["from_name"] = content.get("from_name", "")
        fu["from_email"] = content.get("from_email", "")
        results.append(fu)
    return results


def update_follow_up(user_id: str, follow_up_id: str, data: dict) -> dict | None:
    data["updated_at"] = datetime.now(timezone.utc)
    follow_ups_col().update_one(
        {"_id": ObjectId(follow_up_id), "user_id": user_id}, {"$set": data}
    )
    doc = follow_ups_col().find_one({"_id": ObjectId(follow_up_id), "user_id": user_id})
    return _serialize(doc) if doc else None


def complete_follow_up(user_id: str, follow_up_id: str) -> dict | None:
    return update_follow_up(user_id, follow_up_id, {"status": "completed"})


def delete_follow_up(user_id: str, follow_up_id: str) -> bool:
    r = follow_ups_col().delete_one({"_id": ObjectId(follow_up_id), "user_id": user_id})
    return r.deleted_count > 0


def auto_detect_today_follow_ups(user_id: str) -> dict:
    now = datetime.now(timezone.utc)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    today_date_filter = {"$or": [
        {"original_date": {"$gte": start}},
        {"original_date": None, "date": {"$gte": start}},
    ]}
    q = {"$and": [
        {"user_id": user_id, "archived": False, "trashed": False},
        today_date_filter,
        {"$or": [{"snoozed_until": None}, {"snoozed_until": {"$lte": now}}]},
    ]}
    docs = list(email_metadata_col().find(q).limit(200))
    email_ids = [str(d["_id"]) for d in docs]
    content_map = get_emails_content_batch(email_ids, user_id) if email_ids else {}
    items: list[dict] = []
    for d in docs:
        eid = str(d["_id"])
        content = content_map.get(eid) or {}
        items.append({
            "id": eid,
            "subject": content.get("subject", ""),
            "from_name": content.get("from_name", ""),
            "from_email": content.get("from_email", ""),
            "preview": content.get("preview", ""),
            "body": content.get("body_chunk", ""),
        })
    decisions = []
    created = 0
    existing = 0
    if items:
        payload = json.dumps({"emails": items})
        res = chat_json(
            "Decide follow-ups. Return JSON: {\"decisions\": [{\"id\":\"...\",\"need\":true|false,\"due_hours\":number}]}",
            f"Evaluate and return decisions for these emails:\n{payload}",
            temperature=0.1,
        )
        decisions = res.get("decisions", [])
    for dec in decisions:
        if not dec.get("need"):
            continue
        eid = dec.get("id", "")
        if not eid:
            continue
        exists = follow_ups_col().find_one({
            "user_id": user_id,
            "email_id": eid,
            "status": {"$in": ["pending", "overdue", "snoozed"]},
        })
        if exists:
            existing += 1
            continue
        due_hours = dec.get("due_hours")
        due_date = now + timedelta(hours=int(due_hours)) if isinstance(due_hours, (int, float)) else now + timedelta(hours=24)
        doc = {
            "user_id": user_id,
            "email_id": eid,
            "due_date": due_date,
            "status": "pending",
            "auto_reminder_sent": False,
            "days_waiting": 0,
            "created_at": now,
            "updated_at": now,
        }
        follow_ups_col().insert_one(doc)
        created += 1
    return {"scanned": len(items), "created": created, "skipped_existing": existing}


def _serialize(doc: dict) -> dict:
    return {
        "id": str(doc["_id"]),
        "user_id": doc["user_id"],
        "email_id": doc["email_id"],
        "due_date": doc["due_date"],
        "status": doc["status"],
        "auto_reminder_sent": doc.get("auto_reminder_sent", False),
        "days_waiting": doc.get("days_waiting", 0),
        "created_at": doc.get("created_at"),
    }
