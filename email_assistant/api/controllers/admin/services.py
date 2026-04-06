from datetime import datetime, timezone, timedelta

from bson import ObjectId
from bson.errors import InvalidId
from django.conf import settings as django_settings
from qdrant_client.models import Filter, FieldCondition, MatchValue

from api.admin_auth import user_is_admin
from database.db import (
    users_col,
    user_settings_col,
    mailboxes_col,
    email_metadata_col,
    follow_ups_col,
    refresh_tokens_col,
    email_attachments_col,
    agent_profiles_col,
    feedback_col,
    meetings_col,
    get_db,
    get_qdrant,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _user_summary(doc: dict) -> dict:
    return {
        "id": str(doc["_id"]),
        "email": doc["email"],
        "name": doc["name"],
        "timezone": doc.get("timezone", "UTC"),
        "created_at": doc["created_at"],
        "updated_at": doc.get("updated_at"),
        "is_admin": user_is_admin(doc),
        "disabled": bool(doc.get("disabled")),
    }


def _qdrant_ok() -> bool:
    try:
        from database.db import get_qdrant
        get_qdrant().get_collections()
        return True
    except Exception:
        return False


# ── stats (enhanced) ─────────────────────────────────────────────────────────

def get_stats() -> dict:
    db = get_db()
    try:
        db.command("ping")
        mongo_ok = True
    except Exception:
        mongo_ok = False

    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_ago = now - timedelta(days=7)

    users = users_col()
    mboxes = mailboxes_col()
    emails = email_metadata_col()
    fups = follow_ups_col()

    total_users = users.count_documents({})
    total_mailboxes = mboxes.count_documents({})
    total_emails = emails.count_documents({})
    total_follow_ups = fups.count_documents({})

    signups_today = users.count_documents({"created_at": {"$gte": today_start}})
    signups_week = users.count_documents({"created_at": {"$gte": week_ago}})

    sync_statuses = {}
    for s in ["synced", "syncing", "pending", "error", "cancelled"]:
        sync_statuses[s] = mboxes.count_documents({"sync_status": s})

    active_sessions = refresh_tokens_col().count_documents({"expires_at": {"$gte": now}})
    total_attachments = email_attachments_col().count_documents({})
    total_agent_profiles = agent_profiles_col().count_documents({})
    total_meetings = meetings_col().count_documents({})
    total_feedback = feedback_col().count_documents({})

    daily_signups = []
    for i in range(6, -1, -1):
        day = (now - timedelta(days=i)).replace(hour=0, minute=0, second=0, microsecond=0)
        next_day = day + timedelta(days=1)
        count = users.count_documents({"created_at": {"$gte": day, "$lt": next_day}})
        daily_signups.append({"date": day.strftime("%Y-%m-%d"), "count": count})

    top_users_pipeline = [
        {"$group": {"_id": "$user_id", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 5},
    ]
    top_users_raw = list(emails.aggregate(top_users_pipeline))
    top_users = []
    for row in top_users_raw:
        uid = row["_id"]
        udoc = users.find_one({"_id": ObjectId(uid)}) if uid else None
        top_users.append({
            "user_id": uid,
            "email": udoc["email"] if udoc else "unknown",
            "name": udoc["name"] if udoc else "unknown",
            "email_count": row["count"],
        })

    daily_email_volume = []
    for i in range(6, -1, -1):
        day = (now - timedelta(days=i)).replace(hour=0, minute=0, second=0, microsecond=0)
        next_day = day + timedelta(days=1)
        em_c = emails.count_documents({"date": {"$gte": day, "$lt": next_day}})
        daily_email_volume.append({"date": day.strftime("%Y-%m-%d"), "count": em_c})

    top_mb_pipeline = [
        {"$group": {"_id": "$mailbox_id", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 5},
    ]
    top_mailboxes = []
    for row in emails.aggregate(top_mb_pipeline):
        mid = row["_id"]
        mbdoc = None
        if mid:
            try:
                mbdoc = mboxes.find_one({"_id": ObjectId(mid)})
            except Exception:
                mbdoc = mboxes.find_one({"_id": mid}) if isinstance(mid, str) else None
        uid_mb = (mbdoc or {}).get("user_id", "")
        udoc_mb = users.find_one({"_id": ObjectId(uid_mb)}) if uid_mb else None
        top_mailboxes.append({
            "id": str(mid) if mid is not None else "",
            "name": (mbdoc or {}).get("name", "—"),
            "email": (mbdoc or {}).get("email", "—"),
            "user_id": uid_mb,
            "user_email": udoc_mb["email"] if udoc_mb else "unknown",
            "email_count": row["count"],
        })

    total_read = emails.count_documents({"read": True})
    total_unread = emails.count_documents({"read": False})
    total_starred = emails.count_documents({"starred": True})

    follow_up_statuses: dict[str, int] = {}
    try:
        for fu_row in fups.aggregate([{"$group": {"_id": "$status", "count": {"$sum": 1}}}]):
            k = fu_row.get("_id")
            follow_up_statuses[str(k) if k is not None else "unknown"] = int(fu_row.get("count", 0))
    except Exception:
        pass

    feedback_by_category: dict[str, int] = {}
    for cat in ("general", "idea", "bug"):
        feedback_by_category[cat] = feedback_col().count_documents({"category": cat})
    feedback_by_category["other"] = feedback_col().count_documents(
        {"category": {"$nin": ["general", "idea", "bug"]}}
    )

    users_disabled = users.count_documents({"disabled": True})
    users_admin_flag = users.count_documents({"is_admin": True})
    meetings_conflicting = meetings_col().count_documents({"conflict": True})

    denom_u = max(total_users, 1)
    denom_m = max(total_mailboxes, 1)

    return {
        "mongodb_ok": mongo_ok,
        "qdrant_ok": _qdrant_ok(),
        "users": total_users,
        "mailboxes": total_mailboxes,
        "emails_indexed": total_emails,
        "follow_ups": total_follow_ups,
        "attachments": total_attachments,
        "agent_profiles": total_agent_profiles,
        "meetings": total_meetings,
        "feedback_submissions": total_feedback,
        "active_sessions": active_sessions,
        "signups_today": signups_today,
        "signups_week": signups_week,
        "sync_statuses": sync_statuses,
        "daily_signups": daily_signups,
        "daily_email_volume": daily_email_volume,
        "top_users": top_users,
        "top_mailboxes": top_mailboxes,
        "engagement": {
            "read": total_read,
            "unread": total_unread,
            "starred": total_starred,
        },
        "averages": {
            "emails_per_user": round(total_emails / denom_u, 1),
            "mailboxes_per_user": round(total_mailboxes / denom_u, 2),
            "emails_per_mailbox": round(total_emails / denom_m, 1),
        },
        "users_disabled": users_disabled,
        "users_admin_flag": users_admin_flag,
        "follow_up_statuses": follow_up_statuses,
        "feedback_by_category": feedback_by_category,
        "meetings_conflicting": meetings_conflicting,
    }


# ── recent activity ──────────────────────────────────────────────────────────

def get_recent_activity(limit: int = 30) -> list[dict]:
    events: list[dict] = []

    for doc in users_col().find().sort("created_at", -1).limit(limit):
        events.append({
            "type": "user_registered",
            "timestamp": doc["created_at"],
            "user_email": doc["email"],
            "user_name": doc["name"],
            "user_id": str(doc["_id"]),
        })

    for doc in mailboxes_col().find().sort("created_at", -1).limit(limit):
        uid = doc.get("user_id", "")
        udoc = users_col().find_one({"_id": ObjectId(uid)}) if uid else None
        events.append({
            "type": "mailbox_added",
            "timestamp": doc["created_at"],
            "mailbox_name": doc.get("name", ""),
            "mailbox_email": doc.get("email", ""),
            "user_email": udoc["email"] if udoc else "unknown",
            "user_id": uid,
        })

    for doc in mailboxes_col().find({"last_sync_at": {"$ne": None}}).sort("last_sync_at", -1).limit(limit):
        uid = doc.get("user_id", "")
        udoc = users_col().find_one({"_id": ObjectId(uid)}) if uid else None
        events.append({
            "type": "sync_completed",
            "timestamp": doc.get("last_sync_at"),
            "mailbox_name": doc.get("name", ""),
            "mailbox_email": doc.get("email", ""),
            "sync_status": doc.get("sync_status", ""),
            "user_email": udoc["email"] if udoc else "unknown",
            "user_id": uid,
        })

    for doc in feedback_col().find().sort("created_at", -1).limit(limit):
        uid = doc.get("user_id", "")
        udoc = None
        if uid:
            try:
                udoc = users_col().find_one({"_id": ObjectId(uid)})
            except Exception:
                udoc = None
        msg = doc.get("message") or ""
        events.append({
            "type": "feedback_submitted",
            "timestamp": doc["created_at"],
            "user_id": uid,
            "user_email": udoc["email"] if udoc else (uid or "unknown"),
            "category": doc.get("category", "general"),
            "message_preview": msg[:160] + ("…" if len(msg) > 160 else ""),
            "feedback_id": doc.get("_id"),
        })

    events.sort(key=lambda e: e.get("timestamp") or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    return events[:limit]


# ── mailboxes (all, admin) ───────────────────────────────────────────────────

def list_all_mailboxes(q: str | None, sync_status: str | None, page: int, limit: int) -> dict:
    page = max(1, page)
    limit = min(max(1, limit), 100)
    skip = (page - 1) * limit
    query: dict = {}
    if q and q.strip():
        regex = {"$regex": q.strip(), "$options": "i"}
        query["$or"] = [{"email": regex}, {"name": regex}]
    if sync_status:
        query["sync_status"] = sync_status

    col = mailboxes_col()
    total = col.count_documents(query)
    docs = list(col.find(query).sort("created_at", -1).skip(skip).limit(limit))
    rows = []
    user_cache: dict = {}
    for mb in docs:
        uid = mb.get("user_id", "")
        mid = str(mb["_id"])
        if uid not in user_cache:
            try:
                udoc = users_col().find_one({"_id": ObjectId(uid)})
            except Exception:
                udoc = None
            user_cache[uid] = udoc
        udoc = user_cache[uid]
        em_count = email_metadata_col().count_documents({"user_id": uid, "mailbox_id": mid})
        unread = email_metadata_col().count_documents({"user_id": uid, "mailbox_id": mid, "read": False})
        rows.append({
            "id": mid,
            "name": mb.get("name", ""),
            "email": mb.get("email", ""),
            "color": mb.get("color", "#3b82f6"),
            "sync_status": mb.get("sync_status", ""),
            "last_sync_at": mb.get("last_sync_at"),
            "created_at": mb.get("created_at"),
            "email_count": em_count,
            "unread": unread,
            "user_id": uid,
            "user_email": udoc["email"] if udoc else "unknown",
            "user_name": udoc["name"] if udoc else "unknown",
        })
    return {"mailboxes": rows, "total": total, "page": page, "limit": limit}


# ── feedback (admin) ──────────────────────────────────────────────────────────

def list_feedback(q: str | None, category: str | None, page: int, limit: int) -> dict:
    page = max(1, page)
    limit = min(max(1, limit), 100)
    skip = (page - 1) * limit
    query: dict = {}
    if category and category.strip() and category.strip() != "all":
        query["category"] = category.strip().lower()
    if q and q.strip():
        query["message"] = {"$regex": q.strip(), "$options": "i"}

    col = feedback_col()
    total = col.count_documents(query)
    docs = list(col.find(query).sort("created_at", -1).skip(skip).limit(limit))
    rows = []
    for doc in docs:
        uid = doc.get("user_id", "")
        udoc = None
        if uid:
            try:
                udoc = users_col().find_one({"_id": ObjectId(uid)})
            except Exception:
                udoc = None
        rows.append({
            "id": str(doc.get("_id", "")),
            "user_id": uid,
            "user_email": udoc["email"] if udoc else (uid or "—"),
            "user_name": udoc["name"] if udoc else "—",
            "category": doc.get("category", "general"),
            "message": doc.get("message", ""),
            "created_at": doc.get("created_at"),
        })
    return {"feedback": rows, "total": total, "page": page, "limit": limit}


# ── users ────────────────────────────────────────────────────────────────────

def list_users(q: str | None, page: int, limit: int) -> dict:
    page = max(1, page)
    limit = min(max(1, limit), 100)
    skip = (page - 1) * limit
    query: dict = {}
    if q and q.strip():
        regex = {"$regex": q.strip(), "$options": "i"}
        query["$or"] = [{"email": regex}, {"name": regex}]

    col = users_col()
    total = col.count_documents(query)
    cursor = col.find(query).sort("created_at", -1).skip(skip).limit(limit)
    users = []
    for doc in cursor:
        uid = str(doc["_id"])
        mb_count = mailboxes_col().count_documents({"user_id": uid})
        em_count = email_metadata_col().count_documents({"user_id": uid})
        row = _user_summary(doc)
        row["mailbox_count"] = mb_count
        row["email_count"] = em_count
        users.append(row)
    return {"users": users, "total": total, "page": page, "limit": limit}


def get_user_detail(user_id: str) -> dict | None:
    try:
        oid = ObjectId(user_id)
    except InvalidId:
        return None
    doc = users_col().find_one({"_id": oid})
    if not doc:
        return None
    uid = str(doc["_id"])
    settings = user_settings_col().find_one({"user_id": uid})
    settings_out = None
    if settings:
        settings_out = {k: v for k, v in settings.items() if k != "_id"}

    mbs = list(mailboxes_col().find({"user_id": uid}).sort("created_at", -1))
    mailboxes = []
    for mb in mbs:
        mid = str(mb["_id"])
        total_em = email_metadata_col().count_documents({"user_id": uid, "mailbox_id": mid})
        unread = email_metadata_col().count_documents({"user_id": uid, "mailbox_id": mid, "read": False})
        mailboxes.append({
            "id": mid,
            "name": mb.get("name", ""),
            "email": mb.get("email", ""),
            "color": mb.get("color", "#3b82f6"),
            "sync_status": mb.get("sync_status", ""),
            "last_sync_at": mb.get("last_sync_at"),
            "created_at": mb.get("created_at"),
            "email_count": total_em,
            "unread": unread,
        })

    fu_open = follow_ups_col().count_documents({"user_id": uid, "status": {"$ne": "completed"}})
    fu_total = follow_ups_col().count_documents({"user_id": uid})
    attachment_count = email_attachments_col().count_documents({"user_id": uid})

    profile = agent_profiles_col().find_one({"user_id": uid})
    has_agent_profile = profile is not None

    total_read = email_metadata_col().count_documents({"user_id": uid, "read": True})
    total_unread = email_metadata_col().count_documents({"user_id": uid, "read": False})
    total_starred = email_metadata_col().count_documents({"user_id": uid, "starred": True})

    return {
        "user": _user_summary(doc),
        "settings": settings_out,
        "mailboxes": mailboxes,
        "follow_ups_open": fu_open,
        "follow_ups_total": fu_total,
        "attachment_count": attachment_count,
        "has_agent_profile": has_agent_profile,
        "email_stats": {
            "total_read": total_read,
            "total_unread": total_unread,
            "total_starred": total_starred,
        },
    }


def patch_user(user_id: str, disabled: bool | None, is_admin: bool | None) -> dict | None:
    try:
        oid = ObjectId(user_id)
    except InvalidId:
        return None
    updates: dict = {"updated_at": datetime.now(timezone.utc)}
    if disabled is not None:
        updates["disabled"] = disabled
    if is_admin is not None:
        updates["is_admin"] = is_admin
    if len(updates) == 1:
        doc = users_col().find_one({"_id": oid})
        return _user_summary(doc) if doc else None
    users_col().update_one({"_id": oid}, {"$set": updates})
    doc = users_col().find_one({"_id": oid})
    return _user_summary(doc) if doc else None


def delete_user(user_id: str) -> bool:
    try:
        oid = ObjectId(user_id)
    except InvalidId:
        return False
    doc = users_col().find_one({"_id": oid})
    if not doc:
        return False
    uid = str(doc["_id"])
    email_metadata_col().delete_many({"user_id": uid})
    follow_ups_col().delete_many({"user_id": uid})
    mailboxes_col().delete_many({"user_id": uid})
    user_settings_col().delete_many({"user_id": uid})
    email_attachments_col().delete_many({"user_id": uid})
    agent_profiles_col().delete_many({"user_id": uid})
    refresh_tokens_col().delete_many({"user_id": uid})
    meetings_col().delete_many({"user_id": uid})
    feedback_col().delete_many({"user_id": uid})
    try:
        get_qdrant().delete(
            collection_name=django_settings.QDRANT_COLLECTION_EMAIL_CHUNKS,
            points_selector=Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=uid))]
            ),
        )
    except Exception:
        pass
    users_col().delete_one({"_id": oid})
    return True
