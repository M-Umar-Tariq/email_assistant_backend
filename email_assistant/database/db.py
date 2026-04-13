"""
Singleton clients for MongoDB and Qdrant.
Import `get_db` or `get_qdrant` anywhere in the project.
"""

from functools import lru_cache
from datetime import datetime, timezone, timedelta
from django.conf import settings
from pymongo import MongoClient
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


# ── MongoDB ──────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _mongo_client() -> MongoClient:
    return MongoClient(settings.MONGODB_URI, tz_aware=True)


def get_db():
    """Return the default MongoDB database."""
    return _mongo_client()[settings.MONGODB_DB_NAME]


# Convenience accessors for collections
def users_col():
    return get_db()["users"]

def mailboxes_col():
    return get_db()["mailboxes"]

def email_metadata_col():
    return get_db()["email_metadata"]

def follow_ups_col():
    return get_db()["follow_ups"]

def user_settings_col():
    return get_db()["user_settings"]


def next_user_settings_int_id() -> int:
    """
    Next integer `id` for user_settings documents.

    Djongo creates a unique index on Django's primary key field `id`. Raw PyMongo
    inserts without `id` store null for every row, causing E11000 duplicate key
    on the second insert.
    """
    col = user_settings_col()
    pipeline = [{"$group": {"_id": None, "max_id": {"$max": "$id"}}}]
    rows = list(col.aggregate(pipeline))
    if not rows or rows[0].get("max_id") is None:
        return 1
    try:
        return int(rows[0]["max_id"]) + 1
    except (TypeError, ValueError):
        return 1


def refresh_tokens_col():
    return get_db()["refresh_tokens"]

def email_attachments_col():
    return get_db()["email_attachments"]


def next_email_attachment_int_id() -> int:
    """
    Next integer `id` for email_attachments rows.

    Djongo/Django may create a unique index on the ORM primary key field `id`.
    Raw PyMongo upserts without `id` leave it null, so only one row can exist
    (E11000 duplicate key { id: null }). Use $setOnInsert with this value.
    """
    col = email_attachments_col()
    pipeline = [{"$group": {"_id": None, "max_id": {"$max": "$id"}}}]
    rows = list(col.aggregate(pipeline))
    if not rows or rows[0].get("max_id") is None:
        return 1
    try:
        return int(rows[0]["max_id"]) + 1
    except (TypeError, ValueError):
        return 1

def agent_profiles_col():
    return get_db()["agent_profiles"]


def meetings_col():
    """User calendar meetings (from email AI extraction or manual)."""
    return get_db()["meetings"]


def feedback_col():
    """User-submitted product feedback (bug reports, ideas, etc.)."""
    return get_db()["feedback"]


# ── Qdrant ───────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_qdrant() -> QdrantClient:
    kwargs = {
        "url": settings.QDRANT_URL,
        "check_compatibility": False,
        "timeout": getattr(settings, "QDRANT_TIMEOUT", 10),
    }
    if getattr(settings, "QDRANT_API_KEY", None):
        kwargs["api_key"] = settings.QDRANT_API_KEY
    return QdrantClient(**kwargs)


def ensure_qdrant_collection(vector_size: int = 1536):
    """Create the email_chunks collection if it does not exist, and ensure payload indexes for filtering."""
    client = get_qdrant()
    collection_name = settings.QDRANT_COLLECTION_EMAIL_CHUNKS
    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )
    # Payload indexes for efficient filtering
    keyword_fields = ("email_id", "user_id", "mailbox_id", "category", "priority")
    for field_name in keyword_fields:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema="keyword",
            )
        except Exception:
            pass
    # Integer index for chunk_index (used to fetch chunk_index=0 for metadata)
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name="chunk_index",
            field_schema="integer",
        )
    except Exception:
        pass


# ── MongoDB indexes (call once at startup or via management command) ─────────

def _dedup_email_metadata():
    """Remove duplicate emails keeping the oldest entry per (user_id, mailbox_id, message_id)."""
    col = email_metadata_col()
    pipeline = [
        {"$group": {
            "_id": {"user_id": "$user_id", "mailbox_id": "$mailbox_id", "message_id": "$message_id"},
            "count": {"$sum": 1},
            "keep_id": {"$first": "$_id"},
            "all_ids": {"$push": "$_id"},
        }},
        {"$match": {"count": {"$gt": 1}}},
    ]
    duplicates = list(col.aggregate(pipeline))
    removed = 0
    for group in duplicates:
        ids_to_delete = [eid for eid in group["all_ids"] if eid != group["keep_id"]]
        if ids_to_delete:
            col.delete_many({"_id": {"$in": ids_to_delete}})
            removed += len(ids_to_delete)
    if removed:
        print(f"[DEDUP] Removed {removed} duplicate email(s) from email_metadata")


def _safe_create_index(collection, keys, **kwargs):
    """Create index, silently skip if it already exists (possibly under a different name)."""
    try:
        collection.create_index(keys, **kwargs)
    except Exception:
        pass


def ensure_indexes():
    """Create useful indexes on MongoDB collections."""
    _dedup_email_metadata()
    _safe_create_index(users_col(), "email", unique=True)
    _safe_create_index(mailboxes_col(), [("user_id", 1)])
    _safe_create_index(email_metadata_col(), [("user_id", 1), ("mailbox_id", 1), ("date", -1)])
    _safe_create_index(email_metadata_col(), [("user_id", 1), ("message_id", 1), ("mailbox_id", 1)], unique=True)
    _safe_create_index(follow_ups_col(), [("user_id", 1), ("status", 1)])
    _safe_create_index(refresh_tokens_col(), "expires_at", expireAfterSeconds=0)
    _safe_create_index(agent_profiles_col(), "user_id", unique=True)
    _safe_create_index(meetings_col(), [("user_id", 1), ("start", 1)])
    _safe_create_index(meetings_col(), [("user_id", 1), ("end", 1)])
    _safe_create_index(feedback_col(), [("user_id", 1), ("created_at", -1)])


def reset_stale_syncs(max_age_minutes: int = 15):
    """Reset stale 'syncing' states left over from crashes/restarts."""
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(minutes=max_age_minutes)
    try:
        res = mailboxes_col().update_many(
            {
                "sync_status": "syncing",
                "$or": [
                    {"sync_started_at": {"$exists": False}},
                    {"sync_started_at": {"$lte": cutoff}},
                ],
            },
            {"$set": {"sync_status": "cancelled", "sync_started_at": None}},
        )
        if getattr(res, "modified_count", 0):
            print(f"[STARTUP] Reset {res.modified_count} stale sync lock(s)")
    except Exception as e:
        print(f"[STARTUP] Reset sync locks failed: {e}")

