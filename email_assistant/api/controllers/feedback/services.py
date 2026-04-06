"""Store user feedback in MongoDB."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from database.db import feedback_col

MAX_LEN = 8000
ALLOWED_CATEGORIES = frozenset({"bug", "idea", "general"})


def submit_feedback(user_id: str, message: str, category: str | None) -> dict | None:
    text = (message or "").strip()
    if not text:
        return None
    if len(text) > MAX_LEN:
        return None
    cat = (category or "general").strip().lower()
    if cat not in ALLOWED_CATEGORIES:
        cat = "general"
    doc = {
        "_id": str(uuid.uuid4()),
        "user_id": user_id,
        "message": text,
        "category": cat,
        "created_at": datetime.now(timezone.utc),
    }
    feedback_col().insert_one(doc)
    return {"id": doc["_id"], "status": "received"}
