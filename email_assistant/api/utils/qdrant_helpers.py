"""
Shared helpers for reading email content from Qdrant (primary store).
MongoDB only holds mutable state (read/starred/labels/archived/trashed).
All immutable email content lives in Qdrant payloads.
"""

import json

from django.conf import settings
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

from database.db import get_qdrant


def get_email_content(email_id: str, user_id: str) -> dict | None:
    """Fetch content for a single email from Qdrant (chunk_index=0)."""
    qdrant = get_qdrant()
    results = qdrant.scroll(
        collection_name=settings.QDRANT_COLLECTION_EMAIL_CHUNKS,
        scroll_filter=Filter(
            must=[
                FieldCondition(key="email_id", match=MatchValue(value=email_id)),
                FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                FieldCondition(key="chunk_index", match=MatchValue(value=0)),
            ]
        ),
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    points = results[0] if results else []
    if not points:
        return None
    return _payload_to_content(points[0].payload)


def get_emails_content_batch(email_ids: list[str], user_id: str) -> dict:
    """Fetch content for multiple emails from Qdrant. Returns {email_id: content_dict}."""
    if not email_ids:
        return {}

    qdrant = get_qdrant()
    content_map = {}
    BATCH = 100
    for i in range(0, len(email_ids), BATCH):
        batch_ids = email_ids[i : i + BATCH]
        results = qdrant.scroll(
            collection_name=settings.QDRANT_COLLECTION_EMAIL_CHUNKS,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                    FieldCondition(key="chunk_index", match=MatchValue(value=0)),
                    FieldCondition(key="email_id", match=MatchAny(any=batch_ids)),
                ],
            ),
            limit=len(batch_ids) + 10,
            with_payload=True,
            with_vectors=False,
        )
        points = results[0] if results else []
        for point in points:
            eid = point.payload.get("email_id", "")
            if eid:
                content_map[eid] = _payload_to_content(point.payload)
    return content_map


def scroll_all_chunk0(
    user_id: str,
    mailbox_id: str | None = None,
    category: str | None = None,
    priority: str | None = None,
) -> list[dict]:
    """Scroll all chunk_index=0 records for a user. Returns list of content dicts."""
    qdrant = get_qdrant()
    must_filters = [
        FieldCondition(key="user_id", match=MatchValue(value=user_id)),
        FieldCondition(key="chunk_index", match=MatchValue(value=0)),
    ]
    if mailbox_id:
        must_filters.append(FieldCondition(key="mailbox_id", match=MatchValue(value=mailbox_id)))
    if category:
        must_filters.append(FieldCondition(key="category", match=MatchValue(value=category)))
    if priority:
        must_filters.append(FieldCondition(key="priority", match=MatchValue(value=priority)))

    all_points = []
    offset = None
    while True:
        results = qdrant.scroll(
            collection_name=settings.QDRANT_COLLECTION_EMAIL_CHUNKS,
            scroll_filter=Filter(must=must_filters),
            limit=250,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points, next_offset = results
        all_points.extend(points)
        if next_offset is None:
            break
        offset = next_offset

    return [_payload_to_content(p.payload) for p in all_points]


def _payload_to_content(payload: dict) -> dict:
    """Convert a Qdrant payload into a clean content dictionary."""
    to_raw = payload.get("to", "[]")
    if isinstance(to_raw, str):
        try:
            to_list = json.loads(to_raw)
        except (json.JSONDecodeError, TypeError):
            to_list = []
    else:
        to_list = to_raw if isinstance(to_raw, list) else []

    attachments_raw = payload.get("attachments", "[]")
    if isinstance(attachments_raw, str):
        try:
            attachments_list = json.loads(attachments_raw)
        except (json.JSONDecodeError, TypeError):
            attachments_list = []
    else:
        attachments_list = attachments_raw if isinstance(attachments_raw, list) else []

    return {
        "email_id": payload.get("email_id", ""),
        "mailbox_id": payload.get("mailbox_id", ""),
        "subject": payload.get("subject", ""),
        "from_name": payload.get("from_name", ""),
        "from_email": payload.get("from_email", ""),
        "to": to_list,
        "date": payload.get("date", ""),
        "preview": payload.get("preview", ""),
        "has_attachment": payload.get("has_attachment", False),
        "priority": payload.get("priority", "medium"),
        "category": payload.get("category") or None,
        "body_chunk": payload.get("body_chunk", ""),
        "body_html": payload.get("body_html", ""),
        "total_chunks": payload.get("total_chunks", 0),
        "attachments": attachments_list,
        "attachment_text": payload.get("attachment_text", ""),
    }
