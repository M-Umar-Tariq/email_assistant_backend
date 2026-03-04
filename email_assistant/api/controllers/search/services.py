"""
Semantic search: embed query -> Qdrant -> Cohere rerank -> return emails.
Content comes entirely from Qdrant; mutable state (read/starred) from MongoDB.
"""

from django.conf import settings
from qdrant_client.models import Filter, FieldCondition, MatchValue

from database.db import get_qdrant, email_metadata_col
from api.utils.embedding import embed_text
from api.utils.rerank import rerank


def search_emails(
    user_id: str,
    query: str,
    mailbox_id: str | None = None,
    limit: int = 20,
) -> list[dict]:
    query_vector = embed_text(query)

    must_filters = [FieldCondition(key="user_id", match=MatchValue(value=user_id))]
    if mailbox_id:
        must_filters.append(FieldCondition(key="mailbox_id", match=MatchValue(value=mailbox_id)))

    qdrant = get_qdrant()
    search_response = qdrant.query_points(
        collection_name=settings.QDRANT_COLLECTION_EMAIL_CHUNKS,
        query=query_vector,
        query_filter=Filter(must=must_filters),
        limit=100,
        with_payload=True,
    )
    results = search_response.points

    if not results:
        return []

    documents = [
        f"Subject: {r.payload.get('subject', '')}\n{r.payload.get('body_chunk', '')}"
        for r in results
    ]

    reranked = rerank(query, documents, top_n=min(limit * 3, len(documents)))

    seen_email_ids = []
    email_list = []
    for item in reranked:
        idx = item["index"]
        payload = results[idx].payload
        email_id = payload.get("email_id", "")
        if email_id in seen_email_ids:
            continue
        seen_email_ids.append(email_id)

        # Mutable state from MongoDB
        meta = email_metadata_col().find_one({"_id": email_id})

        email_list.append({
            "id": email_id,
            "mailbox_id": payload.get("mailbox_id", ""),
            "subject": payload.get("subject", ""),
            "from_name": payload.get("from_name", ""),
            "from_email": payload.get("from_email", ""),
            "date": meta.get("date") if meta else None,
            "preview": payload.get("preview", ""),
            "read": meta.get("read", False) if meta else False,
            "starred": meta.get("starred", False) if meta else False,
            "priority": payload.get("priority", "medium"),
            "category": payload.get("category") or None,
            "relevance_score": item["relevance_score"],
        })

        if len(email_list) >= limit:
            break

    return email_list
