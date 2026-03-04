"""Cohere reranker helper."""

from django.conf import settings
import cohere


def rerank(query: str, documents: list[str], top_n: int = 20) -> list[dict]:
    """
    Rerank `documents` by relevance to `query`.
    Returns list of dicts: { "index": int, "relevance_score": float }.
    """
    client = cohere.ClientV2(api_key=settings.COHERE_API_KEY)
    resp = client.rerank(
        model=settings.COHERE_RERANK_MODEL,
        query=query,
        documents=documents,
        top_n=min(top_n, len(documents)),
    )
    return [
        {"index": r.index, "relevance_score": r.relevance_score}
        for r in resp.results
    ]
