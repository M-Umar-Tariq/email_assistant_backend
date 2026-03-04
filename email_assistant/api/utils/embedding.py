"""OpenAI embedding helper."""

from django.conf import settings
from openai import OpenAI


def _client() -> OpenAI:
    return OpenAI(api_key=settings.OPENAI_API_KEY)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Return embeddings for a list of texts (batch call)."""
    resp = _client().embeddings.create(
        input=texts,
        model=settings.OPENAI_EMBEDDING_MODEL,
    )
    return [item.embedding for item in resp.data]


def embed_text(text: str) -> list[float]:
    return embed_texts([text])[0]
