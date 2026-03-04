"""Split text into overlapping chunks by token count using tiktoken."""

from django.conf import settings
import tiktoken


def chunk_text(text: str, model: str | None = None) -> list[str]:
    """
    Split `text` into overlapping chunks of CHUNK_SIZE_TOKENS with
    CHUNK_OVERLAP_TOKENS overlap.  Returns a list of text chunks.
    """
    model = model or settings.OPENAI_EMBEDDING_MODEL
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")

    tokens = enc.encode(text)
    chunk_size = settings.CHUNK_SIZE_TOKENS
    overlap = settings.CHUNK_OVERLAP_TOKENS
    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunks.append(enc.decode(chunk_tokens))
        start += chunk_size - overlap
    return chunks if chunks else [text]
