"""OpenAI chat completion helper with optional vision support."""

import json

from django.conf import settings
from openai import OpenAI


def _client() -> OpenAI:
    return OpenAI(api_key=settings.OPENAI_API_KEY)


def chat(system_prompt: str, user_message: str, temperature: float = 0.7, max_tokens: int = 2048) -> str:
    resp = _client().chat.completions.create(
        model=settings.OPENAI_CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


def chat_multi(messages: list[dict], temperature: float = 0.7, max_tokens: int = 4096, model: str | None = None) -> str:
    """Multi-turn conversation with full message history."""
    resp = _client().chat.completions.create(
        model=model or settings.OPENAI_CHAT_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


def chat_json(system_prompt: str, user_message: str, temperature: float = 0.7, max_tokens: int = 2048) -> dict:
    """Chat completion that returns parsed JSON using response_format."""
    resp = _client().chat.completions.create(
        model=settings.OPENAI_CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content or "{}")


def chat_multi_stream(messages: list[dict], temperature: float = 0.7, max_tokens: int = 4096, model: str | None = None):
    """Stream token chunks from a multi-turn conversation. Yields string deltas."""
    stream = _client().chat.completions.create(
        model=model or settings.OPENAI_CHAT_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def chat_with_images(
    system_prompt: str,
    user_message: str,
    image_urls: list[str],
    temperature: float = 0.7,
) -> str:
    """Send text + images to a vision-capable model (GPT-4o / GPT-4o-mini)."""
    content: list[dict] = [{"type": "text", "text": user_message}]
    for url in image_urls[:5]:
        content.append({
            "type": "image_url",
            "image_url": {"url": url, "detail": "low"},
        })

    resp = _client().chat.completions.create(
        model=settings.OPENAI_CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        temperature=temperature,
        max_tokens=1024,
    )
    return resp.choices[0].message.content or ""
