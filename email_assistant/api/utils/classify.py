"""AI-powered email classification: priority & category."""

import json
import logging

from api.utils.llm import chat

logger = logging.getLogger(__name__)


def classify_emails_batch(
    emails: list[dict],
) -> list[dict]:
    """Classify a batch of emails using LLM. Returns list of {priority, category} dicts.

    Each input dict should have: subject, from_name, from_email, preview.
    Returns same-length list with: {"priority": "high"|"medium"|"low", "category": str|None}
    """
    if not emails:
        return []

    batch_size = 15
    results = []

    for i in range(0, len(emails), batch_size):
        batch = emails[i : i + batch_size]
        batch_results = _classify_batch(batch)
        results.extend(batch_results)

    return results


def _classify_batch(batch: list[dict]) -> list[dict]:
    """Classify a single batch (max ~15 emails) via one LLM call."""
    items_text = "\n".join(
        f"[{idx}] From: {e.get('from_name', '')} <{e.get('from_email', '')}> | "
        f"Subject: {e.get('subject', '')} | Preview: {e.get('preview', '')[:150]}"
        for idx, e in enumerate(batch)
    )

    system_prompt = (
        "You are an email priority classifier. For each email, assign:\n"
        "1. priority: 'high' (urgent, security alerts, important deadlines, boss/client emails), "
        "'medium' (normal correspondence, updates), or 'low' (newsletters, promotions, spam-like)\n"
        "2. category: one of 'important', 'updates', 'promotions', 'social', 'newsletters', 'finance', or null\n\n"
        "Respond with ONLY a JSON array (no markdown, no explanation). Each element: "
        '{\"priority\": \"...\", \"category\": \"...\"}\n'
        "The array must have exactly the same number of elements as input emails, in the same order."
    )

    try:
        raw = chat(
            system_prompt=system_prompt,
            user_message=f"Classify these {len(batch)} emails:\n\n{items_text}",
            temperature=0.1,
        )
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        parsed = json.loads(cleaned)
        if isinstance(parsed, list) and len(parsed) == len(batch):
            return [
                {
                    "priority": item.get("priority", "medium") if item.get("priority") in ("high", "medium", "low") else "medium",
                    "category": item.get("category") if item.get("category") in ("important", "updates", "promotions", "social", "newsletters", "finance") else None,
                }
                for item in parsed
            ]
    except Exception as exc:
        logger.warning("AI classify failed: %s", exc)

    return [{"priority": "medium", "category": None}] * len(batch)
