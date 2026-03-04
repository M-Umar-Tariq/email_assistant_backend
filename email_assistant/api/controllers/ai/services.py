"""
AI assistant: hybrid retrieval — date-range scan for broad queries, vector search for specific ones.
Content from Qdrant; mutable state from MongoDB.
"""

import json
import re
from datetime import datetime, timezone, timedelta

from django.conf import settings
from qdrant_client.models import Filter, FieldCondition, MatchValue

from database.db import get_qdrant, email_metadata_col
from api.utils.embedding import embed_text
from api.utils.rerank import rerank
from api.utils.llm import chat, chat_multi, chat_with_images
from api.utils.qdrant_helpers import get_email_content, get_emails_content_batch


# ── Query analysis ────────────────────────────────────────────────────────────

_TIME_PATTERNS = [
    (r"\btoday\b|\bآج\b|\baaj\b", "today"),
    (r"\byesterday\b|\bkal\b|\bگزشتہ\b", "yesterday"),
    (r"\bthis\s+week\b|\bis\s+week\b|\bis\s+hafte\b", "this_week"),
    (r"\blast\s+week\b|\bpichle\s+hafte\b|\bguzishta\s+hafte\b", "last_week"),
    (r"\bthis\s+month\b|\bis\s+month\b|\bis\s+mahine\b|\bis\s+maheene\b", "this_month"),
    (r"\blast\s+month\b|\bpichle\s+month\b|\bpichle\s+mahine\b", "last_month"),
    (r"\blast\s+(\d+)\s+days?\b", "last_n_days"),
    (r"\bthis\s+year\b|\bis\s+saal\b|\bis\s+year\b", "this_year"),
    (r"\ball\s+emails?\b|\bsari?\s+emails?\b|\bsab\s+emails?\b|\btamam\b", "all"),
]

_BROAD_KEYWORDS = re.compile(
    r"\bsummar|overview|recap|digest|highlights?|briefing|report\b"
    r"|\bcount|how\s+many|kitni|kitne\b"
    r"|\blist\s+(all|every|sab|sari)\b"
    r"|\bsummary\b|\bخلاصہ\b",
    re.I,
)


def _detect_time_range(query: str) -> tuple[str | None, datetime | None, datetime | None]:
    """Return (range_label, start_dt, end_dt) or (None, None, None)."""
    now = datetime.now(timezone.utc)
    q_lower = query.lower()

    for pattern, label in _TIME_PATTERNS:
        m = re.search(pattern, q_lower)
        if not m:
            continue

        if label == "today":
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            return label, start, now
        if label == "yesterday":
            start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            end = now.replace(hour=0, minute=0, second=0, microsecond=0)
            return label, start, end
        if label == "this_week":
            start = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
            return label, start, now
        if label == "last_week":
            this_monday = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
            start = this_monday - timedelta(days=7)
            return label, start, this_monday
        if label == "this_month":
            start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            return label, start, now
        if label == "last_month":
            first_this = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            last_month_end = first_this - timedelta(seconds=1)
            start = last_month_end.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            return label, start, first_this
        if label == "last_n_days":
            days = int(m.group(1))
            start = (now - timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0)
            return label, start, now
        if label == "this_year":
            start = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
            return label, start, now
        if label == "all":
            return label, None, None

    return None, None, None


def _is_broad_query(query: str) -> bool:
    return bool(_BROAD_KEYWORDS.search(query))


# ── Context building ──────────────────────────────────────────────────────────

def _build_email_block_full(content: dict, meta: dict | None) -> str:
    """Rich context block for a single email — used when we have few emails."""
    parts = [f"**Email: {content.get('subject', '(no subject)')}**"]
    parts.append(f"From: {content.get('from_name', '')} <{content.get('from_email', '')}>")

    to_list = content.get("to", [])
    if to_list:
        to_str = ", ".join(
            t.get("email", "") if isinstance(t, dict) else str(t) for t in to_list
        )
        parts.append(f"To: {to_str}")

    parts.append(f"Date: {content.get('date', 'unknown')}")

    tags = []
    if content.get("priority") == "high":
        tags.append("HIGH PRIORITY")
    if content.get("has_attachment"):
        tags.append("HAS ATTACHMENTS")
    if meta:
        if meta.get("starred"):
            tags.append("STARRED")
        if not meta.get("read"):
            tags.append("UNREAD")
        if meta.get("labels"):
            tags.append(f"Labels: {', '.join(meta['labels'])}")
    if tags:
        parts.append(f"[{' | '.join(tags)}]")

    parts.append("")
    body = content.get("body_chunk", "")
    parts.append(body[:2000] if len(body) > 2000 else body)

    att_text = content.get("attachment_text", "")
    if att_text:
        parts.append(f"\n[Attachment content]\n{att_text[:2000]}")

    return "\n".join(parts)


def _build_email_block_compact(content: dict, meta: dict | None) -> str:
    """Compact context block — used when we have many emails to fit more in context."""
    tags = []
    if content.get("priority") == "high":
        tags.append("HIGH")
    if content.get("has_attachment"):
        tags.append("ATT")
    if meta and meta.get("starred"):
        tags.append("★")
    if meta and not meta.get("read"):
        tags.append("UNREAD")
    tag_str = f" [{', '.join(tags)}]" if tags else ""

    preview = content.get("preview", "") or content.get("body_chunk", "")[:200]
    return (
        f"• **{content.get('subject', '(no subject)')}**{tag_str}\n"
        f"  From: {content.get('from_name', '')} <{content.get('from_email', '')}> | "
        f"Date: {content.get('date', '?')}\n"
        f"  {preview[:300]}"
    )


# ── Date-range retrieval (MongoDB + Qdrant) ──────────────────────────────────

def _fetch_emails_by_date(
    user_id: str,
    start: datetime | None,
    end: datetime | None,
    mailbox_id: str | None = None,
    limit: int = 200,
) -> tuple[list[dict], list[dict]]:
    """Fetch emails from MongoDB by date range, then load content from Qdrant.
    Returns (contents, metas) sorted by date descending."""
    query_filter: dict = {"user_id": user_id, "archived": {"$ne": True}, "trashed": {"$ne": True}}
    if start or end:
        date_filter = {}
        if start:
            date_filter["$gte"] = start
        if end:
            date_filter["$lte"] = end
        query_filter["date"] = date_filter
    if mailbox_id:
        query_filter["mailbox_id"] = mailbox_id

    metas = list(
        email_metadata_col()
        .find(query_filter)
        .sort("date", -1)
        .limit(limit)
    )
    if not metas:
        return [], []

    email_ids = [str(m["_id"]) for m in metas]
    content_map = get_emails_content_batch(email_ids, user_id)

    contents = []
    matched_metas = []
    for m in metas:
        eid = str(m["_id"])
        c = content_map.get(eid)
        if c:
            contents.append(c)
            matched_metas.append(m)

    return contents, matched_metas


# ── Vector retrieval (Qdrant + rerank) ────────────────────────────────────────

def _fetch_emails_by_vector(
    user_id: str,
    query: str,
    mailbox_id: str | None = None,
    limit: int = 15,
) -> tuple[list[dict], list[dict]]:
    """Vector search + rerank, returns (contents, metas) deduplicated by email."""
    query_vector = embed_text(query)

    must_filters = [FieldCondition(key="user_id", match=MatchValue(value=user_id))]
    if mailbox_id:
        must_filters.append(FieldCondition(key="mailbox_id", match=MatchValue(value=mailbox_id)))

    qdrant = get_qdrant()
    search_response = qdrant.query_points(
        collection_name=settings.QDRANT_COLLECTION_EMAIL_CHUNKS,
        query=query_vector,
        query_filter=Filter(must=must_filters),
        limit=150,
        with_payload=True,
    )
    search_results = search_response.points
    if not search_results:
        return [], []

    documents = [
        f"Subject: {r.payload.get('subject', '')}\n"
        f"From: {r.payload.get('from_name', '')} <{r.payload.get('from_email', '')}>\n"
        f"Date: {r.payload.get('date', '')}\n\n"
        f"{r.payload.get('body_chunk', '')}"
        for r in search_results
    ]

    reranked = rerank(query, documents, top_n=min(40, len(documents)))

    seen_ids: list[str] = []
    payload_map: dict[str, dict] = {}
    for item in reranked:
        point = search_results[item["index"]]
        eid = point.payload.get("email_id", "")
        if eid not in payload_map:
            payload_map[eid] = point.payload
            seen_ids.append(eid)

    top_ids = seen_ids[:limit]
    content_map = get_emails_content_batch(top_ids, user_id)

    contents = []
    metas = []
    for eid in top_ids:
        c = content_map.get(eid)
        if not c:
            p = payload_map.get(eid, {})
            c = {
                "email_id": eid,
                "subject": p.get("subject", ""),
                "from_name": p.get("from_name", ""),
                "from_email": p.get("from_email", ""),
                "to": [],
                "date": p.get("date", ""),
                "preview": p.get("preview", ""),
                "body_chunk": p.get("body_chunk", ""),
                "has_attachment": p.get("has_attachment", False),
                "priority": p.get("priority", "medium"),
                "attachment_text": p.get("attachment_text", ""),
            }
        meta = email_metadata_col().find_one({"_id": eid, "user_id": user_id})
        contents.append(c)
        metas.append(meta or {})

    return contents, metas


# ── Main ask function ─────────────────────────────────────────────────────────

def ask(user_id: str, query: str, mailbox_id: str | None = None, history: list | None = None) -> dict:
    time_label, start_dt, end_dt = _detect_time_range(query)
    broad = _is_broad_query(query)

    if time_label or broad:
        # Date-range retrieval: fetch ALL matching emails
        max_emails = 200 if time_label == "all" else 150
        contents, metas = _fetch_emails_by_date(user_id, start_dt, end_dt, mailbox_id, limit=max_emails)

        if not contents and not broad:
            # Fallback to vector search if date range gave nothing
            contents, metas = _fetch_emails_by_vector(user_id, query, mailbox_id, limit=20)
    else:
        # Specific query: vector search + rerank
        contents, metas = _fetch_emails_by_vector(user_id, query, mailbox_id, limit=20)

    if not contents:
        return {
            "answer": "I couldn't find any relevant emails matching your query.",
            "sources": [],
        }

    # Build context — compact blocks for many emails, full blocks for few
    use_compact = len(contents) > 20
    context_blocks = []
    sources = []
    for c, m in zip(contents, metas):
        if use_compact:
            context_blocks.append(_build_email_block_compact(c, m))
        else:
            context_blocks.append(_build_email_block_full(c, m))
        sources.append({
            "email_id": c.get("email_id", ""),
            "subject": c.get("subject", ""),
        })

    separator = "\n\n" if use_compact else "\n\n━━━━━━━━━━━━━━━━━━━━\n\n"
    context = separator.join(context_blocks)
    now_str = datetime.now(timezone.utc).strftime("%A, %B %d, %Y %H:%M UTC")

    range_note = ""
    if time_label:
        range_note = f"\nNote: The user asked about '{time_label.replace('_', ' ')}'. All {len(contents)} emails in that range are provided below.\n"

    system_prompt = (
        "You are an AI email assistant with full access to the user's mailbox. "
        "Today is " + now_str + ".\n"
        + range_note +
        "\nINSTRUCTIONS:\n"
        "- Answer based ONLY on the email context provided below.\n"
        "- Always cite specific emails by subject and sender when referencing information.\n"
        "- For time-based questions, use the email dates and today's date to filter accurately.\n"
        "- For counting questions (\"how many\"), give exact counts from the data provided.\n"
        "- For summary/digest questions, organize by topic, sender, or priority. Use bullet points.\n"
        "- For action-item questions, list concrete next steps with deadlines if mentioned.\n"
        "- If an email has attachments, mention the attachment names.\n"
        "- If you cannot find enough information, say so honestly rather than guessing.\n"
        "- Be concise, clear, and actionable. Use markdown formatting.\n"
        "- Respond in the same language the user is asking in.\n"
        "- SCOPE: You are strictly an EMAIL assistant. If the user asks about "
        "anything unrelated to their emails, inbox, contacts, or email-related "
        "tasks (e.g. general knowledge, coding, math, recipes, weather, etc.), "
        "respond warmly and empathetically but gently redirect. Example: "
        "\"That's a great question! But I'm your email assistant — I'm best at "
        "helping you with your inbox, emails, and contacts. Is there anything "
        "email-related I can help with?\"\n\n"
        f"EMAIL CONTEXT ({len(contents)} emails):\n\n{context}"
    )

    messages = [{"role": "system", "content": system_prompt}]

    if history:
        for m in history[-8:]:
            messages.append({
                "role": m.get("role", "user"),
                "content": m.get("content", ""),
            })

    messages.append({"role": "user", "content": query})

    answer = chat_multi(messages, temperature=0.4, max_tokens=2048)

    return {
        "answer": answer,
        "sources": sources[:15],
    }


def _extract_image_urls_from_html(html: str) -> list[str]:
    """Extract image src URLs (http/https and data: URIs) from HTML."""
    import re
    if not html:
        return []
    srcs = re.findall(r'<img[^>]+src=["\']([^"\']+)["\']', html, re.I)
    # Keep data: URIs and external https URLs, skip broken/placeholder ones
    return [s for s in srcs if s.startswith("data:image/") or s.startswith("http")]


def ask_about_email(user_id: str, email_id: str, query: str) -> dict:
    """AI chat scoped to a single email — answers questions using the email content + images as context."""
    meta = email_metadata_col().find_one({"_id": email_id, "user_id": user_id})
    if not meta:
        return {"answer": "Email not found.", "sources": []}

    content = get_email_content(email_id, user_id)
    if not content:
        return {"answer": "Could not load email content.", "sources": []}

    from api.controllers.emails.services import _reassemble_body
    body = _reassemble_body(email_id, user_id)

    body_html = content.get("body_html", "")
    image_urls = _extract_image_urls_from_html(body_html)

    attachment_text = content.get("attachment_text", "")
    attachments_meta = content.get("attachments", [])

    email_context = (
        f"From: {content.get('from_name', '')} <{content.get('from_email', '')}>\n"
        f"To: {', '.join(t.get('email', '') for t in content.get('to', []))}\n"
        f"Subject: {content.get('subject', '')}\n"
        f"Date: {meta.get('date', '')}\n\n"
        f"{body[:4000]}"
    )

    if attachment_text:
        email_context += f"\n\n--- ATTACHMENT CONTENT ---\n{attachment_text[:8000]}"

    system_prompt = (
        "You are an AI email assistant. The user is reading a specific email "
        "and asking questions about it. Use the email content below to answer. "
        "Be concise, helpful, and actionable. If the user asks you to draft a reply, "
        "summarize, extract info, translate, or anything else — do it based on this email."
    )
    if attachments_meta:
        filenames = ", ".join(a.get("filename", "?") for a in attachments_meta if isinstance(a, dict))
        system_prompt += (
            f"\n\nThis email has attachments: {filenames}. "
            "The text content of supported attachments (PDF, Word, Excel, CSV, TXT) "
            "has been extracted and included below. Use this content to answer questions "
            "about the attachments."
        )
    if image_urls:
        system_prompt += (
            "\n\nThis email contains images attached below. Analyze them carefully "
            "when answering the user's questions. Describe what you see in the images if asked."
        )

    system_prompt += f"\n\nEMAIL:\n{email_context}"

    if image_urls:
        answer = chat_with_images(
            system_prompt=system_prompt,
            user_message=query,
            image_urls=image_urls,
            temperature=0.5,
        )
    else:
        answer = chat(
            system_prompt=system_prompt,
            user_message=query,
            temperature=0.5,
        )

    return {
        "answer": answer,
        "sources": [{"email_id": email_id, "subject": content.get("subject", "")}],
    }


def get_instant_replies(user_id: str, email_id: str) -> list[dict]:
    """Generate AI instant reply suggestions for a specific email."""
    meta = email_metadata_col().find_one({"_id": email_id, "user_id": user_id})
    if not meta:
        return []

    content = get_email_content(email_id, user_id)
    if not content:
        return []

    from api.controllers.emails.services import _reassemble_body
    body = _reassemble_body(email_id, user_id)

    prompt = (
        f"From: {content.get('from_name', '')} <{content.get('from_email', '')}>\n"
        f"Subject: {content.get('subject', '')}\n\n{body[:2000]}"
    )

    raw = chat(
        system_prompt=(
            "Generate 3 short reply options for the email below. "
            "Format as JSON array with objects: "
            '[{"label": "short label", "tone": "positive|neutral|negative", "text": "full reply text"}]. '
            "Return ONLY valid JSON."
        ),
        user_message=prompt,
        temperature=0.7,
    )

    import json
    try:
        replies = json.loads(raw)
        for i, r in enumerate(replies):
            r["id"] = f"ir-{i}"
        return replies
    except (json.JSONDecodeError, TypeError):
        return []
