"""
AI assistant: hybrid retrieval — date-range scan for broad queries, vector search for specific ones.
Content from Qdrant; mutable state from MongoDB.
"""

import json
import re
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

from django.conf import settings
from qdrant_client.models import Filter, FieldCondition, MatchValue

from database.db import get_qdrant, email_metadata_col, mailboxes_col
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

# "Latest/newest/most recent email" must use date order, not vector search
_LATEST_EMAIL_PATTERN = re.compile(
    r"\b(latest|newest|most\s+recent|last|recent)\s+(email|mail|message)\b"
    r"|\b(last|latest|newest)\s+email\b"
    r"|\bwhat(?:'s| is)\s+my\s+latest\s+email\b",
    re.I,
)


def _detect_time_range(query: str, tz: ZoneInfo | None = None) -> tuple[str | None, datetime | None, datetime | None]:
    """Return (range_label, start_dt, end_dt) or (None, None, None).
    All boundaries are computed in the user's local timezone, then converted
    to UTC so MongoDB queries (which store dates in UTC) work correctly."""
    now_utc = datetime.now(timezone.utc)
    now_local = now_utc.astimezone(tz) if tz else now_utc
    q_lower = query.lower()

    def _local_midnight(dt: datetime) -> datetime:
        """Midnight of the given day in user's timezone, returned as UTC."""
        midnight = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        return midnight.astimezone(timezone.utc) if tz else midnight

    for pattern, label in _TIME_PATTERNS:
        m = re.search(pattern, q_lower)
        if not m:
            continue

        if label == "today":
            start = _local_midnight(now_local)
            return label, start, now_utc
        if label == "yesterday":
            start = _local_midnight(now_local - timedelta(days=1))
            end = _local_midnight(now_local)
            return label, start, end
        if label == "this_week":
            start = _local_midnight(now_local - timedelta(days=now_local.weekday()))
            return label, start, now_utc
        if label == "last_week":
            this_monday = _local_midnight(now_local - timedelta(days=now_local.weekday()))
            start = this_monday - timedelta(days=7)
            return label, start, this_monday
        if label == "this_month":
            start = _local_midnight(now_local.replace(day=1))
            return label, start, now_utc
        if label == "last_month":
            first_this = _local_midnight(now_local.replace(day=1))
            last_month_end = first_this - timedelta(seconds=1)
            start = _local_midnight(last_month_end.astimezone(tz if tz else timezone.utc).replace(day=1))
            return label, start, first_this
        if label == "last_n_days":
            days = int(m.group(1))
            start = _local_midnight(now_local - timedelta(days=days))
            return label, start, now_utc
        if label == "this_year":
            start = _local_midnight(now_local.replace(month=1, day=1))
            return label, start, now_utc
        if label == "all":
            return label, None, None

    return None, None, None


def _is_broad_query(query: str) -> bool:
    return bool(_BROAD_KEYWORDS.search(query))


# ── Context building ──────────────────────────────────────────────────────────

def _format_date(meta: dict | None, content: dict, tz: ZoneInfo | None = None) -> str:
    """Return a human-readable date string using MongoDB date (same source as inbox).
    Falls back to Qdrant ISO string if MongoDB date is unavailable."""
    dt = None
    if meta:
        dt = meta.get("original_date") or meta.get("date")
    if dt and isinstance(dt, datetime):
        if not dt.tzinfo:
            dt = dt.replace(tzinfo=timezone.utc)
        if tz:
            dt = dt.astimezone(tz)
        return dt.strftime("%B %d, %Y %I:%M %p")
    date_str = content.get("date", "")
    if date_str:
        try:
            dt = datetime.fromisoformat(str(date_str).replace("Z", "+00:00"))
            if tz:
                dt = dt.astimezone(tz)
            return dt.strftime("%B %d, %Y %I:%M %p")
        except (ValueError, TypeError):
            return date_str
    return "unknown"


def _build_email_block_full(content: dict, meta: dict | None, tz: ZoneInfo | None = None) -> str:
    """Rich context block for a single email — used when we have few emails."""
    parts = [f"**Email: {content.get('subject', '(no subject)')}**"]
    parts.append(f"From: {content.get('from_name', '')} <{content.get('from_email', '')}>")

    to_list = content.get("to", [])
    if to_list:
        to_str = ", ".join(
            t.get("email", "") if isinstance(t, dict) else str(t) for t in to_list
        )
        parts.append(f"To: {to_str}")

    parts.append(f"Date: {_format_date(meta, content, tz)}")

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


def _build_email_block_compact(content: dict, meta: dict | None, tz: ZoneInfo | None = None) -> str:
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
        f"Date: {_format_date(meta, content, tz)}\n"
        f"  {preview[:300]}"
    )


# ── Date-range retrieval (MongoDB + Qdrant) ──────────────────────────────────

def _fetch_emails_by_date(
    user_id: str,
    start: datetime | None,
    end: datetime | None,
    mailbox_id: str | None = None,
    limit: int = 0,
) -> tuple[list[dict], list[dict]]:
    """Fetch emails from MongoDB by date range, then load content from Qdrant.
    Returns (contents, metas) sorted by date descending. limit=0 means no limit."""
    query_filter: dict = {"user_id": user_id, "archived": {"$ne": True}, "trashed": {"$ne": True}}
    if start or end:
        date_filter = {}
        if start:
            date_filter["$gte"] = start
        if end:
            date_filter["$lte"] = end
        query_filter["$or"] = [
            {"original_date": date_filter},
            {"original_date": None, "date": date_filter},
        ]
    if mailbox_id:
        query_filter["mailbox_id"] = mailbox_id

    cursor = email_metadata_col().find(query_filter).sort("date", -1)
    if limit > 0:
        cursor = cursor.limit(limit)
    metas = list(cursor)
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
    limit: int | None = None,
) -> tuple[list[dict], list[dict]]:
    """Vector search + rerank, returns (contents, metas) deduplicated by email. limit=None means no limit."""
    query_vector = embed_text(query)

    must_filters = [FieldCondition(key="user_id", match=MatchValue(value=user_id))]
    if mailbox_id:
        must_filters.append(FieldCondition(key="mailbox_id", match=MatchValue(value=mailbox_id)))

    qdrant = get_qdrant()
    search_response = qdrant.query_points(
        collection_name=settings.QDRANT_COLLECTION_EMAIL_CHUNKS,
        query=query_vector,
        query_filter=Filter(must=must_filters),
        limit=500,
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

    reranked = rerank(query, documents, top_n=len(documents))

    seen_ids: list[str] = []
    payload_map: dict[str, dict] = {}
    for item in reranked:
        point = search_results[item["index"]]
        eid = point.payload.get("email_id", "")
        if eid not in payload_map:
            payload_map[eid] = point.payload
            seen_ids.append(eid)

    top_ids = seen_ids if limit is None else seen_ids[:limit]
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

def ask(user_id: str, query: str, mailbox_id: str | None = None, history: list | None = None, user_tz: str | None = None) -> dict:
    tz: ZoneInfo | None = None
    if user_tz:
        try:
            tz = ZoneInfo(user_tz)
        except (KeyError, Exception):
            pass

    time_label, start_dt, end_dt = _detect_time_range(query, tz)
    broad = _is_broad_query(query)
    want_latest = bool(_LATEST_EMAIL_PATTERN.search(query))

    if time_label or broad or want_latest:
        contents, metas = _fetch_emails_by_date(user_id, start_dt, end_dt, mailbox_id, limit=0)

        if not contents and not broad and not want_latest:
            contents, metas = _fetch_emails_by_vector(user_id, query, mailbox_id)
    else:
        contents, metas = _fetch_emails_by_vector(user_id, query, mailbox_id)

    if not contents:
        return {
            "answer": "I couldn't find any relevant emails matching your query.",
            "sources": [],
            "actions": [],
        }

    FULL_CUTOFF = 35
    use_compact = len(contents) > FULL_CUTOFF
    context_blocks = []
    sources = []
    for i, (c, m) in enumerate(zip(contents, metas)):
        if use_compact and i >= FULL_CUTOFF:
            context_blocks.append(_build_email_block_compact(c, m, tz))
        else:
            context_blocks.append(_build_email_block_full(c, m, tz))
        sources.append({
            "email_id": c.get("email_id", ""),
            "subject": c.get("subject", ""),
        })

    separator = "\n\n" if use_compact else "\n\n━━━━━━━━━━━━━━━━━━━━\n\n"
    context = separator.join(context_blocks)
    now = datetime.now(timezone.utc)
    if tz:
        now_local = now.astimezone(tz)
        now_str = now_local.strftime(f"%A, %B %d, %Y %H:%M ({user_tz})")
    else:
        now_str = now.strftime("%A, %B %d, %Y %H:%M UTC")

    range_note = ""
    range_note += (
        "\nIMPORTANT: The emails below are always ordered by date, newest first. "
        "The FIRST email in the list is the user's latest (most recent) email. "
        "When the user asks for 'latest', 'newest', 'most recent', or 'last email', "
        "you must use the first email in the list.\n"
    )
    if time_label:
        range_note += f"\nNote: The user asked about '{time_label.replace('_', ' ')}'. All {len(contents)} emails in that range are provided below.\n"
    if use_compact:
        range_note += (
            f"\nThe first {min(FULL_CUTOFF, len(contents))} emails below have FULL content; "
            "use them to answer questions like 'what does this email say' or 'what is he asking'. "
            "The rest are short previews.\n"
        )

    mbox_docs = list(mailboxes_col().find({"user_id": user_id}))
    mailbox_info = [
        {"id": str(mb["_id"]), "email": mb["email"], "name": mb["name"]}
        for mb in mbox_docs
    ]

    from api.controllers.agent.profile import get_profile
    from api.controllers.settings.services import get_user_preferences_prompt

    profile = get_profile(user_id)
    key_contacts = profile.get("key_contacts", [])
    contacts_lookup = [
        {"name": c.get("name", ""), "email": c.get("email", "")}
        for c in key_contacts
        if c.get("email")
    ]

    user_prefs = get_user_preferences_prompt(user_id)
    prefs_block = f"\n{user_prefs}\n\n" if user_prefs else ""

    system_prompt = (
        "You are an AI email assistant with full access to the user's mailbox. "
        "Today is " + now_str + ".\n"
        + range_note
        + prefs_block +
        f"\nMAILBOXES:\n{json.dumps(mailbox_info, default=str)}\n\n"
        f"KEY CONTACTS (use these to resolve names to email addresses):\n"
        f"{json.dumps(contacts_lookup, default=str)}\n\n"
        "\nINSTRUCTIONS:\n"
        "- Answer based ONLY on the email context provided below.\n"
        "- Always cite specific emails by subject and sender when referencing information.\n"
        "- For time-based questions, use the email dates and today's date to filter accurately.\n"
        "- For counting questions (\"how many\"), give exact counts from the data provided.\n"
        "- For summary/digest questions: do NOT give one-line-per-email. For each email write a "
        "proper summary with 2-4 lines: main point, key details, action needed (if any), and "
        "important names/dates/numbers. Use sub-bullets under each email. Organize by topic, "
        "sender, or date. Use markdown (headers, bullets, bold).\n"
        "- For action-item questions, list concrete next steps with deadlines if mentioned.\n"
        "- If an email has attachments, mention the attachment names.\n"
        "- If you cannot find enough information, say so honestly rather than guessing.\n"
        "- Be thorough: include relevant details from the emails. Respond in the same language the user is asking in.\n\n"
        "CAPABILITIES — you can take real actions. When the user asks you to send, reply, "
        "forward, delete, archive, or mark emails, include a JSON block using this format:\n\n"
        "```actions\n"
        '[{"type":"send_email","to":"x@y.com","subject":"...","body":"...",'
        '"mailbox_id":"...","label":"short desc","description":"details",'
        '"requires_approval":true}]\n'
        "```\n\n"
        "Action types:\n"
        "- send_email: to (email or array), subject, body, mailbox_id (optional). "
        "Example: {\"type\":\"send_email\",\"to\":\"ahmed@example.com\",\"subject\":\"...\",\"body\":\"...\",\"label\":\"...\",\"requires_approval\":true}\n"
        "- forward_email: email_id, to (email or array), mailbox_id (optional), body (optional prefix). "
        "Example: {\"type\":\"forward_email\",\"email_id\":\"...\",\"to\":\"someone@example.com\",\"label\":\"...\",\"requires_approval\":true}\n"
        "- trash_email: email_id (move to trash/delete). "
        "Example: {\"type\":\"trash_email\",\"email_id\":\"...\",\"label\":\"Move to trash\",\"requires_approval\":true}\n"
        "- archive_email: email_id. "
        "Example: {\"type\":\"archive_email\",\"email_id\":\"...\",\"label\":\"Archive email\",\"requires_approval\":true}\n"
        "- mark_read: email_id (mark as read). "
        "Example: {\"type\":\"mark_read\",\"email_id\":\"...\",\"label\":\"Mark as read\",\"requires_approval\":false}\n"
        "- draft_reply (email_id,instructions) | send_reply (email_id,body or instructions,mailbox_id) | "
        "send_whatsapp (to=phone,body) | set_reminder (hours,note)\n\n"
        "ACTION RULES:\n"
        "1. Always requires_approval=true for sending emails/messages and for delete/trash/archive\n"
        "2. Be proactive — when the user asks to send, reply, forward, or take action, DO IT by emitting the action block\n"
        "3. Reference specific emails by subject/sender and include a brief content summary\n"
        "4. Respond in the same language the user speaks\n"
        "5. Use send_reply when the user wants to reply to an existing email. Use send_email for NEW emails.\n"
        "6. Use forward_email when the user wants to forward an existing email.\n"
        "7. When the user says 'delete', 'remove', 'trash' an email, use trash_email. "
        "When they say 'archive', use archive_email. When they say 'mark as read', use mark_read.\n"
        "8. RECIPIENT RESOLUTION: When the user says a name (e.g. 'send email to Ahmed'), "
        "look up their email from KEY CONTACTS. **If multiple contacts share the same or "
        "similar name, you MUST list ALL matching contacts with their email addresses and "
        "ask the user to pick the correct one. NEVER auto-pick one silently.**\n"
        "9. SENDER CONFIRMATION: Before emitting any send_email, send_reply, or forward_email "
        "action, you MUST clearly state:\n"
        "   - **From**: which mailbox/email address the email will be sent from\n"
        "   - **To**: the recipient's full email address\n"
        "   - **Subject** and a brief summary of the body\n"
        "   If the user has multiple mailboxes, ask which one to send from. "
        "Only emit the action block AFTER the user confirms these details or says 'send it'.\n\n"
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

    response_text = chat_multi(messages, temperature=0.4, max_tokens=8192)

    actions = _extract_actions(response_text)
    clean_text = _clean_response(response_text)

    return {
        "answer": clean_text,
        "sources": sources,
        "actions": actions,
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
    from api.controllers.settings.services import get_user_preferences_prompt

    body = _reassemble_body(email_id, user_id)
    user_prefs = get_user_preferences_prompt(user_id)

    prompt = (
        f"From: {content.get('from_name', '')} <{content.get('from_email', '')}>\n"
        f"Subject: {content.get('subject', '')}\n\n{body[:2000]}"
    )

    style_hint = ""
    if user_prefs:
        style_hint = f"\n{user_prefs}\nMatch the user's preferred draft style when writing replies.\n"

    raw = chat(
        system_prompt=(
            "Generate 3 short reply options for the email below. "
            + style_hint +
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


# ── Action extraction helpers ─────────────────────────────────────────────────

def _extract_actions(text: str) -> list[dict]:
    actions: list[dict] = []
    for match in re.findall(r"```actions?\s*\n(.*?)\n```", text, re.DOTALL):
        try:
            parsed = json.loads(match.strip())
            if isinstance(parsed, list):
                ts = int(datetime.now(timezone.utc).timestamp())
                for i, a in enumerate(parsed):
                    a["id"] = f"act-{i}-{ts}"
                    a["status"] = (
                        "awaiting_approval"
                        if a.get("requires_approval", True)
                        else "pending"
                    )
                    a["timestamp"] = datetime.now(timezone.utc).isoformat()
                    actions.append(a)
        except (json.JSONDecodeError, TypeError):
            pass
    return actions


def _clean_response(text: str) -> str:
    return re.sub(r"```actions?\s*\n.*?\n```", "", text, flags=re.DOTALL).strip()
