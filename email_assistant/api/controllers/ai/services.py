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


# ── Query intent → LLM context budget ─────────────────────────────────────────
#
# These limits control how many emails the LLM *sees* — NOT how many an action
# can affect.  The executor resolves bulk actions (mark_read with from_email,
# trash with keywords, etc.) against the full database, so the LLM only needs
# enough context to understand intent and identify filter criteria.

_ACTION_RE = re.compile(
    r"\b(mark\s+as\s+read|mark\s+read|mark\s+as\s+unread|mark\s+unread"
    r"|delete|trash|archive|snooze|forward|reply\s+to|send\s+reply"
    r"|open|read\s+the|read\s+email|read\s+this|show\s+me"
    r"|mark\s+all|read\s+all|unread\s+all|delete\s+all|trash\s+all"
    r"|archive\s+all|all\s+emails?\s+as\s+read|all\s+emails?\s+as\s+unread"
    r"|all\s+emails?\s+from)\b",
    re.I,
)
_COUNT_RE = re.compile(r"\b(how\s+many|count|total|number\s+of|kitni|kitne)\b", re.I)
_SINGLE_EMAIL_RE = re.compile(
    r"\b(this\s+email|that\s+email|the\s+email\s+from|one\s+email)\b", re.I
)

_FAST_ACTION_VERBS = re.compile(
    r"\b(?:mark\b.*\b(?:read|unread)|trash|delete|archive|snooze|remove)\b",
    re.I,
)
_NEEDS_CONTENT_RE = re.compile(
    r"\b(?:summar|explain|what|why|who|show|read\s+(?:the|this|my|that)|open"
    r"|reply|forward|send|compose|draft|write|search|find|list)\b",
    re.I,
)


def _is_fast_action(query: str) -> bool:
    """True for pure state-changing actions that don't need email bodies in the prompt.

    Reference words like "it / this / that / latest / first" are NOT a reason to
    leave the fast path: in fast mode we still provide a compact list of recent
    emails (newest first), and chat history gives the AI enough signal to
    resolve pronouns like "mark it as read" back to the right email ID.
    """
    return (
        bool(_FAST_ACTION_VERBS.search(query))
        and not _NEEDS_CONTENT_RE.search(query)
    )


def _query_context_budget(query: str) -> dict:
    """Decide how many emails the LLM needs in its context window.

    The executor handles the real bulk work at DB level, so the LLM only
    needs enough emails to: understand the user's request, identify the
    sender / subject / keywords, and produce a good answer."""
    q = query.strip()

    if _SINGLE_EMAIL_RE.search(q) or _LATEST_EMAIL_PATTERN.search(q):
        return {"limit": 5, "chunks": 40, "rerank": 25}

    if _ACTION_RE.search(q):
        return {"limit": 5, "chunks": 30, "rerank": 0}

    if _COUNT_RE.search(q):
        return {"limit": 50, "chunks": 150, "rerank": 80}

    if _is_broad_query(q):
        return {"limit": 40, "chunks": 150, "rerank": 80}

    return {"limit": 20, "chunks": 120, "rerank": 70}


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
    email_id = content.get("email_id", "")
    parts = [f"**Email: {content.get('subject', '(no subject)')}**"]
    if email_id:
        parts.append(f"ID: {email_id}")
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
    email_id = content.get("email_id", "")
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

    id_str = f" (ID: {email_id})" if email_id else ""
    preview = content.get("preview", "") or content.get("body_chunk", "")[:200]
    return (
        f"• **{content.get('subject', '(no subject)')}**{tag_str}{id_str}\n"
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
    *,
    search_chunk_limit: int = 500,
    rerank_cap: int | None = None,
) -> tuple[list[dict], list[dict]]:
    """Vector search + rerank, returns (contents, metas) deduplicated by email. limit=None means no limit.

    search_chunk_limit caps Qdrant hits (default 500 for deep inbox search).
    rerank_cap caps Cohere rerank input; None means rerank all retrieved chunks (slow at scale).
    """
    query_vector = embed_text(query)

    must_filters = [FieldCondition(key="user_id", match=MatchValue(value=user_id))]
    if mailbox_id:
        must_filters.append(FieldCondition(key="mailbox_id", match=MatchValue(value=mailbox_id)))

    qdrant = get_qdrant()
    search_response = qdrant.query_points(
        collection_name=settings.QDRANT_COLLECTION_EMAIL_CHUNKS,
        query=query_vector,
        query_filter=Filter(must=must_filters),
        limit=max(1, min(search_chunk_limit, 500)),
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

    seen_ids: list[str] = []
    payload_map: dict[str, dict] = {}

    if rerank_cap is not None and rerank_cap <= 0:
        for point in search_results:
            eid = point.payload.get("email_id", "")
            if eid and eid not in payload_map:
                payload_map[eid] = point.payload
                seen_ids.append(eid)
    else:
        rerank_top = len(documents) if rerank_cap is None else min(rerank_cap, len(documents))
        reranked = rerank(query, documents, top_n=max(1, rerank_top))
        for item in reranked:
            point = search_results[item["index"]]
            eid = point.payload.get("email_id", "")
            if eid and eid not in payload_map:
                payload_map[eid] = point.payload
                seen_ids.append(eid)

    top_ids = seen_ids if limit is None else seen_ids[:limit]
    content_map = get_emails_content_batch(top_ids, user_id)

    # Single batch query instead of N find_one() calls — fixes N+1.
    # Skip stale vector entries that no longer exist in MongoDB metadata
    # (prevents AI from generating actions for already-deleted emails).
    metas_by_id = {
        str(m["_id"]): m
        for m in email_metadata_col().find({"_id": {"$in": top_ids}, "user_id": user_id})
    }

    contents = []
    metas = []
    for eid in top_ids:
        meta = metas_by_id.get(eid)
        if not meta:
            continue
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
        contents.append(c)
        metas.append(meta)

    return contents, metas


# ── Main ask function ─────────────────────────────────────────────────────────

def ask(user_id: str, query: str, mailbox_id: str | None = None, history: list | None = None, user_tz: str | None = None) -> dict:
    tz: ZoneInfo | None = None
    if user_tz:
        try:
            tz = ZoneInfo(user_tz)
        except (KeyError, Exception):
            pass

    is_fast = _is_fast_action(query)
    time_label = ""

    if is_fast:
        # Fast actions (mark read/unread, trash, archive, snooze, delete) don't
        # need full email bodies in the prompt, but the AI still needs IDs so it
        # can target *specific* emails when the user says things like
        # "mark these 3 emails as read", "mark today's emails as read", or
        # "mark the two Amazon emails as read". We provide a compact recent
        # slice of the inbox — optionally narrowed by detected time range —
        # and use the compact email block for low token cost.
        time_label, start_dt, end_dt = _detect_time_range(query, tz)
        contents, metas = _fetch_emails_by_date(
            user_id, start_dt, end_dt, mailbox_id,
            limit=0 if time_label else 25,
        )
        # Cap the list so the prompt stays small even if the time range is wide.
        if len(contents) > 50:
            contents = contents[:50]
            metas = metas[:50]
    else:
        time_label, start_dt, end_dt = _detect_time_range(query, tz)
        broad = _is_broad_query(query)
        want_latest = bool(_LATEST_EMAIL_PATTERN.search(query))
        budget = _query_context_budget(query)

        if time_label or broad or want_latest:
            contents, metas = _fetch_emails_by_date(
                user_id, start_dt, end_dt, mailbox_id,
                limit=budget["limit"],
            )

            if not contents and not broad and not want_latest:
                contents, metas = _fetch_emails_by_vector(
                    user_id, query, mailbox_id,
                    limit=budget["limit"],
                    search_chunk_limit=budget["chunks"],
                    rerank_cap=budget["rerank"],
                )
        else:
            contents, metas = _fetch_emails_by_vector(
                user_id, query, mailbox_id,
                limit=budget["limit"],
                search_chunk_limit=budget["chunks"],
                rerank_cap=budget["rerank"],
            )

        if not contents:
            return {
                "answer": "I couldn't find any relevant emails matching your query.",
                "sources": [],
                "actions": [],
            }

    if contents:
        FULL_CUTOFF = 35
        # For fast state-change actions we never need full bodies — the AI only
        # needs IDs + headers to target specific emails.
        use_compact = is_fast or len(contents) > FULL_CUTOFF
        context_blocks = []
        sources = []
        for i, (c, m) in enumerate(zip(contents, metas)):
            if use_compact or (len(contents) > FULL_CUTOFF and i >= FULL_CUTOFF):
                context_blocks.append(_build_email_block_compact(c, m, tz))
            else:
                context_blocks.append(_build_email_block_full(c, m, tz))
            sources.append({
                "email_id": c.get("email_id", ""),
                "subject": c.get("subject", ""),
            })
        separator = "\n\n" if use_compact else "\n\n━━━━━━━━━━━━━━━━━━━━\n\n"
        context = separator.join(context_blocks)
    else:
        use_compact = False
        context = "(No email content loaded — use filter params: from_email, subject, keywords, read, date_from, date_to, or limit to target emails.)"

    now = datetime.now(timezone.utc)
    if tz:
        now_local = now.astimezone(tz)
        now_str = now_local.strftime(f"%A, %B %d, %Y %H:%M ({user_tz})")
    else:
        now_str = now.strftime("%A, %B %d, %Y %H:%M UTC")

    range_note = ""
    if not is_fast:
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
        "CAPABILITIES — you can take REAL actions on emails. When the user asks you to "
        "perform any action (mark read, send, reply, forward, delete, archive, etc.), "
        "you MUST include a JSON action block. Each email in the context has an ID field — "
        "use that exact ID when targeting a specific email.\n\n"
        "Action block format:\n"
        "```actions\n"
        '[{"type":"<action_type>","email_id":"<from context>","label":"short desc",'
        '"description":"details","requires_approval":true}]\n'
        "```\n\n"
        "ALL ACTION TYPES (use the exact type string):\n\n"
        "TARGETING (applies to mark_read, mark_unread, trash_email, delete_email, "
        "archive_email, snooze_email, open_email, forward_email, etc.):\n"
        "  - email_id:  ONE specific email — use the ID shown in the context.\n"
        "  - email_ids: [\"id1\",\"id2\",...] — TWO OR MORE specific emails from the context.\n"
        "  - from_email: ALL emails from this sender address.\n"
        "  - subject:   ALL emails whose subject matches (case-insensitive substring).\n"
        "  - keywords:  ALL emails matching keywords in subject or preview.\n"
        "  - read:      true / false — filter by read state (e.g. only unread ones).\n"
        "  - date_from, date_to: ISO datetime strings — restrict to a date range.\n"
        "  - limit:     max number of emails affected (e.g. 'mark the latest 5 as read').\n"
        "\n"
        "HOW TO PICK THE RIGHT TARGETING:\n"
        "  1. User points at emails visible in the context OR referenced in a PREVIOUS\n"
        "     assistant turn (\"this email\", \"it\", \"that one\", \"ye wala\", \"woh mail\",\n"
        "     \"the Amazon one\", \"these 3 emails\", \"first email\", \"latest email\"):\n"
        "     use email_id for one, email_ids for two or more. Previous assistant turns\n"
        "     include a bracketed `[Emails referenced in this turn: ...]` block — use those\n"
        "     exact IDs to resolve pronouns. NEVER invent IDs — only use IDs from the\n"
        "     current EMAIL CONTEXT or a previous referenced-emails block.\n"
        "  2. User describes a sender (\"all emails from John\", \"John ki emails\"): from_email.\n"
        "  3. User describes a topic / subject (\"all Amazon emails\", \"newsletters\"): subject or keywords.\n"
        "  4. User describes a time range (\"today's emails\", \"aaj ki emails\", \"last week\"): date_from + date_to.\n"
        "  5. User describes read state (\"all unread emails as read\"): use mark_all_read, "
        "or mark_read with read=false to target only unread ones.\n"
        "  6. User asks for a count (\"mark the latest 5 as read\"): combine filters with limit.\n"
        "  7. For EVERY email in the inbox with no other filter: use mark_all_read / mark_all_unread "
        "(do NOT list IDs).\n"
        "Combine params when needed — e.g. mark_read with from_email + read=false marks only\n"
        "unread emails from that sender as read.\n\n"
        "- read_emails: fetch inbox emails. Extra params: unread_only, limit, folder.\n"
        "- open_email: open a specific email. Needs email_id.\n"
        "- open_latest_email: open the most recent email.\n"
        "- search_emails: search by query. Extra params: query (REQUIRED), limit.\n"
        "- send_email: send a new email. Required: to, subject, body. Optional: cc, mailbox_id.\n"
        "- draft_email: prepare a draft (not sent). Params: to, subject, body, cc, mailbox_id.\n"
        "- draft_reply: AI-generate a reply draft. Needs email_id + instructions.\n"
        "- send_reply: send a reply. Required: email_id + body (the full reply text, ready to send). "
        "Optional: subject, cc, mailbox_id. Only omit body and pass instructions if the user "
        "EXPLICITLY says 'reply automatically' / 'auto-draft and send'.\n"
        "- reply_all: reply to all. Required: email_id + body (full text). Optional: subject, mailbox_id.\n"
        "- forward_email: forward an email. Required: email_id + to. Optional: cc, body (message "
        "above the forwarded content), mailbox_id.\n"
        "- trash_email: move to trash. Accepts email_id OR from_email/subject/keywords for bulk.\n"
        "- delete_email: same as trash_email — move to trash (not permanent erase). "
        "Accepts email_id OR filters for bulk.\n"
        "- archive_email: archive. Accepts email_id OR filters for bulk.\n"
        "- mark_read: mark as read. Accepts email_id OR from_email/subject/keywords for bulk.\n"
        "- mark_unread: mark as unread. Accepts email_id OR filters for bulk.\n"
        "- mark_all_read: mark ALL inbox emails as read. NO email_id needed. Optional: mailbox_id.\n"
        "- mark_all_unread: mark ALL inbox emails as unread. NO email_id needed. Optional: mailbox_id.\n"
        "- snooze_email: snooze. Accepts email_id OR filters for bulk + hours (default 24).\n"
        "- send_whatsapp: send WhatsApp. Params: to (phone REQUIRED), body (REQUIRED).\n"
        "- set_reminder: set reminder. Params: email_id, hours (default 24).\n\n"
        "ACTION RULES:\n"
        "1. ALWAYS set requires_approval=true — the user must confirm before execution.\n"
        "2. Be proactive — when the user asks to do something, emit the action block immediately.\n"
        "3. Pick targeting per the rules above: one ID → email_id, a handful of "
        "specific emails → email_ids, a whole group → from_email/subject/keywords/date/read filters.\n"
        "4. Respond in the same language the user speaks.\n"
        "5. Use send_reply for replies, reply_all for reply-all, send_email for NEW messages.\n"
        "6. Use trash_email for delete/remove/trash. Use archive_email for archive.\n"
        "7. For ALL inbox read/unread with no sender filter use mark_all_read / mark_all_unread.\n"
        "8. RECIPIENT RESOLUTION: When the user says a name, look up email from KEY CONTACTS. "
        "If multiple match, list ALL and ask the user to pick.\n"
        "9. COMPOSE PREVIEW: For send_email / send_reply / reply_all / forward_email, the JSON "
        "action MUST carry the complete, ready-to-send content so the user can review BEFORE "
        "approving. Concretely:\n"
        "     • to: the resolved recipient email address(es) — never a name placeholder.\n"
        "     • subject: the actual subject line (for replies, reuse the original subject; "
        "   the backend adds 'Re:' automatically if missing).\n"
        "     • body: the full message text the user would send. Plain text, no markdown code\n"
        "   fences, no 'Here is the draft:' preamble, no explanatory wrapper — ONLY the email body.\n"
        "     • cc: optional, only if the user asked for a CC.\n"
        "   Also briefly show From / To / Subject / a one-line summary in your chat reply so the\n"
        "   user sees what they're approving, but the authoritative content lives in the JSON.\n"
        "10. If the user's request is ambiguous (\"reply to him\" with no context, no body hint),\n"
        "    ask ONE clarifying question instead of guessing. Only emit the action once you have\n"
        "    enough info to produce a sendable draft.\n"
        "11. If multiple mailboxes exist and the user hasn't picked one, set mailbox_id to the\n"
        "    most contextually appropriate (e.g. the mailbox that received the original email "
        "    you're replying to).\n\n"
        "SCOPE: You are strictly an EMAIL assistant. Gently redirect off-topic questions.\n\n"
        f"EMAIL CONTEXT ({len(contents)} emails):\n\n{context}"
    )

    messages = [{"role": "system", "content": system_prompt}]

    if history:
        for m in history[-8:]:
            raw_role = (m.get("role") or "user").lower()
            role = "assistant" if raw_role in ("assistant", "ai", "bot") else "user"
            content = (m.get("content") or "").strip()
            # Inject previously-cited email IDs into assistant turns so follow-ups
            # like "mark it as read" / "delete that email" can be resolved against
            # the exact emails referenced earlier in the conversation.
            if role == "assistant":
                srcs = m.get("sources") or []
                hint_lines = []
                for s in srcs:
                    if not isinstance(s, dict):
                        continue
                    eid = s.get("email_id") or s.get("emailId") or ""
                    subj = s.get("subject") or ""
                    if eid:
                        hint_lines.append(f'  - ID: {eid}  Subject: "{subj}"')
                if hint_lines:
                    content += (
                        "\n\n[Emails referenced in this turn — use these exact IDs "
                        "when the user follows up with pronouns like 'it', 'this', "
                        "'that', 'these', or 'yeh/woh/ye wala':\n"
                        + "\n".join(hint_lines)
                        + "]"
                    )
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": query})

    response_text = chat_multi(messages, temperature=0.4, max_tokens=2048 if is_fast else 8192)

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
                    a["requires_approval"] = True
                    a["status"] = "awaiting_approval"
                    a["timestamp"] = datetime.now(timezone.utc).isoformat()
                    actions.append(a)
        except (json.JSONDecodeError, TypeError):
            pass
    return actions


def _clean_response(text: str) -> str:
    return re.sub(r"```actions?\s*\n.*?\n```", "", text, flags=re.DOTALL).strip()
