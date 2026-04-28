import json
import re
import base64
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from django.conf import settings as django_settings


# ── In-memory cache for profile / mailboxes / preferences ────────────────────
#
# Voice requests hit MongoDB 3× per turn (profile, mailboxes, prefs) even though
# these rarely change mid-session. Small TTL cache saves ~200-400ms per request.

_CACHE: dict = {}
_TTL_PROFILE = 60
_TTL_MAILBOXES = 30
_TTL_PREFS = 60


def _cached(key: str, ttl: int, fn):
    now = time.time()
    entry = _CACHE.get(key)
    if entry and (now - entry["ts"]) < ttl:
        return entry["val"]
    val = fn()
    _CACHE[key] = {"val": val, "ts": now}
    return val


def invalidate_user_cache(user_id: str) -> None:
    """Call from profile/mailbox/settings update endpoints to drop stale cache."""
    for k in list(_CACHE.keys()):
        if k.endswith(f":{user_id}"):
            _CACHE.pop(k, None)


# Voice-specific LLM model (smaller/faster than the default chat model).
_VOICE_MODEL = getattr(django_settings, "OPENAI_VOICE_MODEL", "gpt-4o-mini")

# Sentence boundary for streaming TTS.
_SENTENCE_END_RE = re.compile(r"[.!?][\s\"')\]]*(?=\s|$)")

from database.db import (
    email_metadata_col,
    follow_ups_col,
    mailboxes_col,
)
from api.controllers.ai.services import (
    _fetch_emails_by_vector,
    _fetch_emails_by_date,
    _LATEST_EMAIL_PATTERN,
    _detect_time_range,
    _build_email_block_compact,
    _build_email_block_full,
)
from api.utils.llm import chat_multi, chat_multi_stream, chat_json
from api.utils.qdrant_helpers import (
    get_email_content,
    get_email_ids_by_sender,
    get_emails_content_batch,
)
from .profile import get_profile
from .executor import execute_action


# Matches bare email addresses in free-form user text.
_EMAIL_RE = re.compile(
    r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"
)


# ── Email limit classifier ────────────────────────────────────────────────────

_LIMIT_CLASSIFIER_PROMPT = (
    "You decide how many emails an AI email assistant needs to answer a user's question.\n"
    "Return JSON: {\"limit\": <number>}\n\n"
    "Guidelines:\n"
    "- Specific question about one email/person/thread: 5-8\n"
    "- General question, recent activity, single topic: 10-15\n"
    "- Broad summary, multiple topics/people, comparisons: 20-30\n"
    "- Full inbox overview, 'all emails', 'how many', bulk operations: 30-50\n"
    "- Minimum 5, maximum 50\n"
)


def _classify_email_limit(message: str) -> int:
    try:
        result = chat_json(
            _LIMIT_CLASSIFIER_PROMPT,
            message,
            temperature=0,
            max_tokens=30,
        )
        limit = int(result.get("limit", 10))
        return max(5, min(50, limit))
    except Exception:
        return 10


_BULK_RE = re.compile(
    r"\b(all|every|bulk|inbox|overview|how many|count|summarize|summary)\b",
    re.I,
)


def _voice_email_limit(message: str) -> int:
    """Fast heuristic for voice — no LLM call, <1ms."""
    if _BULK_RE.search(message):
        return 10
    return 6


_EMAIL_RAG_PATTERN = re.compile(
    r"\b("
    r"email|emails|e-mail|inbox|mailbox|mailboxes|unread|draft|drafts|reply|replies|forward|"
    r"sent|trash|archive|spam|folder|message from|message to|write to|send (?:an? )?email|"
    r"attachment|subject line|sender|recipient|newsletter|meeting invite|invoice"
    r")\b",
    re.I,
)


def _agent_needs_email_rag(message: str) -> bool:
    """Pure chit-chat skips embedding, Qdrant, and Cohere rerank."""
    m = message.strip()
    if len(m) > 160:
        return True
    if _EMAIL_RAG_PATTERN.search(m):
        return True
    if re.search(
        r"\b(find|search|show (?:me )?|summari[sz]e|list (?:my )?|last|latest|recent|"
        r"what did|who (?:wrote|sent)|anything from)\b",
        m,
        re.I,
    ):
        return True
    return False


# ── Agent chat ────────────────────────────────────────────────────────────────

def _extract_sender_emails(message: str) -> list[str]:
    """Extract unique email addresses from the user's message (lowercased)."""
    if not message:
        return []
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in _EMAIL_RE.findall(message):
        addr = raw.strip().lower()
        if addr and addr not in seen:
            seen.add(addr)
            ordered.append(addr)
    return ordered


def _fetch_emails_by_sender_addrs(
    user_id: str,
    addrs: list[str],
    mailbox_id: str | None,
    per_sender_limit: int = 15,
) -> tuple[list[dict], list[dict]]:
    """Deterministic sender-based retrieval.

    Vector search does not reliably match exact email addresses, so when the
    user's message mentions a sender like "delete all emails from x@y.com" we
    still need to surface those emails to the LLM so it emits the correct
    action instead of claiming none were found.
    """
    if not addrs:
        return [], []

    all_ids: list[str] = []
    seen_ids: set[str] = set()
    for addr in addrs:
        ids = get_email_ids_by_sender(user_id, addr, mailbox_id)
        for eid in ids[:per_sender_limit]:
            if eid and eid not in seen_ids:
                seen_ids.add(eid)
                all_ids.append(eid)
    if not all_ids:
        return [], []

    content_map = get_emails_content_batch(all_ids, user_id)
    metas_by_id = {
        str(m["_id"]): m
        for m in email_metadata_col().find(
            {"_id": {"$in": all_ids}, "user_id": user_id, "archived": False, "trashed": False}
        )
    }

    contents: list[dict] = []
    metas: list[dict] = []
    for eid in all_ids:
        c = content_map.get(eid)
        m = metas_by_id.get(eid)
        if c and m:
            contents.append(c)
            metas.append(m)
    return contents, metas


_AGENT_LATEST_RE = re.compile(
    r"\b(latest|newest|most\s+recent|last|recent)\s+(email|mail|message)\b"
    r"|\b(last|latest|newest)\s+email\b"
    r"|\bwhat(?:'s| is)\s+my\s+latest\s+email\b"
    r"|\b(naya|nayi|akhri|recent|new)\s+(email|mail)\b"
    r"|\bnew(?:est)?\s+email\b",
    re.I,
)

_UNREAD_RE = re.compile(
    r"\b(unread|unseen|na\s*parhi?|unread\s*emails?|unread\s*messages?)\b",
    re.I,
)

_STARRED_RE = re.compile(
    r"\b(starred|favourite[ds]?|bookmarked|flagged)\s*(emails?|messages?)?\b",
    re.I,
)

_BROAD_INBOX_RE = re.compile(
    r"\b(inbox|overview|briefing|digest|summary|summarize|recap)\b",
    re.I,
)

# Explicit topic/content search — vector search is correct here even for voice.
_TOPIC_SEARCH_RE = re.compile(
    r"\b(find|search|look\s+for|dhundho?|about|related\s+to|concerning|"
    r"regarding|mentioning|containing|with\s+subject|baray\s+mein)\b",
    re.I,
)


def _fetch_rag_context(
    user_id: str,
    message: str,
    mailbox_id: str | None,
    voice: bool = False,
    time_label: str | None = None,
    start_dt=None,
    end_dt=None,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Run sender lookup and date/vector search in parallel.

    When a time range is detected (today, yesterday, this week …) we use
    _fetch_emails_by_date instead of vector search — vector search ranks by
    semantic similarity and completely ignores email dates, so "today's emails"
    via vector search returns whatever is semantically close to the word
    "today", not what was actually received today.
    """
    sender_addrs = _extract_sender_emails(message)
    needs_rag = _agent_needs_email_rag(message)
    want_latest = bool(_AGENT_LATEST_RE.search(message) or _LATEST_EMAIL_PATTERN.search(message))
    has_time_range = bool(time_label)
    want_unread = bool(_UNREAD_RE.search(message)) and not has_time_range
    want_starred = bool(_STARRED_RE.search(message)) and not has_time_range
    # Broad inbox overview without time filter → use date-sorted recent emails
    want_broad = (
        bool(_BROAD_INBOX_RE.search(message))
        and not has_time_range
        and not want_latest
        and not want_unread
        and not want_starred
        and not sender_addrs
    )

    if voice:
        email_limit = _voice_email_limit(message)
    else:
        email_limit = _classify_email_limit(message)

    fetch_limit = (email_limit * 3) if (has_time_range or want_latest) else email_limit

    # Decide retrieval mode (logged for debugging).
    if has_time_range:
        _mode = f"date({time_label})"
    elif want_latest:
        _mode = "latest"
    elif want_unread:
        _mode = "unread_filter"
    elif want_starred:
        _mode = "starred_filter"
    elif want_broad:
        _mode = "broad_inbox"
    else:
        _mode = "vector"

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool:
        sender_future = (
            pool.submit(_fetch_emails_by_sender_addrs, user_id, sender_addrs, mailbox_id)
            if sender_addrs else None
        )

        if needs_rag:
            if has_time_range:
                # Date-range query — applies even when a sender is also mentioned.
                # Sender results come from sender_future; date fetch gives the
                # time-filtered view for the LLM context.
                vector_future = pool.submit(
                    _fetch_emails_by_date,
                    user_id, start_dt, end_dt, mailbox_id, fetch_limit,
                )
            elif want_latest and not sender_addrs:
                vector_future = pool.submit(
                    _fetch_emails_by_date,
                    user_id, None, None, mailbox_id, fetch_limit,
                )
            elif want_unread and not sender_addrs:
                # "Unread emails" — must filter by read=False, not semantic search.
                vector_future = pool.submit(
                    _fetch_emails_by_date,
                    user_id, None, None, mailbox_id, email_limit,
                    False,   # read_filter=False
                    None,    # starred_filter
                )
            elif want_starred and not sender_addrs:
                # "Starred emails" — must filter by starred=True.
                vector_future = pool.submit(
                    _fetch_emails_by_date,
                    user_id, None, None, mailbox_id, email_limit,
                    None,    # read_filter
                    True,    # starred_filter=True
                )
            elif want_broad:
                # Inbox overview without time range — date-sorted recent emails.
                vector_future = pool.submit(
                    _fetch_emails_by_date,
                    user_id, None, None, mailbox_id, email_limit,
                )
            elif voice and not _TOPIC_SEARCH_RE.search(message):
                # Voice fallback: regex didn't identify a specific pattern.
                # Date-sorted recent emails are safer than vector search for voice
                # because vector search ignores recency and can return stale emails.
                # Exception: explicit topic/content searches still use vector search.
                vector_future = pool.submit(
                    _fetch_emails_by_date,
                    user_id, None, None, mailbox_id, email_limit,
                )
                _mode = "voice_safe_fallback"
            else:
                vector_future = pool.submit(
                    _fetch_emails_by_vector,
                    user_id, message,
                    mailbox_id=mailbox_id,
                    limit=email_limit,
                    search_chunk_limit=60 if voice else 140,
                    rerank_cap=0 if voice else 90,
                )
        else:
            vector_future = None

        sender_contents, sender_metas = sender_future.result() if sender_future else ([], [])
        vector_contents, vector_metas = vector_future.result() if vector_future else ([], [])

    elapsed = time.perf_counter() - t0
    print(
        f"  [RAG]  {elapsed:.2f}s — "
        f"sender:{len(sender_contents)}  vector:{len(vector_contents)}"
        f"  limit:{email_limit}  mode:{_mode}"
        f"{'  (skipped)' if not needs_rag and not sender_addrs else ''}",
        flush=True,
    )
    return sender_contents, sender_metas, vector_contents, vector_metas


def _build_agent_messages(
    user_id: str,
    message: str,
    conversation_history: list | None,
    mailbox_id: str | None,
    voice: bool = False,
) -> tuple[list[dict], int, list[dict]]:
    """Build LLM message list, max_tokens, and sources. Shared by streaming and non-streaming paths."""
    from api.controllers.settings.services import get_user_preferences_prompt

    # Detect time range before spawning RAG so the fetch uses the right date
    # filter and the system prompt can mention the period to the LLM.
    time_label, start_dt, end_dt = _detect_time_range(message)

    # Fan out all independent DB/RAG work in parallel.
    # For voice, cache profile/mailboxes/prefs to skip ~200-400ms of DB work.
    t_fanout = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        if voice:
            profile_f = pool.submit(
                _cached, f"profile:{user_id}", _TTL_PROFILE,
                lambda: get_profile(user_id),
            )
            mailboxes_f = pool.submit(
                _cached, f"mboxes:{user_id}", _TTL_MAILBOXES,
                lambda: list(mailboxes_col().find({"user_id": user_id})),
            )
            prefs_f = pool.submit(
                _cached, f"prefs:{user_id}", _TTL_PREFS,
                lambda: get_user_preferences_prompt(user_id),
            )
        else:
            profile_f = pool.submit(get_profile, user_id)
            mailboxes_f = pool.submit(lambda: list(mailboxes_col().find({"user_id": user_id})))
            prefs_f = pool.submit(get_user_preferences_prompt, user_id)

        rag_f = pool.submit(
            _fetch_rag_context,
            user_id, message, mailbox_id, voice,
            time_label, start_dt, end_dt,
        )

        profile = profile_f.result()
        mbox_docs = mailboxes_f.result()
        user_prefs = prefs_f.result()
        sender_contents, sender_metas, contents, metas = rag_f.result()
    print(f"  [FANOUT] {time.perf_counter() - t_fanout:.2f}s — profile+mailboxes+prefs+RAG in parallel", flush=True)

    if sender_contents:
        seen_ids = {c.get("email_id") for c in sender_contents if c.get("email_id")}
        merged_contents = list(sender_contents)
        merged_metas = list(sender_metas)
        for c, m in zip(contents, metas):
            eid = c.get("email_id")
            if not eid or eid not in seen_ids:
                merged_contents.append(c)
                merged_metas.append(m)
                if eid:
                    seen_ids.add(eid)
        contents, metas = merged_contents, merged_metas

    mailbox_info = [
        {"id": str(mb["_id"]), "email": mb["email"], "name": mb["name"]}
        for mb in mbox_docs
    ]

    email_context = ""
    sources: list[dict] = []
    if contents:
        blocks = []
        # Voice always uses compact blocks — shorter prompt = faster TTFT.
        use_compact = voice or len(contents) > 10
        for c, m in zip(contents, metas):
            blocks.append(
                _build_email_block_compact(c, m)
                if use_compact
                else _build_email_block_full(c, m)
            )
            sources.append({
                "email_id": c.get("email_id", ""),
                "subject": c.get("subject", ""),
            })
        email_context = "\n\n".join(blocks) if use_compact else "\n\n━━━━━━━━━━━━━━━━━━━━\n\n".join(blocks)

    now_str = datetime.now(timezone.utc).strftime("%A, %B %d, %Y %H:%M UTC")
    profile_summary = _format_profile(profile)
    prefs_block = f"\n{user_prefs}\n\n" if user_prefs else ""

    key_contacts = profile.get("key_contacts", [])
    # Voice: trim to 5 contacts (smaller prompt). Text: full list.
    contacts_lookup = [
        {"name": c.get("name", ""), "email": c.get("email", "")}
        for c in key_contacts[:5 if voice else len(key_contacts)]
        if c.get("email")
    ]

    # Context note injected into system prompts so LLM knows exactly what's shown.
    if time_label:
        period = time_label.replace("_", " ")
        _time_note = (
            f"FILTER ACTIVE — The user asked about '{period}'. "
            f"The {len(contents)} email(s) below are ALL emails found in that period. "
            "Base your answer ONLY on these emails. "
            "If the list is empty, say no emails were received in that period.\n\n"
        )
    elif want_unread:
        _time_note = (
            f"FILTER ACTIVE — Showing unread emails only ({len(contents)} found). "
            "Base your answer on these unread emails.\n\n"
        )
    elif want_starred:
        _time_note = (
            f"FILTER ACTIVE — Showing starred emails only ({len(contents)} found). "
            "Base your answer on these starred emails.\n\n"
        )
    else:
        _time_note = ""

    if voice:
        system_prompt = (
            f"You are a real-time voice email assistant. Today: {now_str}.\n"
            "Spoken output — no markdown, no bullets, no lists, no asterisks. "
            "Short sentences, contractions, under 3 sentences when possible. "
            "Respond in the user's language.\n\n"
            f"PROFILE: {profile_summary}\n"
            + prefs_block +
            f"MAILBOXES: {json.dumps(mailbox_info, default=str)}\n"
            f"CONTACTS: {json.dumps(contacts_lookup, default=str)}\n\n"
            + _time_note +
            "EMAILS (sorted newest first — the FIRST email listed IS the most recent/latest):\n"
            "When the user asks for 'latest', 'newest', 'last', 'naya', 'akhri' email, "
            "ALWAYS refer to the FIRST email in the list below.\n"
            f"{email_context or 'None found.'}\n\n"
            "ACTIONS — append a ```actions JSON block when acting. Always "
            'requires_approval=true. Example: [{"type":"mark_read","from_email":"x@y.com",'
            '"label":"Mark all from X","requires_approval":true}]\n'
            "Types: read_emails, open_email, open_latest_email, search_emails, "
            "send_email, draft_email, draft_reply, send_reply, reply_all, forward_email, "
            "trash_email, archive_email, delete_email, mark_read, mark_unread, "
            "mark_all_read, mark_all_unread, "
            "star_email (star one or bulk), unstar_email (unstar one or bulk), "
            "mark_all_starred (ALL inbox → starred), mark_all_unstarred (ALL inbox → unstarred), "
            "snooze_email, send_whatsapp, set_reminder.\n\n"
            "TARGETING: single→email_id. bulk→from_email/subject/keywords/folder/label_name/"
            "read/starred/date_from/date_to. Never list many email_ids.\n"
            "SENDER: if user names an email address, ALWAYS emit the action with that "
            "from_email even if EMAILS above doesn't show a match — executor searches full inbox.\n"
            "REPLY: send_reply/reply_all auto-target the original sender. ALWAYS include "
            "email_id (or from_email) of the exact original email. Never target by keywords alone. "
            "No 'to' field on replies. If ambiguous, ask first.\n"
            "BULK: mark_all_read = everything→read. mark_all_unread = everything→unread. Don't confuse.\n"
            "STARRING: star_email = one/bulk star. unstar_email = one/bulk unstar. "
            "mark_all_starred = ALL inbox → starred (no email_id). "
            "mark_all_unstarred = ALL inbox → unstarred (no email_id). "
            "NEVER list individual email_ids for 'all star/unstar' — the context only shows a preview. "
            "NEVER say starring is not supported — it is.\n"
            "SCOPE: email only. Before sending, confirm mailbox+recipient+gist in one short spoken line."
        )
    else:
        system_prompt = (
            f"You are a real-time VOICE assistant for email. Today is {now_str}.\n"
            "Your responses are spoken aloud via text-to-speech, so write exactly how a "
            "friendly human would TALK — short sentences, natural rhythm, no markdown, no "
            "bullet points, no asterisks, no numbered lists. Use commas and periods for "
            "natural pauses. Keep answers under 3 sentences when possible.\n\n"
            "VOICE STYLE:\n"
            "- Talk like a helpful colleague, warm and concise.\n"
            "- Never use formatting: no **, no ##, no - lists, no numbered lists.\n"
            "- Say things like \"You got an email from Ahmed about the project deadline\" "
            "not \"1. **Ahmed** - Subject: Project Deadline\".\n"
            "- For multiple emails, briefly mention the top 2-3 and offer to go deeper.\n"
            "- Use contractions: \"you've\", \"there's\", \"I'll\".\n"
            "- Use filler-free but natural phrasing. Don't say \"certainly\" or \"absolutely\".\n\n"
            f"USER PROFILE:\n{profile_summary}\n\n"
            + prefs_block +
            f"MAILBOXES:\n{json.dumps(mailbox_info, default=str)}\n\n"
            f"KEY CONTACTS:\n{json.dumps(contacts_lookup, default=str)}\n\n"
            + _time_note +
            f"RELEVANT EMAILS:\n{email_context or 'None found.'}\n\n"
            "ACTIONS — when needed, append a JSON block (the user won't see it, only the "
            "spoken text before it). Each email in the context has an ID.\n"
            "```actions\n"
            '[{"type":"mark_read","from_email":"x@y.com","label":"Mark all from X as read",'
            '"requires_approval":true}]\n'
            "```\n"
            "TARGETING: For one email use email_id. For bulk by sender use from_email. "
            "You can also use subject, keywords, folder, mailbox_id, label_name, read, date_from/date_to "
            "to match multiple emails. "
            "Never list many individual email_ids — use a filter.\n\n"
            "SENDER LOOKUPS: When the user references an email address (e.g. "
            "'delete all emails from x@y.com'), the backend will exactly match "
            "that address against the inbox. ALWAYS emit the action with "
            "from_email set to the mentioned address — even if RELEVANT EMAILS "
            "above doesn't obviously show a match. Do NOT reply 'I couldn't find "
            "any emails from that sender' just because the shown context is short; "
            "the executor searches the full mailbox, not just the shown context.\n\n"
            "Types: read_emails, open_email, open_latest_email, search_emails, "
            "send_email, draft_email, draft_reply, send_reply, reply_all, forward_email, "
            "trash_email, move_to_trash, archive_email, delete_email, "
            "mark_read, mark_unread (accept email_id OR filters for bulk), "
            "mark_all_read (ALL inbox → read, no email_id), "
            "mark_all_unread (ALL inbox → unread, no email_id), "
            "star_email (star one/bulk by filters), unstar_email (unstar one/bulk), "
            "mark_all_starred (ALL inbox → starred, no email_id), "
            "mark_all_unstarred (ALL inbox starred → unstarred, no email_id), "
            "snooze_email (email_id or filters + hours), "
            "send_whatsapp (to, body), set_reminder (email_id, hours).\n"
            "Always set requires_approval=true for EVERY action.\n\n"
            "RULES:\n"
            "1. VOICE FIRST: Every response must sound natural when spoken. No visual formatting.\n"
            "2. BREVITY: 1-3 sentences for simple queries. Up to 5 for email summaries.\n"
            "3. Speech input may have errors — infer names, emails, intent from context and KEY CONTACTS.\n"
            "4. If multiple contacts share a name, ask the user to pick.\n"
            "5. Before sending, confirm: from which mailbox, to whom, about what. Keep it spoken.\n"
            "6. Respond in the same language the user speaks.\n"
            "7. SCOPE: email assistant only. Gently redirect off-topic questions.\n"
            "8. Use send_reply for replies, reply_all for reply-all, send_email for new messages.\n"
            "9. REPLY SAFETY — critical, read carefully:\n"
            "    - send_reply and reply_all ALWAYS go to the sender of the original email. "
            "      You must identify the original email precisely.\n"
            "    - ALWAYS include the exact email_id from RELEVANT EMAILS above when replying. "
            "      Pick the ID that matches what the user is clearly talking about.\n"
            "    - If no email_id is available, include from_email (the original sender's exact address).\n"
            "    - NEVER target a reply with only keywords or subject — that risks replying to the wrong person.\n"
            "    - DO NOT set a 'to' field on send_reply. The reply auto-targets the original sender. "
            "      If the user actually wants to send a new message to a DIFFERENT person, use send_email instead.\n"
            "    - Before emitting a send_reply, confirm in one short spoken sentence: whose email you're replying to "
            "      and what you'll say, so the user can catch a wrong pick.\n"
            "    - If the user's request is ambiguous about WHICH email (e.g. multiple emails from the same sender), "
            "      ask a quick clarifying question before emitting the action.\n"
            "10. For delete/trash/archive/mark-unread/snooze, use the matching action type. "
            "delete_email and trash_email both mean move to trash, not permanent delete.\n"
            "11. BULK READ vs UNREAD — do not confuse them:\n"
            "    - User wants READ / clear unread / mark everything read → type mark_all_read.\n"
            "    - User wants UNREAD / mark all as unread / make everything unread → type mark_all_unread.\n"
            "12. For 'all from sender X', use from_email filter. For 'all inbox', use mark_all_read/mark_all_unread.\n"
            "13. STARRING IS SUPPORTED. When user says star/favourite/bookmark emails:\n"
            "    - One email → star_email with email_id.\n"
            "    - Bulk by sender/subject → star_email with from_email/subject/keywords.\n"
            "    - ALL inbox star → mark_all_starred (no email_id, executor stars EVERY inbox email).\n"
            "    - ALL inbox unstar / remove all stars → mark_all_unstarred (no email_id, executor unstarrs EVERY starred email).\n"
            "    CRITICAL: NEVER list individual email_ids for 'star all' or 'unstar all'. "
            "The EMAILS section only shows a preview — the inbox has far more emails. "
            "Always use mark_all_starred / mark_all_unstarred for any 'all' starring request. "
            "NEVER say starring is not supported or you don't have the capability."
        )

    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    if conversation_history:
        # Voice: 4 turns of history (faster). Text: 8 turns (richer context).
        history_window = 4 if voice else 8
        for m in conversation_history[-history_window:]:
            messages.append({"role": m.get("role", "user"), "content": m.get("content", "")})
    messages.append({"role": "user", "content": message})

    if voice:
        max_out = 300 if contents else 150
    else:
        max_out = 800 if contents else 250
    return messages, max_out, sources[:10]


def agent_chat(
    user_id: str,
    message: str,
    conversation_history: list | None = None,
    mailbox_id: str | None = None,
) -> dict:
    messages, max_out, sources = _build_agent_messages(user_id, message, conversation_history, mailbox_id)
    response_text = chat_multi(messages, temperature=0.6, max_tokens=max_out)
    actions = _extract_actions(response_text, mailbox_id=mailbox_id)
    clean_text = _clean_response(response_text)
    return {"content": clean_text, "actions": actions, "sources": sources}


def _safe_tts(text: str) -> str | None:
    """Generate TTS audio for one sentence. Returns base64 or None on failure."""
    t = (text or "").strip()
    if not t:
        return None
    try:
        return generate_speech(t).get("audio")
    except Exception as e:
        print(f"[TTS] streaming sentence failed: {e}", flush=True)
        return None


def agent_chat_stream(
    user_id: str,
    message: str,
    conversation_history: list | None = None,
    mailbox_id: str | None = None,
    stream_tts: bool = False,
):
    """Stream LLM tokens (+ optional sentence-by-sentence TTS audio).

    Events yielded:
      - {"type": "token", "content": str} — LLM text delta
      - {"type": "audio", "audio": base64, "text": sentence}  (only if stream_tts)
      - {"type": "done", "content": str, "actions": [...], "sources": [...]}
    """
    t_start = time.perf_counter()
    messages, max_out, sources = _build_agent_messages(user_id, message, conversation_history, mailbox_id, voice=True)
    t_rag_done = time.perf_counter()
    print(f"  [PREP] {t_rag_done - t_start:.2f}s — profile+RAG+prompt build", flush=True)

    full_response = ""
    sentence_buffer = ""
    tts_queue: deque = deque()
    past_actions_marker = False  # once ``` appears, stop TTS (action JSON block)
    tts_pool = ThreadPoolExecutor(max_workers=3) if stream_tts else None
    t_first_token = None

    try:
        for token in chat_multi_stream(
            messages,
            temperature=0.3,
            max_tokens=max_out,
            model=_VOICE_MODEL,
        ):
            if t_first_token is None:
                t_first_token = time.perf_counter()
                print(f"  [TTFT] {t_first_token - t_rag_done:.2f}s — time to first LLM token", flush=True)

            full_response += token
            yield {"type": "token", "content": token}

            if not stream_tts:
                continue

            if past_actions_marker:
                # Already passed the actions fence — skip TTS for remaining tokens.
                continue

            sentence_buffer += token

            # If the actions fence appears inside the buffer, flush what comes
            # BEFORE it as a final sentence and stop TTS for the rest.
            if "```" in sentence_buffer:
                before, _, _ = sentence_buffer.partition("```")
                before = before.strip()
                if before:
                    tts_queue.append((before, tts_pool.submit(_safe_tts, before)))
                sentence_buffer = ""
                past_actions_marker = True
                continue

            # Detect complete sentences and launch TTS in parallel.
            while True:
                match = _SENTENCE_END_RE.search(sentence_buffer)
                if not match:
                    break
                sentence = sentence_buffer[: match.end()].strip()
                sentence_buffer = sentence_buffer[match.end():]
                if sentence:
                    tts_queue.append((sentence, tts_pool.submit(_safe_tts, sentence)))

            # Emit any TTS audio that is ready (in-order).
            while tts_queue and tts_queue[0][1].done():
                sent, future = tts_queue.popleft()
                audio = future.result()
                if audio:
                    yield {"type": "audio", "audio": audio, "text": sent}

        t_llm_done = time.perf_counter()
        print(
            f"  [LLM]  {t_llm_done - (t_first_token or t_rag_done):.2f}s — stream duration  "
            f"| {len(full_response)} chars total  "
            f"| TOTAL {t_llm_done - t_start:.2f}s end-to-end",
            flush=True,
        )

        # Flush the tail of the sentence buffer + drain pending TTS.
        if stream_tts:
            if not past_actions_marker and sentence_buffer.strip():
                tail = sentence_buffer.strip()
                tts_queue.append((tail, tts_pool.submit(_safe_tts, tail)))

            for sent, future in list(tts_queue):
                try:
                    audio = future.result(timeout=10)
                except Exception:
                    audio = None
                if audio:
                    yield {"type": "audio", "audio": audio, "text": sent}
            tts_queue.clear()

        actions = _extract_actions(full_response, mailbox_id=mailbox_id)
        clean_text = _clean_response(full_response)
        yield {"type": "done", "content": clean_text, "actions": actions, "sources": sources}
    finally:
        if tts_pool is not None:
            tts_pool.shutdown(wait=False)


# ── Proactive suggestions ────────────────────────────────────────────────────

def get_suggestions(user_id: str) -> list[dict]:
    suggestions: list[dict] = []

    overdue = list(
        follow_ups_col()
        .find({"user_id": user_id, "status": "overdue"})
        .limit(3)
    )
    for fu in overdue:
        content = get_email_content(fu.get("email_id", ""), user_id)
        if content:
            suggestions.append({
                "id": str(fu["_id"]),
                "title": f"Reply to {content.get('from_name') or content.get('from_email','')}",
                "description": f"Overdue: {content.get('subject','')}",
                "urgency": "high",
                "action_label": "Draft Reply",
                "type": "draft_reply",
                "email_id": fu.get("email_id", ""),
            })

    unread_docs = list(
        email_metadata_col()
        .find({
            "user_id": user_id,
            "read": False,
            "archived": False,
            "trashed": False,
        })
        .sort("date", -1)
        .limit(20)
    )
    for doc in unread_docs:
        eid = str(doc["_id"])
        content = get_email_content(eid, user_id)
        if content and content.get("priority") == "high":
            if not any(s["id"] == eid for s in suggestions):
                suggestions.append({
                    "id": eid,
                    "title": f"Urgent from {content.get('from_name','')}",
                    "description": content.get("subject", ""),
                    "urgency": "high",
                    "action_label": "View & Respond",
                    "type": "view",
                    "email_id": eid,
                })
        if len(suggestions) >= 6:
            break

    unread_count = email_metadata_col().count_documents({
        "user_id": user_id,
        "read": False,
        "archived": False,
        "trashed": False,
    })
    if unread_count > 10:
        suggestions.append({
            "id": "triage",
            "title": f"{unread_count} unread emails",
            "description": "Ask me to summarize or prioritize your inbox.",
            "urgency": "medium",
            "action_label": "Prioritize Inbox",
            "type": "chat",
        })

    return suggestions[:6]


# ── Action approve / reject ───────────────────────────────────────────────────

def approve_and_execute(user_id: str, action_data: dict) -> dict:
    return execute_action(user_id, action_data)


def reject_action(action_id: str) -> dict:
    return {"id": action_id, "status": "rejected"}


# ── Profile pass-through ─────────────────────────────────────────────────────

def get_user_profile(user_id: str) -> dict:
    return get_profile(user_id)


def build_user_profile(user_id: str) -> dict:
    return get_profile(user_id, force_rebuild=True)


# ── Text-to-speech ────────────────────────────────────────────────────────────

def generate_speech(text: str) -> dict:
    from openai import OpenAI

    client = OpenAI(api_key=django_settings.OPENAI_API_KEY)
    response = client.audio.speech.create(
        model=getattr(django_settings, "OPENAI_TTS_MODEL", "tts-1"),
        voice=getattr(django_settings, "OPENAI_TTS_VOICE", "nova"),
        input=text[:4096],
        speed=1.0,
    )
    audio_b64 = base64.b64encode(response.content).decode("utf-8")
    return {"audio": audio_b64, "format": "mp3"}


# ── Speech-to-text (Whisper) ──────────────────────────────────────────────────

def transcribe_audio(audio_file) -> dict:
    """Transcribe an uploaded audio file using OpenAI Whisper."""
    import tempfile
    import os
    from openai import OpenAI

    client = OpenAI(api_key=django_settings.OPENAI_API_KEY)

    ext = os.path.splitext(audio_file.name)[1] or ".webm"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        for chunk in audio_file.chunks():
            tmp.write(chunk)
        tmp_path = tmp.name

    file_kb = os.path.getsize(tmp_path) / 1024
    t0 = time.perf_counter()
    try:
        with open(tmp_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language="en",
            )
        elapsed = time.perf_counter() - t0
        text = transcript.text.strip()
        print(
            f"  [STT]  {elapsed:.2f}s — Whisper  |  {file_kb:.1f} KB  |  \"{text[:60]}{'...' if len(text) > 60 else ''}\"",
            flush=True,
        )
        return {"text": text}
    finally:
        os.unlink(tmp_path)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _format_profile(profile: dict) -> str:
    parts: list[str] = []

    style = profile.get("communication_style", {})
    if style and style.get("tone") != "Not enough data yet":
        parts.append(
            f"Communication: {style.get('tone','')} | "
            f"Formality: {style.get('formality','')} | "
            f"Length: {style.get('avg_length','')}"
        )

    contacts = profile.get("key_contacts", [])
    if contacts:
        lines = [
            f"  - {c.get('name','')} <{c.get('email','')}> ({c.get('relationship','')}, {c.get('interaction_frequency','')})"
            for c in contacts[:10]
        ]
        parts.append("Key contacts:\n" + "\n".join(lines))

    topics = profile.get("topics_and_interests", [])
    if topics:
        parts.append(f"Topics: {', '.join(topics[:8])}")

    traits = profile.get("personality_traits", [])
    if traits:
        parts.append(f"Traits: {', '.join(traits)}")

    patterns = profile.get("work_patterns", {})
    if patterns and patterns.get("peak_hours") != "Not enough data":
        parts.append(
            f"Work: Peak hours: {patterns.get('peak_hours','')} | "
            f"Priorities: {patterns.get('priorities','')}"
        )

    prefs = profile.get("response_preferences", {})
    if prefs and prefs.get("urgency_handling") != "Not enough data":
        parts.append(
            f"Urgency: {prefs.get('urgency_handling','')} | "
            f"Delegation: {prefs.get('delegation_style','')}"
        )

    return "\n".join(parts) if parts else "No profile data yet — will learn as more emails come in."


def _normalize_mailbox_scope(mailbox_id: str | None) -> str | None:
    if not mailbox_id:
        return None
    cleaned = str(mailbox_id).strip()
    if not cleaned or cleaned.lower() in {"all", "*", "any"}:
        return None
    return cleaned


def _extract_actions(text: str, mailbox_id: str | None = None) -> list[dict]:
    actions: list[dict] = []
    scoped_mailbox_id = _normalize_mailbox_scope(mailbox_id)
    for match in re.findall(r"```actions?\s*\n(.*?)\n```", text, re.DOTALL):
        try:
            parsed = json.loads(match.strip())
            if isinstance(parsed, list):
                ts = int(datetime.now(timezone.utc).timestamp())
                for i, a in enumerate(parsed):
                    if (
                        scoped_mailbox_id
                        and isinstance(a, dict)
                        and not a.get("mailbox_id")
                    ):
                        a["mailbox_id"] = scoped_mailbox_id
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
