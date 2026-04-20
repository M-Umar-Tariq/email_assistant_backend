import json
import re
import base64
from datetime import datetime, timezone

from django.conf import settings as django_settings

from database.db import (
    email_metadata_col,
    follow_ups_col,
    mailboxes_col,
)
from api.controllers.ai.services import (
    _fetch_emails_by_vector,
    _build_email_block_compact,
    _build_email_block_full,
)
from api.utils.llm import chat_multi, chat_json
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


_EMAIL_RAG_PATTERN = re.compile(
    r"\b("
    r"email|emails|e-mail|inbox|mailbox|mailboxes|unread|draft|drafts|reply|replies|forward|"
    r"sent|trash|archive|spam|folder|message from|message to|write to|send (?:an? )?email|"
    r"attachment|subject line|sender|recipient|newsletter|meeting invite|invoice"
    r")\b",
    re.I,
)


def _agent_needs_email_rag(message: str) -> bool:
    """Pure chit-chat skips embedding, Qdrant, Cohere rerank, and the limit-classifier LLM."""
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


def agent_chat(
    user_id: str,
    message: str,
    conversation_history: list | None = None,
    mailbox_id: str | None = None,
) -> dict:
    profile = get_profile(user_id)

    sender_addrs = _extract_sender_emails(message)
    sender_contents, sender_metas = _fetch_emails_by_sender_addrs(
        user_id, sender_addrs, mailbox_id
    )

    if _agent_needs_email_rag(message):
        email_limit = _classify_email_limit(message)
        contents, metas = _fetch_emails_by_vector(
            user_id,
            message,
            mailbox_id=mailbox_id,
            limit=email_limit,
            search_chunk_limit=140,
            rerank_cap=90,
        )
    else:
        contents, metas = [], []

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

    mbox_docs = list(mailboxes_col().find({"user_id": user_id}))
    mailbox_info = [
        {"id": str(mb["_id"]), "email": mb["email"], "name": mb["name"]}
        for mb in mbox_docs
    ]

    email_context = ""
    sources: list[dict] = []
    if contents:
        blocks = []
        use_compact = len(contents) > 10
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
        email_context = ("\n\n" if use_compact else "\n\n━━━━━━━━━━━━━━━━━━━━\n\n").join(blocks)

    from api.controllers.settings.services import get_user_preferences_prompt

    now_str = datetime.now(timezone.utc).strftime("%A, %B %d, %Y %H:%M UTC")
    profile_summary = _format_profile(profile)
    user_prefs = get_user_preferences_prompt(user_id)
    prefs_block = f"\n{user_prefs}\n\n" if user_prefs else ""

    key_contacts = profile.get("key_contacts", [])
    contacts_lookup = [
        {"name": c.get("name", ""), "email": c.get("email", "")}
        for c in key_contacts
        if c.get("email")
    ]

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
        "12. For 'all from sender X', use from_email filter. For 'all inbox', use mark_all_read/mark_all_unread."
    )

    messages = [{"role": "system", "content": system_prompt}]

    if conversation_history:
        for m in conversation_history[-8:]:
            messages.append({"role": m.get("role", "user"), "content": m.get("content", "")})

    messages.append({"role": "user", "content": message})

    max_out = 800 if contents else 250
    response_text = chat_multi(messages, temperature=0.6, max_tokens=max_out)

    actions = _extract_actions(response_text)
    clean_text = _clean_response(response_text)

    return {
        "content": clean_text,
        "actions": actions,
        "sources": sources[:10],
    }


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
        model=getattr(django_settings, "OPENAI_TTS_MODEL", "tts-1-hd"),
        voice=getattr(django_settings, "OPENAI_TTS_VOICE", "nova"),
        input=text[:4096],
        speed=1.05,
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

    try:
        with open(tmp_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language="en",
            )
        return {"text": transcript.text.strip()}
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
