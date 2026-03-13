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
from api.utils.qdrant_helpers import get_email_content
from .profile import get_profile
from .executor import execute_action


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


# ── Agent chat ────────────────────────────────────────────────────────────────

def agent_chat(
    user_id: str,
    message: str,
    conversation_history: list | None = None,
    mailbox_id: str | None = None,
) -> dict:
    profile = get_profile(user_id)

    email_limit = _classify_email_limit(message)

    contents, metas = _fetch_emails_by_vector(
        user_id, message, mailbox_id=mailbox_id, limit=email_limit
    )

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

    now_str = datetime.now(timezone.utc).strftime("%A, %B %d, %Y %H:%M UTC")
    profile_summary = _format_profile(profile)

    key_contacts = profile.get("key_contacts", [])
    contacts_lookup = [
        {"name": c.get("name", ""), "email": c.get("email", "")}
        for c in key_contacts
        if c.get("email")
    ]

    system_prompt = (
        f"You are a personal AI assistant. Today is {now_str}.\n\n"
        f"USER PROFILE (learned from their emails):\n{profile_summary}\n\n"
        f"MAILBOXES:\n{json.dumps(mailbox_info, default=str)}\n\n"
        f"KEY CONTACTS (use these to resolve spoken names to email addresses):\n"
        f"{json.dumps(contacts_lookup, default=str)}\n\n"
        f"RELEVANT EMAILS:\n{email_context or 'No relevant emails found.'}\n\n"
        "EMAIL SUMMARIZATION:\n"
        "- When the user asks about emails, "
        "summarize the email CONTENT, not just subject/metadata.\n"
        "- Give 1–2 concise lines per email: sender, subject, and the main gist or action items.\n"
        "- If many emails match, group summaries by topic/sender and highlight urgent items.\n\n"
        "CAPABILITIES — you can take real actions. When needed, include a JSON "
        "block using this format:\n\n"
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
        "RULES:\n"
        "1. Always requires_approval=true for sending emails/messages and for delete/trash/archive\n"
        "2. Match the user's communication style from their profile\n"
        "3. Be proactive — suggest actions when relevant\n"
        "4. Reference specific emails by subject/sender and include a brief content summary\n"
        "5. Respond in the same language the user speaks\n"
        "6. Be concise but substantive\n"
        "7. Use send_reply when the user wants to SEND a reply (e.g. 'reply to that email saying yes', "
        "'send a reply to Ahmed'). Use draft_reply only when they want a draft to review first.\n"
        "8. Use send_email for NEW emails (e.g. 'send an email to John about the meeting'). "
        "Use forward_email when the user wants to FORWARD an existing email (e.g. 'forward that email to Sarah'). "
        "Always include the relevant email_id for forward_email from RELEVANT EMAILS.\n"
        "9. When the user says 'delete', 'remove', 'trash' or 'throw away' an email, use trash_email with email_id from RELEVANT EMAILS. "
        "When they say 'archive' an email, use archive_email. When they say 'mark as read' or 'mark read', use mark_read.\n"
        "10. SCOPE: You are strictly an EMAIL assistant. If the user asks about "
        "anything unrelated to their emails, inbox, contacts, or email-related "
        "tasks (e.g. general knowledge, coding, math, recipes, weather, etc.), "
        "respond warmly and empathetically but gently redirect. Example: "
        "\"That's a great question! But I'm your email assistant — I'm best at "
        "helping you with your inbox, emails, and contacts. Is there anything "
        "email-related I can help with?\"\n"
        "11. SPEECH INPUT: User messages come from voice/speech recognition and "
        "may contain errors — misspelled names, broken email addresses (spaces "
        "in emails, 'at' instead of '@', 'dot' instead of '.'), or garbled "
        "words. Use context and the user's known contacts from their profile to "
        "infer the correct names, email addresses, and intent. If a user says "
        "a name (e.g. 'send email to Ahmed'), look up their email from KEY CONTACTS below.\n"
        "12. RECIPIENT RESOLUTION: **If multiple contacts share the same or similar name, "
        "you MUST list ALL matching contacts with their email addresses and ask the user "
        "to pick the correct one. NEVER auto-pick one silently.**\n"
        "13. SENDER CONFIRMATION: Before emitting any send_email, send_reply, or forward_email "
        "action, you MUST clearly state:\n"
        "   - **From**: which mailbox/email address the email will be sent from\n"
        "   - **To**: the recipient's full email address\n"
        "   - **Subject** and a brief summary of the body\n"
        "   If the user has multiple mailboxes, ask which one to send from. "
        "Only emit the action block AFTER the user confirms these details or says 'send it'."
    )

    messages = [{"role": "system", "content": system_prompt}]

    if conversation_history:
        for m in conversation_history[-8:]:
            messages.append({"role": m.get("role", "user"), "content": m.get("content", "")})

    messages.append({"role": "user", "content": message})

    response_text = chat_multi(messages, temperature=0.5, max_tokens=2048)

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
        model=getattr(django_settings, "OPENAI_TTS_MODEL", "tts-1"),
        voice=getattr(django_settings, "OPENAI_TTS_VOICE", "nova"),
        input=text[:4096],
    )
    audio_b64 = base64.b64encode(response.content).decode("utf-8")
    return {"audio": audio_b64, "format": "mp3"}


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
