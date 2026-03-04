import json
from collections import Counter

from api.utils.llm import chat
from api.utils.qdrant_helpers import scroll_all_chunk0


def generate_draft(to: str = "", subject: str = "", context: str = "", tone: str = "formal", sender_name: str = "") -> str:
    name_for_signoff = sender_name.strip() if sender_name else "the sender"
    prompt = (
        f"To: {to}\nSubject: {subject}\n"
        f"Context/instructions: {context}\nTone: {tone}\n"
        f"MY NAME IS: {name_for_signoff}"
    )
    return chat(
        system_prompt=(
            "You are an email drafting assistant. Write a professional email "
            "based on the details below. Use the specified tone. "
            f"IMPORTANT: The sender's name is \"{name_for_signoff}\". "
            f"You MUST sign off using exactly \"{name_for_signoff}\" — "
            "NEVER write [Your Name], [Name], or any placeholder. "
            "Return ONLY the email body text (no subject line, no metadata)."
        ),
        user_message=prompt,
        temperature=0.7,
    )


def rewrite(text: str, action: str = "polish", target_language: str | None = None) -> str:
    instructions = {
        "shorten": "Make this email more concise while preserving all key points.",
        "polish": "Improve the clarity, flow, and professionalism of this email.",
        "translate": f"Translate this email to {target_language or 'Spanish'} while keeping the tone.",
        "rewrite": "Rewrite this email with a fresh approach while keeping the same message.",
    }
    instruction = instructions.get(action, instructions["polish"])

    return chat(
        system_prompt=f"You are an email editing assistant. {instruction} Return ONLY the rewritten email body.",
        user_message=text,
        temperature=0.6,
    )


def proofread(text: str) -> list[dict]:
    raw = chat(
        system_prompt=(
            "You are a professional proofreader. Analyze the email below for grammar, "
            "tone, clarity, style, and wordiness issues. Return a JSON array with objects:\n"
            '[{"type": "grammar|tone|clarity|style|wordiness", '
            '"severity": "error|warning|suggestion", '
            '"original": "the problematic text", '
            '"suggestion": "the corrected text", '
            '"explanation": "brief explanation"}]\n'
            "If no issues found, return an empty array []. Return ONLY valid JSON."
        ),
        user_message=text,
        temperature=0.3,
    )
    try:
        results = json.loads(raw)
        for i, r in enumerate(results):
            r["id"] = f"pr-{i}"
        return results
    except (json.JSONDecodeError, TypeError):
        return []


def get_contact_intelligence(user_id: str, email_address: str) -> dict | None:
    """Aggregate contact info from Qdrant (content store)."""
    all_emails = scroll_all_chunk0(user_id)
    matching = [e for e in all_emails if e.get("from_email") == email_address]

    if not matching:
        return None

    name = next((e.get("from_name", "") for e in matching if e.get("from_name")), "")
    dates = [e.get("date", "") for e in matching if e.get("date")]
    last_contact = max(dates) if dates else None
    subjects = [e.get("subject", "") for e in matching if e.get("subject")]

    return {
        "email": email_address,
        "name": name,
        "total_emails": len(matching),
        "last_contact": last_contact,
        "recent_subjects": subjects[:5],
    }
