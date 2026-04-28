"""AI-powered email classification: priority, category, custom labels & meeting extraction."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone

from api.utils.llm import chat

logger = logging.getLogger(__name__)


# ── Custom user-label assignment ──────────────────────────────────────────────

def assign_labels_batch(
    emails: list[dict],
    label_rules: list[dict],
) -> list[list[str]]:
    """Assign user-defined labels to a batch of emails.

    label_rules: [{"name": "Priority", "instruction": "urgent stuff"}, ...]
    emails: [{"subject": ..., "from_name": ..., "from_email": ..., "preview": ...}, ...]
    Returns: same-length list of lists of matching label names per email.
    """
    if not emails or not label_rules:
        return [[] for _ in emails]

    label_names = [r["name"] for r in label_rules if r.get("name")]
    if not label_names:
        return [[] for _ in emails]

    batch_size = 15
    results: list[list[str]] = []
    for i in range(0, len(emails), batch_size):
        batch = emails[i : i + batch_size]
        results.extend(_assign_labels_batch(batch, label_rules, label_names))
    return results


def _assign_labels_batch(
    batch: list[dict],
    label_rules: list[dict],
    label_names: list[str],
) -> list[list[str]]:
    rules_text = "\n".join(
        f'  - "{r["name"]}": {r.get("instruction", "")}' for r in label_rules if r.get("name")
    )
    items_text = "\n".join(
        f"[{idx}] From: {e.get('from_name', '')} <{e.get('from_email', '')}> | "
        f"Subject: {e.get('subject', '')} | Preview: {e.get('preview', '')[:150]}"
        for idx, e in enumerate(batch)
    )

    system_prompt = (
        "You are an email labeling assistant.\n"
        "Goal: apply ONLY the user-defined labels listed below to each email.\n\n"
        "Available labels (name -> instruction):\n"
        f"{rules_text}\n\n"
        "Decision rules:\n"
        "- Read subject + sender + preview together before deciding.\n"
        "- A label should be applied only when the email clearly matches that label's instruction.\n"
        "- Do NOT infer weak matches from vague wording.\n"
        "- It is valid to assign multiple labels to one email when each has clear evidence.\n"
        "- If no label clearly fits, return an empty array for that email.\n"
        "- Use label names EXACTLY as provided (case-sensitive, no synonyms, no new labels).\n"
        "- Remove duplicates inside each email's label list.\n\n"
        "Output format (STRICT):\n"
        "- Return ONLY raw JSON.\n"
        "- Top-level value: array with exactly "
        f"{len(batch)} elements (same order as input emails).\n"
        "- Each element: array of strings (label names).\n"
        "- No markdown, no prose, no comments, no trailing text."
    )

    try:
        raw = chat(
            system_prompt=system_prompt,
            user_message=f"Label these {len(batch)} emails:\n\n{items_text}",
            temperature=0.1,
        )
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            valid_set = set(label_names)
            out: list[list[str]] = []
            for idx in range(len(batch)):
                if idx < len(parsed):
                    item = parsed[idx]
                    out.append(
                        [lbl for lbl in (item if isinstance(item, list) else []) if lbl in valid_set]
                    )
                else:
                    out.append([])
            return out
    except Exception as exc:
        logger.warning("AI label assignment failed: %s", exc)

    return [[] for _ in batch]


def classify_emails_batch(
    emails: list[dict],
    user_id: str = "",
) -> list[dict]:
    """Classify a batch of emails using LLM. Returns list of {priority, category} dicts.

    Each input dict should have: subject, from_name, from_email, preview.
    Returns same-length list with: {"priority": "high"|"medium"|"low", "category": str|None}
    """
    if not emails:
        return []

    user_prefs_hint = ""
    if user_id:
        try:
            from api.controllers.settings.services import get_user_preferences_prompt
            prefs = get_user_preferences_prompt(user_id)
            if prefs:
                user_prefs_hint = prefs + "\nUse these preferences to improve prioritisation.\n\n"
        except Exception:
            pass

    batch_size = 15
    results = []

    for i in range(0, len(emails), batch_size):
        batch = emails[i : i + batch_size]
        batch_results = _classify_batch(batch, user_prefs_hint)
        results.extend(batch_results)

    return results


def _classify_batch(batch: list[dict], user_prefs_hint: str = "") -> list[dict]:
    """Classify a single batch (max ~15 emails) via one LLM call."""
    items_text = "\n".join(
        f"[{idx}] From: {e.get('from_name', '')} <{e.get('from_email', '')}> | "
        f"Subject: {e.get('subject', '')} | Preview: {e.get('preview', '')[:150]}"
        for idx, e in enumerate(batch)
    )

    system_prompt = (
        "You are an email priority + category classifier.\n"
        "For each email, assign exactly one priority and one category (or null category).\n\n"
        "Priority scale:\n"
        "- high: urgent actions, hard deadlines, security/legal/financial risk, executive/client-critical items.\n"
        "- medium: standard work communication, project updates, routine requests.\n"
        "- low: newsletters, broad promotions, social notifications, low-urgency bulk messages.\n\n"
        "Category options (must be one of):\n"
        "- important\n"
        "- updates\n"
        "- promotions\n"
        "- social\n"
        "- newsletters\n"
        "- finance\n"
        "- null (if none clearly fits)\n\n"
        "Classification guidelines:\n"
        "- Prefer precision over over-classification.\n"
        "- Use sender + subject + preview jointly.\n"
        "- Promotional/marketing language usually maps to promotions or newsletters.\n"
        "- Billing/invoices/payments/cost alerts usually map to finance.\n"
        "- If signals conflict, prioritize direct urgency/actionability for priority.\n"
        "- Never output categories outside the allowed set.\n\n"
        + user_prefs_hint +
        "Output format (STRICT):\n"
        "- Return ONLY raw JSON.\n"
        "- Top-level value: array with exactly the same number of elements as input emails, in order.\n"
        "- Each element is an object with keys:\n"
        '  {"priority": "high|medium|low", "category": "important|updates|promotions|social|newsletters|finance|null"}\n'
        "- No markdown, no explanation, no extra keys."
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


# ── Meeting / calendar extraction from email ──────────────────────────────────

_ICS_BLOCK = re.compile(
    r"BEGIN:VEVENT\s*(.+?)END:VEVENT",
    re.IGNORECASE | re.DOTALL,
)
_DTSTART = re.compile(r"^DTSTART[^:]*:\s*([^\r\n]+)", re.IGNORECASE | re.MULTILINE)
_DTEND = re.compile(r"^DTEND[^:]*:\s*([^\r\n]+)", re.IGNORECASE | re.MULTILINE)
_SUMMARY = re.compile(r"^SUMMARY[^:]*:\s*([^\r\n]+)", re.IGNORECASE | re.MULTILINE)
_LOCATION = re.compile(r"^LOCATION[^:]*:\s*([^\r\n]+)", re.IGNORECASE | re.MULTILINE)
_ATTENDEE = re.compile(r"^ATTENDEE[^:]*:\s*([^\r\n]+)", re.IGNORECASE | re.MULTILINE)


def _unfold_ics_line(line: str) -> str:
    return line.replace("\r\n ", "").replace("\n ", "").strip()


def _parse_ics_datetime_value(raw: str) -> datetime | None:
    """Parse ICS date-time (basic forms). Returns UTC-aware datetime or None."""
    if not raw:
        return None
    raw = raw.strip()
    if raw.startswith(";"):
        return None
    # TZID=America/New_York:20260407T130000
    if ":" in raw and not re.match(r"^\d", raw):
        raw = raw.split(":", 1)[-1].strip()
    raw = raw.rstrip("\r\n")
    if raw.endswith("Z"):
        core = raw[:-1]
        digits = core.replace("T", "").replace("-", "")[:14]
        try:
            if len(digits) >= 14:
                return datetime.strptime(digits[:14], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
            if len(digits) >= 8:
                return datetime.strptime(digits[:8], "%Y%m%d").replace(tzinfo=timezone.utc)
        except ValueError:
            return None
        return None
    if "T" in raw:
        digits = raw.replace("T", "").replace("-", "")[:14]
        try:
            if len(digits) >= 14:
                return datetime.strptime(digits[:14], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
            if len(digits) >= 8:
                return datetime.strptime(digits[:8], "%Y%m%d").replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    if len(raw) >= 8 and raw[:8].isdigit():
        try:
            return datetime.strptime(raw[:8], "%Y%m%d").replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return None


def parse_ics_calendar_blocks(text: str) -> list[dict]:
    """
    Extract VEVENTs from raw iCalendar text. Each dict: title, start, end, location, attendees.
    Datetimes are timezone-aware UTC where possible.
    """
    if not text or "BEGIN:VEVENT" not in text.upper():
        return []
    out: list[dict] = []
    for m in _ICS_BLOCK.finditer(text):
        block = m.group(1)
        ds = _DTSTART.search(block)
        de = _DTEND.search(block)
        if not ds or not de:
            continue
        start = _parse_ics_datetime_value(_unfold_ics_line(ds.group(1)))
        end = _parse_ics_datetime_value(_unfold_ics_line(de.group(1)))
        if not start or not end:
            continue
        sm = _SUMMARY.search(block)
        title = _unfold_ics_line(sm.group(1)) if sm else "Meeting"
        title = title.replace("\\,", ",")
        loc_m = _LOCATION.search(block)
        location = _unfold_ics_line(loc_m.group(1)).replace("\\,", ",") if loc_m else None
        attendees = []
        for am in _ATTENDEE.finditer(block):
            line = _unfold_ics_line(am.group(1))
            if "mailto:" in line.lower():
                attendees.append(line.lower().split("mailto:", 1)[-1].strip())
            elif "@" in line:
                attendees.append(line.strip())
        out.append({
            "title": title[:500],
            "start": start,
            "end": end,
            "location": location,
            "attendees": attendees,
        })
    return out


def extract_calendar_snippet_from_text(body: str, html: str = "") -> str:
    """Return first VCALENDAR block found in body or html, if any."""
    for blob in (body or "", html or ""):
        if "BEGIN:VCALENDAR" in blob.upper():
            start = blob.upper().find("BEGIN:VCALENDAR")
            end = blob.upper().find("END:VCALENDAR")
            if end != -1:
                return blob[start : end + len("END:VCALENDAR")]
    return ""


def extract_meetings_batch(emails: list[dict]) -> list[dict | None]:
    """
    Detect at most one primary meeting per email. Returns same-length list of:
    None, or {"title", "start" (datetime UTC), "end" (datetime UTC), "location", "attendees"}.

    Each input dict: email_id, subject, from_name, from_email, preview, body (optional),
    body_html (optional), ics_extra (optional raw calendar string).
    """
    if not emails:
        return []

    batch_size = 15
    results: list[dict | None] = []

    for i in range(0, len(emails), batch_size):
        batch = emails[i : i + batch_size]
        results.extend(_extract_meetings_subbatch(batch))
    return results


def _normalize_meeting_dict(raw: dict | None) -> dict | None:
    if not raw or not isinstance(raw, dict):
        return None
    title = (raw.get("title") or "Meeting").strip()[:500]
    start_s = raw.get("start") or raw.get("start_time")
    end_s = raw.get("end") or raw.get("end_time")
    if not start_s or not end_s:
        return None
    try:
        start_s = str(start_s).replace("Z", "+00:00")
        end_s = str(end_s).replace("Z", "+00:00")
        start = datetime.fromisoformat(start_s)
        end = datetime.fromisoformat(end_s)
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None
    if end <= start:
        return None
    loc = raw.get("location")
    att = raw.get("attendees") or []
    if isinstance(att, str):
        att = [att]
    if not isinstance(att, list):
        att = []
    att = [str(a).strip() for a in att if a][:50]
    return {"title": title, "start": start, "end": end, "location": loc if loc else None, "attendees": att}


def _extract_meetings_subbatch(batch: list[dict]) -> list[dict | None]:
    """One LLM call for up to 15 emails; ICS in body takes precedence per email."""
    out: list[dict | None] = []
    ics_first: list[dict | None] = []
    need_llm_indices: list[int] = []
    need_llm_items: list[dict] = []

    for idx, e in enumerate(batch):
        merged = ""
        if e.get("ics_extra"):
            merged += e["ics_extra"]
        merged += extract_calendar_snippet_from_text(e.get("body") or "", e.get("body_html") or "")
        parsed = parse_ics_calendar_blocks(merged) if merged else []
        if parsed:
            ev = parsed[0]
            ics_first.append(
                {
                    "title": ev["title"],
                    "start": ev["start"],
                    "end": ev["end"],
                    "location": ev.get("location"),
                    "attendees": ev.get("attendees") or [],
                }
            )
        else:
            ics_first.append(None)
            need_llm_indices.append(idx)
            need_llm_items.append(e)

    llm_results_map: dict[int, dict | None] = {}
    if need_llm_items:
        items_text = "\n".join(
            f"[{j}] Email date (received): {it.get('email_date', '')}\n"
            f"    From: {it.get('from_name', '')} <{it.get('from_email', '')}>\n"
            f"    Subject: {it.get('subject', '')}\n"
            f"    Preview: {(it.get('preview') or '')[:400]}\n"
            f"    Body excerpt: {(it.get('body') or '')[:1200]}"
            for j, it in enumerate(need_llm_items)
        )
        system_prompt = (
            "You extract real scheduled meetings/calls/events from email text.\n"
            "Return one meeting object per email only when scheduling evidence is concrete.\n\n"
            "When to return null:\n"
            "- No clear meeting intent.\n"
            "- Missing start/end and no reliable way to infer duration.\n"
            "- Purely tentative chatter without a concrete slot.\n\n"
            "Extraction rules:\n"
            "- If email uses relative time (e.g., 'tomorrow 3pm'), resolve using the provided 'Email date'.\n"
            "- Always output timezone-aware ISO 8601 for start/end.\n"
            "- If duration is explicit (e.g., '30 minutes'), infer end from start.\n"
            "- Keep title short and specific.\n"
            "- Extract attendees only when clear email addresses exist.\n"
            "- If multiple options are listed, choose the primary/proposed slot.\n\n"
            "Output format (STRICT):\n"
            "- Return ONLY raw JSON.\n"
            "- Top-level value: array of exactly "
            f"{len(need_llm_items)} elements, in order.\n"
            "- Each element is either null or:\n"
            '  {"title": string, "start": ISO8601-with-timezone, "end": ISO8601-with-timezone, "location": string|null, "attendees": string[]}\n'
            "- No markdown, no explanation, no extra keys."
        )
        try:
            raw = chat(
                system_prompt=system_prompt,
                user_message=f"Extract meetings from these {len(need_llm_items)} emails:\n\n{items_text}",
                temperature=0.1,
            )
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            parsed = json.loads(cleaned)
            if isinstance(parsed, list) and len(parsed) == len(need_llm_items):
                for j, item in enumerate(parsed):
                    llm_results_map[need_llm_indices[j]] = _normalize_meeting_dict(item) if item else None
        except Exception as exc:
            logger.warning("AI meeting extraction failed: %s", exc)

    for idx in range(len(batch)):
        if ics_first[idx] is not None:
            out.append(ics_first[idx])
        else:
            out.append(llm_results_map.get(idx))
    return out
