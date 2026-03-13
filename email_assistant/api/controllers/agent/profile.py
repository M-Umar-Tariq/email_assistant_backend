"""
Personality profile builder — analyses user's real emails to learn
communication style, habits, key contacts, and behavioural patterns.
"""

import json
from datetime import datetime, timezone, timedelta

from database.db import email_metadata_col, agent_profiles_col
from api.utils.qdrant_helpers import scroll_all_chunk0
from api.utils.llm import chat

PROFILE_TTL_HOURS = 24


# ── Public API ────────────────────────────────────────────────────────────────

def get_profile(user_id: str, force_rebuild: bool = False) -> dict:
    if not force_rebuild:
        cached = agent_profiles_col().find_one({"user_id": user_id})
        if cached:
            expires = cached.get("expires_at")
            if expires and (
                (expires.tzinfo and expires > datetime.now(timezone.utc))
                or (not expires.tzinfo and expires > datetime.utcnow())
            ):
                return _clean_for_response(cached)
    return build_profile(user_id)


def build_profile(user_id: str) -> dict:
    meta_stats = _analyse_metadata(user_id)
    sample_emails = _get_email_samples(user_id, limit=60)

    if not sample_emails and meta_stats["total_emails"] == 0:
        profile = _empty_profile(user_id)
        _save_profile(profile)
        return _clean_for_response(profile)

    profile_data = _llm_analyse(meta_stats, sample_emails)
    profile = {
        "user_id": user_id,
        "email_count_analyzed": meta_stats["total_emails"],
        "built_at": datetime.now(timezone.utc),
        "expires_at": datetime.now(timezone.utc) + timedelta(hours=PROFILE_TTL_HOURS),
        **profile_data,
    }
    _save_profile(profile)
    return _clean_for_response(profile)


# ── Metadata statistics ───────────────────────────────────────────────────────

def _analyse_metadata(user_id: str) -> dict:
    col = email_metadata_col()
    total = col.count_documents({"user_id": user_id})
    unread = col.count_documents({"user_id": user_id, "read": False})
    starred = col.count_documents({"user_id": user_id, "starred": True})

    # Sender info lives in Qdrant, not MongoDB — aggregate from there
    from collections import Counter
    all_emails = scroll_all_chunk0(user_id)
    sender_counter: Counter = Counter()
    sender_names: dict[str, str] = {}
    for e in all_emails:
        addr = e.get("from_email", "")
        if addr:
            sender_counter[addr] += 1
            if not sender_names.get(addr):
                sender_names[addr] = e.get("from_name", "")
    top_senders = [
        {"email": addr, "name": sender_names.get(addr, ""), "count": count}
        for addr, count in sender_counter.most_common(15)
    ]

    by_hour = list(col.aggregate([
        {"$match": {"user_id": user_id, "date": {"$ne": None}}},
        {"$project": {"hour": {"$hour": "$date"}}},
        {"$group": {"_id": "$hour", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
    ]))

    by_dow = list(col.aggregate([
        {"$match": {"user_id": user_id, "date": {"$ne": None}}},
        {"$project": {"dow": {"$dayOfWeek": "$date"}}},
        {"$group": {"_id": "$dow", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
    ]))

    return {
        "total_emails": total,
        "unread": unread,
        "starred": starred,
        "top_senders": top_senders,
        "activity_by_hour": {str(h["_id"]): h["count"] for h in by_hour},
        "activity_by_day": {str(d["_id"]): d["count"] for d in by_dow},
    }


# ── Email sampling ────────────────────────────────────────────────────────────

def _get_email_samples(user_id: str, limit: int = 60) -> list[dict]:
    all_emails = scroll_all_chunk0(user_id)
    all_emails.sort(key=lambda e: e.get("date", ""), reverse=True)
    return all_emails[:limit]


# ── LLM analysis ─────────────────────────────────────────────────────────────

def _llm_analyse(meta_stats: dict, sample_emails: list[dict]) -> dict:
    summaries = []
    for e in sample_emails[:40]:
        to_str = ", ".join(
            (t.get("email", "") if isinstance(t, dict) else str(t))
            for t in (e.get("to") or [])
        )
        body_preview = (e.get("body_chunk", "") or e.get("preview", ""))[:300]
        summaries.append(
            f"From: {e.get('from_name','')} <{e.get('from_email','')}> → To: {to_str}\n"
            f"Subject: {e.get('subject','')}\n"
            f"Date: {e.get('date','')}\n"
            f"Priority: {e.get('priority','medium')}\n"
            f"Body: {body_preview}"
        )

    emails_text = "\n---\n".join(summaries)
    senders_text = "\n".join(
        f"- {s['name']} <{s['email']}>: {s['count']} emails"
        for s in meta_stats.get("top_senders", [])[:10]
    )

    prompt = (
        f"Analyze these emails and statistics to build a user personality profile.\n\n"
        f"STATISTICS:\n- Total emails: {meta_stats['total_emails']}\n"
        f"- Unread: {meta_stats['unread']}\n- Starred: {meta_stats['starred']}\n\n"
        f"TOP CONTACTS:\n{senders_text}\n\n"
        f"SAMPLE EMAILS ({len(sample_emails)}):\n{emails_text}\n\n"
        "Return a JSON object with this structure:\n"
        '{"communication_style":{"tone":"...","formality":"formal|semi-formal|casual",'
        '"avg_length":"short|medium|long","greeting_pattern":"...","sign_off_pattern":"..."},'
        '"key_contacts":[{"name":"...","email":"...","relationship":"colleague|client|manager|friend|vendor",'
        '"interaction_frequency":"daily|weekly|monthly","primary_topics":["..."]}],'
        '"topics_and_interests":["..."],'
        '"work_patterns":{"peak_hours":"...","communication_style":"...","priorities":"..."},'
        '"personality_traits":["..."],'
        '"response_preferences":{"urgency_handling":"...","delegation_style":"...","follow_up_pattern":"..."}}\n\n'
        "Return ONLY valid JSON."
    )

    try:
        raw = chat(
            system_prompt=(
                "You are an expert behavioural analyst. Analyze email patterns to "
                "build accurate personality profiles. Return only valid JSON."
            ),
            user_message=prompt,
            temperature=0.4,
            max_tokens=2048,
        )
        raw = raw.strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        return json.loads(raw.strip())
    except (json.JSONDecodeError, TypeError):
        return _empty_profile_data()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _empty_profile(user_id: str) -> dict:
    return {
        "user_id": user_id,
        "email_count_analyzed": 0,
        "built_at": datetime.now(timezone.utc),
        "expires_at": datetime.now(timezone.utc) + timedelta(hours=1),
        **_empty_profile_data(),
    }


def _empty_profile_data() -> dict:
    return {
        "communication_style": {
            "tone": "Not enough data yet",
            "formality": "unknown",
            "avg_length": "unknown",
            "greeting_pattern": "",
            "sign_off_pattern": "",
        },
        "key_contacts": [],
        "topics_and_interests": [],
        "work_patterns": {
            "peak_hours": "Not enough data",
            "communication_style": "Not enough data",
            "priorities": "Not enough data",
        },
        "personality_traits": [],
        "response_preferences": {
            "urgency_handling": "Not enough data",
            "delegation_style": "Not enough data",
            "follow_up_pattern": "Not enough data",
        },
    }


def _save_profile(profile: dict):
    agent_profiles_col().update_one(
        {"user_id": profile["user_id"]},
        {"$set": profile},
        upsert=True,
    )


def _clean_for_response(profile: dict) -> dict:
    """Remove MongoDB internals and ensure serialisable output."""
    out = {k: v for k, v in profile.items() if k not in ("_id", "expires_at")}
    if "_id" in profile:
        out["id"] = str(profile["_id"])
    return out
