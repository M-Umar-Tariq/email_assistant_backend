from collections import Counter
from datetime import datetime, timedelta, timezone

from database.db import email_metadata_col, user_settings_col
from api.utils.qdrant_helpers import scroll_all_chunk0


def get_overview(user_id: str, days: int = 7) -> dict:
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days)
    prev_start = start - timedelta(days=days)

    current = {"user_id": user_id, "date": {"$gte": start}}
    previous = {"user_id": user_id, "date": {"$gte": prev_start, "$lt": start}}

    received_now = email_metadata_col().count_documents(current)
    received_prev = email_metadata_col().count_documents(previous)

    return {
        "total_received": received_now,
        "received_change": _pct_change(received_prev, received_now),
        "period_days": days,
    }


def get_volume(user_id: str, days: int = 7) -> list[dict]:
    now = datetime.now(timezone.utc)
    result = []
    for i in range(days - 1, -1, -1):
        day_start = (now - timedelta(days=i)).replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)
        count = email_metadata_col().count_documents({
            "user_id": user_id,
            "date": {"$gte": day_start, "$lt": day_end},
        })
        result.append({
            "date": day_start.strftime("%a"),
            "received": count,
        })
    return result


def get_top_senders(user_id: str, limit: int = 10) -> list[dict]:
    """Aggregate top senders from Qdrant (content store)."""
    all_emails = scroll_all_chunk0(user_id)
    counter: Counter = Counter()
    name_map: dict[str, str] = {}
    for email in all_emails:
        addr = email.get("from_email", "")
        if addr:
            counter[addr] += 1
            if not name_map.get(addr):
                name_map[addr] = email.get("from_name", "")

    return [
        {"email": addr, "name": name_map.get(addr, ""), "count": count}
        for addr, count in counter.most_common(limit)
    ]


def get_categories(user_id: str, days: int = 7) -> list[dict]:
    """Aggregate user label usage from MongoDB email metadata in the last ``days`` days.

    Only labels that appear in the user's ``ai_label_rules`` (by name, case-insensitive)
    are counted. Emails may have multiple labels; each label on an email increments
    that label's count once (slices sum to 100% as share of total label assignments).
    """
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days)

    settings = user_settings_col().find_one({"user_id": user_id}) or {}
    rules = settings.get("ai_label_rules") or []
    name_by_lower: dict[str, str] = {}
    for r in rules:
        if not isinstance(r, dict):
            continue
        name = (r.get("name") or "").strip()
        if name:
            name_by_lower[name.lower()] = name

    if not name_by_lower:
        return []

    counter: Counter = Counter()
    for doc in email_metadata_col().find(
        {"user_id": user_id, "date": {"$gte": start}},
        {"labels": 1},
    ):
        labels = doc.get("labels") or []
        if not isinstance(labels, list):
            continue
        for lab in labels:
            s = str(lab).strip()
            if not s:
                continue
            canonical = name_by_lower.get(s.lower())
            if canonical:
                counter[canonical] += 1

    return [
        {"name": label_name, "value": count}
        for label_name, count in counter.most_common()
    ]


def get_metrics(user_id: str, days: int = 7) -> dict:
    now = datetime.now(timezone.utc)
    period_start = now - timedelta(days=days)
    prev_start = period_start - timedelta(days=days)

    total = email_metadata_col().count_documents({"user_id": user_id})
    unread = email_metadata_col().count_documents({"user_id": user_id, "read": False})

    current_received = email_metadata_col().count_documents(
        {"user_id": user_id, "date": {"$gte": period_start}}
    )
    prev_received = email_metadata_col().count_documents(
        {"user_id": user_id, "date": {"$gte": prev_start, "$lt": period_start}}
    )

    current_unread = email_metadata_col().count_documents(
        {"user_id": user_id, "read": False, "date": {"$gte": period_start}}
    )
    prev_unread = email_metadata_col().count_documents(
        {"user_id": user_id, "read": False, "date": {"$gte": prev_start, "$lt": period_start}}
    )

    all_emails = scroll_all_chunk0(user_id)
    senders_current: set[str] = set()
    senders_prev: set[str] = set()
    all_senders: set[str] = set()
    for e in all_emails:
        addr = e.get("from_email", "")
        if not addr:
            continue
        all_senders.add(addr)
        date_str = e.get("date", "")
        if not date_str:
            continue
        try:
            d = datetime.fromisoformat(str(date_str).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            continue
        if d >= period_start:
            senders_current.add(addr)
        elif d >= prev_start:
            senders_prev.add(addr)

    active_contacts = len(senders_current) if senders_current else len(all_senders)

    return {
        "total_emails": total,
        "unread": unread,
        "active_contacts": active_contacts,
        "total_emails_change": _pct_change(prev_received, current_received),
        "unread_change": _pct_change(prev_unread, current_unread),
        "active_contacts_change": _pct_change(len(senders_prev), len(senders_current)),
    }


def _pct_change(old: int, new: int) -> str:
    if old == 0:
        return "+100%" if new > 0 else "0%"
    change = ((new - old) / old) * 100
    sign = "+" if change >= 0 else ""
    return f"{sign}{change:.0f}%"
