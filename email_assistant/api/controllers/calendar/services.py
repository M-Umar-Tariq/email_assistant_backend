"""User calendar meetings: CRUD, conflict detection, email-sync upsert."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone, timedelta

from bson import ObjectId

from database.db import email_metadata_col, meetings_col


def _intervals_overlap(sa, ea, sb, eb) -> bool:
    return sa < eb and sb < ea


def recompute_conflicts_for_user(user_id: str) -> None:
    col = meetings_col()
    docs = list(col.find({"user_id": user_id}))
    for d in docs:
        col.update_one({"_id": d["_id"]}, {"$set": {"conflict": False}})
    n = len(docs)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = docs[i], docs[j]
            try:
                if _intervals_overlap(a["start"], a["end"], b["start"], b["end"]):
                    col.update_one({"_id": a["_id"]}, {"$set": {"conflict": True}})
                    col.update_one({"_id": b["_id"]}, {"$set": {"conflict": True}})
            except (KeyError, TypeError):
                continue


def _norm_meeting_link(v) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    return s[:2000]


def _serialize(m: dict) -> dict:
    return {
        "id": str(m["_id"]),
        "title": m.get("title", ""),
        "start": m["start"].isoformat() if isinstance(m.get("start"), datetime) else m.get("start"),
        "end": m["end"].isoformat() if isinstance(m.get("end"), datetime) else m.get("end"),
        "location": m.get("location"),
        "meeting_link": m.get("meeting_link"),
        "attendees": m.get("attendees") or [],
        "notes": m.get("notes") or "",
        "source": m.get("source", "manual"),
        "email_id": m.get("email_id"),
        "mailbox_id": _norm_mailbox_id_val(m.get("mailbox_id")),
        "conflict": bool(m.get("conflict")),
    }


def _norm_mailbox_id_val(v) -> str | None:
    if v is None:
        return None
    return str(v)


def _mailbox_id_for_email_batch(user_id: str, email_ids: list[str]) -> dict[str, str | None]:
    """Map email_id string -> mailbox_id for metadata rows.

    email_metadata._id is typically a UUID string (IMAP sync), not ObjectId — querying
    only ObjectIds would miss every row and break per-mailbox calendar filters.
    """
    if not email_ids:
        return {}
    str_ids: list[str] = []
    seen: set[str] = set()
    for eid in email_ids:
        if not eid:
            continue
        s = str(eid).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        str_ids.append(s)

    oids: list[ObjectId] = []
    for s in str_ids:
        try:
            oids.append(ObjectId(s))
        except Exception:
            continue

    if not str_ids and not oids:
        return {}

    # Match UUID/string _id (current sync) and legacy ObjectId _id
    id_clause: dict
    if str_ids and oids:
        id_clause = {"$or": [{"_id": {"$in": str_ids}}, {"_id": {"$in": oids}}]}
    elif oids:
        id_clause = {"_id": {"$in": oids}}
    else:
        id_clause = {"_id": {"$in": str_ids}}

    out: dict[str, str | None] = {}
    for doc in email_metadata_col().find(
        {"user_id": user_id, **id_clause},
        {"mailbox_id": 1},
    ):
        mb = doc.get("mailbox_id")
        out[str(doc["_id"])] = _norm_mailbox_id_val(mb)
    return out


def list_meetings(
    user_id: str,
    start_date: str | None,
    end_date: str | None,
    mailbox_id: str | None = None,
) -> list[dict]:
    col = meetings_col()
    q: dict = {"user_id": user_id}
    if start_date and end_date:
        try:
            sd = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
            ed = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
            if sd.tzinfo is None:
                sd = sd.replace(tzinfo=timezone.utc)
            if ed.tzinfo is None:
                ed = ed.replace(tzinfo=timezone.utc)
            # overlap window: meeting intersects [sd, ed]
            q["$and"] = [{"start": {"$lt": ed}}, {"end": {"$gt": sd}}]
        except (ValueError, TypeError):
            pass
    elif start_date:
        try:
            sd = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
            if sd.tzinfo is None:
                sd = sd.replace(tzinfo=timezone.utc)
            day_end = sd + timedelta(days=1)
            q["$and"] = [{"start": {"$lt": day_end}}, {"end": {"$gt": sd}}]
        except (ValueError, TypeError):
            pass

    cur = col.find(q).sort("start", 1)
    rows = list(cur)

    mb_filter = (mailbox_id or "").strip()
    if not mb_filter or mb_filter.lower() == "all":
        return [_serialize(m) for m in rows]

    eids = [str(m.get("email_id")) for m in rows if m.get("email_id")]
    box_by_email = _mailbox_id_for_email_batch(user_id, eids)

    filtered: list[dict] = []
    for m in rows:
        if _norm_mailbox_id_val(m.get("mailbox_id")) == mb_filter:
            filtered.append(m)
            continue
        eid = m.get("email_id")
        if eid and _norm_mailbox_id_val(box_by_email.get(str(eid))) == mb_filter:
            filtered.append(m)

    return [_serialize(m) for m in filtered]


def _parse_dt(val) -> datetime | None:
    if val is None:
        return None
    if isinstance(val, datetime):
        return val if val.tzinfo else val.replace(tzinfo=timezone.utc)
    try:
        s = str(val).replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _overlapping_titles(user_id: str, start: datetime, end: datetime, exclude_id: str | None) -> list[str]:
    col = meetings_col()
    q = {
        "user_id": user_id,
        "start": {"$lt": end},
        "end": {"$gt": start},
    }
    if exclude_id:
        q["_id"] = {"$ne": exclude_id}
    titles = []
    for m in col.find(q):
        titles.append(m.get("title", "Meeting"))
    return titles


def create_meeting(user_id: str, data: dict) -> tuple[dict | None, list[str]]:
    title = (data.get("title") or "").strip() or "Meeting"
    start = _parse_dt(data.get("start"))
    end = _parse_dt(data.get("end"))
    if not start or not end or end <= start:
        return None, []
    col = meetings_col()
    mid = str(uuid.uuid4())
    overlap = _overlapping_titles(user_id, start, end, None)
    mb_raw = data.get("mailbox_id")
    mb_attach = (
        str(mb_raw).strip()
        if isinstance(mb_raw, str) and mb_raw.strip()
        else None
    )
    doc = {
        "_id": mid,
        "user_id": user_id,
        "title": title,
        "start": start,
        "end": end,
        "location": data.get("location"),
        "meeting_link": _norm_meeting_link(data.get("meeting_link")),
        "attendees": data.get("attendees") if isinstance(data.get("attendees"), list) else [],
        "notes": (data.get("notes") or "")[:4000],
        "source": "manual",
        "email_id": None,
        "mailbox_id": mb_attach,
        "conflict": False,
        "created_at": datetime.now(timezone.utc),
    }
    col.insert_one(doc)
    recompute_conflicts_for_user(user_id)
    saved = col.find_one({"_id": mid})
    return _serialize(saved) if saved else None, overlap


def update_meeting(user_id: str, meeting_id: str, data: dict) -> dict | None:
    col = meetings_col()
    existing = col.find_one({"_id": meeting_id, "user_id": user_id})
    if not existing:
        return None
    updates: dict = {}
    if "title" in data:
        updates["title"] = (data.get("title") or "").strip() or "Meeting"
    if "location" in data:
        updates["location"] = data.get("location")
    if "meeting_link" in data:
        updates["meeting_link"] = _norm_meeting_link(data.get("meeting_link"))
    if "attendees" in data and isinstance(data.get("attendees"), list):
        updates["attendees"] = data["attendees"]
    if "notes" in data:
        updates["notes"] = (data.get("notes") or "")[:4000]
    if "mailbox_id" in data:
        mb_raw = data.get("mailbox_id")
        updates["mailbox_id"] = (
            str(mb_raw).strip()
            if isinstance(mb_raw, str) and str(mb_raw).strip()
            else None
        )
    start = _parse_dt(data.get("start")) if "start" in data else existing["start"]
    end = _parse_dt(data.get("end")) if "end" in data else existing["end"]
    if "start" in data or "end" in data:
        if not start or not end or end <= start:
            return None
        updates["start"] = start
        updates["end"] = end
    if updates:
        col.update_one({"_id": meeting_id}, {"$set": updates})
    recompute_conflicts_for_user(user_id)
    saved = col.find_one({"_id": meeting_id})
    return _serialize(saved) if saved else None


def delete_meeting(user_id: str, meeting_id: str) -> bool:
    col = meetings_col()
    res = col.delete_one({"_id": meeting_id, "user_id": user_id})
    if res.deleted_count:
        recompute_conflicts_for_user(user_id)
        return True
    return False


def upsert_meeting_from_email(
    user_id: str,
    email_id: str,
    meeting_payload: dict,
    mailbox_id: str | None = None,
) -> dict | None:
    """Create or update a meeting linked to an email (AI / ICS extraction)."""
    col = meetings_col()
    title = meeting_payload.get("title") or "Meeting"
    start = meeting_payload.get("start")
    end = meeting_payload.get("end")
    if not isinstance(start, datetime) or not isinstance(end, datetime) or end <= start:
        return None
    now = datetime.now(timezone.utc)
    existing = col.find_one({"user_id": user_id, "email_id": email_id})
    base = {
        "user_id": user_id,
        "email_id": email_id,
        "title": title[:500],
        "start": start,
        "end": end,
        "location": meeting_payload.get("location"),
        "attendees": meeting_payload.get("attendees") or [],
        "notes": "",
        "source": "email",
        "conflict": False,
        "mailbox_id": mailbox_id if (mailbox_id and str(mailbox_id).strip()) else None,
    }
    if existing:
        col.update_one(
            {"_id": existing["_id"]},
            {"$set": {**{k: v for k, v in base.items() if k != "user_id"}}},
        )
        mid = existing["_id"]
    else:
        mid = str(uuid.uuid4())
        col.insert_one({**base, "_id": mid, "created_at": now})
    recompute_conflicts_for_user(user_id)
    saved = col.find_one({"_id": mid})
    return _serialize(saved) if saved else None


def get_today_meeting_stats(user_id: str) -> dict:
    """Counts and next meeting for briefing dashboard (only meetings not ended yet today)."""
    col = meetings_col()
    now = datetime.now(timezone.utc)
    sod = now.replace(hour=0, minute=0, second=0, microsecond=0)
    eod = sod + timedelta(days=1)
    q = {
        "user_id": user_id,
        "start": {"$lt": eod},
        "end": {"$gt": sod},
    }
    today = list(col.find(q).sort("start", 1))
    remaining = [m for m in today if m["end"] > now]
    conflicts = sum(1 for m in remaining if m.get("conflict"))
    next_m = _serialize(remaining[0]) if remaining else None
    return {
        "meetings_today_count": len(remaining),
        "meetings_today_conflicts": conflicts,
        "next_meeting": next_m,
    }
