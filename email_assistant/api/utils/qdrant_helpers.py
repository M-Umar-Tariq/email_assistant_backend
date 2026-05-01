"""
Shared helpers for reading email content from Qdrant (primary store).
MongoDB only holds mutable state (read/starred/labels/archived/trashed).
All immutable email content lives in Qdrant payloads.
"""

import json
import re

from django.conf import settings
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

from database.db import get_qdrant


def get_email_content(email_id: str, user_id: str) -> dict | None:
    """Fetch content for a single email from Qdrant (chunk_index=0)."""
    qdrant = get_qdrant()
    results = qdrant.scroll(
        collection_name=settings.QDRANT_COLLECTION_EMAIL_CHUNKS,
        scroll_filter=Filter(
            must=[
                FieldCondition(key="email_id", match=MatchValue(value=email_id)),
                FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                FieldCondition(key="chunk_index", match=MatchValue(value=0)),
            ]
        ),
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    points = results[0] if results else []
    if not points:
        return None
    return _payload_to_content(points[0].payload)


def get_emails_content_batch(email_ids: list[str], user_id: str) -> dict:
    """Fetch content for multiple emails from Qdrant. Returns {email_id: content_dict}."""
    if not email_ids:
        return {}

    qdrant = get_qdrant()
    content_map = {}
    BATCH = 100
    for i in range(0, len(email_ids), BATCH):
        batch_ids = email_ids[i : i + BATCH]
        results = qdrant.scroll(
            collection_name=settings.QDRANT_COLLECTION_EMAIL_CHUNKS,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                    FieldCondition(key="chunk_index", match=MatchValue(value=0)),
                    FieldCondition(key="email_id", match=MatchAny(any=batch_ids)),
                ],
            ),
            limit=len(batch_ids) + 10,
            with_payload=True,
            with_vectors=False,
        )
        points = results[0] if results else []
        for point in points:
            eid = point.payload.get("email_id", "")
            if eid:
                content_map[eid] = _payload_to_content(point.payload)
    return content_map


def scroll_all_chunk0(
    user_id: str,
    mailbox_id: str | None = None,
    category: str | None = None,
    priority: str | None = None,
) -> list[dict]:
    """Scroll all chunk_index=0 records for a user. Returns list of content dicts."""
    qdrant = get_qdrant()
    must_filters = [
        FieldCondition(key="user_id", match=MatchValue(value=user_id)),
        FieldCondition(key="chunk_index", match=MatchValue(value=0)),
    ]
    if mailbox_id:
        must_filters.append(FieldCondition(key="mailbox_id", match=MatchValue(value=mailbox_id)))
    if category:
        must_filters.append(FieldCondition(key="category", match=MatchValue(value=category)))
    if priority:
        must_filters.append(FieldCondition(key="priority", match=MatchValue(value=priority)))

    all_points = []
    offset = None
    while True:
        results = qdrant.scroll(
            collection_name=settings.QDRANT_COLLECTION_EMAIL_CHUNKS,
            scroll_filter=Filter(must=must_filters),
            limit=500,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points, next_offset = results
        all_points.extend(points)
        if next_offset is None:
            break
        offset = next_offset

    return [_payload_to_content(p.payload) for p in all_points]


def scroll_all_chunks(
    user_id: str,
    mailbox_id: str | None = None,
) -> list[dict]:
    """Scroll EVERY chunk for the user (chunk_index 0..N), not just chunk-0.

    Used when we need the full body of long emails (chunk-0's `body_chunk`
    only carries the first slice). Returns raw payload dicts so the caller
    can group by `email_id` and reconstruct the body in chunk_index order.
    """
    qdrant = get_qdrant()
    must_filters = [FieldCondition(key="user_id", match=MatchValue(value=user_id))]
    if mailbox_id:
        must_filters.append(FieldCondition(key="mailbox_id", match=MatchValue(value=mailbox_id)))

    all_points: list[dict] = []
    offset = None
    while True:
        results = qdrant.scroll(
            collection_name=settings.QDRANT_COLLECTION_EMAIL_CHUNKS,
            scroll_filter=Filter(must=must_filters),
            limit=500,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points, next_offset = results
        for p in points:
            all_points.append(dict(p.payload))
        if next_offset is None:
            break
        offset = next_offset
    return all_points


def get_email_qdrant_haystacks(
    user_id: str,
    mailbox_id: str | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    """Return (body_map, header_map) keyed by email_id.

    `body_map[eid]` is the full searchable text — subject + preview + sender
    + every recipient + body across ALL chunks (sorted by chunk_index) +
    body_html stripped + attachment text.

    `header_map[eid]` is just the participant header — sender + every
    recipient. `get_email_ids_by_participants` uses this to ensure at least
    one matched token actually lives on the From/To header (otherwise emails
    that merely mention two common names anywhere in the body would slip
    through).

    Built from a single Qdrant scroll so we don't pay twice.
    """
    points = scroll_all_chunks(user_id, mailbox_id=mailbox_id)
    grouped: dict[str, list[dict]] = {}
    for p in points:
        eid = p.get("email_id") or ""
        if not eid:
            continue
        grouped.setdefault(eid, []).append(p)

    body_map: dict[str, str] = {}
    header_map: dict[str, str] = {}
    for eid, plist in grouped.items():
        plist.sort(key=lambda x: x.get("chunk_index", 0))
        head = plist[0]

        body_parts: list[str] = [
            str(head.get("subject", "")),
            str(head.get("preview", "")),
            str(head.get("from_name", "")),
            str(head.get("from_email", "")),
        ]
        header_parts: list[str] = [
            str(head.get("from_name", "")),
            str(head.get("from_email", "")),
        ]

        to_raw = head.get("to", "[]")
        to_list: list = []
        if isinstance(to_raw, str):
            try:
                to_list = json.loads(to_raw)
            except (json.JSONDecodeError, TypeError):
                to_list = []
        elif isinstance(to_raw, list):
            to_list = to_raw
        for t in to_list:
            if isinstance(t, dict):
                n = str(t.get("name", ""))
                e = str(t.get("email", ""))
                body_parts.append(n)
                body_parts.append(e)
                header_parts.append(n)
                header_parts.append(e)
            elif isinstance(t, str):
                body_parts.append(t)
                header_parts.append(t)

        # Concatenate body across every chunk (the actual full body).
        for p in plist:
            chunk_text = p.get("body_chunk")
            if chunk_text:
                body_parts.append(str(chunk_text))
        body_html = head.get("body_html", "")
        if body_html:
            body_parts.append(_strip_html(body_html))
        att = head.get("attachment_text", "")
        if att:
            body_parts.append(str(att))

        body_map[eid] = " ".join(body_parts)
        header_map[eid] = " ".join(header_parts)
    return body_map, header_map


def get_email_chunk0_search_haystacks(
    user_id: str,
    mailbox_id: str | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    """Build searchable text from chunk_index=0 payloads only.

    Avoids ``scroll_all_chunks`` (every chunk per email), which is very slow at
    scale. Keyword and participant filters almost always hit subject, preview,
    addresses, or the start of the body. Thread / reply text still comes from
    Mongo via ``extra_text`` / ``header_text`` in the callers.

    Returns the same ``(body_map, header_map)`` shape as
    ``get_email_qdrant_haystacks`` for drop-in use in filter helpers.
    """
    body_map: dict[str, str] = {}
    header_map: dict[str, str] = {}
    for content in scroll_all_chunk0(user_id, mailbox_id=mailbox_id):
        eid = content.get("email_id") or ""
        if not eid:
            continue
        body_map[eid] = _full_text_haystack(content)
        header_map[eid] = _participants_text(content)
    return body_map, header_map


def get_email_qdrant_text_map(
    user_id: str,
    mailbox_id: str | None = None,
) -> dict[str, str]:
    """Backwards-compat shim — returns just the body map."""
    body_map, _ = get_email_chunk0_search_haystacks(user_id, mailbox_id=mailbox_id)
    return body_map


_ANGLE_ADDR_RE = re.compile(r"<([^>]+)>")


def _extract_addr(addr: str) -> str:
    """Return the bare email address from 'Name <addr>' or plain 'addr'."""
    m = _ANGLE_ADDR_RE.search(addr or "")
    return m.group(1).strip().lower() if m else (addr or "").strip().lower()


def get_email_ids_by_sender(
    user_id: str,
    from_email: str,
    mailbox_id: str | None = None,
) -> list[str]:
    """Return email_ids for emails from the given sender (case-insensitive).

    Handles both bare addresses and 'Name <addr>' format stored in Qdrant.
    """
    if not (from_email or "").strip():
        return []
    all_content = scroll_all_chunk0(user_id, mailbox_id=mailbox_id)
    target = _extract_addr(from_email)
    return [
        c["email_id"]
        for c in all_content
        if c.get("email_id") and _extract_addr(c.get("from_email") or "") == target
    ]


_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_REPEAT_RUN_RE = re.compile(r"(.)\1+")


def _normalize_for_match(value: str) -> str:
    cleaned = _NON_ALNUM_RE.sub("", str(value or "").lower())
    return _REPEAT_RUN_RE.sub(r"\1", cleaned)


def _strip_html(html: str) -> str:
    """Cheap HTML→text for keyword matching (avoids pulling in BeautifulSoup)."""
    if not html:
        return ""
    return re.sub(r"<[^>]+>", " ", html)


def _participants_text(content: dict) -> str:
    """Concatenate all participant identifiers (from + to names/emails) lowercased.

    Used by `get_email_ids_by_participants` for cheap substring matching of
    natural-language participant tokens like "umar" or "zubair@x.com".
    """
    parts: list[str] = []
    fn = content.get("from_name") or ""
    fe = content.get("from_email") or ""
    if fn:
        parts.append(str(fn))
    if fe:
        parts.append(str(fe))
    for t in content.get("to") or []:
        if isinstance(t, dict):
            n = t.get("name") or ""
            e = t.get("email") or ""
            if n:
                parts.append(str(n))
            if e:
                parts.append(str(e))
        elif isinstance(t, str):
            parts.append(t)
    return " ".join(parts).lower()


def _full_text_haystack(content: dict) -> str:
    """All searchable text on a single email's chunk-0 payload.

    Pulls subject, preview, both names/addresses, every recipient, and the
    body (text + cheap-stripped HTML). Used by keyword matching so a query
    for "indeed" finds emails from indeed.com, and "ebike feature" finds
    matches deep in the body — not just subject/preview.
    """
    parts: list[str] = []
    parts.append(str(content.get("subject", "")))
    parts.append(str(content.get("preview", "")))
    parts.append(str(content.get("from_name", "")))
    parts.append(str(content.get("from_email", "")))
    for t in content.get("to") or []:
        if isinstance(t, dict):
            parts.append(str(t.get("name", "")))
            parts.append(str(t.get("email", "")))
        elif isinstance(t, str):
            parts.append(t)
    parts.append(str(content.get("body_chunk", "")))
    parts.append(_strip_html(content.get("body_html", "")))
    parts.append(str(content.get("attachment_text", "")))
    return " ".join(parts)


def _all_tokens_match(text: str, tokens: list[str]) -> bool:
    """Each token must appear as a substring of the normalized text."""
    if not tokens:
        return False
    norm = _normalize_for_match(text)
    return all(t in norm for t in tokens if t)


def _any_token_matches(text: str, tokens: list[str]) -> bool:
    if not tokens:
        return False
    norm = _normalize_for_match(text)
    return any(t in norm for t in tokens if t)


def _tokens_for_phrase(phrase: str) -> list[str]:
    """Split a phrase into normalized tokens, dropping 1-char junk unless
    that's all the user gave (mirrors the keyword tokenizer's rules)."""
    tokens = [_normalize_for_match(t) for t in (phrase or "").split() if _normalize_for_match(t)]
    if any(len(t) >= 2 for t in tokens):
        tokens = [t for t in tokens if len(t) >= 2]
    return tokens


def get_email_ids_by_keywords_any(
    user_id: str,
    keyword_phrases: list[str],
    mailbox_id: str | None = None,
    extra_text: dict[str, str] | None = None,
    *,
    chunk0_haystacks: tuple[dict[str, str], dict[str, str]] | None = None,
) -> list[str]:
    """Return email_ids that match AT LEAST ONE of the given phrases.

    Each phrase's words must all appear in the haystack (within-phrase AND);
    an email matches if any single phrase fully matches (across-phrase OR).
    Used for queries like "emails about quarterly report or budget" — phrase
    list ["quarterly report", "budget"], match if either phrase hits.
    """
    if not keyword_phrases:
        return []
    phrase_tokens: list[list[str]] = []
    for ph in keyword_phrases:
        tks = _tokens_for_phrase(ph)
        if tks:
            phrase_tokens.append(tks)
    if not phrase_tokens:
        return []
    body_map, _ = chunk0_haystacks or get_email_chunk0_search_haystacks(
        user_id, mailbox_id=mailbox_id
    )
    extras = extra_text or {}
    out: list[str] = []
    for eid, text in body_map.items():
        combined = _normalize_for_match(text + " " + extras.get(eid, ""))
        if any(all(tok in combined for tok in tks) for tks in phrase_tokens):
            out.append(eid)
    return out


def get_email_ids_by_keywords(
    user_id: str,
    keywords: str,
    mailbox_id: str | None = None,
    extra_text: dict[str, str] | None = None,
    *,
    chunk0_haystacks: tuple[dict[str, str], dict[str, str]] | None = None,
) -> list[str]:
    """Return email_ids whose subject / preview / sender / recipient / first
    body slice / attachment text (chunk 0) contains all keyword tokens.

    `extra_text` lets the caller inject additional searchable text per
    email_id (e.g. thread / sent reply bodies pulled from MongoDB).

    Tokens are split on whitespace; every token must appear (substring,
    case-insensitive, punctuation-insensitive). So "ebike feature" requires
    both words to appear, possibly far apart, possibly inside hyphenated
    forms like "E-Bike Feature".
    """
    raw = (keywords or "").strip()
    if not raw:
        return []
    tokens = [_normalize_for_match(t) for t in raw.split() if _normalize_for_match(t)]
    # Drop trivially-short tokens that would match too greedily; but keep them
    # if they are the only ones (e.g. user typed "AI").
    if any(len(t) >= 2 for t in tokens):
        tokens = [t for t in tokens if len(t) >= 2]
    if not tokens:
        return []
    body_map, _ = chunk0_haystacks or get_email_chunk0_search_haystacks(
        user_id, mailbox_id=mailbox_id
    )
    extras = extra_text or {}
    out: list[str] = []
    for eid, text in body_map.items():
        combined = text + " " + extras.get(eid, "")
        if _all_tokens_match(combined, tokens):
            out.append(eid)
    return out


def get_email_ids_by_participants(
    user_id: str,
    participants: list[str],
    match: str = "all",
    mailbox_id: str | None = None,
    extra_text: dict[str, str] | None = None,
    header_text: dict[str, str] | None = None,
    *,
    chunk0_haystacks: tuple[dict[str, str], dict[str, str]] | None = None,
    seed_ids: list[str] | None = None,
) -> list[str]:
    """Return email_ids whose sender, recipient OR body mentions the participants.

    Each token is matched (case- and punctuation-insensitive) against the
    sender / every recipient / chunk-0 body text (plus stripped HTML) plus
    any `extra_text` (thread / sent reply bodies from MongoDB).

    `header_text` optionally provides a per-email_id "header haystack" — the
    sender + recipients of the original email AND of every reply — so that
    `match="all"` can require at least one token to hit the From/To headers
    (header is required to avoid false positives where two common names
    happen to appear together inside a long body).

    `match="all"` requires every token to appear; `match="any"` keeps emails
    that match at least one token.
    """
    tokens = [_normalize_for_match(t) for t in participants if _normalize_for_match(t)]
    if not tokens:
        return []
    use_all = (match or "all").lower() != "any"
    body_map, qdrant_header_map = chunk0_haystacks or get_email_chunk0_search_haystacks(
        user_id, mailbox_id=mailbox_id
    )
    extras = extra_text or {}
    extra_headers = header_text or {}

    # Header-hit gating exists to avoid false positives when ANDing multiple
    # names (e.g. "Umar AND Zubair" shouldn't match a long email that merely
    # mentions both names in passing). With a single name there's no such
    # risk — body mentions are exactly what "emails related to Qasim" means.
    require_header_hit = use_all and len(tokens) > 1
    out: set[str] = set(seed_ids or [])
    for eid, body_text in body_map.items():
        if eid in out:
            continue
        full_norm = _normalize_for_match(body_text + " " + extras.get(eid, ""))
        header_norm = _normalize_for_match(
            qdrant_header_map.get(eid, "") + " " + extra_headers.get(eid, "")
        )
        if use_all:
            if not all(tok in full_norm for tok in tokens):
                continue
            if not require_header_hit:
                out.add(eid)
            elif header_norm and any(tok in header_norm for tok in tokens):
                out.add(eid)
            elif not header_norm:
                out.add(eid)
        else:
            if any(tok in full_norm for tok in tokens):
                out.add(eid)
    return list(out)


def _payload_to_content(payload: dict) -> dict:
    """Convert a Qdrant payload into a clean content dictionary."""
    to_raw = payload.get("to", "[]")
    if isinstance(to_raw, str):
        try:
            to_list = json.loads(to_raw)
        except (json.JSONDecodeError, TypeError):
            to_list = []
    else:
        to_list = to_raw if isinstance(to_raw, list) else []

    attachments_raw = payload.get("attachments", "[]")
    if isinstance(attachments_raw, str):
        try:
            attachments_list = json.loads(attachments_raw)
        except (json.JSONDecodeError, TypeError):
            attachments_list = []
    else:
        attachments_list = attachments_raw if isinstance(attachments_raw, list) else []

    return {
        "email_id": payload.get("email_id", ""),
        "mailbox_id": payload.get("mailbox_id", ""),
        "subject": payload.get("subject", ""),
        "from_name": payload.get("from_name", ""),
        "from_email": payload.get("from_email", ""),
        "to": to_list,
        "date": payload.get("date", ""),
        "preview": payload.get("preview", ""),
        "has_attachment": payload.get("has_attachment", False),
        "priority": payload.get("priority", "medium"),
        "category": payload.get("category") or None,
        "body_chunk": payload.get("body_chunk", ""),
        "body_html": payload.get("body_html", ""),
        "total_chunks": payload.get("total_chunks", 0),
        "attachments": attachments_list,
        "attachment_text": payload.get("attachment_text", ""),
    }
