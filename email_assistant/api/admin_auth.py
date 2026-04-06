"""Helpers for admin access: env allowlist and optional per-user flag."""

from django.conf import settings


def user_is_admin(user_doc: dict | None) -> bool:
    if not user_doc:
        return False
    if user_doc.get("is_admin") is True:
        return True
    email = (user_doc.get("email") or "").strip().lower()
    return email in getattr(settings, "ADMIN_EMAILS", frozenset())
