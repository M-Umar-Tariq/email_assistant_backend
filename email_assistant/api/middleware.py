"""
JWT authentication helpers.
Provides `@jwt_required` decorator for function-based DRF views.
"""

import jwt
from datetime import datetime, timedelta, timezone
from functools import wraps

from django.conf import settings
from rest_framework.response import Response
from rest_framework import status

from database.db import users_col, refresh_tokens_col
from bson import ObjectId

from api.admin_auth import user_is_admin


# ── Token helpers ────────────────────────────────────────────────────────────

def create_access_token(user_id: str) -> str:
    payload = {
        "user_id": user_id,
        "type": "access",
        "exp": datetime.now(timezone.utc) + settings.JWT_ACCESS_LIFETIME,
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm="HS256")


def create_refresh_token(user_id: str) -> str:
    payload = {
        "user_id": user_id,
        "type": "refresh",
        "exp": datetime.now(timezone.utc) + settings.JWT_REFRESH_LIFETIME,
        "iat": datetime.now(timezone.utc),
    }
    token = jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm="HS256")
    refresh_tokens_col().insert_one({
        "token": token,
        "user_id": user_id,
        "expires_at": datetime.now(timezone.utc) + settings.JWT_REFRESH_LIFETIME,
    })
    return token


def decode_token(token: str) -> dict:
    return jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=["HS256"])


def verify_refresh_token(token: str) -> bool:
    """Check if the refresh token exists in the database (not blacklisted)."""
    return refresh_tokens_col().find_one({"token": token}) is not None


def blacklist_refresh_token(token: str):
    refresh_tokens_col().delete_one({"token": token})


# ── Decorator ────────────────────────────────────────────────────────────────

def jwt_required(view_func):
    """Decorator that validates the JWT access token and injects `request.user_id`."""
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return Response(
                {"error": "Authorization header missing or malformed"},
                status=status.HTTP_401_UNAUTHORIZED,
            )
        token = auth_header.split(" ", 1)[1]
        try:
            payload = decode_token(token)
        except jwt.ExpiredSignatureError:
            return Response({"error": "Token expired"}, status=status.HTTP_401_UNAUTHORIZED)
        except jwt.InvalidTokenError:
            return Response({"error": "Invalid token"}, status=status.HTTP_401_UNAUTHORIZED)

        if payload.get("type") != "access":
            return Response({"error": "Invalid token type"}, status=status.HTTP_401_UNAUTHORIZED)

        request.user_id = payload["user_id"]
        return view_func(request, *args, **kwargs)
    return wrapper


def admin_required(view_func):
    """JWT access token + admin allowlist or `is_admin` on user document."""

    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return Response(
                {"error": "Authorization header missing or malformed"},
                status=status.HTTP_401_UNAUTHORIZED,
            )
        token = auth_header.split(" ", 1)[1]
        try:
            payload = decode_token(token)
        except jwt.ExpiredSignatureError:
            return Response({"error": "Token expired"}, status=status.HTTP_401_UNAUTHORIZED)
        except jwt.InvalidTokenError:
            return Response({"error": "Invalid token"}, status=status.HTTP_401_UNAUTHORIZED)

        if payload.get("type") != "access":
            return Response({"error": "Invalid token type"}, status=status.HTTP_401_UNAUTHORIZED)

        request.user_id = payload["user_id"]
        try:
            user = users_col().find_one({"_id": ObjectId(request.user_id)})
        except Exception:
            user = None
        if not user_is_admin(user):
            return Response({"error": "Admin access required"}, status=status.HTTP_403_FORBIDDEN)
        return view_func(request, *args, **kwargs)

    return wrapper
