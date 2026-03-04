from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from api.middleware import jwt_required
from database.serializers.mailbox_serializers import (
    MailboxCreateSerializer,
    MailboxUpdateSerializer,
)
from . import services

try:
    from qdrant_client.http.exceptions import UnexpectedResponse
except ImportError:
    UnexpectedResponse = None


@api_view(["GET", "POST"])
@jwt_required
def mailbox_list_create(request):
    if request.method == "GET":
        data = services.list_mailboxes(request.user_id)
        return Response(data)

    ser = MailboxCreateSerializer(data=request.data)
    if not ser.is_valid():
        return Response(ser.errors, status=status.HTTP_400_BAD_REQUEST)
    try:
        mb = services.create_mailbox(request.user_id, ser.validated_data)
    except ValueError as e:
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
    return Response(mb, status=status.HTTP_201_CREATED)


@api_view(["GET", "PATCH", "DELETE"])
@jwt_required
def mailbox_detail(request, mailbox_id):
    if request.method == "GET":
        mb = services.get_mailbox(request.user_id, mailbox_id)
        if not mb:
            return Response({"error": "Not found"}, status=status.HTTP_404_NOT_FOUND)
        return Response(mb)

    if request.method == "PATCH":
        ser = MailboxUpdateSerializer(data=request.data)
        if not ser.is_valid():
            return Response(ser.errors, status=status.HTTP_400_BAD_REQUEST)
        mb = services.update_mailbox(request.user_id, mailbox_id, ser.validated_data)
        if not mb:
            return Response({"error": "Not found"}, status=status.HTTP_404_NOT_FOUND)
        return Response(mb)

    ok = services.delete_mailbox(request.user_id, mailbox_id)
    if not ok:
        return Response({"error": "Not found"}, status=status.HTTP_404_NOT_FOUND)
    return Response(status=status.HTTP_204_NO_CONTENT)


@api_view(["POST"])
@jwt_required
def mailbox_stop_sync(request, mailbox_id):
    result = services.stop_sync(request.user_id, mailbox_id)
    return Response(result)


@api_view(["POST"])
@jwt_required
def mailbox_sync(request, mailbox_id):
    body = getattr(request, "data", None) or {}
    if not isinstance(body, dict):
        body = {}
    initial_sync = body.get("initial_sync")
    limit = body.get("limit")
    if limit is not None:
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = None
    try:
        result = services.sync_mailbox(
            request.user_id,
            mailbox_id,
            initial_sync=initial_sync,
            limit=limit,
        )
    except ValueError as e:
        return Response({"error": str(e)}, status=status.HTTP_404_NOT_FOUND)
    except RuntimeError as e:
        msg = str(e)
        # IMAP login/auth failure -> 400 with helpful message
        if "IMAP failed" in msg and ("LOGIN" in msg or "BAD" in msg or "AUTHENTICATIONFAILED" in msg.upper()):
            return Response(
                {"error": "Invalid email or password. For Gmail, use an App Password (Google Account → Security → App passwords). For Outlook, use your password or an app password."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        return Response({"error": msg}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    except Exception as e:
        if UnexpectedResponse and isinstance(e, UnexpectedResponse):
            return Response(
                {"error": "Vector database connection failed. Check QDRANT_URL and QDRANT_API_KEY in .env and that the API key has read/write access."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        raise
    return Response(result)
