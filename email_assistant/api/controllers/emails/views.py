import base64

from django.http import HttpResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from api.middleware import jwt_required
from database.serializers.email_serializers import (
    EmailUpdateSerializer,
    EmailSnoozeSerializer,
    EmailSendSerializer,
    EmailForwardSerializer,
)
from . import services


@api_view(["POST"])
@jwt_required
def email_delete_all(request):
    result = services.delete_all_emails(request.user_id)
    return Response(result)


@api_view(["GET"])
@jwt_required
def email_stats(request):
    data = services.email_stats(request.user_id)
    return Response(data)


@api_view(["GET"])
@jwt_required
def email_list(request):
    data = services.list_emails(
        user_id=request.user_id,
        mailbox_id=request.query_params.get("mailbox_id"),
        category=request.query_params.get("category"),
        unread_only=request.query_params.get("unread_only", "").lower() == "true",
        limit=int(request.query_params.get("limit", 50)),
        offset=int(request.query_params.get("offset", 0)),
    )
    return Response(data)


@api_view(["GET"])
@jwt_required
def email_detail(request, email_id):
    email = services.get_email(request.user_id, email_id)
    if not email:
        return Response({"error": "Not found"}, status=status.HTTP_404_NOT_FOUND)
    return Response(email)


@api_view(["PATCH"])
@jwt_required
def email_update(request, email_id):
    ser = EmailUpdateSerializer(data=request.data)
    if not ser.is_valid():
        return Response(ser.errors, status=status.HTTP_400_BAD_REQUEST)
    email = services.update_email(request.user_id, email_id, ser.validated_data)
    if not email:
        return Response({"error": "Not found"}, status=status.HTTP_404_NOT_FOUND)
    return Response(email)


@api_view(["POST"])
@jwt_required
def email_snooze(request, email_id):
    ser = EmailSnoozeSerializer(data=request.data)
    if not ser.is_valid():
        return Response(ser.errors, status=status.HTTP_400_BAD_REQUEST)
    email = services.snooze_email(request.user_id, email_id, ser.validated_data["hours"])
    if not email:
        return Response({"error": "Not found"}, status=status.HTTP_404_NOT_FOUND)
    return Response(email)


@api_view(["POST"])
@jwt_required
def email_archive(request, email_id):
    ok = services.archive_email(request.user_id, email_id)
    if not ok:
        return Response({"error": "Not found"}, status=status.HTTP_404_NOT_FOUND)
    return Response({"status": "archived"})


@api_view(["POST"])
@jwt_required
def email_trash(request, email_id):
    ok = services.trash_email(request.user_id, email_id)
    if not ok:
        return Response({"error": "Not found"}, status=status.HTTP_404_NOT_FOUND)
    return Response({"status": "trashed"})


@api_view(["POST"])
@jwt_required
def email_spam(request, email_id):
    ok = services.spam_email(request.user_id, email_id)
    if not ok:
        return Response({"error": "Not found"}, status=status.HTTP_404_NOT_FOUND)
    return Response({"status": "spam"})


@api_view(["POST"])
@jwt_required
def email_send(request):
    ser = EmailSendSerializer(data=request.data)
    if not ser.is_valid():
        return Response(ser.errors, status=status.HTTP_400_BAD_REQUEST)
    try:
        result = services.send_email(request.user_id, ser.validated_data)
    except ValueError as e:
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
    return Response(result, status=status.HTTP_201_CREATED)


@api_view(["POST"])
@jwt_required
def email_reply(request, email_id):
    ser = EmailSendSerializer(data=request.data)
    if not ser.is_valid():
        return Response(ser.errors, status=status.HTTP_400_BAD_REQUEST)
    try:
        result = services.reply_email(request.user_id, email_id, ser.validated_data)
    except ValueError as e:
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
    return Response(result, status=status.HTTP_201_CREATED)


@api_view(["POST"])
@jwt_required
def email_forward(request, email_id):
    ser = EmailForwardSerializer(data=request.data)
    if not ser.is_valid():
        return Response(ser.errors, status=status.HTTP_400_BAD_REQUEST)
    try:
        result = services.forward_email(request.user_id, email_id, ser.validated_data)
    except ValueError as e:
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
    return Response(result, status=status.HTTP_201_CREATED)


@api_view(["DELETE"])
@jwt_required
def email_delete_thread_reply(request, email_id, reply_index):
    result = services.delete_thread_reply(request.user_id, email_id, int(reply_index))
    if not result:
        return Response({"error": "Not found"}, status=status.HTTP_404_NOT_FOUND)
    return Response(result)


@api_view(["DELETE"])
@jwt_required
def email_delete_sent_reply(request, email_id, reply_index):
    result = services.delete_sent_reply(request.user_id, email_id, int(reply_index))
    if not result:
        return Response({"error": "Not found"}, status=status.HTTP_404_NOT_FOUND)
    return Response(result)


@api_view(["GET"])
@jwt_required
def email_attachment_download(request, email_id, attachment_index):
    att = services.get_attachment(request.user_id, email_id, int(attachment_index))
    if not att:
        return Response({"error": "Attachment not found"}, status=status.HTTP_404_NOT_FOUND)
    data = base64.b64decode(att["data_b64"])
    response = HttpResponse(data, content_type=att["content_type"])
    response["Content-Disposition"] = f'attachment; filename="{att["filename"]}"'
    response["Content-Length"] = len(data)
    return response
