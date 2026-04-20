import base64
from datetime import datetime

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


def _parse_iso_dt(value: str | None):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


@api_view(["POST"])
@jwt_required
def email_delete_all(request):
    result = services.delete_all_emails(request.user_id)
    return Response(result)


@api_view(["POST"])
@jwt_required
def email_mark_all_read(request):
    """Body/query: optional mailbox_id to scope to one mailbox."""
    mb = request.data.get("mailbox_id") if request.data else None
    if mb is None:
        mb = request.query_params.get("mailbox_id")
    result = services.mark_all_inbox_read(request.user_id, mailbox_id=mb or None)
    return Response(result)


@api_view(["POST"])
@jwt_required
def email_mark_all_unread(request):
    """Body/query: optional mailbox_id to scope to one mailbox."""
    mb = request.data.get("mailbox_id") if request.data else None
    if mb is None:
        mb = request.query_params.get("mailbox_id")
    result = services.mark_all_inbox_unread(request.user_id, mailbox_id=mb or None)
    return Response(result)


@api_view(["GET"])
@jwt_required
def email_stats(request):
    data = services.email_stats(
        request.user_id,
        mailbox_id=request.query_params.get("mailbox_id") or None,
    )
    return Response(data)


@api_view(["GET"])
@jwt_required
def email_unique_senders(request):
    data = services.unique_senders(
        request.user_id,
        mailbox_id=request.query_params.get("mailbox_id") or None,
    )
    return Response(data)


@api_view(["GET"])
@jwt_required
def email_folder_counts(request):
    data = services.folder_counts(
        user_id=request.user_id,
        mailbox_id=request.query_params.get("mailbox_id") or None,
    )
    return Response(data)


@api_view(["GET"])
@jwt_required
def email_list(request):
    date_from = _parse_iso_dt(request.query_params.get("date_from"))
    date_to = _parse_iso_dt(request.query_params.get("date_to"))
    data = services.list_emails(
        user_id=request.user_id,
        mailbox_id=request.query_params.get("mailbox_id"),
        category=request.query_params.get("category"),
        unread_only=request.query_params.get("unread_only", "").lower() == "true",
        from_email=request.query_params.get("from_email") or None,
        subject=request.query_params.get("subject") or None,
        keywords=request.query_params.get("keywords") or None,
        date_from=date_from,
        date_to=date_to,
        label=request.query_params.get("label") or None,
        folder=request.query_params.get("folder") or None,
        limit=int(request.query_params.get("limit", 50)),
        offset=int(request.query_params.get("offset", 0)),
        inbox_preset=request.query_params.get("inbox_preset") or None,
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
def email_move_to_inbox(request, email_id):
    ok = services.move_email_to_inbox(request.user_id, email_id)
    if not ok:
        return Response({"error": "Not found"}, status=status.HTTP_404_NOT_FOUND)
    return Response({"status": "moved_to_inbox"})


@api_view(["POST"])
@jwt_required
def email_spam(request, email_id):
    ok = services.spam_email(request.user_id, email_id)
    if not ok:
        return Response({"error": "Not found"}, status=status.HTTP_404_NOT_FOUND)
    return Response({"status": "spam"})


@api_view(["POST"])
@jwt_required
def email_delete(request, email_id):
    ok = services.delete_email(request.user_id, email_id)
    if not ok:
        return Response({"error": "Not found"}, status=status.HTTP_404_NOT_FOUND)
    return Response({"status": "deleted"})


# ── Bulk endpoints ──────────────────────────────────────────────────────────
#
# One request → one service call → one Mongo batch + one IMAP session per
# mailbox. Prevents the 10s+ delays that occurred when the client fired 50
# individual per-email HTTP requests.


def _parse_bulk_ids(request) -> tuple[list[str] | None, Response | None]:
    body = request.data if isinstance(request.data, dict) else {}
    raw = body.get("email_ids")
    if not isinstance(raw, list) or not raw:
        return None, Response(
            {"error": "email_ids must be a non-empty list"},
            status=status.HTTP_400_BAD_REQUEST,
        )
    cleaned = [str(x) for x in raw if isinstance(x, (str, int)) and str(x).strip()]
    if not cleaned:
        return None, Response(
            {"error": "email_ids must be a non-empty list"},
            status=status.HTTP_400_BAD_REQUEST,
        )
    return cleaned, None


@api_view(["POST"])
@jwt_required
def email_bulk_archive(request):
    ids, err = _parse_bulk_ids(request)
    if err:
        return err
    return Response(services.bulk_archive_emails(request.user_id, ids))


@api_view(["POST"])
@jwt_required
def email_bulk_trash(request):
    ids, err = _parse_bulk_ids(request)
    if err:
        return err
    return Response(services.bulk_trash_emails(request.user_id, ids))


@api_view(["POST"])
@jwt_required
def email_bulk_spam(request):
    ids, err = _parse_bulk_ids(request)
    if err:
        return err
    return Response(services.bulk_spam_emails(request.user_id, ids))


@api_view(["POST"])
@jwt_required
def email_bulk_move_to_inbox(request):
    ids, err = _parse_bulk_ids(request)
    if err:
        return err
    return Response(services.bulk_move_to_inbox_emails(request.user_id, ids))


@api_view(["POST"])
@jwt_required
def email_bulk_delete(request):
    ids, err = _parse_bulk_ids(request)
    if err:
        return err
    return Response(services.bulk_delete_emails(request.user_id, ids))


@api_view(["POST"])
@jwt_required
def email_bulk_update(request):
    ids, err = _parse_bulk_ids(request)
    if err:
        return err
    body = request.data if isinstance(request.data, dict) else {}
    data: dict = {}
    if "read" in body:
        data["read"] = bool(body["read"])
    if "starred" in body:
        data["starred"] = bool(body["starred"])
    if not data:
        return Response(
            {"error": "Provide at least one of: read, starred"},
            status=status.HTTP_400_BAD_REQUEST,
        )
    return Response(services.bulk_update_emails(request.user_id, ids, data))


@api_view(["POST"])
@jwt_required
def email_bulk_snooze(request):
    ids, err = _parse_bulk_ids(request)
    if err:
        return err
    body = request.data if isinstance(request.data, dict) else {}
    try:
        hours = int(body.get("hours"))
    except (TypeError, ValueError):
        return Response(
            {"error": "hours must be an integer"},
            status=status.HTTP_400_BAD_REQUEST,
        )
    if hours <= 0:
        return Response(
            {"error": "hours must be a positive integer"},
            status=status.HTTP_400_BAD_REQUEST,
        )
    return Response(services.bulk_snooze_emails(request.user_id, ids, hours))


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
def email_with_meetings(request):
    """Emails whose AI-detected meeting the user may add to the calendar.

    Query params: `mailbox_id` (optional), `status` (pending|added|dismissed).
    """
    mb = request.query_params.get("mailbox_id") or None
    status_f = request.query_params.get("status") or None
    data = services.list_emails_with_meetings(
        request.user_id, mailbox_id=mb, status_filter=status_f
    )
    return Response(data)


@api_view(["POST"])
@jwt_required
def email_add_meeting(request, email_id):
    """Promote an email's detected meeting into a real calendar event."""
    result = services.add_detected_meeting_to_calendar(request.user_id, email_id)
    if not result:
        return Response(
            {"error": "No detected meeting on this email"},
            status=status.HTTP_404_NOT_FOUND,
        )
    return Response(result, status=status.HTTP_201_CREATED)


@api_view(["POST"])
@jwt_required
def email_dismiss_meeting(request, email_id):
    """Dismiss an email's detected meeting — removes the approval banner."""
    email = services.dismiss_detected_meeting(request.user_id, email_id)
    if not email:
        return Response({"error": "Not found"}, status=status.HTTP_404_NOT_FOUND)
    return Response(email)


@api_view(["GET"])
@jwt_required
def email_attachment_download(request, email_id, attachment_index):
    att = services.get_attachment(request.user_id, email_id, int(attachment_index))
    if not att:
        reason = services.diagnose_missing_attachment(request.user_id, email_id, int(attachment_index))
        print(f"[DOWNLOAD] Attachment not found: email_id={email_id} index={attachment_index} reason={reason}")
        return Response({"error": "Attachment not found", "detail": reason}, status=status.HTTP_404_NOT_FOUND)
    data = base64.b64decode(att["data_b64"])
    response = HttpResponse(data, content_type=att["content_type"])
    response["Content-Disposition"] = f'attachment; filename="{att["filename"]}"'
    response["Content-Length"] = len(data)
    return response
