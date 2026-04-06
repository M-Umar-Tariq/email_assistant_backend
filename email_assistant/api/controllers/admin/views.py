from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from api.middleware import admin_required
from . import services


@api_view(["GET"])
@admin_required
def stats(request):
    return Response(services.get_stats())


@api_view(["GET"])
@admin_required
def recent_activity(request):
    try:
        limit = int(request.query_params.get("limit") or 30)
    except ValueError:
        limit = 30
    return Response(services.get_recent_activity(min(limit, 100)))


@api_view(["GET"])
@admin_required
def feedback_list(request):
    q = request.query_params.get("q")
    category = request.query_params.get("category")
    try:
        page = int(request.query_params.get("page") or 1)
    except ValueError:
        page = 1
    try:
        limit = int(request.query_params.get("limit") or 20)
    except ValueError:
        limit = 20
    return Response(services.list_feedback(q, category, page, limit))


@api_view(["GET"])
@admin_required
def mailboxes_list(request):
    q = request.query_params.get("q")
    sync_status = request.query_params.get("sync_status")
    try:
        page = int(request.query_params.get("page") or 1)
    except ValueError:
        page = 1
    try:
        limit = int(request.query_params.get("limit") or 20)
    except ValueError:
        limit = 20
    return Response(services.list_all_mailboxes(q, sync_status, page, limit))


@api_view(["GET"])
@admin_required
def users_list(request):
    q = request.query_params.get("q")
    try:
        page = int(request.query_params.get("page") or 1)
    except ValueError:
        page = 1
    try:
        limit = int(request.query_params.get("limit") or 20)
    except ValueError:
        limit = 20
    return Response(services.list_users(q, page, limit))


@api_view(["GET", "PATCH", "DELETE"])
@admin_required
def user_detail(request, user_id: str):
    if request.method == "GET":
        data = services.get_user_detail(user_id)
        if not data:
            return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)
        return Response(data)

    if request.method == "DELETE":
        ok = services.delete_user(user_id)
        if not ok:
            return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)
        return Response({"deleted": True})

    disabled = request.data.get("disabled")
    is_admin = request.data.get("is_admin")
    if disabled is None and is_admin is None:
        return Response(
            {"error": "Provide `disabled` and/or `is_admin`"},
            status=status.HTTP_400_BAD_REQUEST,
        )
    if disabled is not None and not isinstance(disabled, bool):
        return Response({"error": "`disabled` must be boolean"}, status=status.HTTP_400_BAD_REQUEST)
    if is_admin is not None and not isinstance(is_admin, bool):
        return Response({"error": "`is_admin` must be boolean"}, status=status.HTTP_400_BAD_REQUEST)

    updated = services.patch_user(user_id, disabled, is_admin)
    if not updated:
        return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)
    return Response({"user": updated})
