from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from api.middleware import jwt_required
from . import services


@api_view(["GET", "POST"])
@jwt_required
def meeting_list_create(request):
    if request.method == "GET":
        mb = request.query_params.get("mailbox_id")
        data = services.list_meetings(
            request.user_id,
            request.query_params.get("start_date"),
            request.query_params.get("end_date"),
            mailbox_id=(mb.strip() if isinstance(mb, str) and mb.strip() else None),
        )
        return Response({"meetings": data})

    body = request.data or {}
    meeting, overlaps = services.create_meeting(request.user_id, body)
    if not meeting:
        return Response(
            {"error": "Invalid start/end times"},
            status=status.HTTP_400_BAD_REQUEST,
        )
    return Response(
        {
            "meeting": meeting,
            "overlapping_titles": overlaps,
            "has_overlap": len(overlaps) > 0,
        },
        status=status.HTTP_201_CREATED,
    )


@api_view(["PATCH", "DELETE"])
@jwt_required
def meeting_detail(request, meeting_id):
    if request.method == "PATCH":
        m = services.update_meeting(request.user_id, meeting_id, request.data or {})
        if not m:
            return Response({"error": "Not found or invalid times"}, status=status.HTTP_404_NOT_FOUND)
        return Response({"meeting": m})

    ok = services.delete_meeting(request.user_id, meeting_id)
    if not ok:
        return Response({"error": "Not found"}, status=status.HTTP_404_NOT_FOUND)
    return Response(status=status.HTTP_204_NO_CONTENT)
