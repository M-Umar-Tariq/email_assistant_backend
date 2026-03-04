from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from api.middleware import jwt_required
from database.serializers.follow_up_serializers import (
    FollowUpCreateSerializer,
    FollowUpUpdateSerializer,
)
from . import services


@api_view(["GET", "POST"])
@jwt_required
def follow_up_list_create(request):
    if request.method == "GET":
        data = services.list_follow_ups(
            request.user_id,
            request.query_params.get("status"),
        )
        return Response(data)

    ser = FollowUpCreateSerializer(data=request.data)
    if not ser.is_valid():
        return Response(ser.errors, status=status.HTTP_400_BAD_REQUEST)
    fu = services.create_follow_up(request.user_id, ser.validated_data)
    return Response(fu, status=status.HTTP_201_CREATED)


@api_view(["PATCH", "DELETE"])
@jwt_required
def follow_up_detail(request, follow_up_id):
    if request.method == "PATCH":
        ser = FollowUpUpdateSerializer(data=request.data)
        if not ser.is_valid():
            return Response(ser.errors, status=status.HTTP_400_BAD_REQUEST)
        fu = services.update_follow_up(request.user_id, follow_up_id, ser.validated_data)
        if not fu:
            return Response({"error": "Not found"}, status=status.HTTP_404_NOT_FOUND)
        return Response(fu)

    ok = services.delete_follow_up(request.user_id, follow_up_id)
    if not ok:
        return Response({"error": "Not found"}, status=status.HTTP_404_NOT_FOUND)
    return Response(status=status.HTTP_204_NO_CONTENT)


@api_view(["POST"])
@jwt_required
def follow_up_complete(request, follow_up_id):
    fu = services.complete_follow_up(request.user_id, follow_up_id)
    if not fu:
        return Response({"error": "Not found"}, status=status.HTTP_404_NOT_FOUND)
    return Response(fu)


@api_view(["POST"])
@jwt_required
def follow_up_auto_today(request):
    result = services.auto_detect_today_follow_ups(request.user_id)
    return Response(result)
