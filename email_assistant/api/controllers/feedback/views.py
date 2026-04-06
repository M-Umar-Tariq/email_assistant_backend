from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from api.middleware import jwt_required
from . import services


@api_view(["POST"])
@jwt_required
def feedback_submit(request):
    body = request.data or {}
    message = body.get("message", "")
    category = body.get("category")
    result = services.submit_feedback(request.user_id, message, category)
    if not result:
        return Response(
            {"error": "Message is required (max 8000 characters)."},
            status=status.HTTP_400_BAD_REQUEST,
        )
    return Response(result, status=status.HTTP_201_CREATED)
