from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from api.middleware import jwt_required
from . import services


@api_view(["GET"])
@jwt_required
def suggestions(request):
    return Response(services.get_suggestions(request.user_id))


@api_view(["POST"])
@jwt_required
def chat(request):
    message = request.data.get("message", "").strip()
    if not message:
        return Response(
            {"error": "message is required"},
            status=status.HTTP_400_BAD_REQUEST,
        )
    history = request.data.get("history", [])
    mailbox_id = request.data.get("mailbox_id") or None
    result = services.agent_chat(request.user_id, message, history or None, mailbox_id=mailbox_id)
    return Response(result)


@api_view(["GET"])
@jwt_required
def profile(request):
    return Response(services.get_user_profile(request.user_id))


@api_view(["POST"])
@jwt_required
def build_profile(request):
    return Response(services.build_user_profile(request.user_id))


@api_view(["POST"])
@jwt_required
def execute(request):
    action_data = request.data
    if not action_data or not action_data.get("type"):
        return Response(
            {"error": "action data with 'type' is required"},
            status=status.HTTP_400_BAD_REQUEST,
        )
    result = services.approve_and_execute(request.user_id, action_data)
    return Response(result)


@api_view(["POST"])
@jwt_required
def reject(request, action_id):
    return Response(services.reject_action(action_id))


@api_view(["POST"])
@jwt_required
def speak(request):
    text = request.data.get("text", "").strip()
    if not text:
        return Response(
            {"error": "text is required"},
            status=status.HTTP_400_BAD_REQUEST,
        )
    try:
        result = services.generate_speech(text)
        return Response(result)
    except Exception:
        return Response(
            {"error": "Speech generation failed. Please try again."},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
