import json

from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status

from api.middleware import jwt_required, decode_token
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


@csrf_exempt
def chat_stream(request):
    """Streaming chat endpoint — yields SSE events (token chunks then done)."""
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return JsonResponse({"error": "Unauthorized"}, status=401)
    try:
        payload = decode_token(auth_header.split(" ", 1)[1])
        if payload.get("type") != "access":
            return JsonResponse({"error": "Invalid token"}, status=401)
        user_id = payload["user_id"]
    except Exception:
        return JsonResponse({"error": "Invalid token"}, status=401)

    try:
        data = json.loads(request.body)
    except Exception:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    message = data.get("message", "").strip()
    if not message:
        return JsonResponse({"error": "message is required"}, status=400)

    history = data.get("history", [])
    mailbox_id = data.get("mailbox_id") or None

    def event_stream():
        try:
            for event in services.agent_chat_stream(user_id, message, history or None, mailbox_id):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Internal error'})}\n\n"
        yield "data: [DONE]\n\n"

    response = StreamingHttpResponse(event_stream(), content_type="text/event-stream")
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    return response


@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
@jwt_required
def transcribe(request):
    audio_file = request.FILES.get("audio")
    if not audio_file:
        return Response(
            {"error": "audio file is required"},
            status=status.HTTP_400_BAD_REQUEST,
        )
    try:
        result = services.transcribe_audio(audio_file)
        return Response(result)
    except Exception:
        return Response(
            {"error": "Transcription failed. Please try again."},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
