from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from api.middleware import jwt_required
from . import services


@api_view(["POST"])
@jwt_required
def generate(request):
    result = services.generate_draft(
        to=request.data.get("to", ""),
        subject=request.data.get("subject", ""),
        context=request.data.get("context", ""),
        tone=request.data.get("tone", "formal"),
        sender_name=request.data.get("sender_name", ""),
        user_id=getattr(request, "user_id", ""),
    )
    return Response({"draft": result})


@api_view(["POST"])
@jwt_required
def rewrite(request):
    text = request.data.get("text", "").strip()
    if not text:
        return Response({"error": "text is required"}, status=status.HTTP_400_BAD_REQUEST)
    result = services.rewrite(
        text=text,
        action=request.data.get("action", "polish"),
        target_language=request.data.get("target_language"),
    )
    return Response({"rewritten": result})


@api_view(["POST"])
@jwt_required
def proofread(request):
    text = request.data.get("text", "").strip()
    if not text:
        return Response({"error": "text is required"}, status=status.HTTP_400_BAD_REQUEST)
    results = services.proofread(text)
    return Response(results)


@api_view(["GET"])
@jwt_required
def contact_intelligence(request):
    email_addr = request.query_params.get("email", "").strip()
    if not email_addr:
        return Response({"error": "email param is required"}, status=status.HTTP_400_BAD_REQUEST)
    data = services.get_contact_intelligence(request.user_id, email_addr)
    if not data:
        return Response({"error": "Contact not found"}, status=status.HTTP_404_NOT_FOUND)
    return Response(data)
