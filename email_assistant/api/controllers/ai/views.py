from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from api.middleware import jwt_required
from . import services


@api_view(["POST"])
@jwt_required
def ask(request):
    query = request.data.get("query", "").strip()
    if not query:
        return Response({"error": "query is required"}, status=status.HTTP_400_BAD_REQUEST)

    mailbox_id = request.data.get("mailbox_id") or None
    history = request.data.get("history") or None
    user_tz = request.data.get("timezone") or None
    result = services.ask(request.user_id, query, mailbox_id=mailbox_id, history=history, user_tz=user_tz)
    return Response(result)


@api_view(["POST"])
@jwt_required
def ask_about_email(request, email_id):
    query = request.data.get("query", "").strip()
    if not query:
        return Response({"error": "query is required"}, status=status.HTTP_400_BAD_REQUEST)
    result = services.ask_about_email(request.user_id, email_id, query)
    return Response(result)


@api_view(["GET"])
@jwt_required
def instant_replies(request, email_id):
    replies = services.get_instant_replies(request.user_id, email_id)
    return Response(replies)


@api_view(["GET"])
def suggested_questions(request):
    return Response([
        "What did the supplier promise about delivery dates?",
        "Summarize all emails from this month",
        "What are my open action items?",
        "Who is waiting on my response?",
        "Show me all contract-related emails",
        "What are the key risks flagged today?",
    ])
