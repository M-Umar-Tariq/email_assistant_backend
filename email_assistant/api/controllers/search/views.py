from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from api.middleware import jwt_required
from . import services


@api_view(["GET"])
@jwt_required
def search(request):
    query = request.query_params.get("q", "").strip()
    if not query:
        return Response({"error": "q param is required"}, status=status.HTTP_400_BAD_REQUEST)

    results = services.search_emails(
        user_id=request.user_id,
        query=query,
        mailbox_id=request.query_params.get("mailbox_id"),
        limit=int(request.query_params.get("limit", 20)),
    )
    return Response(results)
