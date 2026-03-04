from rest_framework.decorators import api_view
from rest_framework.response import Response

from api.middleware import jwt_required
from . import services


@api_view(["GET"])
@jwt_required
def briefing(request):
    data = services.get_briefing(request.user_id)
    return Response(data)


@api_view(["GET"])
@jwt_required
def ai_briefing(request):
    text = services.generate_ai_briefing(request.user_id)
    return Response({"briefing": text})
