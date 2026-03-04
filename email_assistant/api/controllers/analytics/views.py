from rest_framework.decorators import api_view
from rest_framework.response import Response

from api.middleware import jwt_required
from . import services


@api_view(["GET"])
@jwt_required
def overview(request):
    days = int(request.query_params.get("days", 7))
    return Response(services.get_overview(request.user_id, days))


@api_view(["GET"])
@jwt_required
def volume(request):
    days = int(request.query_params.get("days", 7))
    return Response(services.get_volume(request.user_id, days))


@api_view(["GET"])
@jwt_required
def top_senders(request):
    limit = int(request.query_params.get("limit", 10))
    return Response(services.get_top_senders(request.user_id, limit))


@api_view(["GET"])
@jwt_required
def categories(request):
    return Response(services.get_categories(request.user_id))


@api_view(["GET"])
@jwt_required
def metrics(request):
    return Response(services.get_metrics(request.user_id))
