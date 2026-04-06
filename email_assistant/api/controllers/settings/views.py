from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from api.middleware import jwt_required
from database.serializers.settings_serializers import SettingsUpdateSerializer
from . import services


@api_view(["GET", "PATCH"])
@jwt_required
def settings_view(request):
    if request.method == "GET":
        data = services.get_settings(request.user_id)
        return Response(data)

    ser = SettingsUpdateSerializer(data=request.data)
    if not ser.is_valid():
        return Response(ser.errors, status=status.HTTP_400_BAD_REQUEST)
    data = services.update_settings(request.user_id, ser.validated_data)
    return Response(data)


@api_view(["POST"])
@jwt_required
def relabel_emails(request):
    """Re-classify all emails with the user's current label rules + priority."""
    result = services.relabel_all_emails(request.user_id)
    return Response(result)
