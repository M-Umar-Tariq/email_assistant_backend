from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from database.serializers.user_serializers import (
    RegisterSerializer,
    LoginSerializer,
    ProfileUpdateSerializer,
)
from api.middleware import (
    create_access_token,
    create_refresh_token,
    decode_token,
    verify_refresh_token,
    blacklist_refresh_token,
    jwt_required,
)
from api.controllers.admin import services as admin_services
from . import services


@api_view(["POST"])
def register(request):
    ser = RegisterSerializer(data=request.data)
    if not ser.is_valid():
        return Response(ser.errors, status=status.HTTP_400_BAD_REQUEST)

    try:
        user = services.create_user(**ser.validated_data)
    except ValueError as e:
        return Response({"error": str(e)}, status=status.HTTP_409_CONFLICT)

    return Response({
        "user": user,
        "access_token": create_access_token(user["id"]),
        "refresh_token": create_refresh_token(user["id"]),
    }, status=status.HTTP_201_CREATED)


@api_view(["POST"])
def login(request):
    ser = LoginSerializer(data=request.data)
    if not ser.is_valid():
        return Response(ser.errors, status=status.HTTP_400_BAD_REQUEST)

    try:
        user = services.authenticate(**ser.validated_data)
    except ValueError as e:
        return Response({"error": str(e)}, status=status.HTTP_401_UNAUTHORIZED)

    return Response({
        "user": user,
        "access_token": create_access_token(user["id"]),
        "refresh_token": create_refresh_token(user["id"]),
    })


@api_view(["POST"])
def refresh(request):
    token = request.data.get("refresh_token", "")
    if not token:
        return Response({"error": "refresh_token required"}, status=status.HTTP_400_BAD_REQUEST)

    if not verify_refresh_token(token):
        return Response({"error": "Invalid or expired refresh token"}, status=status.HTTP_401_UNAUTHORIZED)

    try:
        payload = decode_token(token)
    except Exception:
        return Response({"error": "Invalid refresh token"}, status=status.HTTP_401_UNAUTHORIZED)

    if payload.get("type") != "refresh":
        return Response({"error": "Invalid token type"}, status=status.HTTP_401_UNAUTHORIZED)

    blacklist_refresh_token(token)
    user_id = payload["user_id"]
    return Response({
        "access_token": create_access_token(user_id),
        "refresh_token": create_refresh_token(user_id),
    })


@api_view(["POST"])
@jwt_required
def logout(request):
    token = request.data.get("refresh_token", "")
    if token:
        blacklist_refresh_token(token)
    return Response({"message": "Logged out"})


@api_view(["GET", "PATCH"])
@jwt_required
def me(request):
    if request.method == "GET":
        user = services.get_user_by_id(request.user_id)
        if not user:
            return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)
        return Response(user)

    ser = ProfileUpdateSerializer(data=request.data)
    if not ser.is_valid():
        return Response(ser.errors, status=status.HTTP_400_BAD_REQUEST)
    user = services.update_user(request.user_id, ser.validated_data)
    return Response(user)


@api_view(["POST"])
@jwt_required
def delete_account(request):
    """Permanently delete the authenticated user and all associated DB + Qdrant data."""
    if not admin_services.delete_user(request.user_id):
        return Response(
            {"error": "Account could not be deleted"},
            status=status.HTTP_400_BAD_REQUEST,
        )
    return Response(status=status.HTTP_204_NO_CONTENT)
