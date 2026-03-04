from django.urls import path
from . import views

urlpatterns = [
    path("register/", views.register, name="auth-register"),
    path("login/", views.login, name="auth-login"),
    path("refresh/", views.refresh, name="auth-refresh"),
    path("logout/", views.logout, name="auth-logout"),
    path("me/", views.me, name="auth-me"),
]
