from django.urls import path
from . import views

urlpatterns = [
    path("", views.settings_view, name="settings"),
    path("relabel/", views.relabel_emails, name="settings-relabel"),
]
