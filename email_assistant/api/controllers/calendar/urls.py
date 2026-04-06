from django.urls import path

from . import views

urlpatterns = [
    path("", views.meeting_list_create, name="calendar-list-create"),
    path("<str:meeting_id>/", views.meeting_detail, name="calendar-detail"),
]
