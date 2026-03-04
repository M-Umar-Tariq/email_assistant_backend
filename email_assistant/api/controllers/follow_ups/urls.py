from django.urls import path
from . import views

urlpatterns = [
    path("", views.follow_up_list_create, name="follow-up-list-create"),
    path("auto/today/", views.follow_up_auto_today, name="follow-up-auto-today"),
    path("<str:follow_up_id>/", views.follow_up_detail, name="follow-up-detail"),
    path("<str:follow_up_id>/complete/", views.follow_up_complete, name="follow-up-complete"),
]
