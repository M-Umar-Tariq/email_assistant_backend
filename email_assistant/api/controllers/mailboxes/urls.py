from django.urls import path
from . import views

urlpatterns = [
    path("", views.mailbox_list_create, name="mailbox-list-create"),
    path("google/start/", views.mailbox_google_start, name="mailbox-google-start"),
    path("google/callback/", views.mailbox_google_callback, name="mailbox-google-callback"),
    path("<str:mailbox_id>/", views.mailbox_detail, name="mailbox-detail"),
    path("<str:mailbox_id>/sync/", views.mailbox_sync, name="mailbox-sync"),
    path("<str:mailbox_id>/stop-sync/", views.mailbox_stop_sync, name="mailbox-stop-sync"),
]
