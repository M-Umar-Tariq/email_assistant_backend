from django.urls import path
from . import views

urlpatterns = [
    path("", views.mailbox_list_create, name="mailbox-list-create"),
    path("<str:mailbox_id>/", views.mailbox_detail, name="mailbox-detail"),
    path("<str:mailbox_id>/sync/", views.mailbox_sync, name="mailbox-sync"),
    path("<str:mailbox_id>/stop-sync/", views.mailbox_stop_sync, name="mailbox-stop-sync"),
]
