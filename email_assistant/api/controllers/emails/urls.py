from django.urls import path
from . import views

urlpatterns = [
    path("", views.email_list, name="email-list"),
    path("stats/", views.email_stats, name="email-stats"),
    path("folder-counts/", views.email_folder_counts, name="email-folder-counts"),
    path("unique-senders/", views.email_unique_senders, name="email-unique-senders"),
    path("send/", views.email_send, name="email-send"),
    path("delete-all/", views.email_delete_all, name="email-delete-all"),
    path("mark-all-read/", views.email_mark_all_read, name="email-mark-all-read"),
    path("mark-all-unread/", views.email_mark_all_unread, name="email-mark-all-unread"),
    path("<str:email_id>/", views.email_detail, name="email-detail"),
    path("<str:email_id>/update/", views.email_update, name="email-update"),
    path("<str:email_id>/snooze/", views.email_snooze, name="email-snooze"),
    path("<str:email_id>/archive/", views.email_archive, name="email-archive"),
    path("<str:email_id>/trash/", views.email_trash, name="email-trash"),
    path("<str:email_id>/spam/", views.email_spam, name="email-spam"),
    path("<str:email_id>/delete/", views.email_delete, name="email-delete"),
    path("<str:email_id>/reply/", views.email_reply, name="email-reply"),
    path("<str:email_id>/forward/", views.email_forward, name="email-forward"),
    path("<str:email_id>/thread-reply/<int:reply_index>/", views.email_delete_thread_reply, name="email-delete-thread-reply"),
    path("<str:email_id>/sent-reply/<int:reply_index>/", views.email_delete_sent_reply, name="email-delete-sent-reply"),
    path("<str:email_id>/attachments/<int:attachment_index>/download/", views.email_attachment_download, name="email-attachment-download"),
]
