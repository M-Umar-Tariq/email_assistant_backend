from django.urls import path
from . import views

urlpatterns = [
    path("stats/", views.stats, name="admin-stats"),
    path("activity/", views.recent_activity, name="admin-activity"),
    path("feedback/", views.feedback_list, name="admin-feedback-list"),
    path("mailboxes/", views.mailboxes_list, name="admin-mailboxes"),
    path("users/", views.users_list, name="admin-users-list"),
    path("users/<str:user_id>/", views.user_detail, name="admin-user-detail"),
]
