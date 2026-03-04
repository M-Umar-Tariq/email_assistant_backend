from django.urls import path
from . import views

urlpatterns = [
    path("suggestions/", views.suggestions, name="agent-suggestions"),
    path("chat/", views.chat, name="agent-chat"),
    path("profile/", views.profile, name="agent-profile"),
    path("profile/build/", views.build_profile, name="agent-profile-build"),
    path("execute/", views.execute, name="agent-execute"),
    path("reject/<str:action_id>/", views.reject, name="agent-reject"),
    path("speak/", views.speak, name="agent-speak"),
]
