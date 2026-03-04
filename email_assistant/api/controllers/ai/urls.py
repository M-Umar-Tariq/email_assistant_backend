from django.urls import path
from . import views

urlpatterns = [
    path("ask/", views.ask, name="ai-ask"),
    path("ask/<str:email_id>/", views.ask_about_email, name="ai-ask-about-email"),
    path("suggested-questions/", views.suggested_questions, name="ai-suggested"),
    path("instant-replies/<str:email_id>/", views.instant_replies, name="ai-instant-replies"),
]
