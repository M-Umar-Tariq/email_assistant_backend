from django.urls import path
from . import views

urlpatterns = [
    path("", views.briefing, name="briefing"),
    path("ai/", views.ai_briefing, name="ai-briefing"),
]
