from django.urls import path

from . import views

urlpatterns = [
    path("", views.feedback_submit, name="feedback-submit"),
]
