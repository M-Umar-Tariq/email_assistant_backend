from django.urls import path
from . import views

urlpatterns = [
    path("overview/", views.overview, name="analytics-overview"),
    path("volume/", views.volume, name="analytics-volume"),
    path("top-senders/", views.top_senders, name="analytics-top-senders"),
    path("categories/", views.categories, name="analytics-categories"),
    path("metrics/", views.metrics, name="analytics-metrics"),
]
