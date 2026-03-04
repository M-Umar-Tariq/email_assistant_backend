from django.urls import path
from . import views

urlpatterns = [
    path("generate/", views.generate, name="compose-generate"),
    path("rewrite/", views.rewrite, name="compose-rewrite"),
    path("proofread/", views.proofread, name="compose-proofread"),
    path("contact/", views.contact_intelligence, name="compose-contact"),
]
