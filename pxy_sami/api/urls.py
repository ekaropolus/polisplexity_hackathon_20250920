# pxy_sami/api/urls.py
from django.urls import path
from .views import sami_health, sami_run

urlpatterns = [
    path("api/sami/health", sami_health, name="sami_health"),
    path("api/sami/run",    sami_run,    name="sami_run"),
]
