from django.urls import path
from . import views

urlpatterns = [
    path("api/contracts/sami.json", views.sami_contracts, name="contracts_sami"),
    path("api/contracts/sites.json", views.sites_contracts, name="contracts_sites"),
]
