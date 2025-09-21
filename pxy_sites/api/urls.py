from django.urls import path
from .views import (
    SitesHealth, SiteSearchView,
    sites_download, sites_geojson, sites_preview, sites_recent_runs
)

urlpatterns = [
    path("api/sites/health", SitesHealth.as_view(), name="sites_health"),
    path("api/sites/search", SiteSearchView.as_view(), name="sites_search"),

    # artifacts
    path("api/sites/download/<str:kind>/<str:search_id>", sites_download, name="sites_download"),
    path("api/sites/geojson/<str:kind>/<str:search_id>", sites_geojson, name="sites_geojson"),
    path("api/sites/preview/<str:kind>/<str:search_id>", sites_preview, name="sites_preview"),
    path("api/sites/runs/recent", sites_recent_runs, name="sites_recent_runs"),
]
