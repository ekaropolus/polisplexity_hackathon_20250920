from django.urls import path
from .formatters import format_sami, format_sites
from .views import agents_list, agents_execute, agents_health

urlpatterns = [
    path("api/agents/list", agents_list, name="agents_list"),
    path("api/agents/execute", agents_execute, name="agents_execute"),
    path("api/agents/format/sami", format_sami, name="agents_format_sami"),
    path("api/agents/format/sites", format_sites, name="agents_format_sites"),
    path("api/agents/health", agents_health),
]
