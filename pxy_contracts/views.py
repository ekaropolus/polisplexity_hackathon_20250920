from django.http import JsonResponse
from django.views.decorators.http import require_GET

# Versión del contrato
try:
    from .version import SPEC_VERSION
except Exception:
    SPEC_VERSION = "0.1.0"

# Modelos Pydantic
from .contracts.sami import SAMIRunRequest, SAMIRunResponse
from .contracts.sites import SiteSearchRequest, SiteSearchResponse


def _schema_of(model_cls):
    """Devuelve JSONSchema para Pydantic v2 o v1."""
    try:
        return model_cls.model_json_schema()  # Pydantic v2
    except Exception:
        return model_cls.schema()             # Pydantic v1


@require_GET
def sami_contracts(request):
    """GET /api/contracts/sami.json"""
    return JsonResponse({
        "spec_version": SPEC_VERSION,
        "request": _schema_of(SAMIRunRequest),
        "response": _schema_of(SAMIRunResponse),
    })


@require_GET
def sites_contracts(request):
    """GET /api/contracts/sites.json"""
    return JsonResponse({
        "spec_version": SPEC_VERSION,
        "request": _schema_of(SiteSearchRequest),
        "response": _schema_of(SiteSearchResponse),
    })
