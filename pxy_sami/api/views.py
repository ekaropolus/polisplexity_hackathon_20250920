# pxy_sami/api/views.py
from __future__ import annotations
import uuid
from rest_framework.decorators import api_view, throttle_classes
from rest_framework.response import Response
from rest_framework import status
from rest_framework.throttling import ScopedRateThrottle
from pydantic import ValidationError

from pxy_contracts.contracts import SAMIRunRequest
from pxy_sami.estimators.sami_core import run_sami
from pxy_dashboard.utils.share import mint_sami_share_url

# NEW:
from core.urlbuild import public_url

def _err(code: str, message: str, hint: str | None = None, http_status: int = 400):
    return Response(
        {"ok": False, "code": code, "message": message, "hint": hint, "trace_id": str(uuid.uuid4())},
        status=http_status,
    )

@api_view(["GET"])
@throttle_classes([ScopedRateThrottle])
def sami_health(request):
    sami_health.throttle_scope = "sami_health"
    try:
        return Response({"ok": True, "service": "sami"})
    except Exception as e:
        return _err("sami_health_error", "SAMI health check failed", str(e),
                    http_status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(["POST"])
@throttle_classes([ScopedRateThrottle])
def sami_run(request):
    sami_run.throttle_scope = "sami_run"
    try:
        req = SAMIRunRequest.model_validate(request.data or {})
    except ValidationError as ve:
        return _err("invalid", "Validation error", hint=str(ve), http_status=status.HTTP_400_BAD_REQUEST)

    try:
        resp = run_sami(req)
        data = resp.model_dump()

        # inject share URL (signed)
        rid = data.get("run_id")
        if rid:
            meta = {
                "indicator": data.get("indicator"),
                "beta": data.get("beta"),
                "r2": data.get("r2"),
                "n": len(data.get("residuals") or []),
            }
            data["share_url"] = mint_sami_share_url(rid, meta=meta, request=request)

        # ABSOLUTIZE any path-like URLs (chart, share)
        for k in ("chart_url", "share_url"):
            if data.get(k):
                data[k] = public_url(data[k], request)

        return Response(data)
    except Exception as e:
        return _err("sami_error", "SAMI run failed", hint=str(e),
                    http_status=status.HTTP_502_BAD_GATEWAY)
