# pxy_sites/api/views.py
from __future__ import annotations

import io
import json
import logging
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image
from django.conf import settings
from django.http import (
    FileResponse,
    HttpRequest,
    HttpResponse,
    HttpResponseNotFound,
    JsonResponse,
)
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.exceptions import ValidationError as DRFValidationError
from rest_framework import status
from rest_framework.throttling import ScopedRateThrottle
from pydantic import ValidationError as PydValidationError

from pxy_contracts.contracts.sites import SiteSearchRequest, SiteSearchResponse
from pxy_sites.models import SiteRun
from pxy_sites.services.site_scoring import run_site_search
from pxy_dashboard.utils.share import mint_sites_share_url

# NEW: public URL helpers (proxy/HTTPS aware)
from core.urlbuild import public_base, public_url

log = logging.getLogger(__name__)

# -------- uniform error envelope helpers --------
def _env(code: str, message: str, *, hint: str | None = None, http: int = 400):
    return Response(
        {"ok": False, "code": code, "message": message, "hint": hint, "trace_id": str(uuid.uuid4())},
        status=http,
    )

def _env_json(code: str, message: str, *, hint: str | None = None, http: int = 400):
    return JsonResponse(
        {"ok": False, "code": code, "message": message, "hint": hint, "trace_id": str(uuid.uuid4())},
        status=http,
    )

# -------- helpers --------
def _pyify(o):
    """Make objects JSONField-safe (NumPy → native Python)."""
    if isinstance(o, (np.floating, np.float32, np.float64)):
        return float(o)
    if isinstance(o, (np.integer, np.int32, np.int64)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)

# -------- DRF API views --------
class SitesHealth(APIView):
    authentication_classes = []
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = "sites_health"

    def get(self, request, *args, **kwargs):
        return Response({"ok": True, "app": "pxy_sites"})

class SiteSearchView(APIView):
    throttle_scope = "sites_search"

    def post(self, request, *args, **kwargs):
        t0 = time.perf_counter()
        # 1) Validate contract
        try:
            req = SiteSearchRequest(**(request.data or {}))
        except PydValidationError as ve:
            return _env("invalid", "Validation error", hint=str(ve), http=status.HTTP_400_BAD_REQUEST)

        # 2) Run scoring
        try:
            resp: SiteSearchResponse = run_site_search(req)
        except Exception as e:
            dur_ms = (time.perf_counter() - t0) * 1000.0
            log.warning(
                "[sites] search_failed city=%s business=%s bands=%s err=%s duration_ms=%.1f",
                getattr(req, "city", None), getattr(req, "business", None), getattr(req, "time_bands", None),
                e, dur_ms,
            )
            return _env("sites_error", "Sites search failed", hint=str(e), http=status.HTTP_502_BAD_GATEWAY)

        data = resp.model_dump()

        # 3) Public base + absolutize any URLs returned by the scorer
        base = public_base(request)
        sid = data.get("search_id")

        # ensure top-level map URLs & share are absolute
        for k in ("map_url", "demand_map_url", "competition_map_url", "share_url"):
            if data.get(k):
                data[k] = public_url(data[k], request)

        # inject (signed) share_url if we minted one
        if sid and not data.get("share_url"):
            data["share_url"] = mint_sites_share_url(sid, request=request)
            data["share_url"] = public_url(data["share_url"], request)

        # derived artifact/geojson/download/preview URLs (absolute)
        def _dl(kind: str) -> str: return f"{base}/api/sites/download/{kind}/{sid}"
        def _gj(kind: str) -> str: return f"{base}/api/sites/geojson/{kind}/{sid}"
        def _pv(kind: str) -> str: return f"{base}/api/sites/preview/{kind}/{sid}"

        if sid and data.get("map_url"):
            data["main_download_url"] = _dl("main")
            data["main_preview_url"]  = _pv("main")
        if sid and data.get("demand_map_url"):
            data["demand_download_url"] = _dl("demand")
            data["demand_preview_url"]  = _pv("demand")
        if sid and data.get("competition_map_url"):
            data["competition_download_url"] = _dl("competition")
            data["competition_preview_url"]  = _pv("competition")
        if sid:
            data["isochrones_geojson_url"]       = _gj("isochrones")
            data["candidates_geojson_url"]       = _gj("candidates")
            data["pois_competition_geojson_url"] = _gj("pois_competition")
            data["popgrid_geojson_url"]          = _gj("popgrid")

        # 4) Persist run in DB (best-effort)
        try:
            safe_payload = json.loads(json.dumps(req.model_dump(), default=_pyify))
            safe_result  = json.loads(json.dumps(data,           default=_pyify))
            SiteRun.objects.create(
                search_id=sid,
                city=safe_result.get("city"),
                business=safe_result.get("business"),
                payload_json=safe_payload,
                result_json=safe_result,
            )
            log.info("[sites] saved SiteRun %s", sid)
        except Exception as e:
            data.setdefault("warnings", []).append(f"persist_failed: {e}")
            log.warning("[sites] persist_failed for %s: %s", sid, e)

        dur_ms = (time.perf_counter() - t0) * 1000.0
        log.info(
            "[sites] search_ok city=%s business=%s bands=%s duration_ms=%.1f",
            data.get("city"), data.get("business"), data.get("time_bands"), dur_ms,
        )
        return Response(data, status=status.HTTP_200_OK)

# -------- Artifacts (FBVs) --------
_KIND_PREFIX = {"main": "sites", "demand": "demand", "competition": "competition"}

@csrf_exempt
def sites_download(request: HttpRequest, kind: str, search_id: str):
    prefix = _KIND_PREFIX.get(kind)
    if not prefix:
        return _env_json("invalid_kind", "Invalid kind", hint=str(list(_KIND_PREFIX)), http=400)
    try:
        uuid.UUID(search_id)
    except Exception:
        return _env_json("invalid_search_id", "search_id must be a UUID", http=400)

    fname = f"{prefix}_{search_id}.png"
    fpath = Path(settings.MEDIA_ROOT) / "sites" / fname
    if not fpath.exists():
        return _env_json("not_found", f"file not found: {fname}", http=404)

    return FileResponse(open(fpath, "rb"), content_type="image/png", as_attachment=True, filename=fname)

_GJ_KEYS = {
    "isochrones": "isochrones_fc",
    "candidates": "candidates_fc",
    "pois_competition": "pois_competition_fc",
    "popgrid": "popgrid_fc",
}

@csrf_exempt
def sites_geojson(request: HttpRequest, kind: str, search_id: str):
    if kind not in _GJ_KEYS:
        return _env_json("invalid_kind", "Invalid kind", hint=str(list(_GJ_KEYS)), http=400)
    try:
        uuid.UUID(search_id)
    except Exception:
        return _env_json("invalid_search_id", "search_id must be a UUID", http=400)

    fpath = Path(settings.MEDIA_ROOT) / "sites" / f"run_{search_id}.json"
    if not fpath.exists():
        return _env_json("not_found", f"artifact not found: run_{search_id}.json", http=404)

    try:
        with open(fpath, "r", encoding="utf-8") as f:
            artifact = json.load(f)
        fc = artifact.get(_GJ_KEYS[kind]) or {"type": "FeatureCollection", "features": []}
        return HttpResponse(json.dumps(fc), content_type="application/geo+json")
    except Exception as e:
        return _env_json("artifact_read_error", "Failed to read artifact", hint=str(e), http=500)

_PREVIEW_PREFIX = {"main": "sites", "demand": "demand", "competition": "competition"}

@csrf_exempt
def sites_preview(request: HttpRequest, kind: str, search_id: str):
    prefix = _PREVIEW_PREFIX.get(kind)
    if not prefix:
        return _env_json("invalid_kind", "Invalid kind", hint=str(list(_PREVIEW_PREFIX)), http=400)
    try:
        uuid.UUID(search_id)
    except Exception:
        return _env_json("invalid_search_id", "search_id must be a UUID", http=400)

    fname = f"{prefix}_{search_id}.png"
    fpath = Path(settings.MEDIA_ROOT) / "sites" / fname
    if not fpath.exists():
        return _env_json("not_found", f"file not found: {fname}", http=404)

    # resize params
    def _clamp_int(val, lo, hi, default=None):
        try:
            v = int(val)
            return max(lo, min(hi, v))
        except Exception:
            return default

    w_q = _clamp_int(request.GET.get("w"), 16, 2000, None)
    h_q = _clamp_int(request.GET.get("h"), 16, 2000, None)
    try:
        scale_q = float(request.GET.get("scale")) if request.GET.get("scale") else None
        if scale_q is not None:
            scale_q = max(0.05, min(3.0, scale_q))
    except Exception:
        scale_q = None

    if not any([w_q, h_q, scale_q]):
        with open(fpath, "rb") as f:
            data = f.read()
        resp = HttpResponse(data, content_type="image/png")
        resp["Cache-Control"] = "public, max-age=3600"
        return resp

    try:
        im = Image.open(fpath)
        im = im.convert("RGBA") if im.mode not in ("RGB", "RGBA") else im
        orig_w, orig_h = im.size

        if scale_q:
            w = int(round(orig_w * scale_q)); h = int(round(orig_h * scale_q))
        elif w_q and h_q:
            w, h = w_q, h_q
        elif w_q:
            ratio = w_q / float(orig_w); w, h = w_q, max(1, int(round(orig_h * ratio)))
        elif h_q:
            ratio = h_q / float(orig_h); w, h = max(1, int(round(orig_w * ratio))), h_q
        else:
            w, h = orig_w, orig_h

        w = max(16, min(2000, w)); h = max(16, min(2000, h))
        im = im.resize((w, h), Image.LANCZOS)

        buf = io.BytesIO(); im.save(buf, format="PNG", optimize=True); buf.seek(0)
        resp = HttpResponse(buf.getvalue(), content_type="image/png")
        resp["Cache-Control"] = "public, max-age=600"
        return resp
    except Exception as e:
        return _env_json("resize_failed", "Image resize failed", hint=str(e), http=500)

@require_GET
def sites_recent_runs(request: HttpRequest):
    """GET /api/sites/runs/recent?limit=10 — list latest runs with handy URLs."""
    try:
        limit = int(request.GET.get("limit", "10"))
    except Exception:
        limit = 10
    limit = max(1, min(limit, 50))

    items = []
    qs = SiteRun.objects.order_by("-created_at")[:limit]
    for r in qs:
        res = r.result_json or {}
        items.append({
            "search_id": r.search_id,
            "city": r.city,
            "business": r.business,
            "created_at": r.created_at.isoformat(),
            "map_url": res.get("map_url"),
            "demand_map_url": res.get("demand_map_url"),
            "competition_map_url": res.get("competition_map_url"),
            "download": {
                "main": res.get("main_download_url"),
                "demand": res.get("demand_download_url"),
                "competition": res.get("competition_download_url"),
            },
            "geojson": {
                "isochrones": res.get("isochrones_geojson_url"),
                "candidates": res.get("candidates_geojson_url"),
                "pois_competition": res.get("pois_competition_geojson_url"),
                "popgrid": res.get("popgrid_geojson_url"),
            },
        })
    return JsonResponse({"items": items})
