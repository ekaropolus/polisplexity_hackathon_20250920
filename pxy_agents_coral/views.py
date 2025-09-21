# polisplexity/pxy_agents_coral/views.py
from __future__ import annotations

import json
import re
from typing import Any, Dict, Tuple
from urllib.parse import urlparse

import requests
from django.conf import settings
from django.http import JsonResponse, HttpRequest
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods, require_POST

# build absolute public URLs from a request + path
from core.urlbuild import public_url

# ----- contracts version (best-effort) -----
try:
    from pxy_contracts.version import SPEC_VERSION
except Exception:
    SPEC_VERSION = "0.1.0"

# ----- INTERNAL CALL BASES -----
# For the generic /api/agents/execute proxy (kept for compatibility)
AGENTS_INTERNAL_BASE = getattr(settings, "AGENTS_INTERNAL_BASE", "")

# For the formatter endpoints we *force* an internal base and never guess from Host.
# Set in .env: AGENTS_INTERNAL_BASE=http://127.0.0.1:8002
# Fallback keeps you safe even if env is missing/misread.
FORMAT_INTERNAL_BASE = (AGENTS_INTERNAL_BASE or "http://127.0.0.1:8002").rstrip("/")

# ===== helpers =====
def _load_body(request: HttpRequest) -> Dict[str, Any]:
    try:
        raw = (request.body or b"").decode("utf-8")
        return json.loads(raw or "{}")
    except Exception:
        return {}

def _extract_payload(body: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """
    Returns (payload, src) where src is 'payload', 'args_raw', or 'empty'.
    Accepts:
      1) {"payload": {...}}
      2) Canonical tg envelope with .input.args_raw like "/sami {...}"
    """
    if isinstance(body.get("payload"), dict):
        return body["payload"], "payload"
    args_raw = (body.get("input", {}) or {}).get("args_raw") or ""
    cleaned = re.sub(r"^/\w+\s*", "", args_raw).strip()
    if cleaned:
        try:
            return json.loads(cleaned), "args_raw"
        except Exception:
            pass
    return {}, "empty"

def _post_underlying(agent: str, payload: Dict[str, Any], timeout: float = 60.0):
    """
    Call the *real* internal APIs via a fixed base (no build_absolute_uri):
      sami  -> /api/sami/run
      sites -> /api/sites/search
    """
    path = "/api/sami/run" if agent == "sami" else "/api/sites/search"
    url = f"{FORMAT_INTERNAL_BASE}{path}"
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        try:
            data = r.json()
        except Exception:
            data = {"code": "NON_JSON", "message": r.text[:2000]}
        return r.status_code, data
    except requests.Timeout:
        return 504, {"code": "UPSTREAM_TIMEOUT", "message": "agent upstream timed out", "_debug_url": url}
    except Exception as e:
        return 500, {"code": "EXEC_ERROR", "message": str(e), "_debug_url": url}

def _normalize_urls_to_public(data: dict, request: HttpRequest) -> None:
    """
    Convert any absolute URLs that may point to 127.0.0.1:8002 into public URLs
    using the current request host while preserving the path.
    """
    if not isinstance(data, dict):
        return

    url_keys = {
        # common keys from sami + sites
        "share_url", "map_url", "demand_map_url", "competition_map_url",
        "main_download_url", "demand_download_url", "competition_download_url",
        "main_preview_url", "demand_preview_url", "competition_preview_url",
        "isochrones_geojson_url", "candidates_geojson_url",
        "pois_competition_geojson_url", "popgrid_geojson_url",
        "chart_url",
    }

    for k in list(url_keys):
        v = data.get(k)
        if not isinstance(v, str) or not v:
            continue
        try:
            p = urlparse(v)
            # only rewrite absolute http(s) URLs; keep relative ones
            if p.scheme in ("http", "https") and p.path:
                data[k] = public_url(request, p.path)
        except Exception:
            # never fail formatting due to a bad URL
            pass

# Tiny text builders for bot replies
def _text_sami(data: Dict[str, Any]) -> str:
    if "beta" in data and "r2" in data:
        lines = [f"SAMI run: β={data['beta']:.3f}, R²={data['r2']:.3f}"]
        for c in sorted(data.get("residuals", []), key=lambda x: x.get("rank", 1e9))[:3]:
            lines.append(f"{c.get('rank')}. {c.get('city')}: {c.get('sami',0):+0.2f}")
        if data.get("share_url"):
            lines += ["", data["share_url"]]
        return "\n".join(lines)
    if data.get("code"):
        return f"⚠️ {data.get('code')}: {data.get('message','')}"
    return "SAMI results ready."

def _text_sites(data: Dict[str, Any]) -> str:
    if isinstance(data.get("candidates"), list):
        city = data.get("city", "?"); biz = data.get("business", "?")
        lines = [f"Top sites for {biz} in {city}:"]
        for i, c in enumerate(data["candidates"][:3], 1):
            lines.append(f"{i}. score={c.get('score',0):.2f} @ ({c.get('lat',0):.5f},{c.get('lon',0):.5f})")
        for k in ("share_url", "isochrones_geojson_url", "candidates_geojson_url"):
            if data.get(k): lines.append(data[k])
        return "\n".join(lines)
    if data.get("code"):
        return f"⚠️ {data.get('code')}: {data.get('message','')}"
    return "Site scoring ready."

# ===== public endpoints =====
@csrf_exempt
@require_http_methods(["GET", "POST"])
def agents_list(request: HttpRequest):
    # use request host only for *outward* links (safe)
    base = request.build_absolute_uri("/")[:-1]
    agents = [
        {
            "agent": "sami",
            "name": "SAMI-Agent",
            "version": "1.0.0",
            "spec_version": SPEC_VERSION,
            "contracts_url": f"{base}/api/contracts/sami.json",
            "execute_url": f"{base}/api/agents/execute",
            "description": "Urban scaling (β, R²) + SAMI residuals + chart",
        },
        {
            "agent": "sites",
            "name": "Sites-Agent",
            "version": "1.0.0",
            "spec_version": SPEC_VERSION,
            "contracts_url": f"{base}/api/contracts/sites.json",
            "execute_url": f"{base}/api/agents/execute",
            "description": "Site scoring (access, demand, competition) with maps",
        },
    ]
    lines = ["Available agents:"]
    for a in agents:
        lines.append(f"- {a['agent']}: {a['description']}")
    lines += [
        "",
        "Try:",
        '/sami {"indicator":"imss_wages_2023","cities":["CDMX","GDL","MTY"]}',
        '/sites {"city":"CDMX","business":"cafe","time_bands":[10,20]}',
    ]
    return JsonResponse({"agents": agents, "text": "\n".join(lines)})

@csrf_exempt
@require_POST
def agents_execute(request: HttpRequest):
    """
    Body: { "agent": "sami"|"sites", "payload": {...} }
    Proxies to the *internal* API using AGENTS_INTERNAL_BASE (or same-host fallback).
    """
    try:
        body = json.loads(request.body.decode("utf-8") or "{}")
        agent = (body.get("agent") or "").strip().lower()
        payload = body.get("payload")

        if agent not in {"sami", "sites"}:
            return JsonResponse({"code": "AGENT_NOT_FOUND", "message": f"unknown agent '{agent}'"}, status=404)
        if payload is None:
            return JsonResponse({"code": "BAD_REQUEST", "message": "missing 'payload'"}, status=400)

        path = "/api/sami/run" if agent == "sami" else "/api/sites/search"
        base = (AGENTS_INTERNAL_BASE or "http://127.0.0.1:8002").rstrip("/")
        url = f"{base}{path}"

        r = requests.post(url, json=payload, timeout=90)
        return JsonResponse(r.json(), status=r.status_code, safe=False)

    except requests.Timeout:
        return JsonResponse({"code": "UPSTREAM_TIMEOUT", "message": "agent upstream timed out"}, status=504)
    except ValueError as ve:
        return JsonResponse({"code": "BAD_JSON", "message": str(ve)}, status=400)
    except Exception as e:
        return JsonResponse({"code": "AGENT_EXEC_ERROR", "message": str(e)}, status=500)

# ----- formatters (call underlying APIs directly via fixed base) -----
@csrf_exempt
@require_http_methods(["GET", "POST"])
def format_sami(request: HttpRequest):
    body = _load_body(request)
    payload, src = _extract_payload(body)
    status, data = _post_underlying("sami", payload, timeout=60.0)
    data = data if isinstance(data, dict) else {"result": data}
    _normalize_urls_to_public(data, request)  # ensure public links
    data.setdefault("_echo", {"src": src, "payload_keys": list(payload.keys())})
    try:
        data["text"] = _text_sami(data)
    except Exception:
        pass
    return JsonResponse(data, status=status, safe=False)

@csrf_exempt
@require_http_methods(["GET", "POST"])
def format_sites(request: HttpRequest):
    body = _load_body(request)
    payload, src = _extract_payload(body)
    status, data = _post_underlying("sites", payload, timeout=60.0)
    data = data if isinstance(data, dict) else {"result": data}
    _normalize_urls_to_public(data, request)  # ensure public links
    data.setdefault("_echo", {"src": src, "payload_keys": list(payload.keys())})
    try:
        data["text"] = _text_sites(data)
    except Exception:
        pass
    return JsonResponse(data, status=status, safe=False)

# pxy_agents_coral/views.py  (add this function)
from django.http import JsonResponse
from django.conf import settings
import requests

def agents_health(request):
    data = {
        "service": "polisplexity-agents",
        "version": "1.0.0",
        "spec_version": settings.SPEC_VERSION if hasattr(settings, "SPEC_VERSION") else "0.1.0",
        "checks": {},
    }
    try:
        r1 = requests.get(f"{request.build_absolute_uri('/')[:-1]}/api/sami/health", timeout=3)
        data["checks"]["sami"] = {"ok": r1.status_code == 200}
    except Exception as e:
        data["checks"]["sami"] = {"ok": False, "error": str(e)}
    try:
        r2 = requests.get(f"{request.build_absolute_uri('/')[:-1]}/api/sites/health", timeout=3)
        data["checks"]["sites"] = {"ok": r2.status_code == 200}
    except Exception as e:
        data["checks"]["sites"] = {"ok": False, "error": str(e)}
    return JsonResponse(data)

