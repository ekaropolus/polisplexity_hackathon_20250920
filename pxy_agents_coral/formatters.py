from __future__ import annotations #
import json
import re
import requests
from typing import Any, Dict, Tuple
from django.http import JsonResponse, HttpRequest
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

# ---------- helpers ----------
def _base(request: HttpRequest) -> str:
    return request.build_absolute_uri("/")[:-1]

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
      2) full canonical envelope with .input.args_raw="/sami {...}"
    """
    # 1) direct payload
    if isinstance(body.get("payload"), dict):
        return body["payload"], "payload"

    # 2) canonical envelope: parse JSON after the command in args_raw
    args_raw = (body.get("input", {}) or {}).get("args_raw") or ""
    # strip leading "/sami " or "/sites "
    cleaned = re.sub(r"^/\w+\s*", "", args_raw).strip()
    if cleaned:
        try:
            return json.loads(cleaned), "args_raw"
        except Exception:
            pass

    return {}, "empty"

def _post_execute(request: HttpRequest, agent: str, payload: Dict[str, Any], timeout: float = 30.0):
    url = f"{_base(request)}/api/agents/execute"
    try:
        r = requests.post(url, json={"agent": agent, "payload": payload}, timeout=timeout)
        # try parse json regardless of status
        try:
            data = r.json()
        except Exception:
            data = {"code": "NON_JSON", "message": r.text[:2000]}
        return r.status_code, data
    except requests.Timeout:
        return 504, {"code": "UPSTREAM_TIMEOUT", "message": "agent upstream timed out"}
    except Exception as e:
        return 500, {"code": "EXEC_ERROR", "message": str(e)}

# ---------- text builders ----------
def _text_sami(data: Dict[str, Any]) -> str:
    if "beta" in data and "r2" in data:
        lines = [f"SAMI run: β={data['beta']:.3f}, R²={data['r2']:.3f}"]
        resid = data.get("residuals") or []
        top = sorted(resid, key=lambda x: x.get("rank", 1e9))[:3]
        for c in top:
            lines.append(f"{c.get('rank')}. {c.get('city')}: {c.get('sami',0):+0.2f}")
        if data.get("share_url"):
            lines += ["", data["share_url"]]
        return "\n".join(lines)
    if data.get("code"):
        return f"⚠️ {data.get('code')}: {data.get('message','')}"
    return "SAMI results ready."

def _text_sites(data: Dict[str, Any]) -> str:
    if isinstance(data.get("candidates"), list):
        city = data.get("city", "?")
        business = data.get("business", "?")
        lines = [f"Top sites for {business} in {city}:"]
        for i, c in enumerate(data["candidates"][:3], 1):
            lat = c.get("lat", 0); lon = c.get("lon", 0); sc = c.get("score", 0)
            lines.append(f"{i}. score={sc:.2f} @ ({lat:.5f},{lon:.5f})")
        for k in ("share_url", "isochrones_geojson_url", "candidates_geojson_url"):
            if data.get(k): lines.append(data[k])
        return "\n".join(lines)
    if data.get("code"):
        return f"⚠️ {data.get('code')}: {data.get('message','')}"
    return "Site scoring ready."

# ---------- views ----------
@csrf_exempt
@require_http_methods(["GET", "POST"])
def format_sami(request: HttpRequest):
    body = _load_body(request)
    payload, src = _extract_payload(body)
    status, data = _post_execute(request, "sami", payload, timeout=30.0)
    # add echo + text
    data = data if isinstance(data, dict) else {"result": data}
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
    status, data = _post_execute(request, "sites", payload, timeout=30.0)
    data = data if isinstance(data, dict) else {"result": data}
    data.setdefault("_echo", {"src": src, "payload_keys": list(payload.keys())})
    try:
        data["text"] = _text_sites(data)
    except Exception:
        pass
    return JsonResponse(data, status=status, safe=False)
