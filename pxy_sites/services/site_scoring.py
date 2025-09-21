# pxy_sites/services/site_scoring.py
from __future__ import annotations
import os, json, uuid, random, math
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime

from django.conf import settings
from pyproj import Geod

# Headless backend para matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

import numpy as np
from shapely.geometry import Point, Polygon
from scipy.stats import gaussian_kde

from pxy_contracts.contracts import (
    SiteSearchRequest, SiteSearchResponse,
    CandidateSite, ScoreBreakdown
)
from pxy_routing.services import get_routing_provider
from pxy_de.providers.base import get_provider


# --------------------------- Helpers geométricos ---------------------------

def _isochrone_area_km2(feature: dict) -> float:
    geom = (feature or {}).get("geometry") or {}
    if geom.get("type") != "Polygon":
        return 0.0
    rings = geom.get("coordinates") or []
    if not rings:
        return 0.0
    coords = rings[0]
    if len(coords) < 4:
        return 0.0
    geod = Geod(ellps="WGS84")
    lons = [float(x[0]) for x in coords]
    lats = [float(x[1]) for x in coords]
    area_m2, _ = geod.polygon_area_perimeter(lons, lats)
    return abs(area_m2) / 1_000_000.0  # m² -> km²


def _polygon_from_feature(feature: dict) -> Optional[Polygon]:
    geom = (feature or {}).get("geometry") or {}
    if geom.get("type") != "Polygon":
        return None
    coords = geom.get("coordinates")
    if not coords or not coords[0]:
        return None
    try:
        ring = [(float(x[0]), float(x[1])) for x in coords[0]]
        if len(ring) < 4:
            return None
        return Polygon(ring)
    except Exception:
        return None


def _extent_from_iso_list(iso_list: List[dict]) -> Optional[Tuple[float, float, float, float]]:
    xs, ys = [], []
    for item in iso_list or []:
        feat = item.get("feature") or {}
        geom = feat.get("geometry") or {}
        if geom.get("type") != "Polygon":
            continue
        coords = geom.get("coordinates") or []
        if not coords:
            continue
        ring = coords[0]
        for x, y in ring:
            xs.append(float(x)); ys.append(float(y))
    if not xs or not ys:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def _build_isochrones(center: Tuple[float, float], time_bands: List[int]) -> List[dict]:
    """
    Build isochrones for the requested minute bands.
    - If the routing provider supports `isochrones(center, minutes_list)`, use it once
      (reduces ORS requests and rate-limit pressure).
    - Otherwise, fall back to one call per band.
    Output schema stays the same as before: a list of dicts with
      {"minutes": int, "feature": Feature(Polygon), "area_km2": float}
    """
    rp = get_routing_provider()
    bands: List[int] = [int(m) for m in (time_bands or [])]
    out: List[dict] = []

    # Try a single batched call first
    if hasattr(rp, "isochrones"):
        try:
            feats = rp.isochrones(center, bands)  # expected same order as requested bands
            n = min(len(bands), len(feats))
            for m, feat in zip(bands[:n], feats[:n]):
                area_km2 = _isochrone_area_km2(feat)
                props = {"minutes": int(m), "area_km2": float(area_km2)}
                f = {"type": "Feature", "geometry": feat.get("geometry"), "properties": props}
                out.append({"minutes": int(m), "feature": f, "area_km2": float(area_km2)})

            # If provider returned fewer features than requested, fill the rest via single calls
            for m in bands[n:]:
                feat = rp.isochrone(center, int(m))
                area_km2 = _isochrone_area_km2(feat)
                props = {"minutes": int(m), "area_km2": float(area_km2)}
                f = {"type": "Feature", "geometry": feat.get("geometry"), "properties": props}
                out.append({"minutes": int(m), "feature": f, "area_km2": float(area_km2)})

            return out
        except Exception:
            # Fall back to per-band calls below if the batch call fails for any reason
            pass

    # Fallback: one request per band (original behavior)
    for m in bands:
        feat = rp.isochrone(center, int(m))
        area_km2 = _isochrone_area_km2(feat)
        props = {"minutes": int(m), "area_km2": float(area_km2)}
        f = {"type": "Feature", "geometry": feat.get("geometry"), "properties": props}
        out.append({"minutes": int(m), "feature": f, "area_km2": float(area_km2)})

    return out



def _access_from_iso_list(iso_list: List[dict]) -> Tuple[float, List[str]]:
    if not iso_list:
        return 0.0, ["no_isochrones"]
    areas = [item["area_km2"] for item in iso_list]
    max_a = max(areas) if areas else 0.0
    if max_a <= 0:
        return 0.0, [f"{item['minutes']} min area ≈ 0.0 km²" for item in iso_list]
    norms = [a / max_a for a in areas]
    access = sum(norms) / len(norms)
    reasons = [f"{item['minutes']} min area ≈ {item['area_km2']:.1f} km²" for item in iso_list]
    return float(access), reasons


# --------------------------- Scores data-driven ---------------------------

def _competition_from_pois(city: str, business: str, iso_list: List[dict]) -> Tuple[float, List[str]]:
    prov = get_provider()
    try:
        pois = prov.denue(city, business)  # DataFrame[name,lat,lon,category]
    except Exception as e:
        return 0.5, [f"competition_fallback: provider_error={e}"]

    if pois.empty or not iso_list:
        return 0.5, ["competition_fallback: no_pois_or_isochrones"]

    largest = max(iso_list, key=lambda x: x["minutes"])
    poly = _polygon_from_feature(largest["feature"])
    if poly is None:
        return 0.5, ["competition_fallback: invalid_polygon"]

    area_km2 = float(largest.get("area_km2") or 0.0)
    if area_km2 <= 0.0:
        return 0.5, ["competition_fallback: zero_area"]

    cnt = 0
    for row in pois.itertuples(index=False):
        try:
            p = Point(float(row.lon), float(row.lat))
            if poly.contains(p):
                cnt += 1
        except Exception:
            continue

    density = cnt / area_km2  # POIs per km²
    D_ref = float(os.getenv("COMP_REF_DENSITY", "5.0"))
    comp = 1.0 / (1.0 + density / D_ref)
    comp = float(max(0.0, min(1.0, comp)))

    reasons = [
        f"largest_band: {largest['minutes']} min, area ≈ {area_km2:.1f} km²",
        f"competitors_inside: {cnt}, density ≈ {density:.2f} /km²",
        f"competition_score = 1/(1 + density/{D_ref:.1f}) ≈ {comp:.2f}",
    ]
    return comp, reasons


def _demand_from_popgrid(city: str, iso_list: List[dict]) -> Tuple[float, List[str]]:
    prov = get_provider()
    try:
        grid = prov.popgrid(city)  # DataFrame[cell_id, lat, lon, pop]
    except Exception as e:
        return 0.5, [f"demand_fallback: provider_error={e}"]

    if grid.empty or not iso_list:
        return 0.5, ["demand_fallback: no_grid_or_isochrones"]

    largest = max(iso_list, key=lambda x: x["minutes"])
    poly = _polygon_from_feature(largest["feature"])
    if poly is None:
        return 0.5, ["demand_fallback: invalid_polygon"]

    area_km2 = float(largest.get("area_km2") or 0.0)
    if area_km2 <= 0.0:
        return 0.5, ["demand_fallback: zero_area"]

    total_pop = 0.0
    for row in grid.itertuples(index=False):
        try:
            p = Point(float(row.lon), float(row.lat))
            if poly.contains(p):
                total_pop += float(row.pop)
        except Exception:
            continue

    density = total_pop / area_km2 if area_km2 > 0 else 0.0
    P_ref = float(os.getenv("DEMAND_REF_POP", "50000"))
    demand = total_pop / (total_pop + P_ref) if (total_pop + P_ref) > 0 else 0.0
    demand = float(max(0.0, min(1.0, demand)))

    reasons = [
        f"largest_band: {largest['minutes']} min, area ≈ {area_km2:.1f} km²",
        f"population_inside ≈ {int(total_pop)}, density ≈ {density:.1f} /km²",
        f"demand_score = pop/(pop+{int(P_ref)}) ≈ {demand:.2f}",
    ]
    return demand, reasons


# --------------------------- Sampling y Mapa principal ---------------------------

def _sample_points_in_polygon(poly: Polygon, n: int, rng: random.Random) -> List[Tuple[float, float]]:
    minx, miny, maxx, maxy = poly.bounds
    pts: List[Tuple[float, float]] = []
    max_tries = n * 50
    tries = 0
    while len(pts) < n and tries < max_tries:
        tries += 1
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        if poly.contains(Point(x, y)):
            pts.append((y, x))  # (lat, lon)
    return pts


def _km_per_deg_lon(lat_deg: float) -> float:
    return 111.320 * math.cos(math.radians(lat_deg))


def _km_per_deg_lat() -> float:
    return 110.574


def _save_sites_map(center: Tuple[float, float], iso_list_for_map: List[dict],
                    search_id: str, city: str, business: str,
                    top_candidates: List[Tuple[float, float, float]]) -> str | None:
    try:
        media_dir = settings.MEDIA_ROOT / "sites"
        media_dir.mkdir(parents=True, exist_ok=True)
        out_path = media_dir / f"sites_{search_id}.png"

        # recolectar polígonos/extent
        lons, lats = [center[1]], [center[0]]
        polys = []
        for item in iso_list_for_map:
            feat = item["feature"]
            geom = feat.get("geometry") or {}
            if geom.get("type") != "Polygon":
                continue
            coords = geom.get("coordinates")[0]
            poly_xy = [(float(x[0]), float(x[1])) for x in coords]
            polys.append({"minutes": item["minutes"], "coords": poly_xy, "area": item["area_km2"]})
            lons.extend([p[0] for p in poly_xy])
            lats.extend([p[1] for p in poly_xy])

        fig, ax = plt.subplots(figsize=(7.6, 7.6))

        band_palette = ["#2E86AB", "#F18F01", "#C73E1D", "#6C5B7B", "#17B890", "#7E57C2"]
        rank_palette = ["#1B998B", "#3A86FF", "#FB5607", "#FFBE0B", "#8338EC", "#FF006E"]

        for i, item in enumerate(sorted(polys, key=lambda d: d["minutes"], reverse=True)):
            poly = MplPolygon(item["coords"], closed=True,
                              facecolor=band_palette[i % len(band_palette)], alpha=0.25,
                              edgecolor=band_palette[i % len(band_palette)], linewidth=1.6,
                              label=f"{item['minutes']} min · {item['area']:.1f} km²")
            ax.add_patch(poly)

        ax.scatter([center[1]], [center[0]], s=68, zorder=6,
                   facecolor="#000", edgecolor="white", linewidth=1.2)
        ax.annotate("center", (center[1], center[0]),
                    xytext=(center[1] + 0.01, center[0] + 0.01),
                    fontsize=9, color="#303030",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75),
                    arrowprops=dict(arrowstyle="-", lw=0.7, color="#666", alpha=0.9))

        sizes = [90, 80, 72, 64, 56, 50, 46, 42, 38, 34]
        legend_rows = []
        for idx, (lat, lon, score) in enumerate(top_candidates, start=1):
            color = rank_palette[(idx - 1) % len(rank_palette)]
            size = sizes[idx - 1] if idx - 1 < len(sizes) else 30
            ax.scatter([lon], [lat], s=size, zorder=7,
                       facecolor=color, edgecolor="white", linewidth=1.0)
            ax.annotate(f"{idx} · {score:.2f}", (lon, lat),
                        xytext=(lon + 0.008, lat + 0.008),
                        fontsize=8, color="#111",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#bbb", alpha=0.9))
            legend_rows.append(f"{idx}. ({score:.2f})  {lat:.4f}, {lon:.4f}")
            lons.append(lon); lats.append(lat)

        if lons and lats:
            minx, maxx = min(lons), max(lons)
            miny, maxy = min(lats), max(lats)
            pad_x = max((maxx - minx) * 0.08, 0.01)
            pad_y = max((maxy - miny) * 0.08, 0.01)
            ax.set_xlim(minx - pad_x, maxx + pad_x)
            ax.set_ylim(miny - pad_y, maxy + pad_y)

        ax.set_title(f"Top sites — {business} @ {city}", fontsize=13, pad=10)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        leg = ax.legend(loc="lower right", frameon=True, fontsize=8, title="Isochrones")
        if leg and leg.get_frame():
            leg.get_frame().set_alpha(0.9)

        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        x_text = x0 + (x1 - x0) * 0.70
        y_text = y0 + (y1 - y0) * 0.97
        ax.text(x_text, y_text,
                "Top-K (score)\n" + "\n".join(legend_rows),
                ha="left", va="top", fontsize=8, color="#111",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", alpha=0.9))

        km_per_deg_x = _km_per_deg_lon(center[0])
        deg_len = 5.0 / km_per_deg_x if km_per_deg_x > 0 else 0.05
        px = x0 + (x1 - x0) * 0.10
        py = y0 + (y1 - y0) * 0.08
        ax.plot([px, px + deg_len], [py, py], lw=3, color="#222")
        ax.plot([px, px], [py - 0.001, py + 0.001], lw=2, color="#222")
        ax.plot([px + deg_len, px + deg_len], [py - 0.001, py + 0.001], lw=2, color="#222")
        ax.text(px + deg_len / 2.0, py + 0.002, "5 km",
                ha="center", va="bottom", fontsize=8, color="#222",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

        return f"{settings.MEDIA_URL}sites/{out_path.name}"
    except Exception:
        return None


# --------------------------- Mapas densidad: Demanda / Competencia ---------------------------

def _grid_kde(xy: np.ndarray, weights: Optional[np.ndarray],
              x_grid: np.ndarray, y_grid: np.ndarray, bw: Optional[float] = None) -> np.ndarray:
    if xy.shape[1] != 2 or xy.shape[0] < 2:
        return np.zeros((y_grid.size, x_grid.size), dtype=float)
    kde = gaussian_kde(xy.T, weights=weights, bw_method=bw)
    Xg, Yg = np.meshgrid(x_grid, y_grid)
    pts = np.vstack([Xg.ravel(), Yg.ravel()])
    z = kde(pts).reshape(Yg.shape)
    z = z - z.min()
    if z.max() > 0:
        z = z / z.max()
    return z


def _render_density_map(kind: str,
                        center: Tuple[float, float],
                        iso_list: List[dict],
                        points_xy: np.ndarray,
                        weights: Optional[np.ndarray],
                        search_id: str,
                        city: str,
                        business: str) -> Optional[str]:
    try:
        extent = _extent_from_iso_list(iso_list)
        if extent is None:
            cx, cy = center[1], center[0]
            extent = (cx - 0.08, cy - 0.08, cx + 0.08, cy + 0.08)
        minx, miny, maxx, maxy = extent
        pad_x = max((maxx - minx) * 0.05, 0.01)
        pad_y = max((maxy - miny) * 0.05, 0.01)
        minx -= pad_x; maxx += pad_x
        miny -= pad_y; maxy += pad_y

        lat0 = max(miny, min(maxy, center[0]))
        kx = _km_per_deg_lon(lat0)
        ky = _km_per_deg_lat()

        if points_xy.size == 0:
            return None
        xs = points_xy[:, 0] * kx
        ys = points_xy[:, 1] * ky

        grid_n = int(os.getenv("HEAT_GRID_N", "220"))
        xg = np.linspace(minx * kx, maxx * kx, grid_n)
        yg = np.linspace(miny * ky, maxy * ky, grid_n)

        z = _grid_kde(np.c_[xs, ys], weights, xg, yg, bw=None)

        media_dir = settings.MEDIA_ROOT / "sites"
        media_dir.mkdir(parents=True, exist_ok=True)
        out_path = media_dir / f"{kind}_{search_id}.png"

        fig, ax = plt.subplots(figsize=(8.0, 7.0))
        im = ax.imshow(z, origin="lower",
                       extent=(minx, maxx, miny, maxy),
                       interpolation="bilinear", alpha=0.85)
        if kind == "demand":
            im.set_cmap("YlOrRd")
            title = f"Demand heat — {business} @ {city}"
        else:
            im.set_cmap("GnBu")
            title = f"Competition heat — {business} @ {city}"

        cs = ax.contour(z, levels=6, linewidths=0.8, alpha=0.8,
                        extent=(minx, maxx, miny, maxy), colors="k")
        ax.clabel(cs, inline=True, fontsize=7, fmt="%.2f")

        for item in sorted(iso_list, key=lambda d: d["minutes"], reverse=True):
            feat = item.get("feature") or {}
            geom = feat.get("geometry") or {}
            if geom.get("type") != "Polygon":
                continue
            coords = geom.get("coordinates")[0]
            ring = np.array([(float(x[0]), float(x[1])) for x in coords])
            ax.plot(ring[:, 0], ring[:, 1], lw=1.2, alpha=0.9)

        ax.scatter([center[1]], [center[0]], s=55, zorder=5,
                   facecolor="#000", edgecolor="white", linewidth=1.0)

        ax.set_title(title, fontsize=13, pad=10)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel("relative intensity", rotation=90, labelpad=8)

        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

        return f"{settings.MEDIA_URL}sites/{out_path.name}"
    except Exception:
        return None


def _render_demand_map(center: Tuple[float, float], iso_list: List[dict],
                       city: str, search_id: str, business: str) -> Optional[str]:
    prov = get_provider()
    try:
        grid = prov.popgrid(city)  # cell_id, lat, lon, pop
    except Exception:
        return None
    if grid.empty:
        return None
    pts = grid[["lon", "lat", "pop"]].dropna().copy()
    points_xy = pts[["lon", "lat"]].to_numpy(dtype=float)
    weights = pts["pop"].to_numpy(dtype=float)
    return _render_density_map("demand", center, iso_list, points_xy, weights, search_id, city, business)


def _render_competition_map(center: Tuple[float, float], iso_list: List[dict],
                            city: str, business: str, search_id: str) -> Optional[str]:
    prov = get_provider()
    try:
        pois = prov.denue(city, business)  # name, lat, lon, category
    except Exception:
        return None
    if pois.empty:
        return None
    pts = pois[["lon", "lat"]].dropna().copy()
    points_xy = pts.to_numpy(dtype=float)
    return _render_density_map("competition", center, iso_list, points_xy, None, search_id, city, business)


# --------------------------- Artefacto GeoJSON por búsqueda ---------------------------

def _fc(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"type": "FeatureCollection", "features": features}

def _candidates_fc(center: Tuple[float,float],
                   top: List[Tuple[float,float,float,ScoreBreakdown,List[str],List[dict]]]) -> Dict[str, Any]:
    feats = []
    for idx, (lat, lon, score, br, _reasons, _iso) in enumerate(top, start=1):
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
            "properties": {
                "rank": idx,
                "score": float(score),
                "access": float(br.access),
                "demand": float(br.demand),
                "competition": float(br.competition),
                "is_center": abs(lat - center[0]) < 1e-9 and abs(lon - center[1]) < 1e-9,
            }
        })
    return _fc(feats)

def _isochrones_fc(iso_list: List[dict]) -> Dict[str, Any]:
    feats = []
    for item in iso_list:
        f = item["feature"]
        # ya tiene properties {"minutes","area_km2"}
        feats.append(f)
    return _fc(feats)

def _pois_fc(pois_df, poly: Polygon) -> Dict[str, Any]:
    feats = []
    if pois_df is None or pois_df.empty:
        return _fc(feats)
    count = 0
    for row in pois_df.itertuples(index=False):
        try:
            lon = float(row.lon); lat = float(row.lat)
            if not poly.contains(Point(lon, lat)):
                continue
            feats.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {
                    "name": getattr(row, "name", None),
                    "category": getattr(row, "category", None),
                }
            })
            count += 1
            if count >= int(os.getenv("MAX_POIS_GEOJSON", "1000")):
                break
        except Exception:
            continue
    return _fc(feats)

def _popgrid_fc(grid_df, poly: Polygon) -> Dict[str, Any]:
    feats = []
    if grid_df is None or grid_df.empty:
        return _fc(feats)
    # filtra dentro del polígono
    inside = []
    for row in grid_df.itertuples(index=False):
        try:
            lon = float(row.lon); lat = float(row.lat); pop = float(row.pop)
            if poly.contains(Point(lon, lat)):
                inside.append((lon, lat, pop))
        except Exception:
            continue
    if not inside:
        return _fc(feats)
    # ordena por población desc y limita
    inside.sort(key=lambda t: t[2], reverse=True)
    cap = int(os.getenv("MAX_POPGRID_GEOJSON", "800"))
    inside = inside[:cap]
    for lon, lat, pop in inside:
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {"pop": pop}
        })
    return _fc(feats)

def _save_run_artifact(search_id: str,
                       req: SiteSearchRequest,
                       chosen_center: Tuple[float,float],
                       top: List[Tuple[float,float,float,ScoreBreakdown,List[str],List[dict]]],
                       iso_list: List[dict]) -> Optional[str]:
    """
    Guarda un JSON con:
      - request_summary
      - candidates_fc
      - isochrones_fc
      - pois_competition_fc
      - popgrid_fc (muestra)
    """
    try:
        media_dir = settings.MEDIA_ROOT / "sites"
        media_dir.mkdir(parents=True, exist_ok=True)
        out_path = media_dir / f"run_{search_id}.json"

        # polígono mayor para recortes
        largest = max(iso_list, key=lambda x: x["minutes"]) if iso_list else None
        poly = _polygon_from_feature(largest["feature"]) if largest else None

        prov = get_provider()
        try:
            pois = prov.denue(req.city, req.business)
        except Exception:
            pois = None
        try:
            grid = prov.popgrid(req.city)
        except Exception:
            grid = None

        artifact = {
            "version": "sites-artifact-1",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "request": req.model_dump(),
            "center": {"lat": chosen_center[0], "lon": chosen_center[1]},
            "candidates_fc": _candidates_fc(chosen_center, top),
            "isochrones_fc": _isochrones_fc(iso_list),
            "pois_competition_fc": _pois_fc(pois, poly) if poly is not None else _fc([]),
            "popgrid_fc": _popgrid_fc(grid, poly) if poly is not None else _fc([]),
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(artifact, f, ensure_ascii=False)
        return str(out_path)
    except Exception:
        return None


# --------------------------- Estimador principal ---------------------------

def run_site_search(req: SiteSearchRequest) -> SiteSearchResponse:
    search_id = str(uuid.uuid4())
    warnings: List[str] = []
    candidates: List[CandidateSite] = []
    map_url: str | None = None
    demand_map_url: Optional[str] = None
    competition_map_url: Optional[str] = None

    w_access = float(os.getenv("WEIGHT_ACCESS", "0.35"))
    w_demand = float(os.getenv("WEIGHT_DEMAND", "0.40"))
    w_comp   = float(os.getenv("WEIGHT_COMP", "0.25"))

    if req.center:
        center = (float(req.center[0]), float(req.center[1]))
        base_iso = _build_isochrones(center, req.time_bands or [])
        largest = max(base_iso, key=lambda x: x["minutes"]) if base_iso else None
        poly = _polygon_from_feature(largest["feature"]) if largest else None

        if poly is None:
            access, access_r = _access_from_iso_list(base_iso)
            comp, comp_r = _competition_from_pois(req.city, req.business, base_iso)
            dem, dem_r = _demand_from_popgrid(req.city, base_iso)
            score = w_access * access + w_demand * dem + w_comp * comp
            score = float(max(0.0, min(1.0, score)))
            breakdown = ScoreBreakdown(demand=dem, competition=comp, access=access)
            reasons = (["Access from isochrone areas (normalized avg)"] + access_r +
                       ["Competition from POI density (largest band)"] + comp_r +
                       ["Demand from population grid (largest band)"] + dem_r)
            candidates.append(CandidateSite(lat=center[0], lon=center[1], score=score,
                                            breakdown=breakdown, reasons=reasons))
            map_url = _save_sites_map(center, base_iso, search_id, req.city, req.business,
                                      [(center[0], center[1], score)])
            warnings.append("sampling_fallback_invalid_polygon")
            demand_map_url = _render_demand_map(center, base_iso, req.city, search_id, req.business)
            competition_map_url = _render_competition_map(center, base_iso, req.city, req.business, search_id)
            # artefacto (solo center)
            _save_run_artifact(
                search_id, req, center,
                [(center[0], center[1], score, breakdown, reasons, base_iso)],
                base_iso
            )
        else:
            rng = random.Random(int(search_id.replace("-", ""), 16) & 0xFFFFFFFF)
            samples = _sample_points_in_polygon(poly, int(req.num_samples), rng)
            cand_points: List[Tuple[float, float]] = [center] + samples

            scored: List[Tuple[float, float, float, ScoreBreakdown, List[str], List[dict]]] = []
            for (lat, lon) in cand_points:
                iso_list = _build_isochrones((lat, lon), req.time_bands or [])
                access, access_r = _access_from_iso_list(iso_list)
                comp, comp_r = _competition_from_pois(req.city, req.business, iso_list)
                dem, dem_r = _demand_from_popgrid(req.city, iso_list)
                score = w_access * access + w_demand * dem + w_comp * comp
                score = float(max(0.0, min(1.0, score)))
                breakdown = ScoreBreakdown(demand=dem, competition=comp, access=access)
                reasons = (["Access from isochrone areas (normalized avg)"] + access_r +
                           ["Competition from POI density (largest band)"] + comp_r +
                           ["Demand from population grid (largest band)"] + dem_r)
                scored.append((lat, lon, score, breakdown, reasons, iso_list))

            scored.sort(key=lambda t: t[2], reverse=True)
            top = scored[: max(1, int(req.max_candidates))]

            for (lat, lon, score, breakdown, reasons, _iso) in top:
                candidates.append(CandidateSite(
                    lat=lat, lon=lon, score=score, breakdown=breakdown, reasons=reasons
                ))

            top1_iso = top[0][5]
            top_points = [(lat, lon, score) for (lat, lon, score, *_rest) in top]
            map_url = _save_sites_map((top[0][0], top[0][1]), top1_iso, search_id,
                                      req.city, req.business, top_points)
            warnings.append("multi_candidate_sampling_ok")

            demand_map_url = _render_demand_map((top[0][0], top[0][1]), top1_iso, req.city, search_id, req.business)
            competition_map_url = _render_competition_map((top[0][0], top[0][1]), top1_iso, req.city, req.business, search_id)

            if demand_map_url: warnings.append("demand_map_saved")
            else: warnings.append("demand_map_failed")
            if competition_map_url: warnings.append("competition_map_saved")
            else: warnings.append("competition_map_failed")

            # artefacto (Top-K + isócronas del Top-1)
            _save_run_artifact(search_id, req, (top[0][0], top[0][1]), top, top1_iso)
    else:
        neutral = ScoreBreakdown(demand=0.5, competition=0.5, access=0.5)
        for i in range(req.max_candidates):
            candidates.append(CandidateSite(
                lat=0.0, lon=0.0, score=0.5,
                breakdown=neutral,
                reasons=[f"stub candidate #{i+1} for {req.business} in {req.city}"],
            ))
        warnings.append("no_center_provided_stub_output")

    return SiteSearchResponse(
        search_id=search_id,
        city=req.city,
        business=req.business,
        time_bands=req.time_bands,
        candidates=candidates,
        map_url=map_url,
        demand_map_url=demand_map_url,
        competition_map_url=competition_map_url,
        data_release=req.data_release,
        warnings=warnings,
    )
