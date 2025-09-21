from __future__ import annotations
from typing import List, Optional, Dict, Tuple
from pydantic import BaseModel, Field, confloat
from ..version import SPEC_VERSION


class ScoreBreakdown(BaseModel):
    demand: confloat(ge=0, le=1) = Field(..., description="Normalized demand component (0–1)")
    competition: confloat(ge=0, le=1) = Field(..., description="Competition penalty (0–1, higher = better after penalty)")
    access: confloat(ge=0, le=1) = Field(..., description="Accessibility component (0–1)")


class CandidateSite(BaseModel):
    lat: float = Field(..., description="Latitude (WGS84)")
    lon: float = Field(..., description="Longitude (WGS84)")
    score: confloat(ge=0, le=1) = Field(..., description="Final normalized score (0–1)")
    breakdown: Optional[ScoreBreakdown] = Field(None, description="Score components")
    reasons: Optional[List[str]] = Field(None, description="Human-readable justifications")
    address: Optional[str] = Field(None, description="Optional address or label")
    grid_id: Optional[str] = Field(None, description="Optional grid/AGEB/cell identifier")


class SiteSearchRequest(BaseModel):
    city: str = Field(..., description="City id/name (e.g., CDMX)")
    business: str = Field(..., description="Business type (e.g., cafe, farmacia)")
    time_bands: List[int] = Field(..., description="Isochrone minutes, e.g., [10,20,30]")
    max_candidates: int = Field(3, description="How many top sites to return")
    data_release: Optional[str] = Field(None, description="Data snapshot id (e.g., denue_2024q4)")
    center: Optional[Tuple[float, float]] = Field(
        None, description="Optional center [lat, lon] for access calculations"
    )
    num_samples: int = Field(
        12, ge=1, le=50,
        description="How many candidate points to sample when center is provided"
    )



class SiteSearchResponse(BaseModel):
    model_id: str = Field("site-score-v0.1.0", description="Model identifier")
    spec_version: str = Field(SPEC_VERSION, description="Contracts spec version")
    search_id: str = Field(..., description="UUID for this search")
    city: str = Field(..., description="Echoed city")
    business: str = Field(..., description="Echoed business type")
    time_bands: List[int] = Field(..., description="Echoed time bands")
    candidates: List[CandidateSite] = Field(..., description="Ranked list of sites")

    # Mapas
    map_url: Optional[str] = Field(None, description="Main map (isochrones + Top-K)")
    demand_map_url: Optional[str] = Field(None, description="Demand heat-style map (PNG)")
    competition_map_url: Optional[str] = Field(None, description="Competition heat-style map (PNG)")

    data_release: Optional[str] = Field(None, description="Data snapshot used")
    warnings: Optional[List[str]] = Field(None, description="Any non-fatal warnings")

