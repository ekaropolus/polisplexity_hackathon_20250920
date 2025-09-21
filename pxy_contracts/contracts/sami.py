from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field
from ..version import SPEC_VERSION


class SAMICity(BaseModel):
    """Per-city SAMI score (size-adjusted residual)."""
    city: str = Field(..., description="City name or code")
    sami: float = Field(..., description="Size-adjusted residual (z-like score)")
    rank: Optional[int] = Field(None, description="Rank among requested cities")


class SAMIPoint(BaseModel):
    """Raw point used in the fit (optionally with logs for client plots)."""
    city: str = Field(..., description="City name or code")
    value: float = Field(..., description="Indicator value (Y)")
    N: float = Field(..., description="Scale variable, typically population (N)")
    log_value: Optional[float] = Field(None, description="log(value) if computed server-side")
    log_N: Optional[float] = Field(None, description="log(N) if computed server-side")


class SAMIRunRequest(BaseModel):
    """Request to run SAMI for an indicator over a set of cities."""
    cities: List[str] = Field(..., description="Cities to evaluate")
    indicator: str = Field(..., description="Indicator id, e.g., imss_wages_2023")
    data_release: Optional[str] = Field(None, description="Data snapshot id, e.g., inegi_sun_2020_r1")


class SAMIRunResponse(BaseModel):
    """SAMI run output (fit metrics, per-city scores, and optional assets)."""
    model_id: str = Field("sami-ols-v2.0.0", description="Model identifier")
    spec_version: str = Field(SPEC_VERSION, description="Contracts spec version")
    run_id: str = Field(..., description="UUID for this run")
    indicator: str = Field(..., description="Indicator id echoed back")
    beta: float = Field(..., description="Scaling exponent β")
    r2: float = Field(..., description="Coefficient of determination")
    residuals: List[SAMICity] = Field(..., description="Per-city SAMI results")
    chart_url: Optional[str] = Field(None, description="PNG/SVG chart URL if available")
    data_release: Optional[str] = Field(None, description="Data snapshot used")
    warnings: Optional[List[str]] = Field(None, description="Any non-fatal warnings")

    # 56B additions (optional for backward compatibility)
    alpha: Optional[float] = Field(None, description="Intercept α of log–log OLS")
    points: Optional[List[SAMIPoint]] = Field(
        None,
        description="Raw per-city points (value, N, logs) used in the fit",
    )
