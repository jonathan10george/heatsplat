"""Request and response models for the Heatsplat API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SplatMode = Literal["rgb", "thermal", "leaks"]


class AnalyzeRequest(BaseModel):
    """Detection thresholds the viewer can tune live. All optional; defaults match the pipeline."""

    min_delta_c: float = Field(1.5, ge=0.1, le=20.0, allow_inf_nan=False, description="Minimum warmth above local baseline")
    min_zscore: float = Field(2.5, ge=0.0, le=20.0, allow_inf_nan=False)
    cluster_eps_m: float = Field(0.2, ge=0.02, le=2.0, allow_inf_nan=False)
    cluster_min_points: int = Field(12, ge=3, le=1000)
    sign: Literal[1, -1] = Field(1, description="+1 warm anomalies (exterior), -1 cold anomalies (interior)")
    price_per_kwh: float = Field(0.15, ge=0.0, le=5.0, allow_inf_nan=False)
    heating_hours_per_year: float = Field(2500.0, ge=0.0, le=8760.0, allow_inf_nan=False)


class SceneSummaryOut(BaseModel):
    scene_id: str
    name: str
    n_gaussians: int
    temp_min_c: float
    temp_max_c: float
    n_leaks: int
    total_cost_per_year: float
    total_watts: float
    currency: str


class ReportOut(BaseModel):
    scene_id: str
    generator: str
    markdown: str
