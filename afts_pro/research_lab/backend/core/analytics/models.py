"""Shared analytics domain models."""

from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field


class SeriesStats(BaseModel):
    """Descriptive statistics for a numeric series."""

    mean: float
    std: float
    min: float
    max: float
    quantiles: Dict[str, float] = Field(default_factory=dict)


class RollingKpiWindow(BaseModel):
    """KPIs computed for a single rolling window of returns."""

    start_index: int
    end_index: int
    profit_factor: float
    win_rate: float
    avg_win: float
    avg_loss: float
    max_drawdown: float


class DriftResult(BaseModel):
    """Result of a simple drift detection test."""

    score: float
    threshold: float
    drift_detected: bool


class RegimeClusteringResult(BaseModel):
    """Result of a regime clustering run."""

    n_clusters: int
    labels: List[int]


__all__ = ["SeriesStats", "RollingKpiWindow", "DriftResult", "RegimeClusteringResult"]
