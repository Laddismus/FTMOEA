"""Analytics endpoints for the Research Lab backend."""

from typing import List, Sequence

from fastapi import APIRouter
from pydantic import BaseModel, Field

from research_lab.backend.core.analytics import (
    DriftDetector,
    FeatureExplorer,
    RegimeClusteringResult,
    RegimeClusteringService,
    RollingKpiEngine,
    RollingKpiWindow,
    SeriesStats,
    DriftResult,
)

router = APIRouter(prefix="/analytics", tags=["analytics"])

_feature_explorer = FeatureExplorer()
_kpi_engine = RollingKpiEngine()
_drift_detector = DriftDetector()
_regime_service = RegimeClusteringService()


class SeriesInput(BaseModel):
    series: List[float]


class KpiRequest(BaseModel):
    returns: List[float]
    window: int


class RollingKpiResponse(BaseModel):
    windows: List[RollingKpiWindow]


class DriftRequest(BaseModel):
    base_window: List[float]
    current_window: List[float]
    threshold: float | None = Field(default=3.0)


class RegimeRequest(BaseModel):
    features: List[List[float]]
    n_clusters: int


@router.post("/stats", response_model=SeriesStats)
def compute_stats(payload: SeriesInput) -> SeriesStats:
    """Compute descriptive statistics for a series."""

    return _feature_explorer.compute_series_stats(payload.series)


@router.post("/kpis", response_model=RollingKpiResponse)
def compute_kpis(payload: KpiRequest) -> RollingKpiResponse:
    """Compute rolling KPIs for returns."""

    windows = _kpi_engine.compute_rolling_kpis(payload.returns, payload.window)
    return RollingKpiResponse(windows=windows)


@router.post("/drift", response_model=DriftResult)
def detect_drift(payload: DriftRequest) -> DriftResult:
    """Detect drift between two windows."""

    threshold = payload.threshold if payload.threshold is not None else 3.0
    return _drift_detector.detect_drift(payload.base_window, payload.current_window, threshold)


@router.post("/regimes", response_model=RegimeClusteringResult)
def cluster_regimes(payload: RegimeRequest) -> RegimeClusteringResult:
    """Cluster regimes using KMeans."""

    return _regime_service.cluster_regimes(payload.features, payload.n_clusters)


__all__ = ["router"]
