"""Analytics core modules for the Research Lab backend."""

from research_lab.backend.core.analytics.feature_explorer import FeatureExplorer
from research_lab.backend.core.analytics.kpi_engine import RollingKpiEngine
from research_lab.backend.core.analytics.drift_detector import DriftDetector
from research_lab.backend.core.analytics.regime_clustering import RegimeClusteringService
from research_lab.backend.core.analytics.models import (
    SeriesStats,
    RollingKpiWindow,
    DriftResult,
    RegimeClusteringResult,
)

__all__ = [
    "FeatureExplorer",
    "RollingKpiEngine",
    "DriftDetector",
    "RegimeClusteringService",
    "SeriesStats",
    "RollingKpiWindow",
    "DriftResult",
    "RegimeClusteringResult",
]
