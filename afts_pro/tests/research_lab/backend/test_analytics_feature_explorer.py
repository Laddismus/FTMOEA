import pytest

from research_lab.backend.core.analytics.feature_explorer import FeatureExplorer


def test_compute_series_stats_basic() -> None:
    series = [1.0, 2.0, 3.0, 4.0]

    stats = FeatureExplorer.compute_series_stats(series)

    assert stats.mean == pytest.approx(2.5)
    assert stats.std == pytest.approx(1.11803, rel=1e-4)
    assert stats.min == 1.0
    assert stats.max == 4.0
    assert stats.quantiles["0.25"] == pytest.approx(1.75)
    assert stats.quantiles["0.5"] == pytest.approx(2.5)
    assert stats.quantiles["0.75"] == pytest.approx(3.25)
