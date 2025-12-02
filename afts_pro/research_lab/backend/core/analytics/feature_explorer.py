"""Feature exploration utilities."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from research_lab.backend.core.analytics.models import SeriesStats


class FeatureExplorer:
    """Compute descriptive statistics for numeric series."""

    @staticmethod
    def compute_series_stats(series: Sequence[float], quantiles: Sequence[float] = (0.25, 0.5, 0.75)) -> SeriesStats:
        """Return descriptive statistics for the provided series.

        Args:
            series: Numeric sequence to analyze.
            quantiles: Quantile levels to compute (between 0 and 1).

        Returns:
            SeriesStats with mean, std, min, max, and quantiles.
        """

        if len(series) == 0:
            raise ValueError("Series must contain at least one value.")

        array = np.asarray(series, dtype=float)
        stats_quantiles = np.quantile(array, quantiles)
        quantile_map: Dict[str, float] = {f"{q}": float(val) for q, val in zip(quantiles, stats_quantiles)}
        return SeriesStats(
            mean=float(np.mean(array)),
            std=float(np.std(array)),
            min=float(np.min(array)),
            max=float(np.max(array)),
            quantiles=quantile_map,
        )

    @staticmethod
    def compute_histogram(series: Sequence[float], bins: int = 20) -> Dict[str, list[float]]:
        """Compute a histogram representation of the series.

        Args:
            series: Numeric sequence to analyze.
            bins: Number of histogram bins.

        Returns:
            Mapping with bin_edges and counts for downstream visualization.
        """

        if len(series) == 0:
            raise ValueError("Series must contain at least one value.")

        counts, bin_edges = np.histogram(series, bins=bins)
        return {"bin_edges": bin_edges.tolist(), "counts": counts.astype(int).tolist()}


__all__ = ["FeatureExplorer"]
