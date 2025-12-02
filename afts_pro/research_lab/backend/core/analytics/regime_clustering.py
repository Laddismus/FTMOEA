"""Simple regime clustering service using KMeans."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.cluster import KMeans

from research_lab.backend.core.analytics.models import RegimeClusteringResult


class RegimeClusteringService:
    """Cluster feature vectors into regimes."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    def cluster_regimes(self, features: Sequence[Sequence[float]], n_clusters: int) -> RegimeClusteringResult:
        """Cluster feature vectors using KMeans.

        Args:
            features: 2D collection of feature vectors.
            n_clusters: Number of clusters to form.

        Returns:
            RegimeClusteringResult with cluster count and labels.
        """

        if n_clusters <= 0:
            raise ValueError("n_clusters must be positive.")

        X = np.asarray(features, dtype=float)
        if X.ndim != 2:
            raise ValueError("features must be a 2D collection.")
        if X.shape[0] < n_clusters:
            raise ValueError("number of samples must be >= n_clusters.")

        model = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        labels = model.fit_predict(X)
        return RegimeClusteringResult(n_clusters=n_clusters, labels=labels.tolist())


__all__ = ["RegimeClusteringService"]
