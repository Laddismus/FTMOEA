from collections import Counter

import numpy as np

from research_lab.backend.core.analytics.regime_clustering import RegimeClusteringService


def test_regime_clustering_creates_two_clusters() -> None:
    cluster_a = np.random.normal(loc=0.0, scale=0.1, size=(20, 2))
    cluster_b = np.random.normal(loc=10.0, scale=0.1, size=(20, 2))
    features = np.vstack([cluster_a, cluster_b]).tolist()

    service = RegimeClusteringService(random_state=42)
    result = service.cluster_regimes(features, n_clusters=2)

    assert len(result.labels) == len(features)
    counts = Counter(result.labels)
    assert len(counts) == 2
    # Majority of first half should belong to one cluster, second half to the other.
    first_half_label = Counter(result.labels[: len(features) // 2]).most_common(1)[0][0]
    second_half_label = Counter(result.labels[len(features) // 2 :]).most_common(1)[0][0]
    assert first_half_label != second_half_label
