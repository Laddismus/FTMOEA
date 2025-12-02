import math

import pytest

from research_lab.backend.core.analytics.drift_detector import DriftDetector


def test_drift_detector_detects_large_shift() -> None:
    detector = DriftDetector()
    base = [1.0, 1.1, 0.9, 1.0, 1.05]
    current = [5.0, 5.1, 4.9, 5.05]

    result = detector.detect_drift(base, current, threshold=3.0)

    assert result.drift_detected is True
    assert result.score >= result.threshold


def test_drift_detector_handles_small_variation_and_low_std() -> None:
    detector = DriftDetector()
    base = [1.0, 1.0, 1.0, 1.0]
    current = [1.0, 1.0, 1.0000001]

    result = detector.detect_drift(base, current, threshold=3.0)

    assert result.drift_detected is False
    assert math.isfinite(result.score)
