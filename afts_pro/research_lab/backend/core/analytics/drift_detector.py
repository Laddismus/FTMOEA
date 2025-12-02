"""Drift detection utilities."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from research_lab.backend.core.analytics.models import DriftResult


class DriftDetector:
    """Perform simple drift detection using z-score comparison."""

    def detect_drift(self, base_window: Sequence[float], current_window: Sequence[float], threshold: float = 3.0) -> DriftResult:
        """Compare two windows and flag drift if the z-score exceeds the threshold.

        Args:
            base_window: Historical reference window.
            current_window: Current window to compare.
            threshold: Z-score threshold for drift detection.

        Returns:
            DriftResult containing score, threshold, and detection flag.
        """

        if len(base_window) == 0 or len(current_window) == 0:
            raise ValueError("Windows must not be empty.")

        base_arr = np.asarray(base_window, dtype=float)
        current_arr = np.asarray(current_window, dtype=float)

        base_mean = float(np.mean(base_arr))
        base_std = float(np.std(base_arr))
        current_mean = float(np.mean(current_arr))

        if math.isclose(base_std, 0.0, abs_tol=1e-8):
            diff = abs(current_mean - base_mean)
            if math.isclose(diff, 0.0, abs_tol=1e-8):
                z_score = 0.0
            else:
                z_score = diff / 1e-6
        else:
            z_score = abs(current_mean - base_mean) / base_std

        drift_detected = z_score >= threshold
        return DriftResult(score=z_score, threshold=threshold, drift_detected=drift_detected)


__all__ = ["DriftDetector"]
