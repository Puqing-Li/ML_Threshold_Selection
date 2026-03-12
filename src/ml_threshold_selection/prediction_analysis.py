#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prediction analysis helpers: thresholds and curves.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Sequence


def find_inflection_threshold(thresholds: np.ndarray, artifact_rates: Sequence[float]):
    try:
        if len(artifact_rates) < 3:
            return None
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(np.asarray(artifact_rates, dtype=float), sigma=1.0)
        second = np.gradient(np.gradient(smoothed))
        idx = int(np.argmax(second))
        if 0 < idx < len(thresholds) - 1:
            return float(thresholds[idx])
        for i in range(1, len(artifact_rates)):
            if artifact_rates[i] - artifact_rates[i - 1] > 0.1:
                return float(thresholds[i])
        return None
    except Exception:
        return None


def compute_dual_thresholds(voxels_cont: np.ndarray, probabilities: np.ndarray, strict_probability_threshold: float = 0.01) -> Tuple[float | None, float | None]:
    """
    Compute dual thresholds for particle filtering.
    
    Parameters:
    -----------
    voxels_cont : np.ndarray
        Continuous voxel volumes
    probabilities : np.ndarray
        Predicted artifact probabilities
    strict_probability_threshold : float, default=0.01
        Probability threshold for strict filtering (P > threshold)
        
    Returns:
    --------
    Tuple[float | None, float | None]
        (inflection_threshold, strict_threshold)
    """
    thresholds = np.logspace(
        np.log10(max(voxels_cont.min(), 1e-12)),
        np.log10(voxels_cont.max()),
        50,
    )
    artifact_rates = []
    for t in thresholds:
        retained = voxels_cont >= t
        artifact_rates.append(float(np.mean(probabilities[retained])) if np.sum(retained) > 0 else 0.0)
    inflection = find_inflection_threshold(thresholds, artifact_rates)
    strict_mask = probabilities > strict_probability_threshold
    strict = float(np.max(voxels_cont[strict_mask])) if np.any(strict_mask) else None
    if inflection is not None and strict is not None and strict < inflection:
        strict = inflection
    return inflection, strict


