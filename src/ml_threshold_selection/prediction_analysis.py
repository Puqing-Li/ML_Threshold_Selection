#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prediction analysis helpers: thresholds and curves.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Sequence


def find_inflection_threshold(thresholds: np.ndarray, retained_mean_probabilities: Sequence[float]):
    try:
        if len(retained_mean_probabilities) < 3:
            return None
        from scipy.ndimage import gaussian_filter1d
        thresholds = np.asarray(thresholds, dtype=float)
        if np.any(thresholds <= 0) or not np.all(np.diff(thresholds) > 0):
            raise ValueError('thresholds must be positive and strictly increasing')
        smoothed = gaussian_filter1d(np.asarray(retained_mean_probabilities, dtype=float), sigma=1.0)
        log_thresholds = np.log10(thresholds)
        first = np.gradient(smoothed, log_thresholds)
        second = np.gradient(first, log_thresholds)
        idx = int(np.argmax(second))
        if 0 < idx < len(thresholds) - 1:
            return float(thresholds[idx])
        return None
    except Exception:
        return None


def compute_dual_thresholds(voxels_cont: np.ndarray, probabilities: np.ndarray, strict_probability_threshold: float = 0.01) -> Tuple[float | None, float | None]:
    """
    Compute the manuscript's dual operating thresholds for object filtering.
    
    Parameters:
    -----------
    voxels_cont : np.ndarray
        Continuous voxel volumes
    probabilities : np.ndarray
        Predicted probabilities of the expert-defined below-threshold class
    strict_probability_threshold : float, default=0.01
        Probability threshold for strict filtering (P > threshold)
        
    Returns:
    --------
    Tuple[float | None, float | None]
        (inflection_threshold, strict_threshold)
    """
    voxels_cont = np.asarray(voxels_cont, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    if voxels_cont.shape != probabilities.shape or voxels_cont.size == 0:
        raise ValueError('voxels_cont and probabilities must be non-empty arrays with matching shapes')
    if not np.all(np.isfinite(voxels_cont)) or np.any(voxels_cont <= 0):
        raise ValueError('voxels_cont must be finite and strictly positive')
    if not np.all(np.isfinite(probabilities)):
        raise ValueError('probabilities must be finite')
    thresholds = np.logspace(
        np.log10(max(voxels_cont.min(), 1e-12)),
        np.log10(voxels_cont.max()),
        50,
    )
    retained_mean_probabilities = []
    for t in thresholds:
        retained = voxels_cont >= t
        retained_mean_probabilities.append(
            float(np.mean(probabilities[retained])) if np.sum(retained) > 0 else 0.0
        )
    inflection = find_inflection_threshold(thresholds, retained_mean_probabilities)
    strict_mask = probabilities > strict_probability_threshold
    if np.any(strict_mask):
        strict = float(np.floor(np.max(voxels_cont[strict_mask])) + 1.0)
    else:
        strict = float(np.ceil(np.min(voxels_cont)))
    if inflection is not None and strict is not None and strict < inflection:
        strict = inflection
    return inflection, strict
