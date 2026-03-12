#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Threshold generation helpers for fabric analysis.
"""

from __future__ import annotations

import numpy as np
from typing import List


def generate_logstep_thresholds(
    volumes: np.ndarray,
    v_loose: float,
    v_strict: float,
    min_particles: int = 50,
    log10_step: float = 0.25,
) -> List[float]:
    """Generate thresholds starting from global min volume with fixed log10 step.

    - Include loose/strict thresholds and return a sorted unique list.
    - Stop extending when retained particles fall below min_particles.
    """
    factor = float(10 ** log10_step)
    v_min_all = float(np.min(volumes))
    if not np.isfinite(v_min_all):
        v_min_all = 0.0

    seq = []
    v_curr = v_min_all
    while True:
        N = int(np.sum(volumes >= v_curr))
        if N < min_particles:
            break
        seq.append(float(v_curr))
        v_next = v_curr * factor
        if v_next <= v_curr * (1 + 1e-12):
            break
        v_curr = v_next

    v_set = set(seq)
    v_set.add(float(v_loose))
    v_set.add(float(v_strict))
    return sorted(v_set)


