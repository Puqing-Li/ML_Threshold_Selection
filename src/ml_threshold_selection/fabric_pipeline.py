#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline orchestrator for fabric analysis boxplots.
"""

from __future__ import annotations

import os
import time
import numpy as np
import pandas as pd
from typing import Dict, Tuple

from .fabric_thresholds import generate_logstep_thresholds
from .fabric_bootstrap import build_spinel_block, precompute_logE_block, bootstrap_tp_samples
from .fabric_boxplots_dual_thresholds import plot_param_boxplot_by_volume_thresholds


def run_fabric_boxplots(
    df: pd.DataFrame,
    voxel_size_mm: float,
    loose_threshold_vox: int,
    strict_threshold_vox: int,
    logger,
    outputs_dir: str = 'outputs',
    n_bootstrap: int = 1000,
    min_particles: int = 50,
) -> Tuple[str, str]:
    t_start_all = time.time()

    # Preconditions are assumed validated by caller
    voxel_vol = float(voxel_size_mm) ** 3
    volumes = df['Volume3d (mm^3) '].astype(float).values
    v_loose = float(loose_threshold_vox) * voxel_vol
    v_strict = float(strict_threshold_vox) * voxel_vol

    thresholds_mm3 = generate_logstep_thresholds(volumes, v_loose, v_strict, min_particles=min_particles, log10_step=0.25)

    # Prepare tensors
    spinel_block_full = build_spinel_block(df)
    logE_stack = precompute_logE_block(spinel_block_full)

    # Bootstrap across thresholds
    bootstrap_T: Dict[float, list] = {}
    bootstrap_P: Dict[float, list] = {}
    particle_counts: Dict[float, int] = {}

    for vt in thresholds_mm3:
        mask = volumes >= vt
        retained_idx = np.flatnonzero(mask)
        N = int(retained_idx.size)
        particle_counts[vt] = N
        logger.info(f"🔹 Threshold ≥ {vt:.6g} mm³ | N={N}")
        if N < min_particles:
            bootstrap_T[vt] = []
            bootstrap_P[vt] = []
            logger.info("   ↳ skipped: insufficient particles")
            continue
        t0 = time.time()
        logE_retained = logE_stack[retained_idx]
        t_samples, p_samples = bootstrap_tp_samples(logE_retained, n_bootstrap)
        bootstrap_T[vt] = t_samples
        bootstrap_P[vt] = p_samples
        logger.info(f"   ↳ valid T samples: {len(t_samples)}, valid P' samples: {len(p_samples)}, elapsed {time.time()-t0:.1f}s")

    os.makedirs(outputs_dir, exist_ok=True)
    t_path = os.path.join(outputs_dir, 'Fabric_T_boxplot.png')
    p_path = os.path.join(outputs_dir, 'Fabric_Pprime_boxplot.png')

    plot_param_boxplot_by_volume_thresholds(
        bootstrap_T,
        param='T',
        inflection_threshold=v_loose,
        zero_artifact_threshold=v_strict,
        particle_counts=particle_counts,
        title='T Parameter Across Volume Thresholds',
        save_path=t_path,
        show=False,
    )
    plot_param_boxplot_by_volume_thresholds(
        bootstrap_P,
        param="P'",
        inflection_threshold=v_loose,
        zero_artifact_threshold=v_strict,
        particle_counts=particle_counts,
        title="P' Parameter Across Volume Thresholds",
        save_path=p_path,
        show=False,
    )

    logger.info("✅ Fabric boxplots generated")
    logger.info(f"   - T boxplot: {t_path}")
    logger.info(f"   - P' boxplot: {p_path}")
    logger.info(f"⏱️ Total time: {time.time()-t_start_all:.1f}s")

    return t_path, p_path
