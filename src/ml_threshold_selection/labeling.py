#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Label generation based on expert thresholds (voxel-domain).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Callable


def generate_labels_from_thresholds(
    training_data: pd.DataFrame,
    expert_thresholds: Dict[str, float],
    voxel_sizes: Dict[str, float],
    sample_list: List[str],
    log: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    if 'SampleID' not in training_data.columns:
        if log: log("❌ SampleID column missing in training data")
        return training_data
    if not voxel_sizes:
        if log: log("❌ Please input voxel sizes first (mm)")
        return training_data

    sample_threshold_vox = {}
    for sample_id, t_abs in expert_thresholds.items():
        if sample_id in voxel_sizes:
            try:
                voxel_mm = float(voxel_sizes[sample_id])
                voxel_vol = voxel_mm ** 3
                sample_threshold_vox[sample_id] = int(np.ceil(float(t_abs) / voxel_vol))
            except Exception:
                continue

    labels = []
    for _, row in training_data.iterrows():
        sample_id = row.get('SampleID')
        volume_mm3 = row.get('Volume3d (mm^3) ', None)
        if sample_id in sample_threshold_vox and volume_mm3 is not None and sample_id in voxel_sizes:
            voxel_mm = float(voxel_sizes[sample_id])
            voxel_vol = voxel_mm ** 3
            v_vox = float(volume_mm3) / voxel_vol
            t_vox = sample_threshold_vox[sample_id]
            label = 1 if v_vox < t_vox else 0
        else:
            label = -1
        labels.append(label)

    training_data = training_data.copy()
    training_data['label'] = labels
    label_counts = training_data['label'].value_counts().to_dict()
    if log:
        log("📊 Label distribution:")
        log(f"   - Normal (0): {label_counts.get(0, 0)}")
        log(f"   - Artifact (1): {label_counts.get(1, 0)}")
        log(f"   - Unknown (-1): {label_counts.get(-1, 0)}")
    if -1 in label_counts and label_counts[-1] > 0:
        training_data = training_data[training_data['label'] != -1].copy()
        if log: log(f"⚠️ Removed {label_counts[-1]} particles without expert thresholds")
    return training_data


