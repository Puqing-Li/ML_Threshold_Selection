#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Label generation based on expert thresholds in physical volume units."""

from __future__ import annotations

import math
from typing import Callable, Dict, List

import pandas as pd

from ml_threshold_selection.voxel_config import require_voxel_sizes


def generate_labels_from_thresholds(
    training_data: pd.DataFrame,
    expert_thresholds: Dict[str, float],
    voxel_sizes: Dict[str, float],
    sample_list: List[str],
    log: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    if 'SampleID' not in training_data.columns:
        raise ValueError('SampleID column missing in training data')

    sample_ids = training_data['SampleID'].astype(str).unique().tolist()
    require_voxel_sizes(voxel_sizes, sample_ids)

    valid_thresholds = {}
    for sample_id, threshold_mm3 in expert_thresholds.items():
        try:
            threshold_mm3 = float(threshold_mm3)
        except (TypeError, ValueError):
            continue
        if math.isfinite(threshold_mm3) and threshold_mm3 > 0:
            valid_thresholds[str(sample_id)] = threshold_mm3

    missing_thresholds = [
        sample_id for sample_id in sample_ids
        if sample_id not in valid_thresholds
    ]
    if missing_thresholds:
        raise ValueError(
            'Missing valid expert thresholds (mm^3) for samples: '
            + ', '.join(missing_thresholds)
        )

    labels = []
    for _, row in training_data.iterrows():
        sample_id = str(row.get('SampleID'))
        volume_mm3 = row.get('Volume3d (mm^3) ', None)
        if volume_mm3 is None:
            raise ValueError(f'Missing physical volume for sample {sample_id}')
        try:
            volume_mm3 = float(volume_mm3)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'Invalid physical volume for sample {sample_id}') from exc
        if not math.isfinite(volume_mm3) or volume_mm3 <= 0:
            raise ValueError(f'Invalid physical volume for sample {sample_id}')
        label = 1 if volume_mm3 < valid_thresholds[sample_id] else 0
        labels.append(label)

    training_data = training_data.copy()
    training_data['label'] = labels
    label_counts = training_data['label'].value_counts().to_dict()
    if log:
        log("Label distribution:")
        log(f"   - Retained class (0): {label_counts.get(0, 0)}")
        log(f"   - Below-threshold class (1): {label_counts.get(1, 0)}")
        log(f"   - Unknown (-1): {label_counts.get(-1, 0)}")
    return training_data
