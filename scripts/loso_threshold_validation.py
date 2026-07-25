#!/usr/bin/env python3
"""Audit whether LOSO object predictions recover sample-level expert Vmin."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

import cross_validation as cv
from src.ml_threshold_selection.prediction_analysis import compute_dual_thresholds


def expert_threshold_units(threshold_mm3: float, voxel_size_mm: float) -> tuple[float, int]:
    """Return the continuous voxel equivalent and integer retention boundary."""
    equivalent_voxels = float(threshold_mm3 / voxel_size_mm**3)
    return equivalent_voxels, int(np.ceil(equivalent_voxels))


def candidate_threshold_units(
    continuous_voxels: float, voxel_size_mm: float
) -> tuple[int, float]:
    """Return the applied integer voxel threshold and its physical volume."""
    applied_voxels = int(np.ceil(continuous_voxels))
    return applied_voxels, float(applied_voxels * voxel_size_mm**3)


def project_decisions_to_vmin(voxel_counts: np.ndarray, decisions: np.ndarray) -> tuple[float, float]:
    """Return scalar Vmin minimizing disagreement with binary object decisions."""
    order = np.argsort(voxel_counts)
    volumes = voxel_counts[order]
    labels = decisions[order].astype(int)
    prefix_below = np.r_[0, np.cumsum(labels)]
    k = np.arange(len(labels) + 1)
    errors = (k - prefix_below) + (prefix_below[-1] - prefix_below)
    best = int(np.argmin(errors))

    if best == 0:
        threshold = float(np.nextafter(volumes[0], -np.inf))
    elif best == len(volumes):
        threshold = float(np.nextafter(volumes[-1], np.inf))
    else:
        threshold = float((volumes[best - 1] + volumes[best]) / 2.0)
    return threshold, float(errors[best] / len(labels))


def run(repo: Path) -> pd.DataFrame:
    config = cv.load_config(repo / "examples" / "expert_thresholds.csv")
    samples = cv.load_samples(repo / "training_data", config)
    prototype, _ = cv.make_classifier()
    rows = []

    for held_out in samples:
        training_ids = [sample_id for sample_id in samples if sample_id != held_out]
        x_train = np.vstack([samples[sample_id]["X"] for sample_id in training_ids])
        y_train = np.concatenate([samples[sample_id]["y"] for sample_id in training_ids])
        scaler = StandardScaler().fit(x_train)
        model = clone(prototype).fit(
            pd.DataFrame(scaler.transform(x_train), columns=cv.FEATURE_NAMES), y_train
        )

        x_test = samples[held_out]["X"]
        y_test = samples[held_out]["y"]
        probability = model.predict_proba(
            pd.DataFrame(scaler.transform(x_test), columns=cv.FEATURE_NAMES)
        )[:, 1]
        decision = probability >= 0.5
        voxel_count = x_test[:, 0]
        projected_vmin, projection_disagreement = project_decisions_to_vmin(
            voxel_count, decision
        )
        loose_vmin, strict_vmin = compute_dual_thresholds(voxel_count, probability, 0.01)
        expert_threshold_mm3 = float(config[held_out]["thr"])
        voxel_size_mm = float(config[held_out]["vox"])
        expert_voxel_equivalent, expert_integer_boundary = expert_threshold_units(
            expert_threshold_mm3, voxel_size_mm
        )
        projected_applied_vox, projected_mm3 = candidate_threshold_units(
            projected_vmin, voxel_size_mm
        )
        loose_applied_vox, loose_mm3 = candidate_threshold_units(
            loose_vmin, voxel_size_mm
        )
        strict_applied_vox, strict_mm3 = candidate_threshold_units(
            strict_vmin, voxel_size_mm
        )
        tn, fp, fn, tp = confusion_matrix(y_test, decision, labels=[0, 1]).ravel()

        rows.append(
            {
                "Sample": held_out,
                "n": len(y_test),
                "BelowThresholdPrevalence": y_test.mean(),
                "AUC": roc_auc_score(y_test, probability),
                "AccuracyAtP0.5": accuracy_score(y_test, decision),
                "BalancedAccuracyAtP0.5": balanced_accuracy_score(y_test, decision),
                "BrierScore": brier_score_loss(y_test, probability),
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "TP": tp,
                "ExpertThreshold_mm3": expert_threshold_mm3,
                "VoxelSize_mm": voxel_size_mm,
                "ExpertEquivalentVoxelCount": expert_voxel_equivalent,
                "ExpertMinimumRetainedIntegerVoxelCount": expert_integer_boundary,
                "ProjectedVmin_continuous_vox": projected_vmin,
                "ProjectedVmin_applied_vox": projected_applied_vox,
                "ProjectedVmin_mm3": projected_mm3,
                "ProjectedRelativeError": projected_mm3 / expert_threshold_mm3 - 1.0,
                "ProjectionDisagreement": projection_disagreement,
                "LooseVmin_continuous_vox": loose_vmin,
                "LooseVmin_applied_vox": loose_applied_vox,
                "LooseVmin_mm3": loose_mm3,
                "LooseRelativeError": loose_mm3 / expert_threshold_mm3 - 1.0,
                "StrictVmin_continuous_vox": strict_vmin,
                "StrictVmin_applied_vox": strict_applied_vox,
                "StrictVmin_mm3": strict_mm3,
                "StrictRelativeError": strict_mm3 / expert_threshold_mm3 - 1.0,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    repository = Path(__file__).resolve().parents[1]
    result = run(repository)
    output = repository / "outputs" / "loso_threshold_validation.csv"
    output.parent.mkdir(exist_ok=True)
    result.to_csv(output, index=False)
    with pd.option_context("display.max_columns", None, "display.width", 240):
        print(result.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(
        "Median absolute relative errors in physical threshold volume: "
        f"projected={np.median(np.abs(result['ProjectedRelativeError'])):.1%}, "
        f"loose={np.median(np.abs(result['LooseRelativeError'])):.1%}, "
        f"strict={np.median(np.abs(result['StrictRelativeError'])):.1%}"
    )
    print(f"Wrote {output}")
