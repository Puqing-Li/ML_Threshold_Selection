#!/usr/bin/env python3
"""Audit scan-axis locking around each configured expert threshold."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from axis_locking_validation import _axis_distance_degrees, _summary
from cross_validation import VOL, _read_table, load_config
from loso_threshold_validation import run as run_loso_validation


def run(repo: Path, angle_degrees: float = 5.0) -> pd.DataFrame:
    config = load_config(repo / "examples" / "expert_thresholds.csv")
    loso = run_loso_validation(repo).set_index("Sample")
    rows = []
    for sample_id, metadata in config.items():
        table = _read_table(repo / "training_data" / f"total{sample_id}.xlsx")
        volumes_mm3 = table[VOL].to_numpy(float)
        voxel_volume = metadata["vox"] ** 3
        aligned = _axis_distance_degrees(table, 3) <= angle_degrees
        thresholds_mm3 = {
            "expert": float(metadata["thr"]),
            "loso_projected": float(loso.loc[sample_id, "ProjectedVmin_mm3"]),
            "loso_loose": float(loso.loc[sample_id, "LooseVmin_mm3"]),
            "loso_strict": float(loso.loc[sample_id, "StrictVmin_mm3"]),
        }
        for threshold_name, threshold_mm3 in thresholds_mm3.items():
            voxel_equivalent = threshold_mm3 / voxel_volume
            for group_name, mask in {
                "below": volumes_mm3 < threshold_mm3,
                "at_or_above": volumes_mm3 >= threshold_mm3,
            }.items():
                summary = _summary(mask, aligned)
                rows.append(
                    {
                        "Sample": sample_id,
                        "ThresholdType": threshold_name,
                        "Group": group_name,
                        "VoxelSize_mm": metadata["vox"],
                        "ExpertThreshold_mm3": metadata["thr"],
                        "ExpertEquivalentVoxelCount": metadata["thr"] / voxel_volume,
                        "Threshold_mm3": threshold_mm3,
                        "Threshold_vox_equivalent": voxel_equivalent,
                        "MinimumRetainedIntegerVoxelCount": int(np.ceil(voxel_equivalent)),
                        "Aligned": summary["aligned"],
                        "n": summary["total"],
                        "Fraction": summary["fraction"],
                        "Wilson95Low": summary["wilson_95_low"],
                        "Wilson95High": summary["wilson_95_high"],
                    }
                )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    repository = Path(__file__).resolve().parents[1]
    result = run(repository)
    output = repository / "outputs" / "training_axis_locking_audit.csv"
    output.parent.mkdir(exist_ok=True)
    result.to_csv(output, index=False)
    result["Percent"] = 100.0 * result["Fraction"]
    result["Wilson95LowPercent"] = 100.0 * result["Wilson95Low"]
    result["Wilson95HighPercent"] = 100.0 * result["Wilson95High"]
    print(
        result[
            [
                "Sample",
                "ThresholdType",
                "Group",
                "Threshold_mm3",
                "Threshold_vox_equivalent",
                "ExpertThreshold_mm3",
                "Aligned",
                "n",
                "Percent",
                "Wilson95LowPercent",
                "Wilson95HighPercent",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )
    print(f"Wrote {output}")
