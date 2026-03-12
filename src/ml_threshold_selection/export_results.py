#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export helpers for filtered results and threshold report.
"""

from __future__ import annotations

import os
import pandas as pd


def export_filtered_results(
    results_df: pd.DataFrame,
    probabilities,
    loose_threshold_vox: int,
    strict_threshold_vox: int,
    voxel_size_mm: float,
    outputs_dir: str = 'outputs',
    strict_probability_threshold: float = 0.01,
    sample_id: str | None = None,
) -> str:
    os.makedirs(outputs_dir, exist_ok=True)
    voxel_vol = voxel_size_mm ** 3
    loose_threshold_mm = loose_threshold_vox * voxel_vol
    strict_threshold_mm = strict_threshold_vox * voxel_vol

    df = results_df.copy()
    df['predicted_probability'] = probabilities
    df['predicted_label'] = (probabilities > 0.5).astype(int)

    loose_filtered = df[df['Volume3d (mm^3) '] >= loose_threshold_mm].copy()
    loose_filtered['threshold_type'] = 'Loose (Inflection)'
    loose_filtered['threshold_value_mm3'] = loose_threshold_mm
    loose_filtered['threshold_value_vox'] = loose_threshold_vox

    strict_filtered = df[df['Volume3d (mm^3) '] >= strict_threshold_mm].copy()
    strict_filtered['threshold_type'] = f'Strict (P>{strict_probability_threshold})'
    strict_filtered['threshold_value_mm3'] = strict_threshold_mm
    strict_filtered['threshold_value_vox'] = strict_threshold_vox

    prefix = f"{sample_id}_" if sample_id else ""
    loose_filename = os.path.join(outputs_dir, f"{prefix}Loose_Threshold_Results_{loose_threshold_vox:.0f}vox_{loose_threshold_mm:.2e}mm3.xlsx")
    strict_filename = os.path.join(outputs_dir, f"{prefix}Strict_Threshold_Results_{strict_threshold_vox:.0f}vox_{strict_threshold_mm:.2e}mm3.xlsx")

    loose_filtered.to_excel(loose_filename, index=False)
    strict_filtered.to_excel(strict_filename, index=False)

    return loose_filename, strict_filename


def export_threshold_report(
    out_path: str,
    total_rows: int,
    voxel_size_mm: float,
    loose_threshold_vox: int,
    strict_threshold_vox: int,
    loose_threshold_mm: float,
    strict_threshold_mm: float,
    loose_kept: int,
    strict_kept: int,
):
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("ML Threshold Selection - Dual Threshold Analysis Report\n")
        f.write("=" * 60 + "\n\n")
        import pandas as pd
        f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Test Data: {total_rows} particles\n")
        f.write(f"Voxel Size: {voxel_size_mm:.6f} mm\n\n")
        f.write("DUAL THRESHOLDS:\n")
        f.write("-" * 30 + "\n")
        f.write("Loose Threshold (Inflection):\n")
        f.write(f"  - Voxel Count: {loose_threshold_vox:.0f} vox\n")
        f.write(f"  - Volume: {loose_threshold_mm:.2e} mm³\n")
        f.write(f"  - Particles Kept: {loose_kept}\n\n")
        f.write("Strict Threshold (P>0.05):\n")
        f.write(f"  - Voxel Count: {strict_threshold_vox:.0f} vox\n")
        f.write(f"  - Volume: {strict_threshold_mm:.2e} mm³\n")
        f.write(f"  - Particles Kept: {strict_kept}\n\n")
