#!/usr/bin/env python3
"""Quantify scan-axis alignment without using labels at model-derived thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in ('.xlsx', '.xls'):
        return pd.read_excel(path)
    return pd.read_csv(path, float_precision='round_trip')


def _wilson(k, n, z=1.959963984540054):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return center - half, center + half


def _axis_distance_degrees(df, axis_number):
    vectors = df[[
        f'EigenVec{axis_number}X',
        f'EigenVec{axis_number}Y',
        f'EigenVec{axis_number}Z',
    ]].to_numpy(dtype=float)
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(~np.isfinite(vectors)) or np.any(norms == 0):
        raise ValueError(f'EigenVec{axis_number} contains invalid vectors')
    unit = vectors / norms[:, None]
    return np.degrees(np.arccos(np.clip(np.max(np.abs(unit), axis=1), 0.0, 1.0)))


def _summary(mask, aligned):
    n = int(mask.sum())
    k = int(np.sum(aligned[mask]))
    low, high = _wilson(k, n)
    return {
        'aligned': k,
        'total': n,
        'fraction': k / n if n else None,
        'wilson_95_low': low,
        'wilson_95_high': high,
    }


def _draw_symmetry_schematic(axis):
    """Illustrate exact cancellation for a mirror-symmetric voxel set."""
    pairs = [((-2, -1), (2, -1)), ((-2, 1), (2, 1)), ((-1, 0), (1, 0)), ((-1, 2), (1, 2))]
    centers = [(0, -1), (0, 1)]
    colors = ('#D95F59', '#4C78A8')
    axis.axvline(0, color='#333333', linewidth=1.2, linestyle='--')
    for left, right in pairs:
        for point, color in zip((left, right), colors):
            axis.add_patch(Rectangle(
                (point[0] - 0.45, point[1] - 0.45), 0.9, 0.9,
                facecolor=color, edgecolor='white', linewidth=1.0,
            ))
        axis.plot([left[0], right[0]], [left[1], right[1]], color='#888888',
                  linewidth=0.8, linestyle=':', zorder=0)
    for point in centers:
        axis.add_patch(Rectangle(
            (point[0] - 0.45, point[1] - 0.45), 0.9, 0.9,
            facecolor='#B8B8B8', edgecolor='white', linewidth=1.0,
        ))
    axis.annotate('', xy=(3.0, -2.2), xytext=(0.0, -2.2),
                  arrowprops={'arrowstyle': '->', 'color': '#222222', 'lw': 1.2})
    axis.text(3.05, -2.2, 'x', va='center', ha='left')
    axis.text(0.08, 2.75, r'$x=\bar{x}$', va='top', ha='left', fontsize=9)
    axis.text(
        0.5, -0.17,
        r'paired $+\Delta x$ and $-\Delta x$ terms' '\n'
        r'$C_{xy}=C_{xz}=0\;\Rightarrow\;\hat{x}$ is an eigenvector',
        transform=axis.transAxes, ha='center', va='top', fontsize=9,
    )
    axis.set_xlim(-3.2, 3.2)
    axis.set_ylim(-2.7, 3.0)
    axis.set_aspect('equal')
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_frame_on(False)
    axis.set_title('(A) Mirror symmetry on the voxel grid')


def run(args):
    df = _read_table(args.data)
    volumes = df['Volume3d (mm^3) '].to_numpy(dtype=float)
    voxel_counts = volumes / args.voxel_size_mm ** 3
    distances = {number: _axis_distance_degrees(df, number) for number in (1, 2, 3)}
    aligned = {number: distances[number] <= args.angle_degrees for number in distances}
    isotropic = 3.0 * (1.0 - np.cos(np.deg2rad(args.angle_degrees)))

    summaries = {}
    for number in (1, 2, 3):
        summaries[f'EigenVec{number}'] = {
            'below_loose': _summary(voxel_counts < args.loose_vox, aligned[number]),
            'at_or_above_loose': _summary(voxel_counts >= args.loose_vox, aligned[number]),
            'at_or_above_strict': _summary(voxel_counts >= args.strict_vox, aligned[number]),
        }

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.2))
    _draw_symmetry_schematic(axes[0])
    colors = {1: '#B33A3A', 2: '#2E7D5B', 3: '#345995'}
    edges = np.geomspace(max(1.0, voxel_counts.min()), voxel_counts.max() + 1.0, 16)
    rows = []
    for number in (1, 2, 3):
        x_values, y_values, low_values, high_values = [], [], [], []
        for left, right in zip(edges[:-1], edges[1:]):
            mask = (voxel_counts >= left) & (voxel_counts < right)
            n = int(mask.sum())
            if n < args.min_bin_count:
                continue
            k = int(aligned[number][mask].sum())
            low, high = _wilson(k, n)
            x = float(np.exp(np.mean(np.log(voxel_counts[mask]))))
            fraction = k / n
            x_values.append(x)
            y_values.append(fraction)
            low_values.append(low)
            high_values.append(high)
            rows.append({
                'principal_axis': number,
                'bin_left_vox': left,
                'bin_right_vox': right,
                'geometric_mean_vox': x,
                'aligned': k,
                'total': n,
                'fraction': fraction,
                'wilson_95_low': low,
                'wilson_95_high': high,
            })
        y = np.asarray(y_values)
        axes[1].errorbar(
            x_values,
            100.0 * y,
            yerr=[
                100.0 * np.maximum(0.0, y - np.asarray(low_values)),
                100.0 * np.maximum(0.0, np.asarray(high_values) - y),
            ],
            marker='o',
            markersize=4,
            linewidth=1.5,
            capsize=2,
            color=colors[number],
            label=f'Principal axis {number}',
        )

    retained_thresholds = np.unique(np.ceil(np.geomspace(1.0, voxel_counts.max(), 80)))
    retained_rows = []
    for threshold in retained_thresholds:
        mask = voxel_counts >= threshold
        n = int(mask.sum())
        if n < args.min_retained_count:
            continue
        k = int(aligned[3][mask].sum())
        low, high = _wilson(k, n)
        retained_rows.append((threshold, k / n, low, high, n, k))
    retained_array = np.asarray(retained_rows, dtype=float)
    axes[2].plot(
        retained_array[:, 0], 100.0 * retained_array[:, 1], color=colors[3], linewidth=2
    )
    axes[2].fill_between(
        retained_array[:, 0],
        100.0 * retained_array[:, 2],
        100.0 * retained_array[:, 3],
        color=colors[3],
        alpha=0.2,
        linewidth=0,
    )

    for axis in axes[1:]:
        axis.axhline(100.0 * isotropic, color='black', linestyle='--', linewidth=1.2,
                     label=f'Isotropic expectation ({100.0 * isotropic:.2f}%)')
        axis.axvline(args.loose_vox, color='#666666', linestyle=':', linewidth=1.5,
                     label=f'Loose ({args.loose_vox:g} vox)')
        axis.axvline(args.strict_vox, color='#D97706', linestyle=':', linewidth=1.5,
                     label=f'Strict ({args.strict_vox:g} vox)')
        axis.set_xscale('log')
        axis.set_xlabel('Object size (voxels)')
        axis.set_ylabel(f'Within {args.angle_degrees:g} degrees of a scan axis (%)')
        axis.grid(axis='y', color='#D9D9D9', linewidth=0.7)
    axes[1].set_title('(B) Alignment within object-size bins')
    axes[2].set_title('(C) Minimum-axis alignment after filtering')
    axes[1].legend(fontsize=8)
    axes[2].legend(fontsize=8)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix('.png'), dpi=300, bbox_inches='tight')
    fig.savefig(args.output.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)

    pd.DataFrame(rows).to_csv(args.output.with_name(args.output.name + '_binned.csv'), index=False)
    pd.DataFrame(retained_rows, columns=[
        'threshold_vox', 'fraction', 'wilson_95_low', 'wilson_95_high', 'total', 'aligned'
    ]).to_csv(args.output.with_name(args.output.name + '_retained.csv'), index=False)
    result = {
        'angle_degrees': args.angle_degrees,
        'isotropic_unoriented_axis_fraction': isotropic,
        'loose_vox': args.loose_vox,
        'strict_vox': args.strict_vox,
        'principal_axis_summaries': summaries,
    }
    with open(args.output.with_suffix('.json'), 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--voxel-size-mm', type=float, required=True)
    parser.add_argument('--loose-vox', type=float, required=True)
    parser.add_argument('--strict-vox', type=float, required=True)
    parser.add_argument('--angle-degrees', type=float, default=5.0)
    parser.add_argument('--min-bin-count', type=int, default=50)
    parser.add_argument('--min-retained-count', type=int, default=200)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
