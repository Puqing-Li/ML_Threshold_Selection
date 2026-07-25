#!/usr/bin/env python3
"""Assess loose/strict threshold sensitivity to model seed and strict tolerance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rebuild_revision_results import _load_training, _read_table
from features.res_aware_feature_engineering import ResolutionAwareFeatureEngineer
from ml_threshold_selection.prediction_analysis import compute_dual_thresholds
from ml_threshold_selection.training_pipeline import train_model_pipeline


def run(args):
    training, _, voxel_sizes, _ = _load_training(args.config, args.training_data)
    test = _read_table(args.test_data)
    volumes = test['Volume3d (mm^3) '].astype(float).to_numpy()
    voxel_counts = volumes / args.voxel_size_mm ** 3
    rows = []
    probability_runs = []

    for seed in range(args.seed_start, args.seed_start + args.seed_count):
        engineer = ResolutionAwareFeatureEngineer()
        model, _, _ = train_model_pipeline(
            training_data=training,
            voxel_sizes=voxel_sizes,
            resolution_aware_engineer=engineer,
            lightgbm_available=True,
            random_seed=seed,
        )
        test_features = engineer.extract(test, args.voxel_size_mm, fit_scaler=False)
        probabilities = np.asarray(model.predict(test_features), dtype=float)
        probability_runs.append(probabilities)
        for tolerance in args.tolerances:
            loose, strict = compute_dual_thresholds(
                voxel_counts, probabilities, tolerance
            )
            rows.append({
                'seed': seed,
                'strict_probability_tolerance': tolerance,
                'loose_threshold_vox': loose,
                'strict_threshold_vox': strict,
                'loose_retained_n': int(np.sum(voxel_counts >= loose)) if loose else None,
                'strict_retained_n': int(np.sum(voxel_counts >= strict)) if strict else None,
            })

    results = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output.with_suffix('.csv'), index=False)

    ensemble_probabilities = np.mean(np.vstack(probability_runs), axis=0)
    summary = {}
    for tolerance, group in results.groupby('strict_probability_tolerance'):
        ensemble_loose, ensemble_strict = compute_dual_thresholds(
            voxel_counts, ensemble_probabilities, tolerance
        )
        summary[str(tolerance)] = {
            'loose_vox_min': float(group['loose_threshold_vox'].min()),
            'loose_vox_median': float(group['loose_threshold_vox'].median()),
            'loose_vox_max': float(group['loose_threshold_vox'].max()),
            'strict_vox_min': float(group['strict_threshold_vox'].min()),
            'strict_vox_median': float(group['strict_threshold_vox'].median()),
            'strict_vox_max': float(group['strict_threshold_vox'].max()),
            'ensemble_mean_probability_loose_vox': float(ensemble_loose),
            'ensemble_mean_probability_strict_vox': float(ensemble_strict),
        }
    with open(args.output.with_suffix('.json'), 'w', encoding='utf-8') as handle:
        json.dump({
            'seed_start': args.seed_start,
            'seed_count': args.seed_count,
            'sample_id': args.sample_id,
            'summary_by_tolerance': summary,
        }, handle, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    loose = results.drop_duplicates('seed').sort_values('seed')
    axes[0].plot(loose['seed'], loose['loose_threshold_vox'], 'o-', color='#345995')
    axes[0].set(
        xlabel='LightGBM random seed',
        ylabel='Loose threshold (voxels)',
        title='(A) Loose-threshold seed sensitivity',
    )

    groups = [
        results.loc[
            results['strict_probability_tolerance'] == tolerance,
            'strict_threshold_vox',
        ].to_numpy()
        for tolerance in args.tolerances
    ]
    axes[1].boxplot(groups, tick_labels=[f'{value:g}' for value in args.tolerances])
    axes[1].set(
        xlabel='Strict probability tolerance',
        ylabel='Strict threshold (voxels)',
        title='(B) Strict-threshold sensitivity',
    )
    for axis in axes:
        axis.grid(axis='y', color='#D9D9D9', linewidth=0.7)
    fig.tight_layout()
    fig.savefig(args.output.with_suffix('.png'), dpi=300, bbox_inches='tight')
    fig.savefig(args.output.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path('examples/expert_thresholds.csv'))
    parser.add_argument('--training-data', type=Path, default=Path('trained model'))
    parser.add_argument('--test-data', type=Path, default=Path('examples/Quantity_LE01.xlsx'))
    parser.add_argument('--sample-id', default='LE01')
    parser.add_argument('--voxel-size-mm', type=float, default=0.03)
    parser.add_argument('--seed-start', type=int, default=0)
    parser.add_argument('--seed-count', type=int, default=20)
    parser.add_argument('--tolerances', type=float, nargs='+', default=[0.005, 0.01, 0.02, 0.05])
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
