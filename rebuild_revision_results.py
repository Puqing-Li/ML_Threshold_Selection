#!/usr/bin/env python3
"""Rebuild the revision results from one documented, non-interactive workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))

from features.res_aware_feature_engineering import ResolutionAwareFeatureEngineer
from ml_threshold_selection.fabric_pipeline import run_fabric_boxplots
from ml_threshold_selection.io_persistence import load_last, save_portable
from ml_threshold_selection.labeling import generate_labels_from_thresholds
from ml_threshold_selection.mean_fabric_calculator import (
    compute_mean_fabric_single,
    export_mean_fabric_txt,
)
from ml_threshold_selection.prediction_analysis import compute_dual_thresholds
from ml_threshold_selection.training_pipeline import train_model_pipeline
from ml_threshold_selection.voxel_config import FEATURE_SCHEMA_VERSION


class _Logger:
    def info(self, message):
        print(str(message).encode('ascii', errors='replace').decode('ascii'))


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in ('.xlsx', '.xls'):
        return pd.read_excel(path)
    return pd.read_csv(path, float_precision='round_trip')


def _load_training(config_path: Path, data_dir: Path):
    config = pd.read_csv(config_path)
    frames = []
    thresholds = {}
    voxel_sizes = {}
    training_files = []
    for row in config.itertuples(index=False):
        sample_id = str(row.SampleID)
        hits = (
            sorted(data_dir.glob(f'{sample_id}.xlsx'))
            or sorted(data_dir.glob(f'total{sample_id}.xlsx'))
            or sorted(data_dir.glob(f'*{sample_id}*.xlsx'))
            or sorted(data_dir.glob(f'{sample_id}.csv.gz'))
            or sorted(data_dir.glob(f'total{sample_id}.csv.gz'))
            or sorted(data_dir.glob(f'*{sample_id}*.csv.gz'))
            or sorted(data_dir.glob(f'*{sample_id}*.csv'))
        )
        if not hits:
            raise FileNotFoundError(f'No training table found for {sample_id}')
        frame = _read_table(hits[0])
        frame['SampleID'] = sample_id
        frames.append(frame)
        thresholds[sample_id] = float(row.ExpertThreshold_mm3)
        voxel_sizes[sample_id] = float(row.VoxelSize_mm)
        training_files.append(str(hits[0]))
    training = pd.concat(frames, ignore_index=True)
    training = generate_labels_from_thresholds(
        training,
        expert_thresholds=thresholds,
        voxel_sizes=voxel_sizes,
        sample_list=list(thresholds),
    )
    return training, thresholds, voxel_sizes, training_files


def _mean_result(df, sample_id, threshold_name, threshold_vox, voxel_size, out_dir):
    threshold_mm3 = threshold_vox * voxel_size ** 3
    matrix, vectors, values, t_value, p_prime, n_particles = compute_mean_fabric_single(
        df, threshold_mm3
    )
    if matrix is None:
        raise RuntimeError(f'Insufficient objects at the {threshold_name} threshold')
    export_mean_fabric_txt(
        sample_id=sample_id,
        threshold_type=threshold_name,
        volume_threshold_mm3=threshold_mm3,
        mean_ellipsoid_matrix=matrix,
        eigenvectors=vectors,
        eigenvalues=values,
        T=t_value,
        P_prime=p_prime,
        n_particles=n_particles,
        output_dir=str(out_dir),
    )
    return {
        'threshold_vox': int(threshold_vox),
        'threshold_mm3': float(threshold_mm3),
        'retained_objects': int(n_particles),
        'T': float(t_value),
        'P_prime': float(p_prime),
        'mean_semiaxes_mm': np.diag(values).astype(float).tolist(),
        'principal_axes_columns': np.asarray(vectors, dtype=float).tolist(),
    }


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    training, thresholds, voxel_sizes, training_files = _load_training(
        args.config, args.training_data
    )
    if args.retrain:
        engineer = ResolutionAwareFeatureEngineer()
        model, features, training_results = train_model_pipeline(
            training_data=training,
            voxel_sizes=voxel_sizes,
            resolution_aware_engineer=engineer,
            lightgbm_available=True,
            random_seed=args.seed,
        )
        source = 'retrained in this run'
    else:
        # Load the released bundle. The strict threshold is the largest object whose
        # artifact probability still exceeds --strict-probability, so it is a maximum
        # over the model's low-probability tail and shifts when the model is refitted.
        # Reproducing the reported thresholds therefore requires the released model.
        bundle = load_last(str(args.model_dir))
        schema = bundle.get('feature_schema_version')
        if schema != FEATURE_SCHEMA_VERSION:
            raise RuntimeError(
                f'Model bundle in {args.model_dir} carries feature schema {schema!r}; '
                f'this code requires {FEATURE_SCHEMA_VERSION!r}'
            )
        model = bundle['model']
        engineer = bundle['resolution_aware_engineer']
        features = bundle.get('features')
        training_results = bundle.get('training_results') or {}
        source = f'released bundle in {args.model_dir}'
    print(f'model: {source}')

    test = _read_table(args.test_data)
    test_features = engineer.extract(test, args.voxel_size_mm, fit_scaler=False)
    probabilities = np.asarray(model.predict(test_features), dtype=float)
    volumes = test['Volume3d (mm^3) '].astype(float).to_numpy()
    voxel_counts = volumes / args.voxel_size_mm ** 3
    loose_raw, strict_raw = compute_dual_thresholds(
        voxel_counts, probabilities, args.strict_probability
    )
    if loose_raw is None or strict_raw is None:
        raise RuntimeError('The dual-threshold calculation did not return both thresholds')
    loose_vox = int(np.ceil(loose_raw))
    strict_vox = int(np.ceil(strict_raw))

    predictions = test.copy()
    predictions['VoxelCount'] = voxel_counts
    predictions['predicted_below_threshold_probability'] = probabilities
    predictions.to_csv(
        args.output / f'{args.sample_id}_predictions.csv.gz',
        index=False,
        compression='gzip',
    )

    loose = _mean_result(
        test, args.sample_id, 'Loose', loose_vox, args.voxel_size_mm, args.output
    )
    strict = _mean_result(
        test, args.sample_id, 'Strict', strict_vox, args.voxel_size_mm, args.output
    )
    run_fabric_boxplots(
        df=test,
        voxel_size_mm=args.voxel_size_mm,
        loose_threshold_vox=loose_vox,
        strict_threshold_vox=strict_vox,
        logger=_Logger(),
        outputs_dir=str(args.output),
        n_bootstrap=args.bootstrap,
        random_seed=args.seed,
    )

    if args.retrain:
        save_portable({
            'model': model,
            'training_data': training,
            'expert_thresholds': thresholds,
            'voxel_sizes': voxel_sizes,
            'training_files': training_files,
            'features': features,
            'training_results': training_results,
            'ellipsoid_analysis_results': {},
            'resolution_aware_engineer': engineer,
        }, str(args.output))

    summary = {
        'sample_id': args.sample_id,
        'test_file': str(args.test_data.resolve()),
        'voxel_size_mm': args.voxel_size_mm,
        'random_seed': args.seed,
        'bootstrap_resamples': args.bootstrap,
        'label_definition': 'expert-defined below-threshold pseudo-label if Volume3d_mm3 < expert threshold_mm3',
        'feature_definition': 'sample-specific VoxelCount plus symmetric dimensionless log-ellipsoid Mandel components (a_ref = 1 mm)',
        'model_source': source,
        'training_objects': int(len(training)),
        'training_auc_resubstitution': (
            float(training_results['train_auc'])
            if 'train_auc' in training_results else None),
        'training_accuracy_resubstitution': (
            float(training_results['train_accuracy'])
            if 'train_accuracy' in training_results else None),
        'loose': loose,
        'strict': strict,
    }
    with open(args.output / f'{args.sample_id}_revision_summary.json', 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT / 'examples' / 'expert_thresholds.csv')
    parser.add_argument('--training-data', type=Path, default=ROOT / 'trained model')
    parser.add_argument('--test-data', type=Path, default=ROOT / 'examples' / 'Quantity_LE01.xlsx')
    parser.add_argument('--sample-id', default='LE01')
    parser.add_argument('--voxel-size-mm', type=float, default=0.03)
    parser.add_argument('--model-dir', type=Path, default=ROOT / 'trained model',
                        help='directory holding the released model bundle')
    parser.add_argument('--retrain', action='store_true',
                        help='refit the classifier instead of loading the released '
                             'bundle; the strict threshold will differ because it is '
                             'a maximum over the model probability tail')
    parser.add_argument('--strict-probability', type=float, default=0.01)
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
