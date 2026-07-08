#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the version-portable model persistence bundle (io_persistence).

The portable bundle exists so that a model trained in one environment
(e.g. pandas 3 / NumPy 2) can be loaded in a different one without relying on
pickle, which is not stable across pandas / NumPy / scikit-learn major versions.
These tests assert a lossless, prediction-identical round trip and that the
bundle contains no pickled objects.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml_threshold_selection import io_persistence as iop
from features.res_aware_feature_engineering import ResolutionAwareFeatureEngineer

lgb = pytest.importorskip("lightgbm")


def _raw_frame(n=40, seed=0):
    rng = np.random.default_rng(seed)
    eig = np.sort(rng.uniform(1e-4, 4e-3, size=(n, 3)), axis=1)[:, ::-1]
    return pd.DataFrame({
        'Volume3d (mm^3) ': rng.uniform(1e-5, 5e-3, size=n),
        'EigenVal1': eig[:, 0], 'EigenVal2': eig[:, 1], 'EigenVal3': eig[:, 2],
        'EigenVec1X': 1.0, 'EigenVec1Y': 0.0, 'EigenVec1Z': 0.0,
        'EigenVec2X': 0.0, 'EigenVec2Y': 1.0, 'EigenVec2Z': 0.0,
        'EigenVec3X': 0.0, 'EigenVec3Y': 0.0, 'EigenVec3Z': 1.0,
    })


def _build_model_data():
    raw = _raw_frame()
    eng = ResolutionAwareFeatureEngineer()
    feats = eng.extract(raw, voxel_size_mm=0.03, fit_scaler=True)  # fits the scaler
    y = (raw['Volume3d (mm^3) '].values > raw['Volume3d (mm^3) '].median()).astype(int)
    booster = lgb.train(
        {'objective': 'binary', 'num_leaves': 7, 'verbose': -1},
        lgb.Dataset(feats.values, label=y), num_boost_round=8)
    return {
        'model': booster,
        'training_data': raw,
        'expert_thresholds': {'sampleA': 1.0e-3, 'sampleB': 2.5e-3},
        'voxel_sizes': {'sampleA': 0.03, 'sampleB': 0.035},
        'training_files': ['sampleA', 'sampleB'],
        'features': feats,
        'training_results': {
            'X': feats, 'y': y, 'train_proba': booster.predict(feats.values),
            'train_auc': 0.9, 'precision': 0.8,
        },
        'ellipsoid_analysis_results': {
            'feature_stats': {'L11': {'mean': 0.1, 'is_significant': np.bool_(True)}},
            'selected_features': {'significant_features': ['L11', 'L22']},
            'correlation_matrix': pd.DataFrame(np.eye(2), columns=['a', 'b'], index=['a', 'b']),
        },
        'resolution_aware_engineer': eng,
    }, feats, y


def test_portable_roundtrip_is_prediction_identical(tmp_path):
    md, feats, _ = _build_model_data()
    p_ref = md['model'].predict(feats.values)

    iop.save_portable(md, str(tmp_path))
    loaded = iop.load_portable(str(tmp_path))

    # all top-level keys preserved
    assert sorted(loaded.keys()) == sorted(md.keys())

    # model predicts identically
    p_new = loaded['model'].predict(feats.values)
    assert np.array_equal(p_ref, p_new)

    # scaler statistics restored exactly -> engineer reproduces features exactly
    e_new = loaded['resolution_aware_engineer']
    assert np.array_equal(e_new.scaler.mean_, md['resolution_aware_engineer'].scaler.mean_)
    assert np.array_equal(e_new.scaler.scale_, md['resolution_aware_engineer'].scaler.scale_)
    f_new = e_new.extract(md['training_data'], voxel_size_mm=0.03, fit_scaler=False)
    np.testing.assert_array_equal(f_new.values, feats.values)


def test_portable_preserves_types_and_shapes(tmp_path):
    md, _, _ = _build_model_data()
    iop.save_portable(md, str(tmp_path))
    loaded = iop.load_portable(str(tmp_path))

    assert loaded['training_data'].shape == md['training_data'].shape
    assert loaded['expert_thresholds'] == md['expert_thresholds']
    assert loaded['training_files'] == md['training_files']
    # numpy bool scalar comes back as a JSON-safe python bool, value preserved
    sig = loaded['ellipsoid_analysis_results']['feature_stats']['L11']['is_significant']
    assert isinstance(sig, bool) and sig is True


def test_bundle_contains_no_pickle(tmp_path):
    md, _, _ = _build_model_data()
    iop.save_portable(md, str(tmp_path))
    bundle = tmp_path / iop.PORTABLE_DIRNAME
    assert (bundle / 'manifest.json').exists()
    bad = [p.name for p in bundle.rglob('*') if p.suffix in ('.pkl', '.pickle', '.joblib')]
    assert bad == []
    manifest = json.loads((bundle / 'manifest.json').read_text(encoding='utf-8'))

    def has_unsupported(node):
        if not isinstance(node, dict):
            return False
        if node.get('kind') == 'unsupported':
            return True
        if node.get('kind') == 'dict':
            return any(has_unsupported(v) for v in node['items'].values())
        if node.get('kind') == 'list':
            return any(has_unsupported(v) for v in node['items'])
        return False

    assert not any(has_unsupported(v) for v in manifest['keys'].values())


def test_load_last_prefers_portable_over_pickle(tmp_path):
    md, feats, _ = _build_model_data()
    # write a portable bundle
    iop.save_portable(md, str(tmp_path))
    # write a deliberately broken pickle alongside; load_last must ignore it
    (tmp_path / iop.PICKLE_FILENAME).write_bytes(b'not a real pickle')
    loaded = iop.load_last(str(tmp_path))
    assert np.array_equal(loaded['model'].predict(feats.values), md['model'].predict(feats.values))


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
