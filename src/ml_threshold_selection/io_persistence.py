#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Persistence helpers: auto-save / load the last session model and state.

Two on-disk formats are supported:

* Legacy single-file pickle ``last_time_model.pkl`` - kept for backward
  compatibility and same-machine round trips.
* A version-portable bundle directory ``last_time_model_portable`` that stores
  the LightGBM model as native text, DataFrames as gzip-compressed CSV, NumPy
  arrays as ``.npy`` and everything else as JSON. This avoids pickling
  pandas / NumPy / scikit-learn objects, which otherwise fails to unpickle
  across library major versions (e.g. a model saved with pandas 3 / NumPy 2
  cannot be read by an older environment).

``load_last`` prefers the portable bundle when present and transparently falls
back to the pickle, so existing installations keep working unchanged.
"""

from __future__ import annotations

import json
import os
import pickle

import numpy as np
import pandas as pd

PORTABLE_DIRNAME = 'last_time_model_portable'
PICKLE_FILENAME = 'last_time_model.pkl'


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _is_json_scalar(x) -> bool:
    return x is None or isinstance(x, (bool, int, float, str))


def _to_list(a):
    if a is None:
        return None
    return np.asarray(a).tolist()


def _to_num(x):
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    return x


def _sanitize(name: str) -> str:
    return ''.join(c if c.isalnum() else '_' for c in str(name))[:60]


# --------------------------------------------------------------------------- #
# Portable (version-independent) serialization
# --------------------------------------------------------------------------- #
def _save_node(value, bundle_dir: str, name: str):
    """Serialize one object into portable files; return a JSON-able descriptor."""
    # LightGBM model -> native text format (stable across lightgbm versions)
    try:
        import lightgbm as lgb
        if isinstance(value, lgb.Booster):
            fn = name + '.model.txt'
            value.save_model(os.path.join(bundle_dir, fn))
            return {'kind': 'lgb_booster', 'file': fn}
    except Exception:
        pass

    # Fitted feature engineer -> plain scaler statistics (no sklearn pickle)
    try:
        from features.res_aware_feature_engineering import ResolutionAwareFeatureEngineer
        if isinstance(value, ResolutionAwareFeatureEngineer):
            sc = value.scaler
            return {
                'kind': 'engineer',
                'is_fitted': bool(getattr(value, 'is_fitted', False)),
                'current_voxel_size_mm': _to_num(getattr(value, 'current_voxel_size_mm', None)),
                'scaler': {
                    'mean': _to_list(getattr(sc, 'mean_', None)),
                    'scale': _to_list(getattr(sc, 'scale_', None)),
                    'var': _to_list(getattr(sc, 'var_', None)),
                    'n_features_in': int(getattr(sc, 'n_features_in_', 0)) or None,
                    'feature_names_in': _to_list(getattr(sc, 'feature_names_in_', None)),
                    'n_samples_seen': _to_num(getattr(sc, 'n_samples_seen_', None)),
                },
            }
    except Exception:
        pass

    if isinstance(value, pd.DataFrame):
        fn = name + '.csv.gz'
        value.to_csv(os.path.join(bundle_dir, fn), index=True, compression='gzip')
        return {'kind': 'dataframe', 'file': fn}

    if isinstance(value, pd.Series):
        fn = name + '.series.csv.gz'
        value.to_frame(name='__value__').to_csv(
            os.path.join(bundle_dir, fn), index=True, compression='gzip')
        return {'kind': 'series', 'file': fn}

    if isinstance(value, np.ndarray) and value.dtype != object:
        fn = name + '.npy'
        np.save(os.path.join(bundle_dir, fn), value, allow_pickle=False)
        return {'kind': 'ndarray', 'file': fn}

    if isinstance(value, dict):
        items = {}
        for k, v in value.items():
            items[str(k)] = _save_node(v, bundle_dir, name + '__' + _sanitize(k))
        return {'kind': 'dict', 'items': items}

    if isinstance(value, (list, tuple)):
        if all(_is_json_scalar(x) for x in value):
            return {'kind': 'json', 'value': list(value)}
        items = [_save_node(v, bundle_dir, name + '__%d' % i) for i, v in enumerate(value)]
        return {'kind': 'list', 'items': items}

    if isinstance(value, np.generic):  # np.bool_/np.integer/np.floating/np.str_ ...
        return {'kind': 'json', 'value': value.item()}
    if _is_json_scalar(value):
        return {'kind': 'json', 'value': value}

    # Never silently fall back to pickle (that would reintroduce the bug).
    return {'kind': 'unsupported', 'repr': repr(value)[:200]}


def _load_node(desc, bundle_dir: str):
    kind = desc.get('kind')
    if kind == 'json':
        return desc.get('value')
    if kind == 'lgb_booster':
        import lightgbm as lgb
        return lgb.Booster(model_file=os.path.join(bundle_dir, desc['file']))
    if kind == 'engineer':
        from features.res_aware_feature_engineering import ResolutionAwareFeatureEngineer
        eng = ResolutionAwareFeatureEngineer()
        eng.is_fitted = bool(desc.get('is_fitted'))
        eng.current_voxel_size_mm = desc.get('current_voxel_size_mm')
        sc_desc = desc.get('scaler') or {}
        if sc_desc.get('mean') is not None:
            eng.scaler.mean_ = np.asarray(sc_desc['mean'], dtype=float)
            eng.scaler.scale_ = np.asarray(sc_desc['scale'], dtype=float)
            if sc_desc.get('var') is not None:
                eng.scaler.var_ = np.asarray(sc_desc['var'], dtype=float)
            if sc_desc.get('n_features_in'):
                eng.scaler.n_features_in_ = int(sc_desc['n_features_in'])
            if sc_desc.get('feature_names_in') is not None:
                eng.scaler.feature_names_in_ = np.asarray(sc_desc['feature_names_in'], dtype=object)
            if sc_desc.get('n_samples_seen') is not None:
                eng.scaler.n_samples_seen_ = sc_desc['n_samples_seen']
        return eng
    if kind == 'dataframe':
        return pd.read_csv(os.path.join(bundle_dir, desc['file']), index_col=0, compression='infer')
    if kind == 'series':
        df = pd.read_csv(os.path.join(bundle_dir, desc['file']), index_col=0, compression='infer')
        return df['__value__']
    if kind == 'ndarray':
        return np.load(os.path.join(bundle_dir, desc['file']), allow_pickle=False)
    if kind == 'dict':
        return {k: _load_node(v, bundle_dir) for k, v in desc['items'].items()}
    if kind == 'list':
        return [_load_node(v, bundle_dir) for v in desc['items']]
    # 'unsupported' or unknown -> None (do not crash the whole load)
    return None


def save_portable(model_data: dict, outputs_dir: str = 'outputs') -> str:
    """Write a version-portable bundle directory and return its path."""
    bundle_dir = os.path.join(outputs_dir, PORTABLE_DIRNAME)
    os.makedirs(bundle_dir, exist_ok=True)
    manifest = {'format_version': 1, 'keys': {}}
    for key, value in model_data.items():
        manifest['keys'][key] = _save_node(value, bundle_dir, _sanitize(key))
    with open(os.path.join(bundle_dir, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return bundle_dir


def load_portable(outputs_dir_or_bundle: str) -> dict:
    """Load a portable bundle. Accepts either the outputs dir or the bundle dir."""
    bundle_dir = outputs_dir_or_bundle
    if not os.path.exists(os.path.join(bundle_dir, 'manifest.json')):
        bundle_dir = os.path.join(outputs_dir_or_bundle, PORTABLE_DIRNAME)
    with open(os.path.join(bundle_dir, 'manifest.json'), 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    return {key: _load_node(desc, bundle_dir) for key, desc in manifest['keys'].items()}


# --------------------------------------------------------------------------- #
# Public API (unchanged signatures)
# --------------------------------------------------------------------------- #
def auto_save(model, training_data, expert_thresholds, voxel_sizes, training_files, features, training_results, ellipsoid_analysis_results, resolution_aware_engineer, outputs_dir: str = 'outputs'):
    os.makedirs(outputs_dir, exist_ok=True)
    model_data = {
        'model': model,
        'training_data': training_data,
        'expert_thresholds': expert_thresholds,
        'voxel_sizes': voxel_sizes,
        'training_files': training_files,
        'features': features,
        'training_results': training_results,
        'ellipsoid_analysis_results': ellipsoid_analysis_results,
        'resolution_aware_engineer': resolution_aware_engineer,
    }
    # Legacy pickle (fast same-machine round trip).
    with open(os.path.join(outputs_dir, PICKLE_FILENAME), 'wb') as f:
        pickle.dump(model_data, f)
    # Portable bundle (cross-version). Never let a bundle hiccup break training.
    try:
        save_portable(model_data, outputs_dir)
    except Exception:
        pass


def load_last(outputs_dir: str = 'outputs'):
    # Prefer the version-portable bundle when it exists.
    bundle_dir = os.path.join(outputs_dir, PORTABLE_DIRNAME)
    if os.path.exists(os.path.join(bundle_dir, 'manifest.json')):
        try:
            return load_portable(bundle_dir)
        except Exception:
            pass  # fall back to the pickle below
    model_file = os.path.join(outputs_dir, PICKLE_FILENAME)
    if not os.path.exists(model_file):
        raise FileNotFoundError('No last_time_model found (portable bundle or pickle)')
    with open(model_file, 'rb') as f:
        return pickle.load(f)
