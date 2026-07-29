#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training pipeline: feature extraction (resolution-aware) + model training + metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Tuple


def lightgbm_parameters(random_seed: Optional[int] = None) -> Dict[str, Any]:
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
    }
    if random_seed is None:
        # Explicitly preserve the seed profile used by the released manuscript
        # model. These are LightGBM 4.6.0's recorded component seeds.
        params.update({
            'bagging_seed': 3,
            'feature_fraction_seed': 2,
            'data_random_seed': 1,
            'extra_seed': 6,
            'drop_seed': 4,
            'objective_seed': 5,
        })
    else:
        params.update({
            'seed': random_seed,
            'feature_fraction_seed': random_seed,
            'bagging_seed': random_seed,
            'data_random_seed': random_seed,
            'deterministic': True,
            'force_col_wise': True,
        })
    return params


def train_model_pipeline(
    training_data: pd.DataFrame,
    voxel_sizes: Dict[str, float],
    resolution_aware_engineer,
    lightgbm_available: bool,
    random_seed: Optional[int] = None,
) -> Tuple[Any, pd.DataFrame, Dict[str, Any]]:
    # Verify voxel sizes
    missing = []
    sample_ids = training_data['SampleID'].unique().tolist() if 'SampleID' in training_data.columns else []
    for sid in sample_ids:
        if sid not in voxel_sizes:
            missing.append(sid)
    if missing:
        raise ValueError(f"Missing voxel sizes (mm) for samples: {missing}")

    features = resolution_aware_engineer.extract_by_sample(
        training_data,
        voxel_sizes=voxel_sizes,
        fit_scaler=True,
    )

    X = features.fillna(0)
    y = training_data['label'].values

    model = None
    if lightgbm_available:
        import lightgbm as lgb
        train_data = lgb.Dataset(X, label=y)
        params = lightgbm_parameters(random_seed)
        model = lgb.train(params, train_data, num_boost_round=100)
        proba = model.predict(X)
    else:
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42 if random_seed is None else random_seed,
            class_weight='balanced'
        )
        clf.fit(X, y)
        model = clf
        proba = clf.predict_proba(X)[:, 1]

    # Metrics
    from sklearn.metrics import (
        roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
    )
    train_auc = roc_auc_score(y, proba)
    pred_bin = (proba > 0.5).astype(int)
    train_accuracy = accuracy_score(y, pred_bin)
    precision = precision_score(y, pred_bin)
    recall = recall_score(y, pred_bin)
    f1 = f1_score(y, pred_bin)

    training_results = {
        'X': X,
        'y': y,
        'train_proba': proba,
        'train_pred': pred_bin,
        'train_auc': train_auc,
        'train_accuracy': train_accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'features': features,
    }

    return model, features, training_results
