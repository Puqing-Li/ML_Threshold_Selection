#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training pipeline: feature extraction (resolution-aware) + model training + metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any


def train_model_pipeline(
    training_data: pd.DataFrame,
    voxel_sizes: Dict[str, float],
    resolution_aware_engineer,
    lightgbm_available: bool,
) -> Tuple[Any, pd.DataFrame, Dict[str, Any]]:
    # Verify voxel sizes
    missing = []
    # Use first sample id present in data for normalization
    sample_ids = training_data['SampleID'].unique().tolist() if 'SampleID' in training_data.columns else []
    for sid in sample_ids:
        if sid not in voxel_sizes:
            missing.append(sid)
    if missing:
        raise ValueError(f"Missing voxel sizes (mm) for samples: {missing}")

    first_sample = sample_ids[0] if sample_ids else list(voxel_sizes.keys())[0]
    voxel_mm = float(voxel_sizes[first_sample])

    # Extract features (fit scaler inside)
    features = resolution_aware_engineer.extract(training_data, voxel_size_mm=voxel_mm, fit_scaler=True)

    X = features.fillna(0)
    y = training_data['label'].values

    model = None
    if lightgbm_available:
        import lightgbm as lgb
        train_data = lgb.Dataset(X, label=y)
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        model = lgb.train(params, train_data, num_boost_round=100)
        proba = model.predict(X)
    else:
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
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


