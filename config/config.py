#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration file for ML Threshold Selection system.
"""

# Strict probability threshold for artifact filtering
# This determines the P>threshold criterion for strict filtering
# Default: 0.01 (P>0.01 means particles with >1% artifact probability are filtered)
STRICT_PROBABILITY_THRESHOLD = 0.01

# Other configurable parameters
DEFAULT_VOXEL_SIZE_MM = 0.03
MIN_PARTICLES_FOR_ANALYSIS = 50
BOOTSTRAP_ITERATIONS = 1000

# Feature engineering parameters
FEATURE_ENGINEERING = {
    'resolution_aware': True,
    'log_ellipsoid_features': True,
    'standardize_features': True
}

# Model training parameters
MODEL_PARAMS = {
    'lightgbm': {
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
    }
}
