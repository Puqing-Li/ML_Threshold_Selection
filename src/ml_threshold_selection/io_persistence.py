#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Persistence helpers: auto-save/load last session model and state.
"""

from __future__ import annotations

import os
import pickle


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
    with open(os.path.join(outputs_dir, 'last_time_model.pkl'), 'wb') as f:
        pickle.dump(model_data, f)


def load_last(outputs_dir: str = 'outputs'):
    model_file = os.path.join(outputs_dir, 'last_time_model.pkl')
    if not os.path.exists(model_file):
        raise FileNotFoundError('No last_time_model.pkl found')
    with open(model_file, 'rb') as f:
        return pickle.load(f)
