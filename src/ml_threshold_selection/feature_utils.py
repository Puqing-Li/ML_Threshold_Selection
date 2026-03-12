#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility feature extraction for quick, non-pipeline use.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def extract_simple_features(df: pd.DataFrame) -> pd.DataFrame:
    features = {}
    features['log_volume'] = np.log10(df['Volume3d (mm^3) '].values)
    features['volume'] = df['Volume3d (mm^3) '].values
    if 'EigenVal1' in df.columns:
        eigenvals = df[['EigenVal1', 'EigenVal2', 'EigenVal3']].values
        a, b, c = np.sqrt(eigenvals[:, 0]), np.sqrt(eigenvals[:, 1]), np.sqrt(eigenvals[:, 2])
        features['a'] = a
        features['b'] = b
        features['c'] = c
        features['a_b_ratio'] = a / b
        features['a_c_ratio'] = a / c
        features['b_c_ratio'] = b / c
        features['sphericity'] = c / a
        lambda_sum = np.sum(eigenvals, axis=1)
        features['anisotropy'] = (eigenvals[:, 0] - eigenvals[:, 2]) / lambda_sum
        features['lambda_diff_12'] = np.abs(eigenvals[:, 0] - eigenvals[:, 1])
        features['lambda_diff_23'] = np.abs(eigenvals[:, 1] - eigenvals[:, 2])
    if 'Elongation' in df.columns:
        features['elongation'] = df['Elongation'].values
    if 'Flatness' in df.columns:
        features['flatness'] = df['Flatness'].values
    if all(col in df.columns for col in ['EigenVec1X', 'EigenVec1Y', 'EigenVec1Z']):
        eigenvec1 = df[['EigenVec1X', 'EigenVec1Y', 'EigenVec1Z']].values
        x_axis = np.array([1, 0, 0])
        y_axis = np.array([0, 1, 0])
        z_axis = np.array([0, 0, 1])
        features['eigenvec1_x_alignment'] = np.abs(np.dot(eigenvec1, x_axis))
        features['eigenvec1_y_alignment'] = np.abs(np.dot(eigenvec1, y_axis))
        features['eigenvec1_z_alignment'] = np.abs(np.dot(eigenvec1, z_axis))
        features['eigenvec1_max_alignment'] = np.maximum.reduce([
            features['eigenvec1_x_alignment'],
            features['eigenvec1_y_alignment'],
            features['eigenvec1_z_alignment']
        ])
    if all(col in df.columns for col in ['EigenVec2X', 'EigenVec2Y', 'EigenVec2Z']):
        eigenvec2 = df[['EigenVec2X', 'EigenVec2Y', 'EigenVec2Z']].values
        x_axis = np.array([1, 0, 0])
        y_axis = np.array([0, 1, 0])
        z_axis = np.array([0, 0, 1])
        features['eigenvec2_x_alignment'] = np.abs(np.dot(eigenvec2, x_axis))
        features['eigenvec2_y_alignment'] = np.abs(np.dot(eigenvec2, y_axis))
        features['eigenvec2_z_alignment'] = np.abs(np.dot(eigenvec2, z_axis))
        features['eigenvec2_max_alignment'] = np.maximum.reduce([
            features['eigenvec2_x_alignment'],
            features['eigenvec2_y_alignment'],
            features['eigenvec2_z_alignment']
        ])
    if all(col in df.columns for col in ['EigenVec3X', 'EigenVec3Y', 'EigenVec3Z']):
        eigenvec3 = df[['EigenVec3X', 'EigenVec3Y', 'EigenVec3Z']].values
        x_axis = np.array([1, 0, 0])
        y_axis = np.array([0, 1, 0])
        z_axis = np.array([0, 0, 1])
        features['eigenvec3_x_alignment'] = np.abs(np.dot(eigenvec3, x_axis))
        features['eigenvec3_y_alignment'] = np.abs(np.dot(eigenvec3, y_axis))
        features['eigenvec3_z_alignment'] = np.abs(np.dot(eigenvec3, z_axis))
        features['eigenvec3_max_alignment'] = np.maximum.reduce([
            features['eigenvec3_x_alignment'],
            features['eigenvec3_y_alignment'],
            features['eigenvec3_z_alignment']
        ])
    features['is_small_volume'] = (features['volume'] < 1e-6).astype(int)
    features['is_very_small_volume'] = (features['volume'] < 1e-7).astype(int)
    if 'elongation' in features and 'flatness' in features:
        features['elongation_flatness_product'] = features['elongation'] * features['flatness']
        features['is_high_elongation'] = (features['elongation'] > 0.8).astype(int)
        features['is_high_flatness'] = (features['flatness'] > 0.8).astype(int)
    if 'eigenvec1_max_alignment' in features:
        features['is_voxel_aligned'] = (features['eigenvec1_max_alignment'] > 0.9).astype(int)
    return pd.DataFrame(features)


