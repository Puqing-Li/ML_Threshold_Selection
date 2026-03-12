#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Legacy Ellipsoid Feature Engineering (file renamed from joshua_feature_engineering.py)
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class JoshuaFeatureEngineer:
    """Log-ellipsoid tensor feature engineering (7 core features)"""
    
    def __init__(self, voxel_size_mm: Optional[float] = None):
        self.voxel_size_mm = voxel_size_mm

    def extract_joshua_features(self, df: pd.DataFrame) -> pd.DataFrame:
        volume = df['Volume3d (mm^3) '].values
        eigenvals = df[['EigenVal1', 'EigenVal2', 'EigenVal3']].values
        eigenvec1 = df[['EigenVec1X', 'EigenVec1Y', 'EigenVec1Z']].values
        eigenvec2 = df[['EigenVec2X', 'EigenVec2Y', 'EigenVec2Z']].values
        eigenvec3 = df[['EigenVec3X', 'EigenVec3Y', 'EigenVec3Z']].values

        a1, a2, a3 = np.sqrt(eigenvals[:, 0]), np.sqrt(eigenvals[:, 1]), np.sqrt(eigenvals[:, 2])
        l1, l2, l3 = np.log(a1), np.log(a2), np.log(a3)

        Q = np.stack([eigenvec1, eigenvec2, eigenvec3], axis=1)  # (n, 3, 3)
        vec_norms = np.linalg.norm(Q, axis=2)
        if not np.allclose(vec_norms, 1.0, atol=1e-6):
            Q = Q / vec_norms[:, :, np.newaxis]

        log_E_tilde = np.zeros((len(df), 3, 3))
        log_E_tilde[:, 0, 0] = l1
        log_E_tilde[:, 1, 1] = l2
        log_E_tilde[:, 2, 2] = l3

        L = np.zeros((len(df), 3, 3))
        for i in range(len(df)):
            try:
                L[i] = Q[i].T @ log_E_tilde[i] @ Q[i]
                if np.isnan(L[i]).any() or np.isinf(L[i]).any():
                    L[i] = np.zeros((3, 3))
            except Exception:
                L[i] = np.zeros((3, 3))

        l_vector = np.zeros((len(df), 6))
        l_vector[:, 0] = L[:, 0, 0]
        l_vector[:, 1] = L[:, 1, 1]
        l_vector[:, 2] = L[:, 2, 2]
        l_vector[:, 3] = np.sqrt(2) * L[:, 0, 1]
        l_vector[:, 4] = np.sqrt(2) * L[:, 0, 2]
        l_vector[:, 5] = np.sqrt(2) * L[:, 1, 2]

        features = {
            'Volume': volume,
            'L11': l_vector[:, 0],
            'L22': l_vector[:, 1],
            'L33': l_vector[:, 2],
            'sqrt2_L12': l_vector[:, 3],
            'sqrt2_L13': l_vector[:, 4],
            'sqrt2_L23': l_vector[:, 5],
        }
        if self.voxel_size_mm is not None:
            features['Volume'] = features['Volume'] / (self.voxel_size_mm ** 3)

        return pd.DataFrame(features, index=df.index)

    def get_feature_names(self) -> list:
        return ['Volume', 'L11', 'L22', 'L33', 'sqrt2_L12', 'sqrt2_L13', 'sqrt2_L23']

    def get_feature_descriptions(self) -> Dict[str, str]:
        return {
            'Volume': 'Particle volume (mm³)',
            'L11': 'Log-ellipsoid tensor diagonal element L₁₁',
            'L22': 'Log-ellipsoid tensor diagonal element L₂₂',
            'L33': 'Log-ellipsoid tensor diagonal element L₃₃',
            'sqrt2_L12': 'Log-ellipsoid tensor off-diagonal element √2L₁₂',
            'sqrt2_L13': 'Log-ellipsoid tensor off-diagonal element √2L₁₃',
            'sqrt2_L23': 'Log-ellipsoid tensor off-diagonal element √2L₂₃',
        }


