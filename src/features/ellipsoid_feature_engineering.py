#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ellipsoid Feature Engineering (renamed from joshua_feature_engineering_fixed.py)
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class JoshuaFeatureEngineerFixed:
    """Fixed feature engineer - resolves voxel size and feature scaling issues"""
    
    def __init__(self, voxel_size_um: Optional[float] = None):
        self.voxel_size_um = voxel_size_um
        self.scaler = StandardScaler()
        self.is_fitted = False

    def extract_joshua_features(self, df: pd.DataFrame, fit_scaler: bool = True) -> pd.DataFrame:
        volume = df['Volume3d (mm^3) '].values
        eigenvals = df[['EigenVal1', 'EigenVal2', 'EigenVal3']].values
        eigenvec1 = df[['EigenVec1X', 'EigenVec1Y', 'EigenVec1Z']].values
        eigenvec2 = df[['EigenVec2X', 'EigenVec2Y', 'EigenVec2Z']].values
        eigenvec3 = df[['EigenVec3X', 'EigenVec3Y', 'EigenVec3Z']].values

        if self.voxel_size_um is not None:
            voxel_size_mm = self.voxel_size_um / 1000.0
            volume_voxels = np.ceil(volume / (voxel_size_mm ** 3)).astype(np.int64)
            volume_voxels = np.maximum(volume_voxels, 1)
            volume_normalized = volume_voxels.astype(float)
        else:
            raise ValueError("Voxel size (μm) is required to normalize volume into voxel counts (>=1). Please input voxel sizes.")

        a1, a2, a3 = np.sqrt(eigenvals[:, 0]), np.sqrt(eigenvals[:, 1]), np.sqrt(eigenvals[:, 2])
        l1, l2, l3 = np.log(a1), np.log(a2), np.log(a3)
        Q = np.stack([eigenvec1, eigenvec2, eigenvec3], axis=1)
        vec_norms = np.linalg.norm(Q, axis=2)
        if not np.allclose(vec_norms, 1.0, atol=1e-6):
            Q = Q / vec_norms[:, :, np.newaxis]
        log_E_tilde = np.zeros((len(df), 3, 3))
        log_E_tilde[:, 0, 0] = l1
        log_E_tilde[:, 1, 1] = l2
        log_E_tilde[:, 2, 2] = l3
        L = np.zeros((len(df), 3, 3))
        for i in range(len(df)):
            L[i] = Q[i].T @ log_E_tilde[i] @ Q[i]
        l_vector = np.zeros((len(df), 6))
        l_vector[:, 0] = L[:, 0, 0]
        l_vector[:, 1] = L[:, 1, 1]
        l_vector[:, 2] = L[:, 2, 2]
        l_vector[:, 3] = np.sqrt(2) * L[:, 0, 1]
        l_vector[:, 4] = np.sqrt(2) * L[:, 0, 2]
        l_vector[:, 5] = np.sqrt(2) * L[:, 1, 2]
        result_df = pd.DataFrame({
            'Volume': volume_normalized,
            'L11': l_vector[:, 0],
            'L22': l_vector[:, 1],
            'L33': l_vector[:, 2],
            'sqrt2_L12': l_vector[:, 3],
            'sqrt2_L13': l_vector[:, 4],
            'sqrt2_L23': l_vector[:, 5],
        }, index=df.index)
        if fit_scaler and not self.is_fitted:
            result_df = pd.DataFrame(self.scaler.fit_transform(result_df), columns=result_df.columns, index=result_df.index)
            self.is_fitted = True
        elif self.is_fitted:
            result_df = pd.DataFrame(self.scaler.transform(result_df), columns=result_df.columns, index=result_df.index)
        return result_df

    def get_feature_names(self) -> list:
        return ['Volume', 'L11', 'L22', 'L33', 'sqrt2_L12', 'sqrt2_L13', 'sqrt2_L23']

    def get_feature_descriptions(self) -> Dict[str, str]:
        return {
            'Volume': 'Normalized particle volume (voxel count)',
            'L11': 'Log-ellipsoid tensor diagonal element L₁₁',
            'L22': 'Log-ellipsoid tensor diagonal element L₂₂',
            'L33': 'Log-ellipsoid tensor diagonal element L₃₃',
            'sqrt2_L12': 'Log-ellipsoid tensor off-diagonal element √2L₁₂',
            'sqrt2_L13': 'Log-ellipsoid tensor off-diagonal element √2L₁₃',
            'sqrt2_L23': 'Log-ellipsoid tensor off-diagonal element √2L₂₃',
        }
