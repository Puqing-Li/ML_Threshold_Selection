#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resolution-aware feature engineering: Absolute Volume (mm^3), Voxel Size (um),
Voxel Count (continuous, not rounded), plus 6 log-ellipsoid tensor features.
Outputs 9 features with StandardScaler (fit/transform modes).
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict
from sklearn.preprocessing import StandardScaler


class ResolutionAwareFeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.current_voxel_size_mm: Optional[float] = None

    def _compute_joshua_tensor(self, df: pd.DataFrame) -> np.ndarray:
        eigenvals = df[['EigenVal1', 'EigenVal2', 'EigenVal3']].values
        eigenvec1 = df[['EigenVec1X', 'EigenVec1Y', 'EigenVec1Z']].values
        eigenvec2 = df[['EigenVec2X', 'EigenVec2Y', 'EigenVec2Z']].values
        eigenvec3 = df[['EigenVec3X', 'EigenVec3Y', 'EigenVec3Z']].values
        a1, a2, a3 = np.sqrt(eigenvals[:, 0]), np.sqrt(eigenvals[:, 1]), np.sqrt(eigenvals[:, 2])
        l1, l2, l3 = np.log(a1), np.log(a2), np.log(a3)
        Q = np.stack([eigenvec1, eigenvec2, eigenvec3], axis=1)  # (n,3,3)
        norms = np.linalg.norm(Q, axis=2, keepdims=True)
        norms[norms == 0] = 1.0
        Q = Q / norms
        logE = np.zeros((len(df), 3, 3))
        logE[:, 0, 0] = -2 * l1
        logE[:, 1, 1] = -2 * l2
        logE[:, 2, 2] = -2 * l3
        L = np.einsum('nij,njk,nlk->nil', Q.transpose(0, 2, 1), logE, Q)
        out = np.zeros((len(df), 6))
        out[:, 0] = L[:, 0, 0]
        out[:, 1] = L[:, 1, 1]
        out[:, 2] = L[:, 2, 2]
        out[:, 3] = np.sqrt(2.0) * L[:, 0, 1]
        out[:, 4] = np.sqrt(2.0) * L[:, 0, 2]
        out[:, 5] = np.sqrt(2.0) * L[:, 1, 2]
        return out

    def extract(self, df: pd.DataFrame, voxel_size_mm: Optional[float], fit_scaler: bool) -> pd.DataFrame:
        if voxel_size_mm is None or voxel_size_mm <= 0:
            raise ValueError('voxel_size_mm must be provided (>0) for resolution-aware features')
        self.current_voxel_size_mm = voxel_size_mm
        voxel_mm = voxel_size_mm
        abs_volume = df['Volume3d (mm^3) '].values.astype(float)
        voxel_count = abs_volume / (voxel_mm ** 3)  # continuous, not rounded
        joshua6 = self._compute_joshua_tensor(df)
        # Build features: use VoxelCount (continuous) + 6 ellipsoid-tensor features. Do NOT include AbsVolume or VoxelSize.
        features = pd.DataFrame({
            'VoxelCount': voxel_count,
            'L11': joshua6[:, 0],
            'L22': joshua6[:, 1],
            'L33': joshua6[:, 2],
            'sqrt2_L12': joshua6[:, 3],
            'sqrt2_L13': joshua6[:, 4],
            'sqrt2_L23': joshua6[:, 5],
        }, index=df.index)
        # scale
        if fit_scaler and not self.is_fitted:
            scaled = self.scaler.fit_transform(features)
            self.is_fitted = True
        else:
            scaled = self.scaler.transform(features)
        return pd.DataFrame(scaled, columns=features.columns, index=features.index)

    @staticmethod
    def feature_names() -> Dict[str, str]:
        return {
            'VoxelCount': 'abs_volume(mm^3) / voxel_mm^3 (continuous)',
            'L11': 'log-ellipsoid L11',
            'L22': 'log-ellipsoid L22',
            'L33': 'log-ellipsoid L33',
            'sqrt2_L12': 'sqrt(2) * L12',
            'sqrt2_L13': 'sqrt(2) * L13',
            'sqrt2_L23': 'sqrt(2) * L23',
        }
