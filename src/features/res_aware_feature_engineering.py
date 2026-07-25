#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resolution-aware feature engineering: a continuous Voxel Count (absolute volume
divided by the cube of the voxel edge length) plus the 6 log-ellipsoid tensor
components. Outputs 7 features with StandardScaler (fit/transform modes).
Absolute Volume and Voxel Size are intentionally excluded from the model input.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict
from sklearn.preprocessing import StandardScaler


class ResolutionAwareFeatureEngineer:
    REFERENCE_LENGTH_MM = 1.0

    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.current_voxel_size_mm: Optional[float] = None

    def _compute_joshua_tensor(self, df: pd.DataFrame) -> np.ndarray:
        eigenvals = df[['EigenVal1', 'EigenVal2', 'EigenVal3']].values.astype(float)
        if not np.all(np.isfinite(eigenvals)) or np.any(eigenvals <= 0):
            bad = int((~np.isfinite(eigenvals)).any(axis=1).sum()
                      + (eigenvals <= 0).any(axis=1).sum())
            raise ValueError(
                f'EigenVal1-3 must be finite and strictly positive, but {bad} of '
                f'{len(eigenvals)} objects fail this. A raw Avizo Label-Analysis '
                'export still contains objects whose fitted ellipsoid is '
                'degenerate. Prepare the table first with "0. Prepare Raw Data" '
                '(tools/BatchFile.py) and load the resulting table.'
            )
        eigenvec1 = df[['EigenVec1X', 'EigenVec1Y', 'EigenVec1Z']].values
        eigenvec2 = df[['EigenVec2X', 'EigenVec2Y', 'EigenVec2Z']].values
        eigenvec3 = df[['EigenVec3X', 'EigenVec3Y', 'EigenVec3Z']].values
        semiaxes = np.sqrt(5.0 * eigenvals)
        a1, a2, a3 = semiaxes[:, 0], semiaxes[:, 1], semiaxes[:, 2]
        l1, l2, l3 = (
            np.log(a1 / self.REFERENCE_LENGTH_MM),
            np.log(a2 / self.REFERENCE_LENGTH_MM),
            np.log(a3 / self.REFERENCE_LENGTH_MM),
        )
        Q = np.stack([eigenvec1, eigenvec2, eigenvec3], axis=1)  # (n,3,3)
        if not np.all(np.isfinite(Q)):
            raise ValueError('EigenVec1-3 must be finite')
        u, _, vh = np.linalg.svd(Q)
        Q = u @ vh
        logE = np.zeros((len(df), 3, 3))
        logE[:, 0, 0] = -2 * l1
        logE[:, 1, 1] = -2 * l2
        logE[:, 2, 2] = -2 * l3
        L = Q.transpose(0, 2, 1) @ logE @ Q
        L = 0.5 * (L + L.transpose(0, 2, 1))
        out = np.zeros((len(df), 6))
        out[:, 0] = L[:, 0, 0]
        out[:, 1] = L[:, 1, 1]
        out[:, 2] = L[:, 2, 2]
        out[:, 3] = np.sqrt(2.0) * L[:, 0, 1]
        out[:, 4] = np.sqrt(2.0) * L[:, 0, 2]
        out[:, 5] = np.sqrt(2.0) * L[:, 1, 2]
        return out

    def _build_unscaled_features(self, df: pd.DataFrame, voxel_size_mm: float) -> pd.DataFrame:
        abs_volume = df['Volume3d (mm^3) '].values.astype(float)
        voxel_count = abs_volume / (voxel_size_mm ** 3)
        joshua6 = self._compute_joshua_tensor(df)
        return pd.DataFrame({
            'VoxelCount': voxel_count,
            'L11': joshua6[:, 0],
            'L22': joshua6[:, 1],
            'L33': joshua6[:, 2],
            'sqrt2_L12': joshua6[:, 3],
            'sqrt2_L13': joshua6[:, 4],
            'sqrt2_L23': joshua6[:, 5],
        }, index=df.index)

    def _scale_features(self, features: pd.DataFrame, fit_scaler: bool) -> pd.DataFrame:
        if fit_scaler and not self.is_fitted:
            scaled = self.scaler.fit_transform(features)
            self.is_fitted = True
        else:
            scaled = self.scaler.transform(features)
        return pd.DataFrame(scaled, columns=features.columns, index=features.index)

    def extract(self, df: pd.DataFrame, voxel_size_mm: Optional[float], fit_scaler: bool) -> pd.DataFrame:
        if voxel_size_mm is None or voxel_size_mm <= 0:
            raise ValueError('voxel_size_mm must be provided (>0) for resolution-aware features')
        self.current_voxel_size_mm = voxel_size_mm
        features = self._build_unscaled_features(df, voxel_size_mm)
        return self._scale_features(features, fit_scaler)

    def extract_by_sample(
        self,
        df: pd.DataFrame,
        voxel_sizes: Dict[str, float],
        fit_scaler: bool,
        sample_column: str = 'SampleID',
    ) -> pd.DataFrame:
        if sample_column not in df.columns:
            raise ValueError(f'{sample_column} column is required for sample-specific voxel sizes')
        features = pd.DataFrame(index=df.index, columns=[
            'VoxelCount', 'L11', 'L22', 'L33', 'sqrt2_L12', 'sqrt2_L13', 'sqrt2_L23'
        ], dtype=float)
        for sample_id in df[sample_column].unique():
            if sample_id not in voxel_sizes or float(voxel_sizes[sample_id]) <= 0:
                raise ValueError(f'Missing valid voxel size for sample {sample_id}')
            mask = df[sample_column] == sample_id
            features.loc[mask] = self._build_unscaled_features(
                df.loc[mask], float(voxel_sizes[sample_id])
            ).values
        return self._scale_features(features, fit_scaler)

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
