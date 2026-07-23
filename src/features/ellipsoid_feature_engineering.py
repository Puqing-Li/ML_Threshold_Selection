#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility wrapper for the corrected seven-feature representation.

The paper pipeline uses :class:`ResolutionAwareFeatureEngineer` directly. This
module keeps the historical public class name while delegating its mathematics
to that implementation so the two routes cannot drift.
"""

import pandas as pd
from typing import Optional, Dict
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from .res_aware_feature_engineering import ResolutionAwareFeatureEngineer


class JoshuaFeatureEngineerFixed:
    """Backward-compatible interface to the corrected seven-feature engineer."""
    
    def __init__(self, voxel_size_um: Optional[float] = None):
        self.voxel_size_um = voxel_size_um
        self.scaler = StandardScaler()
        self.is_fitted = False

    def extract_joshua_features(self, df: pd.DataFrame, fit_scaler: bool = True) -> pd.DataFrame:
        if self.voxel_size_um is None or self.voxel_size_um <= 0:
            raise ValueError("Voxel size (um) must be provided and positive")

        voxel_size_mm = self.voxel_size_um / 1000.0
        corrected = ResolutionAwareFeatureEngineer()._build_unscaled_features(
            df, voxel_size_mm
        )
        result_df = corrected.rename(columns={'VoxelCount': 'Volume'})
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
            'Volume': 'Continuous voxel count: volume (mm^3) / voxel volume (mm^3)',
            'L11': 'Log-ellipsoid tensor diagonal element L11',
            'L22': 'Log-ellipsoid tensor diagonal element L22',
            'L33': 'Log-ellipsoid tensor diagonal element L33',
            'sqrt2_L12': 'Mandel off-diagonal element sqrt(2) * L12',
            'sqrt2_L13': 'Mandel off-diagonal element sqrt(2) * L13',
            'sqrt2_L23': 'Mandel off-diagonal element sqrt(2) * L23',
        }
