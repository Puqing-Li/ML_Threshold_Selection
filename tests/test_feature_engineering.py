#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for feature engineering module
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml_threshold_selection.feature_engineering import FeatureEngineer


class TestFeatureEngineer:
    """Test cases for FeatureEngineer"""
    
    def setup_method(self):
        """Set up test data"""
        self.engineer = FeatureEngineer(voxel_size_mm=0.01)
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'Volume3d (mm^3) ': [1e-6, 2e-6, 5e-7],
            'EigenVal1': [0.001, 0.002, 0.0005],
            'EigenVal2': [0.0009, 0.0018, 0.0004],
            'EigenVal3': [0.0008, 0.0016, 0.0003],
            'EigenVec1X': [1.0, 0.866, 0.707],
            'EigenVec1Y': [0.0, 0.500, 0.707],
            'EigenVec1Z': [0.0, 0.000, 0.000],
            'EigenVec2X': [0.0, 0.000, 0.000],
            'EigenVec2Y': [1.0, 0.000, 0.000],
            'EigenVec2Z': [0.0, 1.000, 1.000],
            'EigenVec3X': [0.0, 0.500, 0.707],
            'EigenVec3Y': [0.0, 0.866, 0.707],
            'EigenVec3Z': [1.0, 0.000, 0.000],
            'ExtentMin1': [0, 1, 0],
            'ExtentMax1': [2, 3, 1],
            'ExtentMin2': [0, 1, 0],
            'ExtentMax2': [2, 3, 1],
            'ExtentMin3': [0, 1, 0],
            'ExtentMax3': [2, 3, 1],
            'VoxelFaceArea': [24, 32, 18]
        })
    
    def test_extract_geometric_features(self):
        """Test geometric feature extraction"""
        features = self.engineer.extract_geometric_features(self.sample_data)
        
        # Check that features are extracted
        assert len(features) == len(self.sample_data)
        assert 'log_volume' in features.columns
        assert 'a' in features.columns
        assert 'b' in features.columns
        assert 'c' in features.columns
        assert 'a_c_ratio' in features.columns
        
        # Check log_volume calculation
        expected_log_vol = np.log10(self.sample_data['Volume3d (mm^3) '].values)
        np.testing.assert_array_almost_equal(features['log_volume'], expected_log_vol)
        
        # Check axis length calculation
        expected_a = np.sqrt(self.sample_data['EigenVal1'].values)
        np.testing.assert_array_almost_equal(features['a'], expected_a)
    
    def test_extract_orientation_features(self):
        """Test orientation feature extraction"""
        features = self.engineer.extract_orientation_features(self.sample_data)
        
        # Check that features are extracted
        assert len(features) == len(self.sample_data)
        assert 'phi1_voxel_angle' in features.columns
        assert 'phi3_voxel_angle' in features.columns
        assert 'anisotropy' in features.columns
        
        # Check angle calculations
        assert all(features['phi1_voxel_angle'] >= 0)
        assert all(features['phi1_voxel_angle'] <= np.pi/2)
        assert all(features['phi3_voxel_angle'] >= 0)
        assert all(features['phi3_voxel_angle'] <= np.pi/2)
    
    def test_extract_volume_features(self):
        """Test volume feature extraction"""
        features = self.engineer.extract_volume_features(self.sample_data)
        
        # Check that features are extracted
        assert len(features) == len(self.sample_data)
        assert 'volume_percentile' in features.columns
        assert 'volume_relative_to_mean' in features.columns
        
        # Check percentile calculation
        assert all(features['volume_percentile'] >= 0)
        assert all(features['volume_percentile'] <= 1)
    
    def test_extract_all_features(self):
        """Test extraction of all features"""
        features = self.engineer.extract_all_features(self.sample_data)
        
        # Check that all feature types are included
        assert 'log_volume' in features.columns  # Geometric
        assert 'phi1_voxel_angle' in features.columns  # Orientation
        assert 'volume_percentile' in features.columns  # Volume
        
        # Check no missing values
        assert not features.isnull().any().any()
    
    def test_missing_columns(self):
        """Test handling of missing columns"""
        # Remove some optional columns
        minimal_data = self.sample_data.drop(columns=['ExtentMin1', 'ExtentMax1', 'VoxelFaceArea'])
        
        features = self.engineer.extract_all_features(minimal_data)
        
        # Should still work without optional columns
        assert len(features) == len(minimal_data)
        assert not features.isnull().any().any()
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame"""
        empty_df = pd.DataFrame()
        
        with pytest.raises(KeyError):
            self.engineer.extract_all_features(empty_df)
    
    def test_get_feature_names(self):
        """Test feature name retrieval"""
        feature_names = self.engineer.get_feature_names()
        
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
        assert all(isinstance(name, str) for name in feature_names)


if __name__ == "__main__":
    pytest.main([__file__])
