#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature engineering module for particle analysis
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
import warnings

warnings.filterwarnings('ignore')


class FeatureEngineer:
    """Feature engineering for particle analysis"""
    
    def __init__(self, voxel_size_mm: Optional[float] = None):
        """Initialize feature engineer
        
        Args:
            voxel_size_mm: Voxel size in mm for spatial calculations
        """
        self.voxel_size_mm = voxel_size_mm
        
    def extract_geometric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract geometric features from particle data
        
        Args:
            df: DataFrame containing particle data
            
        Returns:
            DataFrame with geometric features
        """
        features = {}
        
        # Basic geometric quantities
        features['log_volume'] = np.log10(df['Volume3d (mm^3) '].values)
        
        # Ellipsoid axis lengths
        eigenvals = df[['EigenVal1', 'EigenVal2', 'EigenVal3']].values
        a, b, c = np.sqrt(eigenvals[:, 0]), np.sqrt(eigenvals[:, 1]), np.sqrt(eigenvals[:, 2])
        features['a'] = a
        features['b'] = b  
        features['c'] = c
        features['a_c_ratio'] = a / c
        features['a_b_ratio'] = a / b
        features['b_c_ratio'] = b / c
        
        # Sphericity
        features['sphericity'] = c / a  # min_axis / max_axis
        
        # Shape features (if available)
        if 'Elongation' in df.columns:
            features['elongation'] = df['Elongation'].values
        if 'Flatness' in df.columns:
            features['flatness'] = df['Flatness'].values
        
        # Compactness (if available)
        if 'VoxelFaceArea ' in df.columns and self.voxel_size_mm:
            surface_area = df['VoxelFaceArea '].values * (self.voxel_size_mm ** 2)
            volume = df['Volume3d (mm^3) '].values
            features['compactness'] = (36 * np.pi * volume**2) ** (1/3) / surface_area
        
        # Shortest bounding box axis (if available)
        bbox_cols = ['ExtentMin1 (mm) ', 'ExtentMax1 (mm) ', 'ExtentMin2 (mm) ', 'ExtentMax2 (mm) ', 'ExtentMin3 (mm) ', 'ExtentMax3 (mm) ']
        if all(col in df.columns for col in bbox_cols):
            extents = df[bbox_cols].values
            bbox_sizes = extents[:, [1,3,5]] - extents[:, [0,2,4]]  # Max - Min
            if self.voxel_size_mm:
                bbox_sizes_voxel = bbox_sizes / self.voxel_size_mm
                features['c_vox'] = np.min(bbox_sizes_voxel, axis=1)
        
        # Barycenter features (if available)
        if 'BaryCenterX (mm) ' in df.columns:
            features['barycenter_x'] = df['BaryCenterX (mm) '].values
        if 'BaryCenterY (mm) ' in df.columns:
            features['barycenter_y'] = df['BaryCenterY (mm) '].values
        if 'BaryCenterZ (mm) ' in df.columns:
            features['barycenter_z'] = df['BaryCenterZ (mm) '].values
        
        # Second moment features (if available)
        if 'BinMom2x (mm^2) ' in df.columns:
            features['bin_mom_2x'] = df['BinMom2x (mm^2) '].values
        if 'BinMom2y (mm^2) ' in df.columns:
            features['bin_mom_2y'] = df['BinMom2y (mm^2) '].values
        if 'BinMom2z (mm^2) ' in df.columns:
            features['bin_mom_2z'] = df['BinMom2z (mm^2) '].values
        if 'BinMomxy (mm^2) ' in df.columns:
            features['bin_mom_xy'] = df['BinMomxy (mm^2) '].values
        if 'BinMomxz (mm^2) ' in df.columns:
            features['bin_mom_xz'] = df['BinMomxz (mm^2) '].values
        if 'BinMomyz (mm^2) ' in df.columns:
            features['bin_mom_yz'] = df['BinMomyz (mm^2) '].values
        
        # Additional features
        if 'BorderVoxelCount ' in df.columns:
            features['border_voxel_count'] = df['BorderVoxelCount '].values
        if 'GreyMass (mm^3) ' in df.columns:
            features['grey_mass'] = df['GreyMass (mm^3) '].values
        
        return pd.DataFrame(features)
    
    def extract_orientation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract orientation-related features
        
        Args:
            df: DataFrame containing particle data
            
        Returns:
            DataFrame with orientation features
        """
        features = {}
        
        # Ellipsoid principal axes
        eigenvec1 = df[['EigenVec1X', 'EigenVec1Y', 'EigenVec1Z']].values
        eigenvec3 = df[['EigenVec3X', 'EigenVec3Y', 'EigenVec3Z']].values
        
        # Voxel axes
        voxel_axes = np.array([[1,0,0], [0,1,0], [0,0,1]])
        
        # Principal axis angles with voxel axes
        phi1_dots = np.abs(np.dot(eigenvec1, voxel_axes.T))
        phi3_dots = np.abs(np.dot(eigenvec3, voxel_axes.T))
        
        features['phi1_voxel_angle'] = np.min(np.arccos(np.clip(phi1_dots, 0, 1)), axis=1)
        features['phi3_voxel_angle'] = np.min(np.arccos(np.clip(phi3_dots, 0, 1)), axis=1)
        
        # Anisotropy indicators
        eigenvals = df[['EigenVal1', 'EigenVal2', 'EigenVal3']].values
        lambda_sum = np.sum(eigenvals, axis=1)
        features['lambda_diff_12'] = np.abs(eigenvals[:, 0] - eigenvals[:, 1])
        features['lambda_diff_23'] = np.abs(eigenvals[:, 1] - eigenvals[:, 2])
        features['anisotropy'] = (eigenvals[:, 0] - eigenvals[:, 2]) / lambda_sum
        
        return pd.DataFrame(features)
    
    def extract_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract volume-related features
        
        Args:
            df: DataFrame containing particle data
            
        Returns:
            DataFrame with volume features
        """
        features = {}
        
        volumes = df['Volume3d (mm^3) '].values
        
        # Volume percentiles
        sorted_volumes = np.sort(volumes)
        percentiles = np.searchsorted(sorted_volumes, volumes) / len(volumes)
        features['volume_percentile'] = percentiles
        
        # Volume statistics
        features['volume_relative_to_mean'] = volumes / np.mean(volumes)
        features['volume_relative_to_median'] = volumes / np.median(volumes)
        
        # Grey mass features (if available)
        if 'GreyMass (mm^3) ' in df.columns:
            grey_mass = df['GreyMass (mm^3) '].values
            features['grey_mass_ratio'] = grey_mass / volumes
            features['grey_mass_percentile'] = np.searchsorted(np.sort(grey_mass), grey_mass) / len(grey_mass)
        
        return pd.DataFrame(features)
    
    def extract_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract all features from particle data
        
        Args:
            df: DataFrame containing particle data
            
        Returns:
            DataFrame with all extracted features
        """
        # Extract different feature groups
        geo_features = self.extract_geometric_features(df)
        ori_features = self.extract_orientation_features(df)
        vol_features = self.extract_volume_features(df)
        
        # Combine features
        all_features = pd.concat([geo_features, ori_features, vol_features], axis=1)
        
        # Handle missing values
        all_features = all_features.fillna(0)
        
        return all_features
    
    def get_feature_names(self) -> list:
        """Get list of feature names
        
        Returns:
            List of feature names
        """
        # This would be populated based on the actual features extracted
        return [
            'log_volume', 'a', 'b', 'c', 'a_c_ratio', 'a_b_ratio', 'b_c_ratio',
            'sphericity', 'phi1_voxel_angle', 'phi3_voxel_angle',
            'lambda_diff_12', 'lambda_diff_23', 'anisotropy',
            'volume_percentile', 'volume_relative_to_mean', 'volume_relative_to_median',
            'elongation', 'flatness', 'compactness', 'c_vox',
            'barycenter_x', 'barycenter_y', 'barycenter_z',
            'bin_mom_2x', 'bin_mom_2y', 'bin_mom_2z', 'bin_mom_xy', 'bin_mom_xz', 'bin_mom_yz',
            'border_voxel_count', 'grey_mass', 'grey_mass_ratio', 'grey_mass_percentile'
        ]
