#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data format validator for particle analysis data
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class DataValidator:
    """Validator for particle analysis data format"""
    
    def __init__(self):
        """Initialize validator with expected column formats"""
        self.required_columns = [
            'Volume3d (mm^3) ',
            'EigenVal1', 'EigenVal2', 'EigenVal3',
            'EigenVec1X', 'EigenVec1Y', 'EigenVec1Z',
            'EigenVec2X', 'EigenVec2Y', 'EigenVec2Z',
            'EigenVec3X', 'EigenVec3Y', 'EigenVec3Z'
        ]
        
        self.optional_columns = [
            'BaryCenterX (mm) ', 'BaryCenterY (mm) ', 'BaryCenterZ (mm) ',
            'Anisotropy', 'Elongation', 'Flatness',
            'ExtentMin1 (mm) ', 'ExtentMax1 (mm) ',
            'ExtentMin2 (mm) ', 'ExtentMax2 (mm) ',
            'ExtentMin3 (mm) ', 'ExtentMax3 (mm) ',
            'BinMom2x (mm^2) ', 'BinMom2y (mm^2) ', 'BinMom2z (mm^2) ',
            'BinMomxy (mm^2) ', 'BinMomxz (mm^2) ', 'BinMomyz (mm^2) ',
            'VoxelFaceArea ', 'BorderVoxelCount ', 'GreyMass (mm^3) ',
            'indexMaterials', 'SampleID'
        ]
        
        self.supervised_columns = ['label']
    
    def validate_csv(self, filepath: str, mode: str = 'both') -> Dict[str, any]:
        """Validate CSV file format
        
        Args:
            filepath: Path to CSV file
            mode: Validation mode ('supervised', 'semi_supervised', or 'both')
            
        Returns:
            Dictionary with validation results
        """
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            return {
                'valid': False,
                'error': f"Failed to read CSV file: {e}",
                'suggestions': ["Check if file exists and is readable", "Verify CSV format"]
            }
        
        return self.validate_dataframe(df, mode)
    
    def validate_dataframe(self, df: pd.DataFrame, mode: str = 'both') -> Dict[str, any]:
        """Validate DataFrame format
        
        Args:
            df: DataFrame to validate
            mode: Validation mode ('supervised', 'semi_supervised', or 'both')
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'suggestions': [],
            'missing_required': [],
            'missing_optional': [],
            'extra_columns': [],
            'data_quality': {}
        }
        
        # Check required columns
        missing_required = [col for col in self.required_columns if col not in df.columns]
        if missing_required:
            results['valid'] = False
            results['missing_required'] = missing_required
            results['errors'].append(f"Missing required columns: {missing_required}")
        
        # Check mode-specific columns
        if mode in ['supervised', 'both']:
            missing_supervised = [col for col in self.supervised_columns if col not in df.columns]
            if missing_supervised:
                results['valid'] = False
                results['missing_required'].extend(missing_supervised)
                results['errors'].append(f"Missing supervised learning columns: {missing_supervised}")
        
        # Check optional columns
        missing_optional = [col for col in self.optional_columns if col not in df.columns]
        if missing_optional:
            results['missing_optional'] = missing_optional
            results['warnings'].append(f"Missing optional columns: {missing_optional}")
        
        # Check for extra columns
        all_expected = self.required_columns + self.optional_columns + self.supervised_columns
        extra_columns = [col for col in df.columns if col not in all_expected]
        if extra_columns:
            results['extra_columns'] = extra_columns
            results['warnings'].append(f"Extra columns found: {extra_columns}")
        
        # Check data quality
        if results['valid']:
            results['data_quality'] = self._check_data_quality(df)
        
        # Generate suggestions
        results['suggestions'] = self._generate_suggestions(results)
        
        return results
    
    def _check_data_quality(self, df: pd.DataFrame) -> Dict[str, any]:
        """Check data quality issues
        
        Args:
            df: DataFrame to check
            
        Returns:
            Dictionary with data quality metrics
        """
        quality = {
            'n_particles': len(df),
            'missing_values': {},
            'invalid_values': {},
            'data_types': {},
            'value_ranges': {}
        }
        
        # Check missing values
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                quality['missing_values'][col] = missing_count
        
        # Check data types
        for col in df.columns:
            quality['data_types'][col] = str(df[col].dtype)
        
        # Check value ranges for key columns
        key_columns = ['Volume3d (mm^3) ', 'EigenVal1', 'EigenVal2', 'EigenVal3']
        for col in key_columns:
            if col in df.columns:
                values = df[col].dropna()
                if len(values) > 0:
                    quality['value_ranges'][col] = {
                        'min': float(values.min()),
                        'max': float(values.max()),
                        'mean': float(values.mean()),
                        'std': float(values.std())
                    }
        
        # Check for invalid values
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                invalid_count = np.isinf(df[col]).sum() + np.isnan(df[col]).sum()
                if invalid_count > 0:
                    quality['invalid_values'][col] = int(invalid_count)
        
        return quality
    
    def _generate_suggestions(self, results: Dict[str, any]) -> List[str]:
        """Generate suggestions based on validation results
        
        Args:
            results: Validation results
            
        Returns:
            List of suggestions
        """
        suggestions = []
        
        if results['missing_required']:
            suggestions.append("Add missing required columns to your CSV file")
            suggestions.append("Check column names for typos (note trailing spaces in unit columns)")
        
        if results['missing_optional']:
            suggestions.append("Consider adding optional columns for better feature extraction")
            suggestions.append("Optional columns can improve model performance")
        
        if results['extra_columns']:
            suggestions.append("Extra columns will be ignored during processing")
            suggestions.append("Consider removing unused columns to reduce file size")
        
        if results['data_quality'].get('missing_values'):
            suggestions.append("Handle missing values before processing")
            suggestions.append("Consider filling missing values with appropriate defaults")
        
        if results['data_quality'].get('invalid_values'):
            suggestions.append("Remove or fix invalid values (NaN, Inf)")
            suggestions.append("Check data processing pipeline for errors")
        
        return suggestions
    
    def get_column_mapping(self, df: pd.DataFrame) -> Dict[str, str]:
        """Get mapping of actual columns to expected columns
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary mapping actual columns to expected columns
        """
        mapping = {}
        
        # Check for exact matches
        for col in df.columns:
            if col in self.required_columns + self.optional_columns + self.supervised_columns:
                mapping[col] = col
        
        # Check for close matches (case insensitive, space variations)
        for actual_col in df.columns:
            if actual_col not in mapping:
                for expected_col in self.required_columns + self.optional_columns + self.supervised_columns:
                    # Remove trailing spaces and compare
                    actual_clean = actual_col.rstrip()
                    expected_clean = expected_col.rstrip()
                    
                    if actual_clean.lower() == expected_clean.lower():
                        mapping[actual_col] = expected_col
                        break
        
        return mapping
    
    def suggest_column_fixes(self, df: pd.DataFrame) -> List[str]:
        """Suggest fixes for column name issues
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            List of suggested fixes
        """
        fixes = []
        
        # Check for missing trailing spaces
        for col in df.columns:
            if col in ['Volume3d (mm^3)', 'BaryCenterX (mm)', 'BaryCenterY (mm)', 'BaryCenterZ (mm)',
                      'ExtentMin1 (mm)', 'ExtentMax1 (mm)', 'ExtentMin2 (mm)', 'ExtentMax2 (mm)',
                      'ExtentMin3 (mm)', 'ExtentMax3 (mm)', 'BinMom2x (mm^2)', 'BinMom2y (mm^2)',
                      'BinMom2z (mm^2)', 'BinMomxy (mm^2)', 'BinMomxz (mm^2)', 'BinMomyz (mm^2)',
                      'VoxelFaceArea', 'BorderVoxelCount', 'GreyMass (mm^3)']:
                fixes.append(f"Column '{col}' should have a trailing space: '{col} '")
        
        # Check for extra spaces
        for col in df.columns:
            if col.endswith('  '):  # Double space
                fixes.append(f"Column '{col}' has extra spaces, should be: '{col.rstrip()} '")
        
        return fixes


def validate_data_file(filepath: str, mode: str = 'both') -> None:
    """Convenience function to validate a data file and print results
    
    Args:
        filepath: Path to CSV file
        mode: Validation mode
    """
    validator = DataValidator()
    results = validator.validate_csv(filepath, mode)
    
    print(f"Data Validation Results for: {filepath}")
    print("=" * 50)
    
    if results['valid']:
        print("✅ Data format is valid!")
    else:
        print("❌ Data format has errors:")
        for error in results['errors']:
            print(f"  - {error}")
    
    if results['warnings']:
        print("\n⚠️  Warnings:")
        for warning in results['warnings']:
            print(f"  - {warning}")
    
    if results['suggestions']:
        print("\n💡 Suggestions:")
        for suggestion in results['suggestions']:
            print(f"  - {suggestion}")
    
    if results['data_quality']:
        print(f"\n📊 Data Quality:")
        print(f"  - Particles: {results['data_quality']['n_particles']}")
        if results['data_quality']['missing_values']:
            print(f"  - Missing values: {results['data_quality']['missing_values']}")
        if results['data_quality']['invalid_values']:
            print(f"  - Invalid values: {results['data_quality']['invalid_values']}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python data_validator.py <csv_file> [mode]")
        print("Modes: supervised, semi_supervised, both (default)")
        sys.exit(1)
    
    filepath = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else 'both'
    
    validate_data_file(filepath, mode)
