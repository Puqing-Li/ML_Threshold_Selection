#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semi-supervised learning system using expert thresholds
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any, List
import joblib
import warnings
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

from .feature_engineering import FeatureEngineer
from .threshold_finder import AdaptiveThresholdFinder

warnings.filterwarnings('ignore')


class ExpertThresholdProcessor:
    """Processor for expert-determined thresholds"""
    
    def __init__(self):
        """Initialize processor"""
        self.expert_thresholds = {}  # {sample_id: threshold_info}
        self.sample_data = {}  # {sample_id: dataframe}
        
    def add_expert_threshold(self, sample_id: str, threshold: float, 
                           confidence: float = 1.0, notes: str = ""):
        """Add expert-determined threshold
        
        Args:
            sample_id: Sample identifier
            threshold: Volume threshold (mm³)
            confidence: Confidence level (0-1)
            notes: Additional notes
        """
        self.expert_thresholds[sample_id] = {
            'threshold': threshold,
            'confidence': confidence,
            'notes': notes
        }
    
    def load_sample_data(self, sample_id: str, csv_path: str) -> bool:
        """Load sample data from CSV
        
        Args:
            sample_id: Sample identifier
            csv_path: Path to CSV file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            df = pd.read_csv(csv_path)
            self.sample_data[sample_id] = df
            return True
        except Exception as e:
            print(f"Error loading sample {sample_id}: {e}")
            return False
    
    def generate_pseudo_labels(self, sample_id: str, method: str = 'threshold_based') -> Optional[np.ndarray]:
        """Generate pseudo-labels based on expert threshold
        
        Args:
            sample_id: Sample identifier
            method: Pseudo-label generation method
            
        Returns:
            Array of pseudo-labels or None if failed
        """
        if sample_id not in self.sample_data or sample_id not in self.expert_thresholds:
            return None
        
        df = self.sample_data[sample_id]
        threshold = self.expert_thresholds[sample_id]['threshold']
        
        if method == 'threshold_based':
            # Direct threshold-based labeling
            pseudo_labels = (df['Volume3d (mm^3) '] < threshold).astype(int)
            
        elif method == 'threshold_with_features':
            # Soft labeling with feature consideration
            pseudo_labels = self._generate_probabilistic_labels(df, threshold)
            
        elif method == 'threshold_with_uncertainty':
            # Uncertainty-aware labeling
            pseudo_labels = self._generate_uncertainty_labels(df, threshold)
        
        else:
            raise ValueError(f"Unknown pseudo-label method: {method}")
        
        # Apply confidence weighting
        confidence = self.expert_thresholds[sample_id]['confidence']
        pseudo_labels = pseudo_labels * confidence
        
        return pseudo_labels
    
    def _generate_probabilistic_labels(self, df: pd.DataFrame, threshold: float) -> np.ndarray:
        """Generate probabilistic pseudo-labels
        
        Args:
            df: Sample data
            threshold: Volume threshold
            
        Returns:
            Array of probabilistic labels
        """
        volumes = df['Volume3d (mm^3) '].values
        log_volumes = np.log10(volumes)
        log_threshold = np.log10(threshold)
        
        # Sigmoid function for soft labeling
        soft_labels = 1 / (1 + np.exp(-(log_threshold - log_volumes) * 5))
        return soft_labels
    
    def _generate_uncertainty_labels(self, df: pd.DataFrame, threshold: float) -> np.ndarray:
        """Generate uncertainty-aware pseudo-labels
        
        Args:
            df: Sample data
            threshold: Volume threshold
            
        Returns:
            Array of uncertainty-aware labels
        """
        volumes = df['Volume3d (mm^3) '].values
        
        # Define uncertainty region
        uncertainty_factor = 0.1
        lower_bound = threshold * (1 - uncertainty_factor)
        upper_bound = threshold * (1 + uncertainty_factor)
        
        # Hard labels for certain regions
        hard_labels = np.zeros(len(volumes))
        hard_labels[volumes < lower_bound] = 1  # Certain artifacts
        hard_labels[volumes > upper_bound] = 0  # Certain normal particles
        
        # Probabilistic labels for uncertainty region
        uncertain_mask = (volumes >= lower_bound) & (volumes <= upper_bound)
        if np.any(uncertain_mask):
            uncertain_volumes = volumes[uncertain_mask]
            prob_labels = (upper_bound - uncertain_volumes) / (upper_bound - lower_bound)
            hard_labels[uncertain_mask] = prob_labels
        
        return hard_labels
    
    def create_training_dataset(self, method: str = 'threshold_based') -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray], Optional[np.ndarray]]:
        """Create training dataset from expert thresholds
        
        Args:
            method: Pseudo-label generation method
            
        Returns:
            Tuple of (training_df, labels, sample_ids)
        """
        all_data = []
        all_labels = []
        all_sample_ids = []
        
        for sample_id in self.sample_data:
            if sample_id not in self.expert_thresholds:
                continue
                
            df = self.sample_data[sample_id].copy()
            pseudo_labels = self.generate_pseudo_labels(sample_id, method)
            
            if pseudo_labels is not None:
                df['pseudo_label'] = pseudo_labels
                df['sample_id'] = sample_id
                
                all_data.append(df)
                all_labels.extend(pseudo_labels)
                all_sample_ids.extend([sample_id] * len(df))
        
        if not all_data:
            return None, None, None
        
        # Combine all data
        training_df = pd.concat(all_data, ignore_index=True)
        training_labels = np.array(all_labels)
        sample_ids = np.array(all_sample_ids)
        
        return training_df, training_labels, sample_ids


class SemiSupervisedThresholdLearner:
    """Semi-supervised learning system using expert thresholds"""
    
    def __init__(self, random_state: int = 42):
        """Initialize semi-supervised learner
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.model = None
        self.calibrator = None
        self.feature_columns = None
        self.feature_engineer = FeatureEngineer()
        self.threshold_finder = AdaptiveThresholdFinder()
        self.processor = ExpertThresholdProcessor()
        
    def add_expert_threshold(self, sample_id: str, threshold: float, 
                           confidence: float = 1.0, notes: str = ""):
        """Add expert threshold
        
        Args:
            sample_id: Sample identifier
            threshold: Volume threshold
            confidence: Confidence level
            notes: Additional notes
        """
        self.processor.add_expert_threshold(sample_id, threshold, confidence, notes)
    
    def load_sample_data(self, sample_id: str, csv_path: str) -> bool:
        """Load sample data
        
        Args:
            sample_id: Sample identifier
            csv_path: Path to CSV file
            
        Returns:
            True if successful
        """
        return self.processor.load_sample_data(sample_id, csv_path)
    
    def train(self, method: str = 'threshold_based', 
              model_type: str = 'lightgbm') -> Dict[str, Any]:
        """Train semi-supervised model
        
        Args:
            method: Pseudo-label generation method
            model_type: Model type ('lightgbm' or 'random_forest')
            
        Returns:
            Dictionary with training results
        """
        # Create training dataset
        training_df, training_labels, sample_ids = self.processor.create_training_dataset(method)
        
        if training_df is None:
            raise ValueError("No training data available")
        
        # Extract features
        features = self.feature_engineer.extract_all_features(training_df)
        
        # Train model
        if model_type == 'lightgbm':
            self._train_lightgbm(features, training_labels, sample_ids)
        elif model_type == 'random_forest':
            self._train_random_forest(features, training_labels)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.feature_columns = features.columns.tolist()
        
        # Calculate training metrics
        train_proba = self.predict_proba(features)
        if np.any((training_labels > 0) & (training_labels < 1)):
            # Soft labels - use MSE
            train_score = np.mean((train_proba - training_labels) ** 2)
        else:
            # Hard labels - use AUC
            from sklearn.metrics import roc_auc_score
            train_score = roc_auc_score(training_labels, train_proba)
        
        return {
            'n_features': len(self.feature_columns),
            'n_samples': len(features),
            'train_score': train_score,
            'feature_importance': self._get_feature_importance()
        }
    
    def _train_lightgbm(self, X: pd.DataFrame, y: np.ndarray, sample_ids: Optional[np.ndarray] = None):
        """Train LightGBM model
        
        Args:
            X: Feature matrix
            y: Target labels
            sample_ids: Sample identifiers
        """
        # Check for soft labels
        has_soft_labels = np.any((y > 0) & (y < 1))
        
        if LIGHTGBM_AVAILABLE:
            if has_soft_labels:
                # Use regression for soft labels
                lgb_params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.9,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'verbose': -1,
                    'random_state': self.random_state
                }
                self.model = lgb.LGBMRegressor(**lgb_params)
            else:
                # Use classification for hard labels
                lgb_params = {
                    'objective': 'binary',
                    'metric': 'auc',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.9,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'verbose': -1,
                    'random_state': self.random_state,
                    'class_weight': 'balanced'
                }
                self.model = lgb.LGBMClassifier(**lgb_params)
        else:
            # Fallback to RandomForest
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                class_weight='balanced'
            )
        
        # Train model
        self.model.fit(X, y)
        
        # Calibration
        if has_soft_labels:
            # For regression, use sigmoid calibration
            self.calibrator = CalibratedClassifierCV(
                self.model, 
                method='sigmoid',
                cv=3
            )
            # Convert to hard labels for calibration
            y_hard = (y > 0.5).astype(int)
            self.calibrator.fit(X, y_hard)
        else:
            # For classification, use isotonic calibration
            self.calibrator = CalibratedClassifierCV(
                self.model, 
                method='isotonic',
                cv=3
            )
            self.calibrator.fit(X, y)
    
    def _train_random_forest(self, X: pd.DataFrame, y: np.ndarray):
        """Train Random Forest model
        
        Args:
            X: Feature matrix
            y: Target labels
        """
        # Convert soft labels to hard labels for Random Forest
        y_hard = (y > 0.5).astype(int)
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=self.random_state,
            class_weight='balanced'
        )
        
        self.model.fit(X, y_hard)
        self.calibrator = None  # Random Forest doesn't need calibration
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict artifact probabilities
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of artifact probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        X_ordered = X[self.feature_columns]
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_ordered)[:, 1]
        else:
            # Regression model
            pred = self.model.predict(X_ordered)
            return np.clip(pred, 0, 1)
    
    def find_threshold(self, volumes: np.ndarray, probabilities: np.ndarray) -> Tuple[float, float]:
        """Find optimal threshold
        
        Args:
            volumes: Array of particle volumes
            probabilities: Array of artifact probabilities
            
        Returns:
            Tuple of (optimal_threshold, uncertainty)
        """
        return self.threshold_finder.find_threshold(volumes, probabilities)
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores
        
        Returns:
            Dictionary of feature importance
        """
        if self.model is None:
            return {}
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            return dict(zip(self.feature_columns, importance))
        return {}
    
    def save_model(self, filepath: str):
        """Save trained model
        
        Args:
            filepath: Path to save model
        """
        model_data = {
            'model': self.model,
            'calibrator': self.calibrator,
            'feature_columns': self.feature_columns,
            'feature_engineer': self.feature_engineer,
            'threshold_finder': self.threshold_finder,
            'expert_thresholds': self.processor.expert_thresholds
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str):
        """Load trained model
        
        Args:
            filepath: Path to model file
        """
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.calibrator = model_data['calibrator']
        self.feature_columns = model_data['feature_columns']
        self.feature_engineer = model_data.get('feature_engineer', FeatureEngineer())
        self.threshold_finder = model_data.get('threshold_finder', AdaptiveThresholdFinder())
        self.processor.expert_thresholds = model_data.get('expert_thresholds', {})
    
    def analyze_sample(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze a single sample
        
        Args:
            df: Sample data DataFrame
            
        Returns:
            Dictionary with analysis results
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Extract features
        features = self.feature_engineer.extract_all_features(df)
        
        # Predict probabilities
        probabilities = self.predict_proba(features)
        
        # Find threshold
        volumes = df['Volume3d (mm^3) '].values
        threshold, uncertainty = self.find_threshold(volumes, probabilities)
        
        # Calculate statistics
        retained_mask = volumes >= threshold
        n_retained = np.sum(retained_mask)
        n_total = len(volumes)
        removal_rate = (n_total - n_retained) / n_total * 100
        
        return {
            'probabilities': probabilities,
            'threshold': threshold,
            'uncertainty': uncertainty,
            'n_retained': n_retained,
            'n_total': n_total,
            'removal_rate': removal_rate,
            'mean_probability': np.mean(probabilities),
            'high_prob_count': np.sum(probabilities > 0.5)
        }
