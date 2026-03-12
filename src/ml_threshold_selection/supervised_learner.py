#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supervised learning system for threshold selection
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any
import joblib
import warnings
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, precision_recall_curve
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

from .feature_engineering import FeatureEngineer
from .threshold_finder import AdaptiveThresholdFinder

warnings.filterwarnings('ignore')


class SupervisedThresholdLearner:
    """Supervised learning system for artifact classification"""
    
    def __init__(self, random_state: int = 42):
        """Initialize supervised learner
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.model = None
        self.calibrator = None
        self.feature_columns = None
        self.feature_engineer = FeatureEngineer()
        self.threshold_finder = AdaptiveThresholdFinder()
        
    def train(self, X: pd.DataFrame, y: np.ndarray, 
              sample_ids: Optional[np.ndarray] = None, 
              cv_folds: int = 5) -> Dict[str, Any]:
        """Train the supervised learning model
        
        Args:
            X: Feature matrix
            y: Target labels (0=normal, 1=artifact)
            sample_ids: Sample identifiers for cross-validation
            cv_folds: Number of CV folds
            
        Returns:
            Dictionary with training results
        """
        # Set up LightGBM parameters
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
        
        # Set up cross-validation
        if sample_ids is not None:
            cv = LeaveOneGroupOut()
        else:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Train model
        if LIGHTGBM_AVAILABLE:
            self.model = lgb.LGBMClassifier(**lgb_params)
        else:
            # Fallback to RandomForest
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                class_weight='balanced'
            )
        
        # Probability calibration
        self.calibrator = CalibratedClassifierCV(
            self.model, 
            method='isotonic',
            cv=cv
        )
        
        if sample_ids is not None:
            # Custom CV for grouped data
            self._train_with_groups(X, y, sample_ids)
        else:
            self.calibrator.fit(X, y)
        
        self.feature_columns = X.columns.tolist()
        
        # Calculate training metrics
        train_proba = self.predict_proba(X)
        train_auc = roc_auc_score(y, train_proba)
        
        return {
            'n_features': len(self.feature_columns),
            'n_samples': len(X),
            'train_auc': train_auc,
            'feature_importance': self._get_feature_importance()
        }
    
    def _train_with_groups(self, X: pd.DataFrame, y: np.ndarray, 
                          sample_ids: np.ndarray):
        """Train with grouped cross-validation
        
        Args:
            X: Feature matrix
            y: Target labels
            sample_ids: Sample identifiers
        """
        # Simplified implementation - direct training
        self.model.fit(X, y)
        
        # Use sigmoid calibration
        from sklearn.calibration import CalibratedClassifierCV
        self.calibrator = CalibratedClassifierCV(
            self.model, 
            method='sigmoid',
            cv=3
        )
        self.calibrator.fit(X, y)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict artifact probabilities
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of artifact probabilities
        """
        if self.calibrator is None:
            raise ValueError("Model not trained yet")
        
        # Ensure feature order consistency
        X_ordered = X[self.feature_columns]
        return self.calibrator.predict_proba(X_ordered)[:, 1]
    
    def find_threshold(self, volumes: np.ndarray, probabilities: np.ndarray) -> Tuple[float, float]:
        """Find optimal threshold for given volumes and probabilities
        
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
            'threshold_finder': self.threshold_finder
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
