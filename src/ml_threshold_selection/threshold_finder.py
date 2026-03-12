#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptive threshold finder for particle analysis
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
import warnings

warnings.filterwarnings('ignore')


class AdaptiveThresholdFinder:
    """Adaptive threshold finder using A(V) curve analysis"""
    
    def __init__(self, epsilon: float = 0.03, tau: float = 0.02, n_min: int = 50):
        """Initialize threshold finder
        
        Args:
            epsilon: Tolerance for artifact rate (default: 0.03)
            tau: Platform detection threshold (default: 0.02)
            n_min: Minimum number of particles to retain (default: 50)
        """
        self.epsilon = epsilon
        self.tau = tau
        self.n_min = n_min
        
    def find_threshold(self, volumes: np.ndarray, probabilities: np.ndarray, 
                      method: str = 'kneedle') -> Tuple[float, float]:
        """Find optimal threshold using A(V) curve analysis
        
        Args:
            volumes: Array of particle volumes
            probabilities: Array of artifact probabilities
            method: Method for threshold selection ('kneedle' or 'first_valid')
            
        Returns:
            Tuple of (optimal_threshold, uncertainty)
        """
        # Sort by volume
        sort_idx = np.argsort(volumes)
        volumes_sorted = volumes[sort_idx]
        probs_sorted = probabilities[sort_idx]
        
        # Calculate expected artifact rate A(V)
        n_particles = len(volumes)
        artifact_rates = []
        volume_thresholds = []
        
        for i in range(n_particles):
            if i < self.n_min - 1:  # Ensure at least n_min particles retained
                continue
                
            v_thresh = volumes_sorted[i]
            retained_probs = probs_sorted[i:]
            a_rate = np.mean(retained_probs)
            
            artifact_rates.append(a_rate)
            volume_thresholds.append(v_thresh)
        
        artifact_rates = np.array(artifact_rates)
        volume_thresholds = np.array(volume_thresholds)
        
        # Find valid thresholds
        valid_mask = (
            (artifact_rates <= self.epsilon) &  # Tolerance condition
            (np.arange(len(artifact_rates)) >= self.n_min - 1)  # Min particles condition
        )
        
        if not np.any(valid_mask):
            # If no valid thresholds, return most lenient threshold
            return volumes_sorted[self.n_min - 1], 0.0
        
        valid_indices = np.where(valid_mask)[0]
        
        if method == 'kneedle':
            # Use Kneedle algorithm to find knee point
            v_min_star = self._find_kneedle_point(volume_thresholds, artifact_rates, valid_indices)
        else:
            # Select first valid threshold
            v_min_star = volume_thresholds[valid_indices[0]]
        
        # Calculate uncertainty
        uncertainty = self._estimate_uncertainty(volumes, probabilities, v_min_star)
        
        return v_min_star, uncertainty
    
    def _find_kneedle_point(self, volumes: np.ndarray, rates: np.ndarray, 
                           valid_indices: np.ndarray) -> float:
        """Find knee point using Kneedle algorithm
        
        Args:
            volumes: Volume thresholds
            rates: Artifact rates
            valid_indices: Indices of valid thresholds
            
        Returns:
            Optimal threshold value
        """
        if len(valid_indices) < 3:
            return volumes[valid_indices[0]] if len(valid_indices) > 0 else volumes[0]
        
        # Use valid thresholds only
        valid_volumes = volumes[valid_indices]
        valid_rates = rates[valid_indices]
        
        # Calculate first and second derivatives
        log_volumes = np.log10(valid_volumes)
        dy = np.gradient(valid_rates, log_volumes)
        d2y = np.gradient(dy, log_volumes)
        
        # Find maximum second derivative point (steepest knee)
        knee_idx = np.argmax(d2y)
        
        return valid_volumes[knee_idx]
    
    def _estimate_uncertainty(self, volumes: np.ndarray, probabilities: np.ndarray, 
                             threshold: float) -> float:
        """Estimate threshold uncertainty
        
        Args:
            volumes: Array of particle volumes
            probabilities: Array of artifact probabilities
            threshold: Threshold value
            
        Returns:
            Uncertainty estimate
        """
        retained_mask = volumes >= threshold
        if np.sum(retained_mask) == 0:
            return 0.0
        
        retained_probs = probabilities[retained_mask]
        return np.std(retained_probs) / np.sqrt(len(retained_probs))
    
    def plot_av_curve(self, volumes: np.ndarray, probabilities: np.ndarray, 
                     threshold: Optional[float] = None) -> Dict[str, Any]:
        """Plot A(V) curve for visualization
        
        Args:
            volumes: Array of particle volumes
            probabilities: Array of artifact probabilities
            threshold: Optional threshold to highlight
            
        Returns:
            Dictionary with plot data
        """
        # Calculate A(V) curve
        sort_idx = np.argsort(volumes)
        volumes_sorted = volumes[sort_idx]
        probs_sorted = probabilities[sort_idx]
        
        artifact_rates = []
        volume_thresholds = []
        
        for i in range(len(volumes)):
            if i < self.n_min - 1:
                continue
            v_thresh = volumes_sorted[i]
            retained_probs = probs_sorted[i:]
            a_rate = np.mean(retained_probs)
            artifact_rates.append(a_rate)
            volume_thresholds.append(v_thresh)
        
        return {
            'volumes': np.array(volume_thresholds),
            'artifact_rates': np.array(artifact_rates),
            'threshold': threshold,
            'epsilon': self.epsilon,
            'n_min': self.n_min
        }
    
    def update_parameters(self, epsilon: Optional[float] = None, 
                         tau: Optional[float] = None, 
                         n_min: Optional[int] = None):
        """Update threshold finder parameters
        
        Args:
            epsilon: New tolerance value
            tau: New platform threshold
            n_min: New minimum particle count
        """
        if epsilon is not None:
            self.epsilon = epsilon
        if tau is not None:
            self.tau = tau
        if n_min is not None:
            self.n_min = n_min
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameters
        
        Returns:
            Dictionary of current parameters
        """
        return {
            'epsilon': self.epsilon,
            'tau': self.tau,
            'n_min': self.n_min
        }
