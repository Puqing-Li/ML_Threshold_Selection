#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mean fabric tensor calculator: Compute mean fabric without bootstrap for dual thresholds
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import os

try:
    from scipy.linalg import expm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    expm = None

from .fabric_bootstrap import (
    build_spinel_block,
    precompute_logE_block,
    eigvals_from_logMean,
    calculate_T_Pprime_from_vals
)


def compute_mean_fabric_single(
    df: pd.DataFrame,
    volume_threshold_mm3: float
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[float], Optional[float], int]:
    """
    Compute mean fabric tensor for a single threshold (no bootstrap).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Particle data with required columns
    volume_threshold_mm3 : float
        Volume threshold in mm³
        
    Returns:
    --------
    Tuple containing:
    - mean_ellipsoid_matrix: 3x3 mean ellipsoid tensor (or None if insufficient particles)
    - eigenvectors: 3x3 matrix with eigenvectors as columns (or None)
    - eigenvalues: 3x3 diagonal matrix with eigenvalues (or None)
    - T: T parameter (or None)
    - P_prime: P' parameter (or None)
    - n_particles: Number of particles used
    """
    volumes = df['Volume3d (mm^3) '].astype(float).values
    mask = volumes >= volume_threshold_mm3
    retained_df = df[mask].copy()
    
    n_particles = len(retained_df)
    
    if n_particles < 3:  # Need at least 3 particles
        return None, None, None, None, None, n_particles
    
    # Build spinel block and compute log-Euclidean tensors
    spinel_block = build_spinel_block(retained_df)
    logE_stack = precompute_logE_block(spinel_block)
    
    # Compute mean in log-space
    logMean = logE_stack.mean(axis=0)
    # Ensure symmetry
    logMean = (logMean + logMean.T) * 0.5
    
    # Get eigenvalues and eigenvectors from log-space (as done in bootstrap)
    # Compute mean ellipsoid matrix using matrix exponential (as in reference code)
    # Use scipy.linalg.expm for matrix exponential, or reconstruct via eigendecomposition
    if SCIPY_AVAILABLE:
        # Method 1: Use scipy.linalg.expm (matrix exponential) - matches reference code
        mean_ellipsoid_matrix = expm(logMean)
    else:
        # Method 2: Reconstruct via eigendecomposition (fallback if scipy not available)
        eigenvals_log, eigenvecs_log = np.linalg.eigh(logMean)
        sort_idx = np.argsort(eigenvals_log)[::-1]
        eigenvals_log_sorted = eigenvals_log[sort_idx]
        eigenvecs_log_sorted = eigenvecs_log[:, sort_idx]
        eigenvals_sorted = np.exp(eigenvals_log_sorted)
        diag_evals = np.diag(eigenvals_sorted)
        mean_ellipsoid_matrix = eigenvecs_log_sorted @ diag_evals @ eigenvecs_log_sorted.T
    
    # Ensure symmetry
    mean_ellipsoid_matrix = (mean_ellipsoid_matrix + mean_ellipsoid_matrix.T) * 0.5
    
    # Compute eigendecomposition from mean ellipsoid matrix (as in reference code)
    # This ensures eigenvectors match the mean ellipsoid matrix
    eigenvals, eigenvecs = np.linalg.eig(mean_ellipsoid_matrix)
    
    # Sort in descending order (largest first)
    sort_idx = np.argsort(eigenvals)[::-1]
    eigenvals_sorted = eigenvals[sort_idx]
    eigenvecs_sorted = eigenvecs[:, sort_idx]
    
    # Normalize sign convention: ensure first non-zero element of each eigenvector is positive
    # This makes results consistent across different runs
    for i in range(3):
        vec = eigenvecs_sorted[:, i]
        # Find first non-zero element
        first_nonzero_idx = np.where(np.abs(vec) > 1e-10)[0]
        if len(first_nonzero_idx) > 0:
            if vec[first_nonzero_idx[0]] < 0:
                eigenvecs_sorted[:, i] = -vec
    
    # Create diagonal matrix for eigenvalues
    eigenvalues_matrix = np.diag(eigenvals_sorted)
    
    # Calculate T and P' parameters using eigenvalues from mean ellipsoid matrix
    # Note: For symmetric matrices, these eigenvalues are equivalent to exp(logMean eigenvalues)
    T, P_prime = calculate_T_Pprime_from_vals(eigenvals_sorted)
    
    return mean_ellipsoid_matrix, eigenvecs_sorted, eigenvalues_matrix, T, P_prime, n_particles


def export_mean_fabric_txt(
    sample_id: str,
    threshold_type: str,  # "Loose" or "Strict"
    volume_threshold_mm3: float,
    mean_ellipsoid_matrix: np.ndarray,
    eigenvectors: np.ndarray,
    eigenvalues: np.ndarray,
    T: float,
    P_prime: float,
    n_particles: int,
    output_dir: str = 'outputs'
) -> str:
    """
    Export mean fabric results to a formatted txt file.
    
    Parameters:
    -----------
    sample_id : str
        Sample identifier
    threshold_type : str
        "Loose" or "Strict"
    volume_threshold_mm3 : float
        Volume threshold in mm³
    mean_ellipsoid_matrix : np.ndarray
        3x3 mean ellipsoid tensor
    eigenvectors : np.ndarray
        3x3 matrix with eigenvectors as columns
        Column 0 = Eigen1 (XYZ), Column 1 = Eigen2 (XYZ), Column 2 = Eigen3 (XYZ)
    eigenvalues : np.ndarray
        3x3 diagonal matrix with eigenvalues
    T : float
        T parameter
    P_prime : float
        P' parameter
    n_particles : int
        Number of particles used
    output_dir : str
        Output directory
        
    Returns:
    --------
    str: Path to the generated txt file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"{sample_id}_{threshold_type}_MeanFabric.txt"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        # Header
        f.write(f"At threshold {volume_threshold_mm3:.6e} mm³ for sample {sample_id}:\n")
        f.write(f"Threshold Type: {threshold_type}\n")
        f.write("=" * 60 + "\n\n")
        
        # Mean Ellipsoid Matrix
        f.write("Mean Ellipsoid Matrix (meanEll):\n")
        f.write("[\n")
        for i in range(3):
            f.write("  [")
            for j in range(3):
                f.write(f" {mean_ellipsoid_matrix[i, j]:.8e}")
                if j < 2:
                    f.write(",")
            f.write(" ]")
            if i < 2:
                f.write(",")
            f.write("\n")
        f.write("]\n\n")
        
        # Eigenvectors (V matrix) - simple format like original
        f.write("V matrix (eigenvectors):\n")
        f.write("[\n")
        for i in range(3):
            f.write("  [")
            for j in range(3):
                f.write(f" {eigenvectors[i, j]:.8f}")
                if j < 2:
                    f.write(" ")
            f.write(" ]")
            if i < 2:
                f.write("")
            f.write("\n")
        f.write("]\n\n")
        
        # Eigenvalues (D matrix)
        f.write("D matrix (Eigenvalues - diagonal matrix):\n")
        f.write("[\n")
        for i in range(3):
            f.write("  [")
            for j in range(3):
                f.write(f" {eigenvalues[i, j]:.8e}")
                if j < 2:
                    f.write(",")
            f.write(" ]")
            if i < 2:
                f.write(",")
            f.write("\n")
        f.write("]\n\n")
        
        # Statistics
        f.write("Number of particles (spinels): {}\n".format(n_particles))
        f.write("T: {}\n".format(T))
        f.write("P': {}\n".format(P_prime))
        f.write("\n")
        
        # Additional information
        f.write("=" * 60 + "\n")
        f.write("Notes:\n")
        f.write("- V matrix: Each COLUMN is an eigenvector (principal axis). Column 0=Eigen1, Column 1=Eigen2, Column 2=Eigen3\n")
        f.write("- For Avizo Clipping Plane: Foliation plane uses Eigen1 (col 0) and Eigen2 (col 1); Kinematic plane uses Eigen1 (col 0) and Eigen3 (col 2)\n")
    
    return filepath


def format_mean_fabric_for_display(
    sample_id: str,
    threshold_type: str,
    volume_threshold_mm3: float,
    mean_ellipsoid_matrix: np.ndarray,
    eigenvectors: np.ndarray,
    eigenvalues: np.ndarray,
    T: float,
    P_prime: float,
    n_particles: int
) -> str:
    """
    Format mean fabric results as a string for display in GUI.
    
    Parameters:
    -----------
    Same as export_mean_fabric_txt
    
    Returns:
    --------
    str: Formatted text string for display
    """
    lines = []
    
    # Header
    lines.append(f"At threshold {volume_threshold_mm3:.6e} mm³ for sample {sample_id}:")
    lines.append(f"Threshold Type: {threshold_type}")
    lines.append("=" * 60)
    lines.append("")
    
    # Mean Ellipsoid Matrix
    lines.append("Mean Ellipsoid Matrix (meanEll):")
    lines.append("[")
    for i in range(3):
        row_str = "  ["
        for j in range(3):
            row_str += f" {mean_ellipsoid_matrix[i, j]:.8e}"
            if j < 2:
                row_str += ","
        row_str += " ]"
        if i < 2:
            row_str += ","
        lines.append(row_str)
    lines.append("]")
    lines.append("")
    
    # Eigenvectors (V matrix) - simple format like original
    lines.append("V matrix (eigenvectors):")
    lines.append("[")
    for i in range(3):
        row_str = "  ["
        for j in range(3):
            row_str += f" {eigenvectors[i, j]:.8f}"
            if j < 2:
                row_str += " "
        row_str += " ]"
        lines.append(row_str)
    lines.append("]")
    lines.append("")
    
    # Eigenvalues (D matrix)
    lines.append("D matrix (Eigenvalues - diagonal matrix):")
    lines.append("[")
    for i in range(3):
        row_str = "  ["
        for j in range(3):
            row_str += f" {eigenvalues[i, j]:.8e}"
            if j < 2:
                row_str += ","
        row_str += " ]"
        if i < 2:
            row_str += ","
        lines.append(row_str)
    lines.append("]")
    lines.append("")
    
    # Statistics
    lines.append(f"Number of particles (spinels): {n_particles}")
    lines.append(f"T: {T}")
    lines.append(f"P': {P_prime}")
    lines.append("")
    
    # Additional information
    lines.append("=" * 60)
    lines.append("Notes:")
    lines.append("- V matrix: Each COLUMN is an eigenvector (principal axis). Column 0=Eigen1, Column 1=Eigen2, Column 2=Eigen3")
    lines.append("- For Avizo Clipping Plane: Foliation plane uses Eigen1 (col 0) and Eigen2 (col 1); Kinematic plane uses Eigen1 (col 0) and Eigen3 (col 2)")
    
    return "\n".join(lines)

