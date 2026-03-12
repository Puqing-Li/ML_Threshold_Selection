#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fabric analysis bootstrap helpers: log-Euclidean averaging of ellipsoid tensors
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


def gram_schmidt(A: np.ndarray) -> np.ndarray:
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    for j in range(n):
        v = A[:, j].copy()
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], v)
            v -= R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j] if R[j, j] > 0 else v
    return Q


def build_spinel_block(df) -> np.ndarray:
    return df[[
        'EigenVal1','EigenVal2','EigenVal3',
        'EigenVec1X','EigenVec1Y','EigenVec1Z',
        'EigenVec2X','EigenVec2Y','EigenVec2Z',
        'EigenVec3X','EigenVec3Y','EigenVec3Z'
    ]].astype(float).values


def precompute_logE_block(spinel_block: np.ndarray) -> np.ndarray:
    N_all = spinel_block.shape[0]
    logE_stack = np.empty((N_all, 3, 3), dtype=np.float32)
    for i in range(N_all):
        eigenvals = spinel_block[i, :3].astype(np.float64).copy()
        eigenvals[eigenvals <= 0] = 1e-8
        q = np.vstack([
            spinel_block[i, 3:6],
            spinel_block[i, 6:9],
            spinel_block[i, 9:12]
        ]).astype(np.float64)
        q = gram_schmidt(q)
        logETilde = np.diag(np.log(eigenvals))
        logE = q.T @ logETilde @ q
        logE = (logE + logE.T) * 0.5
        logE_stack[i] = logE.astype(np.float32)
    return logE_stack


def calculate_T_Pprime_from_vals(vals_sorted: np.ndarray) -> Tuple[float, float]:
    l1, l2, l3 = [float(x) for x in vals_sorted]
    if l1 <= 0 or l2 <= 0 or l3 <= 0:
        return np.nan, np.nan
    ln1, ln2, ln3 = np.log(l1), np.log(l2), np.log(l3)
    denom = (ln2 - ln3) + (ln1 - ln2)
    if abs(denom) < 1e-10:
        T = 0.0
    else:
        T = (ln2 - ln3 - ln1 + ln2) / denom
    lm = (l1 + l2 + l3) / 3.0
    ln_m = np.log(lm)
    Pp = float(np.exp(np.sqrt(2.0 * ((ln1 - ln_m) ** 2 + (ln2 - ln_m) ** 2 + (ln3 - ln_m) ** 2))))
    return T, Pp


def eigvals_from_logMean(logMean: np.ndarray) -> np.ndarray:
    evals_log, _ = np.linalg.eigh((logMean + logMean.T) * 0.5)
    evals_log_sorted = np.sort(evals_log)[::-1]
    vals = np.exp(evals_log_sorted)
    return vals


def bootstrap_tp_samples(logE_retained: np.ndarray, n_bootstrap: int) -> Tuple[list, list]:
    N = int(logE_retained.shape[0])
    t_samples, p_samples = [], []
    for _ in range(n_bootstrap):
        idx_local = np.random.randint(0, N, size=N)
        logMean = logE_retained[idx_local].mean(axis=0)
        vals = eigvals_from_logMean(logMean)
        T_val, Pp_val = calculate_T_Pprime_from_vals(vals)
        if not np.isnan(T_val):
            t_samples.append(float(T_val))
        if not np.isnan(Pp_val):
            p_samples.append(float(Pp_val))
    return t_samples, p_samples


