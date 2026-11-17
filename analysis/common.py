"""common.py --- the subset of the thesis analysis module that the paper uses.

Vendored so this repository runs on its own. It holds the variable set, the
operating point (lambda1, tau), the data loader, and the legacy solver that
the paper is about. The data are read from ../data/nepal_districts.csv.

Convention: every solver minimises ||X - XW||_F^2, so W[i, j] != 0 means i -> j.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

SEED = 42
DATA_CSV = pathlib.Path(__file__).resolve().parents[1] / "data" / "nepal_districts.csv"

# Structural subset: the closed 8-variable system given to NOTEARS.
STRUCT_VARS = [
    "Literacy", "Poverty", "Infrastructure", "Teacher_Quality",
    "Digital_Access", "Dropout", "STR", "Urban_Rural_Gap",
]

# Expert-assumed DAG (7 edges) used as the theory-based baseline.
ASSUMED_EDGES = [
    ("Poverty", "Literacy"),
    ("Infrastructure", "Literacy"),
    ("Teacher_Quality", "Literacy"),
    ("Digital_Access", "Literacy"),
    ("Poverty", "Dropout"),
    ("Infrastructure", "Teacher_Quality"),
    ("Poverty", "Urban_Rural_Gap"),
]

NOTEARS_LAMBDA1 = 0.15
NOTEARS_TAU = 0.08


def load_data():
    """Return (df_dist, None, None), matching the thesis loader's signature."""
    return pd.read_csv(DATA_CSV), None, None


def standardise(X: np.ndarray) -> np.ndarray:
    """Zero mean, unit variance (population SD)."""
    return (X - X.mean(axis=0)) / X.std(axis=0)


def struct_matrix(df_dist: pd.DataFrame):
    """Standardised structural-stage design matrix and its column names."""
    X = df_dist[STRUCT_VARS].dropna().to_numpy(dtype=float)
    return standardise(X), list(STRUCT_VARS)


def notears_legacy(X, lambda1=NOTEARS_LAMBDA1, max_iter=50, h_tol=1e-8,
                   rho_max=1e16):
    """The original pipeline's projected-gradient solver, retained ONLY to
    reproduce the empty-graph failure the paper studies.

    This solver does not converge. Its inner loop takes 50 fixed-step gradient
    steps with the gradient clipped to norm 10; once the augmented-Lagrangian
    multipliers grow, the clipped gradient is dominated by the acyclicity
    term, whose gradient is proportional to W and so points at the origin.
    Each outer iteration therefore pulls W back toward its zero initialisation.

    Do not use for inference.
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    W = np.zeros((d, d))
    rho, alpha = 1.0, 0.0

    def _h(W):
        M = W * W
        E = np.linalg.matrix_power(np.eye(d) + M / d, d)
        return np.trace(E) - d

    def _grad(W):
        R = X - X @ W
        grad_loss = -1.0 / n * X.T @ R
        M = W * W
        E = np.linalg.matrix_power(np.eye(d) + M / d, d)
        grad_h = E.T * W * 2
        g = grad_loss + (rho * _h(W) + alpha) * grad_h + lambda1 * np.sign(W)
        gnorm = np.linalg.norm(g)
        if gnorm > 10.0:
            g = g * (10.0 / gnorm)
        return g

    for _ in range(max_iter):
        W_old = W.copy()
        for step in range(50):
            lr = 0.01 / (1 + step * 0.01)
            W_new = W - lr * _grad(W)
            W_new = np.sign(W_new) * np.maximum(np.abs(W_new) - lr * lambda1, 0)
            np.fill_diagonal(W_new, 0)
            if not np.isfinite(W_new).all():
                break
            W = W_new
        h_val = _h(W)
        if not np.isfinite(h_val):
            W = W_old
            break
        if h_val > 0.25 * _h(W_old) and _h(W_old) > h_tol:
            rho *= 10
        alpha += rho * h_val
        if h_val < h_tol or rho > rho_max:
            break
    return W


def h_exact(W):
    """Exact acyclicity certificate tr(exp(W o W)) - d via scipy.linalg.expm."""
    from scipy.linalg import expm
    return float(np.trace(expm(W * W)) - W.shape[0])
