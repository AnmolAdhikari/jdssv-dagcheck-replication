"""gate.py --- the diagnostic that consults an external reference.

This is the only check in the package that can detect a solver which ignores
its input, and it is the one that belongs at import.
"""
from __future__ import annotations

import numpy as np

from .reference import simulate_linear_sem, score_recovery


class SolverGateError(AssertionError):
    """Raised when a solver fails to recover structure it was given."""


def compute_recovery_curve(fit, n_grid=(40, 77, 150, 300, 1000), d=8,
                           n_edges=None, tau=0.08, reps=10,
                           weight_range=(0.5, 2.0), standardise=False,
                           seed=500):
    """Recovery of known structure across sample sizes, at your operating point.

    ``fit`` is a callable ``fit(X) -> W``. Set ``standardise`` to match what
    your pipeline does to its design matrix: leaving it False when your real
    data is standardised will overstate the power available to you, because the
    simulated data then carries varsortability that your real data does not.
    Run it both ways and compare; the gap is the part of apparent recovery that
    varsortability supplies.
    """
    n_edges = d if n_edges is None else n_edges
    rows = []
    for n in n_grid:
        scores = []
        for rep in range(reps):
            X, W_true = simulate_linear_sem(
                n, d, n_edges, weight_range=weight_range, seed=seed + rep,
                standardise=standardise)
            scores.append(score_recovery(fit(X), W_true, tau))
        rows.append({
            "n": n,
            "mean_tpr": float(np.mean([s["tpr"] for s in scores])),
            "mean_fdr": float(np.mean([s["fdr"] for s in scores])),
            "mean_shd": float(np.mean([s["shd"] for s in scores])),
            "mean_n_pred": float(np.mean([s["n_pred"] for s in scores])),
        })
    return rows


def assert_solver_valid(fit, n=500, d=8, n_edges=None, tau=0.08, reps=5,
                        min_tpr=0.30, weight_range=(0.5, 2.0),
                        standardise=False, seed=0, verbose=False):
    """Raise unless ``fit`` recovers structure it is given at this operating point.

    The default is deliberately lenient. ``min_tpr=0.30`` is far below what a
    working solver achieves on strong effects and far above what a solver that
    returns a constant achieves, which is 0. The gate is meant to separate
    "running" from "not running", not to score quality; use
    `compute_recovery_curve` for that.

    Raises
    ------
    SolverGateError
        if mean TPR falls below ``min_tpr``, with a message reporting what the
        solver returned, including the degenerate case of an always-empty graph.
    """
    n_edges = d if n_edges is None else n_edges
    scores, max_abs = [], []
    for rep in range(reps):
        X, W_true = simulate_linear_sem(
            n, d, n_edges, weight_range=weight_range, seed=seed + rep,
            standardise=standardise)
        W = np.asarray(fit(X), dtype=float)
        if W.shape != (d, d):
            raise SolverGateError(
                f"solver returned shape {W.shape}, expected ({d}, {d})")
        scores.append(score_recovery(W, W_true, tau))
        max_abs.append(float(np.abs(W).max()))

    tpr = float(np.mean([s["tpr"] for s in scores]))
    fdr = float(np.mean([s["fdr"] for s in scores]))
    n_pred = float(np.mean([s["n_pred"] for s in scores]))
    report = (f"TPR={tpr:.3f} FDR={fdr:.3f} edges={n_pred:.1f} "
              f"max|W|={float(np.mean(max_abs)):.4f} "
              f"(n={n}, d={d}, tau={tau}, {reps} replications)")
    if verbose:
        print(f"dagcheck: {report}")

    if tpr < min_tpr:
        detail = report
        if n_pred == 0:
            detail += ("\n  The thresholded graph was EMPTY on every replication. "
                       "A solver that returns the same answer whatever it is "
                       "given will pass every stability check you run; it "
                       "cannot pass this one.")
            if float(np.mean(max_abs)) > 0:
                detail += (f"\n  Note the estimate is not exactly zero "
                           f"(max|W|={float(np.mean(max_abs)):.4g}), so a test "
                           f"for an all-zero matrix would not have caught it.")
        raise SolverGateError(
            f"solver did not recover known structure: {detail}\n"
            f"  Expected mean TPR >= {min_tpr}.")
    return {"tpr": tpr, "fdr": fdr, "mean_n_pred": n_pred,
            "mean_max_abs_w": float(np.mean(max_abs))}
