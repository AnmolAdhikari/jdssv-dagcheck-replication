"""e3_collapse.py --- mapping the region of solver configurations that collapse.

This is what separates a methods paper from a bug report. The failure in this
paper was not a typo; it was a point in a configuration space that many
hand-rolled augmented-Lagrangian solvers occupy. The experiment sweeps that
space and asks, at each point, whether the solver recovers a DAG it is given
the data of.

Axes swept
  * clip   --- the norm the gradient is clipped to, including no clipping
  * lr0    --- the initial step size of the fixed-step inner loop
  * inner  --- number of inner gradient steps per outer iteration

Everything else is held at the legacy solver's settings. Recovery is scored on
synthetic linear-Gaussian DAGs with known structure, so "collapse" is defined
against ground truth rather than against a reference implementation:

    collapsed  <=>  max |W| < tau, i.e. the thresholded graph is empty

Relation to prior work: Nazaret et al. (2024) describe V-instability, where the
acyclicity CONSTRAINT VALUE vanishes and stops carrying signal. What is seen
here is the ITERATE vanishing, which then makes the constraint vanish with it.
The two are plainly related --- both end with a vacuously satisfied constraint
--- but they are not the same failure, and the manuscript should say so rather
than claim their mechanism or a wholly new one.

Output: results/e3_collapse.json
"""
from __future__ import annotations

import numpy as np

import jdssv_common as J
import structure_learners as SL

CLIPS = [0.1, 1.0, 10.0, 100.0, 1e3, 1e4, np.inf]
LR0S = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
INNERS = [10, 50, 200]
N_SEEDS = 8
N, D, N_EDGES = 500, 8, 8


def fit_projected_gradient(X, lambda1=J.LAMBDA1, clip=10.0, lr0=0.01,
                           inner=50, max_iter=50, h_tol=1e-8, rho_max=1e16):
    """The legacy solver's scheme with its three fixed constants exposed.

    With clip=10.0, lr0=0.01, inner=50 this is exactly `common.notears_legacy`.
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
        if np.isfinite(clip) and gnorm > clip:
            g = g * (clip / gnorm)
        return g

    for _ in range(max_iter):
        W_old = W.copy()
        for step in range(inner):
            lr = lr0 / (1 + step * 0.01)
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


def map_collapse_region():
    import common as C
    datasets = []
    for s in range(N_SEEDS):
        X, W_true = SL.simulate_linear_sem(N, D, N_EDGES, seed=1000 + s)
        datasets.append((C.standardise(X), W_true))

    cells = []
    for clip in CLIPS:
        for lr0 in LR0S:
            for inner in INNERS:
                tprs, fdrs, shds, maxws, collapsed = [], [], [], [], 0
                for X, W_true in datasets:
                    W = fit_projected_gradient(X, clip=clip, lr0=lr0,
                                               inner=inner)
                    sc = SL.score_recovery(W, W_true, J.TAU)
                    tprs.append(sc["tpr"])
                    fdrs.append(sc["fdr"])
                    shds.append(sc["shd"])
                    mw = float(np.abs(W).max())
                    maxws.append(mw)
                    collapsed += int(mw < J.TAU)
                cells.append({
                    "clip": None if not np.isfinite(clip) else float(clip),
                    "clip_label": "none" if not np.isfinite(clip) else f"{clip:g}",
                    "lr0": lr0, "inner": inner,
                    "mean_tpr": float(np.mean(tprs)),
                    "mean_fdr": float(np.mean(fdrs)),
                    "mean_shd": float(np.mean(shds)),
                    "mean_max_abs_w": float(np.mean(maxws)),
                    "share_collapsed": collapsed / len(datasets),
                })
    return cells


def main():
    J.print_banner("e3 --- the collapse region")
    print(f"  {len(CLIPS)} clip x {len(LR0S)} lr0 x {len(INNERS)} inner "
          f"x {N_SEEDS} seeds = "
          f"{len(CLIPS) * len(LR0S) * len(INNERS) * N_SEEDS} solver runs")
    print(f"  known DAGs: n={N}, d={D}, {N_EDGES} edges; "
          f"collapse <=> max|W| < tau = {J.TAU}")

    cells = map_collapse_region()
    n_collapsed = sum(c["share_collapsed"] == 1.0 for c in cells)
    n_partial = sum(0 < c["share_collapsed"] < 1.0 for c in cells)
    best = max(cells, key=lambda c: c["mean_tpr"])

    # The legacy solver's own configuration.
    legacy_cell = [c for c in cells
                   if c["clip_label"] == "10" and c["lr0"] == 0.01
                   and c["inner"] == 50][0]

    print(f"\n  configurations collapsing on every dataset: "
          f"{n_collapsed}/{len(cells)} ({100 * n_collapsed / len(cells):.1f}%)")
    print(f"  configurations collapsing on some but not all: {n_partial}")
    print(f"  best configuration: clip={best['clip_label']}, lr0={best['lr0']}, "
          f"inner={best['inner']} -> TPR={best['mean_tpr']:.3f}, "
          f"FDR={best['mean_fdr']:.3f}")
    print(f"  the legacy configuration (clip=10, lr0=0.01, inner=50): "
          f"TPR={legacy_cell['mean_tpr']:.3f}, "
          f"collapsed on {legacy_cell['share_collapsed'] * 100:.0f}% of datasets")

    # How much of the collapse is attributable to clipping alone?
    by_clip = {}
    for c in cells:
        by_clip.setdefault(c["clip_label"], []).append(c["share_collapsed"])
    print("\n  share of configurations that collapse, by clip norm:")
    for k, v in by_clip.items():
        print(f"    clip = {k:>6s}: {100 * float(np.mean(v)):5.1f}%")

    out = {
        "axes": {"clip": [None if not np.isfinite(c) else c for c in CLIPS],
                 "lr0": LR0S, "inner": INNERS},
        "n_seeds": N_SEEDS, "n": N, "d": D, "n_edges": N_EDGES, "tau": J.TAU,
        "n_cells": len(cells),
        "n_cells_fully_collapsed": n_collapsed,
        "n_cells_partially_collapsed": n_partial,
        "share_cells_fully_collapsed": n_collapsed / len(cells),
        "legacy_configuration": legacy_cell,
        "best_configuration": best,
        "collapse_share_by_clip": {k: float(np.mean(v))
                                   for k, v in by_clip.items()},
        "cells": cells,
    }
    J.save_json("e3_collapse", out)
    return out


if __name__ == "__main__":
    main()
