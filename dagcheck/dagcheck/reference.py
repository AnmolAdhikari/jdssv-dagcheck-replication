"""reference.py --- the external reference the gate consults.

Everything here exists to produce data whose structure is known, so that a
solver's output can be compared against something that did not come from the
solver.
"""
from __future__ import annotations

import numpy as np


def simulate_linear_sem(n, d, n_edges, weight_range=(0.5, 2.0), seed=0,
                        noise_scale=1.0, standardise=False):
    """Draw n observations from a random sparse linear-Gaussian DAG.

    Returns ``(X, W_true)`` with ``W_true[i, j] != 0`` meaning ``i -> j``, so
    that ``X = X W_true + Z`` holds exactly.

    ``standardise`` defaults to False, which leaves the varsortability of the
    generated data intact. Set it to True to match a pipeline that standardises
    its design matrix; see `compute_varsortability` for why this matters.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(d)
    W = np.zeros((d, d))
    candidates = [(order[a], order[b]) for b in range(d) for a in range(b)]
    rng.shuffle(candidates)
    for parent, child in candidates[:n_edges]:
        W[parent, child] = rng.uniform(*weight_range) * rng.choice([-1.0, 1.0])
    X = np.zeros((n, d))
    Z = rng.normal(scale=noise_scale, size=(n, d))
    for v in order:
        X[:, v] = Z[:, v] + X @ W[:, v]
    if standardise:
        X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X, W


def compute_varsortability(X, W_true, rtol=1e-9):
    """Share of true edges running from lower to higher marginal variance.

    0.5 is chance; 1.0 means the causal order is readable from the marginal
    variances alone, so a solver can appear to recover structure without doing
    any structure learning (Reisach et al., 2021). Standardising the columns
    drives this to exactly 0.5.

    Ties are compared with a relative tolerance rather than exactly. This is
    not cosmetic: after standardising, the column variances are 1 only to
    within floating-point error, so an exact comparison counts almost every
    pair as strictly ordered and returns a spurious value away from 0.5 --- the
    very artefact the statistic exists to measure.
    """
    var = np.asarray(X).var(axis=0)
    d = W_true.shape[0]
    agree = total = 0.0
    for i in range(d):
        for j in range(d):
            if i != j and W_true[i, j] != 0:
                total += 1
                if np.isclose(var[j], var[i], rtol=rtol, atol=0.0):
                    agree += 0.5
                elif var[j] > var[i]:
                    agree += 1.0
    return float(agree / total) if total else float("nan")


def score_recovery(W_est, W_true, tau):
    """Edge-level TPR, FDR and SHD of a thresholded estimate against truth."""
    E = np.abs(np.asarray(W_est)) > tau
    T = np.asarray(W_true) != 0
    np.fill_diagonal(E, False)
    tp = int((E & T).sum())
    fp = int((E & ~T).sum())
    fn = int((~E & T).sum())
    return {"n_true": int(T.sum()), "n_pred": int(E.sum()),
            "tp": tp, "fp": fp, "fn": fn,
            "tpr": tp / max(int(T.sum()), 1),
            "fdr": fp / max(tp + fp, 1),
            "shd": fp + fn}
