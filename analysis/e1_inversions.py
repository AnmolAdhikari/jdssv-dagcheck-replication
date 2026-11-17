"""e1_inversions.py --- the paper's central result.

Scores the same five diagnostics on the same data for two solvers: the
non-convergent legacy solver and the validated one. The claim the paper makes
is that the legacy solver wins on four of the five, and that the one it loses
is the only one that consults an external reference.

The five diagnostics
  1. bootstrap edge stability    across resampled DATA POINTS
  2. self-compatibility          across subsets of VARIABLES
  3. hyperparameter invariance   across the (lambda1, tau) grid
  4. acyclicity certificate      h(W) after thresholding
  5. ground-truth recovery       against known DAGs at the operating point

Output: results/e1_inversions.json
"""
from __future__ import annotations

import itertools
import time

import numpy as np

import jdssv_common as J
import structure_learners as SL

B_BOOT = 1000
LAMBDAS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]
TAUS = [0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]
N_RECOVERY_REPS = 20


# ── 1. stability across resampled data points ─────────────────────────────
def compute_bootstrap_stability(X, names, solver, B=B_BOOT, seed=J.SEED):
    """Edge inclusion frequencies under resampling of rows (districts).

    Also records the share of resamples whose edge set is identical to the
    modal one, which is the quantity that saturates for a constant map.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    counts: dict = {}
    sets = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        W = solver(X[idx])
        E = J.edge_set(W, names)
        sets.append(frozenset(E))
        for e in E:
            counts[e] = counts.get(e, 0) + 1
    modal, modal_count = None, 0
    for s in set(sets):
        c = sets.count(s)
        if c > modal_count:
            modal, modal_count = s, c
    bif = {f"{p}->{c}": k / B for (p, c), k in
           sorted(counts.items(), key=lambda kv: -kv[1])}
    return {
        "B": B,
        "modal_count": modal_count,
        "max_bif": max(bif.values()) if bif else 0.0,
        "n_edges_bif_ge_060": sum(v >= 0.60 for v in bif.values()),
        "n_edges_bif_ge_090": sum(v >= 0.90 for v in bif.values()),
        "mean_edges_per_resample": float(np.mean([len(s) for s in sets])),
        "n_distinct_edge_sets": len(set(sets)),
        "modal_set_share": modal_count / B,
        "modal_set_is_empty": len(modal) == 0 if modal is not None else None,
        "bif": bif,
    }


# ── 2. stability across subsets of variables ──────────────────────────────
def compute_self_compatibility(X, names, solver, subset_sizes=(7, 6)):
    """Agreement between a graph learned on a variable subset and the full
    graph restricted to that subset, in the spirit of Faller et al. (2024).

    This is our own implementation of a subset-agreement score, not the
    authors' estimator, and is used only to show how such a criterion behaves
    on a constant map. Agreement is Jaccard overlap of the two edge sets.

    The convention for the degenerate case matters, and is the point: when
    BOTH edge sets are empty there is no disagreement to measure, so agreement
    is 1 by any reading --- the criterion is either undefined or maximal
    exactly where the solver has failed. We record how often that happens.
    """
    d = len(names)
    W_full = solver(X)
    E_full = J.edge_set(W_full, names)

    scores, both_empty, rows = [], 0, []
    for size in subset_sizes:
        for cols in itertools.combinations(range(d), size):
            sub_names = [names[i] for i in cols]
            W_sub = solver(X[:, list(cols)])
            E_sub = J.edge_set(W_sub, sub_names)
            E_restricted = {(p, c) for (p, c) in E_full
                            if p in sub_names and c in sub_names}
            union = E_sub | E_restricted
            if not union:
                score = 1.0
                both_empty += 1
            else:
                score = len(E_sub & E_restricted) / len(union)
            scores.append(score)
            rows.append({"size": size, "n_sub": len(E_sub),
                         "n_restricted": len(E_restricted), "score": score})
    return {
        "subset_sizes": list(subset_sizes),
        "n_subsets": len(scores),
        "mean_agreement": float(np.mean(scores)),
        "min_agreement": float(np.min(scores)),
        "n_subsets_both_empty": both_empty,
        "share_both_empty": both_empty / len(scores),
        "per_subset": rows,
    }


# ── 3. invariance across the hyperparameter grid ──────────────────────────
def compute_hyperparameter_invariance(X, names, solver):
    """Edge count at every (lambda1, tau) operating point.

    A solver that returns the same answer everywhere looks maximally robust
    under a sensitivity analysis, which is how the failure survives review.
    """
    grid, sets = [], []
    cache = {}
    for lam in LAMBDAS:
        if lam not in cache:
            cache[lam] = solver(X, lambda1=lam)
        W = cache[lam]
        for tau in TAUS:
            E = J.edge_set(W, names, tau=tau)
            sets.append(frozenset(E))
            grid.append({"lambda1": lam, "tau": tau, "n_edges": len(E),
                         "max_abs_weight": float(np.abs(W).max())})
    modal_count = max(sets.count(s) for s in set(sets))
    modal_share = modal_count / len(sets)
    counts = [g["n_edges"] for g in grid]
    return {
        "n_points": len(grid),
        "modal_count": modal_count,
        "n_points_empty": sum(c == 0 for c in counts),
        "edge_count_min": int(min(counts)),
        "edge_count_max": int(max(counts)),
        "n_distinct_edge_sets": len(set(sets)),
        "modal_set_share": modal_share,
        "grid": grid,
    }


# ── 4. the acyclicity certificate ─────────────────────────────────────────
def compute_acyclicity(X, solver):
    import common as C
    W = J.threshold_W(solver(X))
    return {"h_exact_after_threshold": C.h_exact(W),
            "max_abs_weight": float(np.abs(solver(X)).max())}


# ── 5. the only diagnostic that consults an external reference ────────────
def compute_ground_truth_recovery(solver, n=77, d=8, n_edges=8,
                                  reps=N_RECOVERY_REPS, seed=J.SEED):
    """TPR / FDR / SHD against known DAGs at the operating point."""
    import common as C
    out = []
    for r in range(reps):
        X, W_true = SL.simulate_linear_sem(n, d, n_edges, seed=seed + r)
        W_est = solver(C.standardise(X))
        out.append(SL.score_recovery(W_est, W_true, J.TAU))
    return {
        "n": n, "d": d, "n_edges": n_edges, "reps": reps,
        "tpr_per_rep": [o["tpr"] for o in out],
        "fdr_per_rep": [o["fdr"] for o in out],
        "mean_tpr": float(np.mean([o["tpr"] for o in out])),
        "se_tpr": float(np.std([o["tpr"] for o in out], ddof=1)
                        / np.sqrt(reps)),
        "se_fdr": float(np.std([o["fdr"] for o in out], ddof=1)
                        / np.sqrt(reps)),
        "mean_fdr": float(np.mean([o["fdr"] for o in out])),
        "mean_shd": float(np.mean([o["shd"] for o in out])),
        "mean_n_pred": float(np.mean([o["n_pred"] for o in out])),
    }


def main():
    X, names = J.load_nepal()
    J.print_banner("e1 --- the five diagnostics, two solvers")
    print(f"  Nepal structural matrix: {X.shape[0]} districts x {X.shape[1]} variables")
    print(f"  operating point: lambda1 = {J.LAMBDA1}, tau = {J.TAU}")

    out = {"n": int(X.shape[0]), "d": int(X.shape[1]), "variables": names,
           "lambda1": J.LAMBDA1, "tau": J.TAU, "B_bootstrap": B_BOOT,
           "lambda_grid": LAMBDAS, "tau_grid": TAUS, "solvers": {}}

    for label, solver in J.SOLVERS.items():
        print(f"\n  --- {label} solver ---")
        t0 = time.time()
        res = {}
        res["acyclicity"] = compute_acyclicity(X, solver)
        print(f"    h(W) after threshold      = "
              f"{res['acyclicity']['h_exact_after_threshold']:.3e}")

        res["hyperparameter_invariance"] = compute_hyperparameter_invariance(
            X, names, solver)
        hi = res["hyperparameter_invariance"]
        print(f"    grid points empty          = {hi['n_points_empty']}/{hi['n_points']}"
              f"   modal-set share = {hi['modal_set_share']:.3f}")

        res["self_compatibility"] = compute_self_compatibility(X, names, solver)
        sc = res["self_compatibility"]
        print(f"    self-compatibility (mean)  = {sc['mean_agreement']:.3f}"
              f"   both-empty subsets = {sc['n_subsets_both_empty']}/{sc['n_subsets']}")

        res["ground_truth_recovery"] = compute_ground_truth_recovery(solver)
        gt = res["ground_truth_recovery"]
        print(f"    ground-truth TPR / FDR     = {gt['mean_tpr']:.3f} / {gt['mean_fdr']:.3f}"
              f"   SHD = {gt['mean_shd']:.2f}")

        res["bootstrap_stability"] = compute_bootstrap_stability(X, names, solver)
        bs = res["bootstrap_stability"]
        print(f"    bootstrap max BIF          = {bs['max_bif']:.3f}"
              f"   modal-set share = {bs['modal_set_share']:.3f}"
              f"   distinct sets = {bs['n_distinct_edge_sets']}")
        print(f"    [{time.time() - t0:.1f}s]")
        out["solvers"][label] = res

    J.save_json("e1_inversions", out)
    return out


if __name__ == "__main__":
    main()
