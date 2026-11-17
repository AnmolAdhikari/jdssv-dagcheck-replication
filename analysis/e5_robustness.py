"""e5_robustness.py --- does the inversion survive a change of data?

The inversion in e1 is measured on one dataset with one graph. A referee is
entitled to ask whether it is a property of that dataset. This script repeats
the four comparable diagnostics across data-generating processes that vary the
dimension, the edge density, and the sample size.

Two deliberate choices, both conservative:

  * every design matrix is standardised before fitting, which removes the
    varsortability signal (Section 5.2) and therefore handicaps the CORRECTED
    solver, not the legacy one. If the inversion survives that, it is not an
    artefact of favourable simulation.
  * the known structure is drawn afresh per replication, so nothing is tuned to
    a particular graph.

Output: results/e5_robustness.json
"""
from __future__ import annotations

import itertools
import time

import numpy as np

import jdssv_common as J
import structure_learners as SL

B_BOOT = 100
LAMBDAS = [0.01, 0.05, 0.15, 0.30]
TAUS = [0.02, 0.08, 0.20]
N_REPS_GT = 10

# (label, d, n_edges, n)
SETTINGS = [
    ("d=5, sparse, n=100",   5,  5, 100),
    ("d=8, sparse, n=100",   8,  8, 100),
    ("d=8, dense, n=100",    8, 16, 100),
    ("d=8, sparse, n=50",    8,  8,  50),
    ("d=8, sparse, n=500",   8,  8, 500),
    ("d=15, sparse, n=100", 15, 15, 100),
]


def make_dataset(d, n_edges, n, seed):
    import common as C
    X, W_true = SL.simulate_linear_sem(n, d, n_edges, seed=seed)
    return C.standardise(X), W_true


def compute_diagnostics(X, names, solver, seed):
    """The four diagnostics of e1 that transfer to synthetic data."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]

    # stability across resampled rows
    sets = []
    for _ in range(B_BOOT):
        idx = rng.integers(0, n, size=n)
        sets.append(frozenset(J.edge_set(solver(X[idx]), names)))
    boot_modal = max(sets.count(s) for s in set(sets)) / B_BOOT

    # stability across variable subsets
    W_full = solver(X)
    E_full = J.edge_set(W_full, names)
    scores = []
    for drop in range(len(names)):
        cols = [k for k in range(len(names)) if k != drop]
        sub_names = [names[k] for k in cols]
        E_sub = J.edge_set(solver(X[:, cols]), sub_names)
        E_res = {(p, c) for (p, c) in E_full
                 if p in sub_names and c in sub_names}
        union = E_sub | E_res
        scores.append(1.0 if not union
                      else len(E_sub & E_res) / len(union))
    self_compat = float(np.mean(scores))

    # invariance across the hyperparameter grid
    grid_sets = []
    for lam in LAMBDAS:
        W = solver(X, lambda1=lam)
        for tau in TAUS:
            grid_sets.append(frozenset(J.edge_set(W, names, tau=tau)))
    grid_modal = max(grid_sets.count(s) for s in set(grid_sets)) / len(grid_sets)

    return {"bootstrap_modal_share": boot_modal,
            "self_compatibility": self_compat,
            "hyperparameter_modal_share": grid_modal}


def main():
    import common as C
    J.print_banner("e5 --- does the inversion survive a change of data?")
    print(f"  {len(SETTINGS)} data-generating processes, B = {B_BOOT}, "
          f"all design matrices standardised")

    out = {"B": B_BOOT, "lambda_grid": LAMBDAS, "tau_grid": TAUS,
           "n_reps_ground_truth": N_REPS_GT, "settings": [], "results": {}}

    for label, d, n_edges, n in SETTINGS:
        names = [f"V{k}" for k in range(d)]
        X, W_true = make_dataset(d, n_edges, n, seed=7000)
        out["settings"].append({"label": label, "d": d,
                                "n_edges": n_edges, "n": n})
        print(f"\n  {label}")
        for solver_label, solver in J.SOLVERS.items():
            t0 = time.time()
            res = compute_diagnostics(X, names, solver, seed=J.SEED)
            # ground-truth recovery, fresh graphs per replication
            tprs = []
            for r in range(N_REPS_GT):
                Xr, Wr = make_dataset(d, n_edges, n, seed=8000 + r)
                tprs.append(SL.score_recovery(solver(Xr), Wr, J.TAU)["tpr"])
            res["ground_truth_tpr"] = float(np.mean(tprs))
            out["results"][f"{label}|{solver_label}"] = res
            print(f"    {solver_label:10s} boot={res['bootstrap_modal_share']:.3f} "
                  f"selfcompat={res['self_compatibility']:.3f} "
                  f"grid={res['hyperparameter_modal_share']:.3f} "
                  f"TPR={res['ground_truth_tpr']:.3f}  [{time.time()-t0:.0f}s]")

    # how often does the legacy solver win each self-consistency diagnostic?
    wins = {k: 0 for k in ("bootstrap_modal_share", "self_compatibility",
                           "hyperparameter_modal_share")}
    gt_wins = 0
    for s in out["settings"]:
        L = out["results"][f"{s['label']}|legacy"]
        C_ = out["results"][f"{s['label']}|corrected"]
        for k in wins:
            wins[k] += int(L[k] >= C_[k])
        gt_wins += int(C_["ground_truth_tpr"] > L["ground_truth_tpr"])
    out["legacy_wins_or_ties"] = wins
    out["corrected_wins_ground_truth"] = gt_wins
    out["n_settings"] = len(out["settings"])

    print(f"\n  across {len(out['settings'])} data-generating processes:")
    for k, v in wins.items():
        print(f"    legacy >= corrected on {k}: {v}/{len(out['settings'])}")
    print(f"    corrected > legacy on ground-truth recovery: "
          f"{gt_wins}/{len(out['settings'])}")

    J.save_json("e5_robustness", out)
    return out


if __name__ == "__main__":
    main()
