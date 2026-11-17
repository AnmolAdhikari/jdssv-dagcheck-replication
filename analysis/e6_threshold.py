"""e6_threshold.py --- does the inversion survive below the attractor floor?

Proposition 1 applies to a solver that is a constant map. At the operating
point the legacy solver is effectively one: it returns max |W| = 0.032, so at
tau = 0.08 the thresholded graph is empty for every input. Below that floor it
is NOT constant --- the small entries do vary with the data --- so the empty
graph gives way to a noise graph, and the protection the proposition describes
should weaken.

This script tests that directly, computing the three self-consistency
diagnostics and ground-truth recovery across a range of tau for both solvers.
Each solver is fitted once per resample and the resulting W thresholded at
every tau, so the comparison across tau is paired.

Output: results/e6_threshold.json
"""
from __future__ import annotations

import itertools

import numpy as np

import jdssv_common as J
import structure_learners as SL

TAUS = [0.005, 0.01, 0.02, 0.05, 0.08, 0.15]
B_BOOT = 300
LAMBDAS = [0.01, 0.05, 0.15, 0.30]
N_REPS_GT = 15


def main():
    import common as C
    X, names = J.load_nepal()
    J.print_banner("e6 --- the inversion as a function of tau")
    print(f"  B = {B_BOOT}, tau in {TAUS}")

    out = {"taus": TAUS, "B": B_BOOT, "lambda_grid": LAMBDAS,
           "n_reps_ground_truth": N_REPS_GT, "results": {}}

    for label, solver in J.SOLVERS.items():
        rng = np.random.default_rng(J.SEED)
        n = X.shape[0]

        # one fit per resample; threshold the same W at every tau
        boot_W = [solver(X[rng.integers(0, n, size=n)]) for _ in range(B_BOOT)]
        W_full = solver(X)
        sub_W = {}
        for drop in range(len(names)):
            cols = [k for k in range(len(names)) if k != drop]
            sub_W[drop] = (solver(X[:, cols]), [names[k] for k in cols])
        grid_W = {lam: solver(X, lambda1=lam) for lam in LAMBDAS}

        gt_W = []
        for r in range(N_REPS_GT):
            Xr, Wr = SL.simulate_linear_sem(77, 8, 8, seed=9000 + r)
            gt_W.append((solver(C.standardise(Xr)), Wr))

        rows = []
        for tau in TAUS:
            sets = [frozenset(J.edge_set(W, names, tau=tau)) for W in boot_W]
            boot_modal = max(sets.count(s) for s in set(sets)) / B_BOOT

            E_full = J.edge_set(W_full, names, tau=tau)
            scores = []
            for drop, (Ws, sub_names) in sub_W.items():
                E_sub = J.edge_set(Ws, sub_names, tau=tau)
                E_res = {(p, c) for (p, c) in E_full
                         if p in sub_names and c in sub_names}
                union = E_sub | E_res
                scores.append(1.0 if not union
                              else len(E_sub & E_res) / len(union))

            gsets = [frozenset(J.edge_set(grid_W[lam], names, tau=tau))
                     for lam in LAMBDAS]
            grid_modal = max(gsets.count(s) for s in set(gsets)) / len(gsets)

            tprs = [SL.score_recovery(W, Wt, tau)["tpr"] for W, Wt in gt_W]
            fdrs = [SL.score_recovery(W, Wt, tau)["fdr"] for W, Wt in gt_W]

            rows.append({
                "tau": tau,
                "n_edges_point_estimate": len(E_full),
                "bootstrap_modal_share": boot_modal,
                "self_compatibility": float(np.mean(scores)),
                "hyperparameter_modal_share": grid_modal,
                "mean_tpr": float(np.mean(tprs)),
                "mean_fdr": float(np.mean(fdrs)),
            })
        out["results"][label] = rows
        print(f"\n  {label}")
        print(f"    {'tau':>6} {'edges':>6} {'boot':>7} {'self':>7} "
              f"{'grid':>7} {'TPR':>7}")
        for r in rows:
            print(f"    {r['tau']:>6} {r['n_edges_point_estimate']:>6} "
                  f"{r['bootstrap_modal_share']:>7.3f} "
                  f"{r['self_compatibility']:>7.3f} "
                  f"{r['hyperparameter_modal_share']:>7.3f} "
                  f"{r['mean_tpr']:>7.3f}")

    # where does the legacy solver stop winning each diagnostic?
    L = {r["tau"]: r for r in out["results"]["legacy"]}
    C_ = {r["tau"]: r for r in out["results"]["corrected"]}
    keys = ["bootstrap_modal_share", "self_compatibility",
            "hyperparameter_modal_share"]
    out["legacy_wins_by_tau"] = {
        str(t): {k: bool(L[t][k] >= C_[t][k]) for k in keys} for t in TAUS}
    out["legacy_always_empty_by_tau"] = {
        str(t): bool(L[t]["n_edges_point_estimate"] == 0) for t in TAUS}

    print("\n  does the legacy solver still match or beat the corrected one?")
    for t in TAUS:
        w = out["legacy_wins_by_tau"][str(t)]
        print(f"    tau={t:<6} empty={out['legacy_always_empty_by_tau'][str(t)]!s:<5} "
              f"boot={w['bootstrap_modal_share']!s:<5} "
              f"self={w['self_compatibility']!s:<5} "
              f"grid={w['hyperparameter_modal_share']!s:<5}")

    J.save_json("e6_threshold", out)
    return out


if __name__ == "__main__":
    main()
