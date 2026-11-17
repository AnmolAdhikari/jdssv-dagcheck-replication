"""e4_recovery.py --- recovery at the operating point, and a reconciliation.

Two earlier runs of what looked like the same experiment disagreed: structure
recovery of a known 8-node DAG at n = 77 with strong weights came out at
TPR = 0.93 in one and TPR = 0.41 in the other. Neither was wrong. One passed
the simulated data to the solver as drawn; the other standardised the columns
first.

That gap is not noise, it is varsortability (Reisach et al., 2021). In data
drawn from a linear-Gaussian SEM, marginal variance tends to increase along the
causal order, so the order is partly readable from the variances alone, and a
solver can appear to recover structure by exploiting it. Standardising the
columns destroys that signal. The difference between the two curves therefore
measures how much of the apparent recovery was varsortability rather than
structure learning --- which makes it worth reporting as a diagnostic in its own
right, not a nuisance to be resolved by picking one convention.

Caveat recorded with the result: standardising changes the structural
coefficients, so a threshold tau calibrated against the generating weights is
no longer on their scale. Edge SUPPORT is unaffected, which is all that
score_recovery compares, but the thresholded estimate is harder to calibrate.

Output: results/e4_recovery.json
"""
from __future__ import annotations

import numpy as np

import jdssv_common as J
import structure_learners as SL

REGIMES = [("weak", (0.1, 0.3)), ("moderate", (0.3, 0.7)), ("strong", (0.5, 2.0))]
NS = [40, 77, 150, 300, 1000]
REPS = 20


# Varsortability comes from the released package rather than a local copy, so
# the number in the paper and the number a reader gets from dagcheck agree.
compute_varsortability = J.dc.compute_varsortability


def compute_recovery_curve(solver, weight_range, standardise, reps=REPS):
    import common as C
    rows = []
    for n in NS:
        sc, vs = [], []
        for rep in range(reps):
            X, W_true = SL.simulate_linear_sem(n, 8, 8,
                                               weight_range=weight_range,
                                               seed=500 + rep)
            vs.append(compute_varsortability(X, W_true))
            Xi = C.standardise(X) if standardise else X
            W_est = solver(Xi)
            sc.append(SL.score_recovery(W_est, W_true, tau=J.TAU))
        rows.append({
            "n": n,
            "se_tpr": float(np.std([s["tpr"] for s in sc], ddof=1)
                            / np.sqrt(len(sc))),
            "se_fdr": float(np.std([s["fdr"] for s in sc], ddof=1)
                            / np.sqrt(len(sc))),
            "mean_tpr": float(np.mean([s["tpr"] for s in sc])),
            "mean_fdr": float(np.mean([s["fdr"] for s in sc])),
            "mean_shd": float(np.mean([s["shd"] for s in sc])),
            "mean_n_pred": float(np.mean([s["n_pred"] for s in sc])),
            "mean_varsortability": float(np.mean(vs)),
        })
    return rows


def main():
    J.print_banner("e4 --- recovery at the operating point")
    print(f"  lambda1 = {J.LAMBDA1}, tau = {J.TAU}, {REPS} reps per point")

    out = {"lambda1": J.LAMBDA1, "tau": J.TAU, "reps": REPS, "n_grid": NS,
           "regimes": {label: list(wr) for label, wr in REGIMES},
           "curves": {}}

    for solver_label, solver in J.SOLVERS.items():
        for regime, wr in REGIMES:
            for scale_label, std in [("raw", False), ("standardised", True)]:
                key = f"{solver_label}|{regime}|{scale_label}"
                rows = compute_recovery_curve(solver, wr, std)
                out["curves"][key] = rows
                at77 = next(r for r in rows if r["n"] == 77)
                print(f"    {key:38s} n=77: TPR={at77['mean_tpr']:.3f}  "
                      f"FDR={at77['mean_fdr']:.3f}  "
                      f"varsort={at77['mean_varsortability']:.3f}")

    # The varsortability gap, which is the reconciliation.
    gaps = {}
    for regime, _ in REGIMES:
        raw = next(r for r in out["curves"][f"corrected|{regime}|raw"]
                   if r["n"] == 77)
        std = next(r for r in out["curves"][f"corrected|{regime}|standardised"]
                   if r["n"] == 77)
        gaps[regime] = {
            "tpr_raw": raw["mean_tpr"],
            "tpr_standardised": std["mean_tpr"],
            "tpr_gap": raw["mean_tpr"] - std["mean_tpr"],
            "varsortability_raw": raw["mean_varsortability"],
        }
    out["varsortability_gap_at_n77"] = gaps

    print("\n  varsortability gap at n = 77 (corrected solver):")
    for regime, g in gaps.items():
        print(f"    {regime:9s} TPR raw {g['tpr_raw']:.3f} -> standardised "
              f"{g['tpr_standardised']:.3f}   gap {g['tpr_gap']:+.3f}   "
              f"(varsortability {g['varsortability_raw']:.3f})")

    # The Nepal data for comparison: already standardised, so by construction
    # it carries no varsortability signal to exploit.
    X, names = J.load_nepal()
    out["nepal_column_variances"] = {
        "min": float(X.var(axis=0).min()), "max": float(X.var(axis=0).max())}
    print(f"\n  Nepal design matrix column variances: "
          f"{X.var(axis=0).min():.3f} to {X.var(axis=0).max():.3f} "
          f"(standardised, so no varsortability signal is available)")

    J.save_json("e4_recovery", out)
    return out


if __name__ == "__main__":
    main()
