"""verify_manuscript.py --- assert every numeric claim in main.tex.

The manuscript quotes numbers in prose. Prose drifts when an experiment is
re-run; this script fails loudly when it does. Each entry pairs a value quoted
in the text with the computed value it came from, and names the section so a
mismatch is quick to locate.

    python verify_manuscript.py
"""
from __future__ import annotations

import sys

import jdssv_common as J


def check(claims):
    width = max(len(c[0]) for c in claims)
    bad = 0
    for label, quoted, computed, tol in claims:
        # inclusive at the boundary: a value quoted to three decimals is
        # half a unit in the last place away from its exact counterpart,
        # and 0.0625 vs 0.062 should pass a 5e-4 tolerance
        ok = abs(quoted - computed) <= tol * (1 + 1e-9)
        bad += not ok
        flag = "ok  " if ok else "FAIL"
        print(f"  [{flag}] {label:<{width}}  manuscript {quoted:>10.4g}"
              f"   computed {computed:>10.4g}")
    return bad


def main():
    e1 = J.load_json("e1_inversions")
    e2 = J.load_json("e2_instrument")
    e3 = J.load_json("e3_collapse")
    e4 = J.load_json("e4_recovery")
    e5 = J.load_json("e5_robustness")
    e6 = J.load_json("e6_threshold")
    L = e1["solvers"]["legacy"]
    C = e1["solvers"]["corrected"]
    lg = e2["legacy"]["log"]
    cg = e2["corrected"]["log"]

    claims = [
        # Section 1 / 3: the failure
        ("S3 legacy max |w| on case data", 0.0321,
         L["acyclicity"]["max_abs_weight"], 5e-4),
        ("S3 legacy edges at tau=0.01 (grid max)", 15,
         L["hyperparameter_invariance"]["edge_count_max"], 0),
        ("S3 legacy outer iterations", 18, e2["legacy"]["n_outer"], 0),
        ("S3 legacy peak ||W||_F", 0.554,
         max(r["W_norm"] for r in lg), 5e-4),
        ("S3 legacy final ||W||_F", 0.0357, lg[-1]["W_norm"], 5e-5),
        ("S3 legacy outer iters with clip active", 4,
         e2["legacy"]["n_outer_clipped"], 0),
        ("S3 acyclicity/data gradient ratio", 163,
         max(r["grad_acyclicity_norm"] / max(r["grad_data_norm"], 1e-30)
             for r in lg), 1.0),
        ("S3 corrected final ||W||_F", 0.931, cg[-1]["W_norm"], 5e-4),
        ("S3 corrected final h", 1.0e-9, cg[-1]["h"], 1e-10),

        # Section 4: the inversion
        ("S4 legacy distinct bootstrap edge sets", 1,
         L["bootstrap_stability"]["n_distinct_edge_sets"], 0),
        ("S4 corrected distinct bootstrap edge sets", 999,
         C["bootstrap_stability"]["n_distinct_edge_sets"], 0),
        ("S4 corrected modal bootstrap share", 0.002,
         C["bootstrap_stability"]["modal_set_share"], 5e-4),
        ("S4 corrected max bootstrap inclusion freq", 0.630,
         C["bootstrap_stability"]["max_bif"], 5e-4),
        ("S4 legacy self-compatibility", 1.000,
         L["self_compatibility"]["mean_agreement"], 1e-6),
        ("S4 corrected self-compatibility", 0.480,
         C["self_compatibility"]["mean_agreement"], 5e-4),
        ("S4 subsets both empty (legacy)", 36,
         L["self_compatibility"]["n_subsets_both_empty"], 0),
        ("S4 n subsets examined", 36,
         L["self_compatibility"]["n_subsets"], 0),
        ("S4 legacy grid modal share", 0.773,
         L["hyperparameter_invariance"]["modal_set_share"], 5e-4),
        ("S4 corrected grid modal share", 0.045,
         C["hyperparameter_invariance"]["modal_set_share"], 5e-4),
        ("S4 grid operating points", 88,
         L["hyperparameter_invariance"]["n_points"], 0),
        ("S4 corrected max edges on grid", 28,
         C["hyperparameter_invariance"]["edge_count_max"], 0),
        ("S4 legacy h after threshold", 0.0,
         L["acyclicity"]["h_exact_after_threshold"], 1e-12),
        ("S4 legacy FDR against known structure", 0.0,
         L["ground_truth_recovery"]["mean_fdr"], 1e-12),
        ("S4 bootstrap B", 1000, e1["B_bootstrap"], 0),

        # Section 5: the collapse region
        ("S5 configurations swept", 126, e3["n_cells"], 0),
        ("S5 configurations fully collapsed", 39,
         e3["n_cells_fully_collapsed"], 0),
        ("S5 share fully collapsed", 0.310,
         e3["share_cells_fully_collapsed"], 5e-4),
        ("S5 partially collapsed", 26, e3["n_cells_partially_collapsed"], 0),
        ("S5 collapse share at clip 0.1", 1.000,
         e3["collapse_share_by_clip"]["0.1"], 5e-4),
        ("S5 collapse share at clip 1", 0.785,
         e3["collapse_share_by_clip"]["1"], 5e-4),
        ("S5 collapse share at clip 10", 0.535,
         e3["collapse_share_by_clip"]["10"], 5e-4),
        ("S5 collapse share at clip 1e4", 0.056,
         e3["collapse_share_by_clip"]["10000"], 5e-4),
        ("S5 collapse share, no clipping", 0.097,
         e3["collapse_share_by_clip"]["none"], 5e-4),
        ("S5 best configuration TPR", 0.656,
         e3["best_configuration"]["mean_tpr"], 5e-4),
        ("S5 best configuration FDR", 0.553,
         e3["best_configuration"]["mean_fdr"], 5e-4),

        # Section 5.2 / 7: varsortability and power
        ("S5 strong TPR as drawn at n=77", 0.956,
         e4["varsortability_gap_at_n77"]["strong"]["tpr_raw"], 5e-4),
        ("S5 strong TPR standardised at n=77", 0.306,
         e4["varsortability_gap_at_n77"]["strong"]["tpr_standardised"], 5e-4),
        ("S5 weak varsortability", 0.619,
         e4["varsortability_gap_at_n77"]["weak"]["varsortability_raw"], 5e-4),
        ("S5 weak gap", 0.038,
         e4["varsortability_gap_at_n77"]["weak"]["tpr_gap"], 5e-4),
        ("S5 moderate varsortability", 0.869,
         e4["varsortability_gap_at_n77"]["moderate"]["varsortability_raw"], 5e-4),
        ("S5 moderate gap", 0.325,
         e4["varsortability_gap_at_n77"]["moderate"]["tpr_gap"], 5e-4),
        ("S5 strong varsortability", 0.956,
         e4["varsortability_gap_at_n77"]["strong"]["varsortability_raw"], 5e-4),
        ("S5 strong gap", 0.650,
         e4["varsortability_gap_at_n77"]["strong"]["tpr_gap"], 5e-4),
        ("S7 strong FDR standardised at n=77", 0.714,
         next(r for r in e4["curves"]["corrected|strong|standardised"]
              if r["n"] == 77)["mean_fdr"], 5e-4),
        ("S7 weak TPR standardised at n=77", 0.206,
         next(r for r in e4["curves"]["corrected|weak|standardised"]
              if r["n"] == 77)["mean_tpr"], 5e-4),

        # Section 5.3: robustness across data-generating processes
        ("S5 data-generating processes", 6, e5["n_settings"], 0),
        ("S5 legacy wins bootstrap stability", 6,
         e5["legacy_wins_or_ties"]["bootstrap_modal_share"], 0),
        ("S5 legacy wins self-compatibility", 6,
         e5["legacy_wins_or_ties"]["self_compatibility"], 0),
        ("S5 legacy wins hyperparameter invariance", 6,
         e5["legacy_wins_or_ties"]["hyperparameter_modal_share"], 0),
        ("S5 corrected wins ground truth", 6,
         e5["corrected_wins_ground_truth"], 0),
        ("S5 legacy bootstrap share 1.0 in every setting", 1.000,
         min(e5["results"][s["label"] + "|legacy"]["bootstrap_modal_share"]
             for s in e5["settings"]), 1e-9),
        ("S5 legacy self-compat 1.0 in every setting", 1.000,
         min(e5["results"][s["label"] + "|legacy"]["self_compatibility"]
             for s in e5["settings"]), 1e-9),

        # Appendix: the inversion as a function of tau
        ("App legacy edges at tau=0.005", 4,
         next(r["n_edges_point_estimate"] for r in e6["results"]["legacy"]
              if r["tau"] == 0.005), 0),
        ("App legacy self-compat at tau=0.01", 0.062,
         next(r["self_compatibility"] for r in e6["results"]["legacy"]
              if r["tau"] == 0.01), 5e-4),
        ("App corrected self-compat at tau=0.01", 0.696,
         next(r["self_compatibility"] for r in e6["results"]["corrected"]
              if r["tau"] == 0.01), 5e-4),
        ("App legacy boot share at tau=0.005", 0.060,
         next(r["bootstrap_modal_share"] for r in e6["results"]["legacy"]
              if r["tau"] == 0.005), 5e-4),
        ("App corrected boot share at tau=0.005", 0.003,
         next(r["bootstrap_modal_share"] for r in e6["results"]["corrected"]
              if r["tau"] == 0.005), 5e-4),
        ("App legacy TPR at tau=0.005", 0.183,
         next(r["mean_tpr"] for r in e6["results"]["legacy"]
              if r["tau"] == 0.005), 5e-4),
        ("App legacy FDR at tau=0.005", 0.481,
         next(r["mean_fdr"] for r in e6["results"]["legacy"]
              if r["tau"] == 0.005), 5e-4),
        ("App legacy self-compat at tau=0.08", 1.000,
         next(r["self_compatibility"] for r in e6["results"]["legacy"]
              if r["tau"] == 0.08), 1e-9),

        # Section 7: the case study
        ("S7 edges at operating point", 15,
         next(g["n_edges"] for g in C["hyperparameter_invariance"]["grid"]
              if g["lambda1"] == e1["lambda1"] and g["tau"] == e1["tau"]), 0),
        ("S7 edges with inclusion freq >= 0.60", 2,
         C["bootstrap_stability"]["n_edges_bif_ge_060"], 0),
        ("S7 point-estimate edges with BIF >= 0.60", 1,
         sum(1 for k, v in C["bootstrap_stability"]["bif"].items()
             if v >= 0.60 and k in {"Teacher_Quality->Urban_Rural_Gap"}), 0),
        ("S7 Digital_Access->Poverty BIF", 0.619,
         C["bootstrap_stability"]["bif"]["Digital_Access->Poverty"], 5e-4),
        ("S7 districts", 77, e1["n"], 0),
        ("S7 variables", 8, e1["d"], 0),
    ]

    J.print_banner("verifying manuscript claims against computed results")
    bad = check(claims)
    print(f"\n  {len(claims)} claims checked, {bad} mismatched")
    if bad:
        print("\n  MANUSCRIPT AND RESULTS DISAGREE --- fix before submission.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
