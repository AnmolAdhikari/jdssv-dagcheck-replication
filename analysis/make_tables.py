"""make_tables.py --- emit the manuscript's tables as LaTeX fragments.

main.tex \input{}s these, so no number in a table is typed by hand. Fragments
are written to output/tables/.
"""
from __future__ import annotations

import pathlib

import numpy as np

import jdssv_common as J
import structure_learners as SL

TABLES = J.FIGURES.parent / "tables"
TABLES.mkdir(parents=True, exist_ok=True)


def write(name, body):
    (TABLES / f"{name}.tex").write_text(body)
    print(f"  wrote tables/{name}.tex")


def make_scorecard_table():
    e1 = J.load_json("e1_inversions")
    L, C = e1["solvers"]["legacy"], e1["solvers"]["corrected"]
    rows = [
        ("Bootstrap stability", "share of $B$ resamples giving the modal graph",
         f"{L['bootstrap_stability']['modal_set_share']:.3f}",
         f"{C['bootstrap_stability']['modal_set_share']:.3f}", "legacy"),
        ("Self-compatibility", "mean agreement across variable subsets",
         f"{L['self_compatibility']['mean_agreement']:.3f}",
         f"{C['self_compatibility']['mean_agreement']:.3f}", "legacy"),
        ("Hyperparameter invariance",
         "share of the $(\\lambda_1,\\tau)$ grid giving it",
         f"{L['hyperparameter_invariance']['modal_set_share']:.3f}",
         f"{C['hyperparameter_invariance']['modal_set_share']:.3f}", "legacy"),
        ("Acyclicity certificate", "$h(\\West)$ after thresholding",
         f"{L['acyclicity']['h_exact_after_threshold']:.2f}",
         f"{C['acyclicity']['h_exact_after_threshold']:.2f}", "tie"),
        ("Ground-truth recovery", "TPR against known DAGs",
         f"{L['ground_truth_recovery']['mean_tpr']:.3f}",
         f"{C['ground_truth_recovery']['mean_tpr']:.3f}", "corrected"),
    ]
    verdict = {"legacy": "legacy", "corrected": "\\textbf{corrected}", "tie": "tie"}
    lines = [
        "\\begin{table}[t]", "\\centering",
        "\\caption{The same five diagnostics applied to the same data by two "
        "solvers, one of which never leaves its initialisation. The first four "
        "are functions of the estimates alone; only the last consults an "
        "external reference. Higher is more reassuring in every row. Three rows "
        "favour the legacy solver strictly, the acyclicity certificate is "
        "satisfied exactly by both, and only ground-truth recovery favours the "
        "solver that works. "
        f"$B = {e1['B_bootstrap']}$, $\\lambda_1 = {e1['lambda1']}$, "
        f"$\\tau = {e1['tau']}$, $n = {e1['n']}$, $d = {e1['d']}$.}}",
        "\\label{tab:scorecard}",
        "\\small", "\\begin{tabular}{@{}lp{5.3cm}rrl@{}}", "\\toprule",
        "Diagnostic & Quantity & Legacy & Corrected & Favours \\\\",
        "\\midrule",
    ]
    for name, qty, lv, cv, who in rows[:4]:
        lines.append(f"{name} & {qty} & {lv} & {cv} & {verdict[who]} \\\\")
    lines += ["\\midrule"]
    name, qty, lv, cv, who = rows[4]
    lines.append(f"{name} & {qty} & {lv} & {cv} & {verdict[who]} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("tab1_scorecard", "\n".join(lines))


def make_recovery_table():
    e4 = J.load_json("e4_recovery")
    lines = [
        "\\begin{table}[!ht]", "\\centering",
        "\\caption{Structure recovery of a known 8-node DAG at $n = 77$, the "
        "size of the case study, with the validated solver at the operating "
        "point. Each regime is run twice: on the data as drawn, and with the "
        "columns standardised. The gap between them is the part of apparent "
        "recovery that varsortability supplies rather than structure learning, "
        f"over {e4['reps']} replications.}}",
        "\\label{tab:recovery}",
        "\\begin{tabular}{lrrrrrr}", "\\toprule",
        "& & \\multicolumn{2}{c}{As drawn} & \\multicolumn{2}{c}{Standardised} & \\\\",
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
        "Regime & Varsort. & TPR & FDR & TPR & FDR & TPR gap \\\\",
        "\\midrule",
    ]
    for reg, wr in e4["regimes"].items():
        raw = next(r for r in e4["curves"][f"corrected|{reg}|raw"] if r["n"] == 77)
        std = next(r for r in e4["curves"][f"corrected|{reg}|standardised"]
                   if r["n"] == 77)
        gap = raw["mean_tpr"] - std["mean_tpr"]
        lines.append(
            f"{reg.capitalize()} $|w| \\in [{wr[0]}, {wr[1]}]$ & "
            f"{raw['mean_varsortability']:.3f} & "
            f"{raw['mean_tpr']:.3f} & {raw['mean_fdr']:.3f} & "
            f"{std['mean_tpr']:.3f} & {std['mean_fdr']:.3f} & "
            f"${gap:+.3f}$ \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("tab2_recovery", "\n".join(lines))


def make_edges_table():
    """Appendix: the learned structure, with how often resampling keeps it."""
    e1 = J.load_json("e1_inversions")
    X, names = J.load_nepal()
    W = J.compute_corrected_W(X)
    edges = sorted(SL.edges_from_W(J.threshold_W(W), names),
                   key=lambda e: -abs(e[2]))
    bif = e1["solvers"]["corrected"]["bootstrap_stability"]["bif"]
    assumed = {(p, c) for p, c in e1.get("assumed_edges", [])} or \
        {(p, c) for p, c in J.ASSUMED_EDGES}

    lines = [
        "\\begin{table}[!ht]", "\\centering",
        "\\caption{The structure recovered from the 77 districts by the "
        "validated solver, ordered by absolute weight, with the share of "
        f"$B = {e1['B_bootstrap']}$ district resamples in which each edge "
        "reappears. Only one edge listed here exceeds 0.60, and none exceeds "
        "0.63. A second edge exceeds 0.60 across the bootstrap without "
        "appearing in this point estimate at all "
        "(\\code{Digital\\_Access} $\\to$ \\code{Poverty}, 0.619), which is "
        "itself a measure of how unstable the estimate is. The rightmost column "
        "marks the edges the theory-based graph also asserts.}",
        "\\label{tab:edges}",
        "\\begin{tabular}{llrrc}", "\\toprule",
        "Parent & Child & Weight & Resample share & In theory graph \\\\",
        "\\midrule",
    ]
    for p, c, w in edges:
        key = f"{p}->{c}"
        p_tex, c_tex = p.replace("_", "\\_"), c.replace("_", "\\_")
        in_theory = "yes" if (p, c) in assumed else ""
        lines.append(
            f"\\code{{{p_tex}}} & \\code{{{c_tex}}} & "
            f"${w:+.3f}$ & {bif.get(key, 0.0):.3f} & "
            f"{in_theory} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("tab3_edges", "\n".join(lines))
    return edges, bif


def make_threshold_table():
    """Appendix: how the inversion depends on the edge threshold."""
    e6 = J.load_json("e6_threshold")
    L = {r["tau"]: r for r in e6["results"]["legacy"]}
    C_ = {r["tau"]: r for r in e6["results"]["corrected"]}
    lines = [
        "\\begin{table}[!ht]", "\\centering", "\\small",
        "\\caption{The three self-consistency diagnostics and ground-truth "
        "recovery as the edge threshold $\\tau$ varies, on the case-study data. "
        "The legacy solver's largest absolute weight is $0.032$, so above that "
        "floor it returns the empty graph and is a constant map; below it the "
        "graph is non-empty noise. Bold marks the solver each diagnostic "
        f"favours. $B = {e6['B']}$ resamples, {e6['n_reps_ground_truth']} "
        "replications for recovery.}",
        "\\label{tab:threshold}",
        "\\begin{tabular}{rrrrrrrrr}", "\\toprule",
        "& \\multicolumn{2}{c}{Edges} & \\multicolumn{2}{c}{Bootstrap "
        "stability} & \\multicolumn{2}{c}{Self-compat.} & "
        "\\multicolumn{2}{c}{Recovery (TPR)} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}",
        "$\\tau$ & leg. & corr. & leg. & corr. & leg. & corr. & leg. & corr. \\\\",
        "\\midrule",
    ]
    def b(x, y, higher_is_better=True):
        """Bold whichever value the diagnostic favours."""
        win = x >= y if higher_is_better else x <= y
        fx = f"\\textbf{{{x:.3f}}}" if win else f"{x:.3f}"
        fy = f"{y:.3f}" if win else f"\\textbf{{{y:.3f}}}"
        return fx, fy
    for tau in e6["taus"]:
        l, c = L[tau], C_[tau]
        bl, bc = b(l["bootstrap_modal_share"], c["bootstrap_modal_share"])
        sl, sc = b(l["self_compatibility"], c["self_compatibility"])
        tl, tc = b(l["mean_tpr"], c["mean_tpr"])
        lines.append(
            f"{tau:g} & {l['n_edges_point_estimate']} & "
            f"{c['n_edges_point_estimate']} & {bl} & {bc} & {sl} & {sc} & "
            f"{tl} & {tc} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("tab4_threshold", "\n".join(lines))


def main():
    J.print_banner("rendering tables")
    make_threshold_table()
    make_scorecard_table()
    make_recovery_table()
    edges, bif = make_edges_table()
    print(f"\n  {len(edges)} edges; max resample share "
          f"{max(bif.values()):.3f}; "
          f"{sum(v >= 0.60 for v in bif.values())} edges at or above 0.60")


if __name__ == "__main__":
    main()
