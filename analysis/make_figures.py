"""make_figures.py --- render every figure in the manuscript from saved results.

The LaTeX source performs no computation: it includes PDFs written here, and
every number quoted in the text is asserted against these same JSON files by
verify_manuscript.py.

    python make_figures.py

Design decisions, held fixed across all six figures:
  * Two entities recur throughout --- the legacy solver and the corrected one ---
    so they keep one colour each everywhere, and colour follows the entity, never
    its rank or its value.
  * Palette is Okabe-Ito, checked for colour-vision separation (worst adjacent
    pair dE 21.9 protan for the two-series case, 11.0 deutan for three).
    Line style and marker vary with the series as well, so the figures survive
    greyscale printing where colour alone would not.
  * Sequential fields use a single hue, light to dark. No rainbow maps.
  * Grid and axes are recessive; data-ink carries the reading.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import jdssv_common as J

# ── palette (validated; see module docstring) ─────────────────────────────
LEGACY = "#D55E00"      # vermillion   --- the non-convergent solver
CORRECT = "#0072B2"     # blue         --- the validated solver
THIRD = "#009E73"       # bluish green --- third categorical slot
INK = "#2B2B29"
MUTED = "#6E6E6A"
GRID = "#DCDCD8"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#F4F8FB", "#9CC4DE", CORRECT, "#00405F"])

STYLE = {"legacy": dict(color=LEGACY, marker="s", linestyle="--", lw=1.8, ms=5),
         "corrected": dict(color=CORRECT, marker="o", linestyle="-", lw=1.8, ms=5)}
LABEL = {"legacy": "Legacy solver (non-convergent)",
         "corrected": "Corrected solver (validated)"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "axes.axisbelow": True,
})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(J.FIGURES / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf")


# ── Figure 1: the scorecard ───────────────────────────────────────────────
def plot_scorecard():
    """Five diagnostics, two solvers. The legacy solver looks better on four."""
    e1 = J.load_json("e1_inversions")
    L, C = e1["solvers"]["legacy"], e1["solvers"]["corrected"]

    def _prop(s, key):
        """A proportion diagnostic, with a Wilson interval."""
        n = s[key]["B"] if key == "bootstrap_stability" else s[key]["n_points"]
        k = s[key]["modal_count"]
        lo, hi = J.wilson_interval(k, n)
        return k / n, lo, hi

    def _mean(values):
        """A mean diagnostic, with a normal interval on the mean."""
        v = np.asarray(values, dtype=float)
        m = float(v.mean())
        half = 1.96 * float(v.std(ddof=1)) / np.sqrt(len(v)) if len(v) > 1 else 0.0
        return m, max(0.0, m - half), min(1.0, m + half)

    def _row(s, kind, key):
        if kind == "prop":
            return _prop(s, key)
        if kind == "selfcompat":
            return _mean([r["score"] for r in s["self_compatibility"]["per_subset"]])
        if kind == "tpr":
            return _mean(s["ground_truth_recovery"]["tpr_per_rep"])
        return 1.0, 1.0, 1.0            # acyclicity: exact, no sampling error

    rows = [
        ("Bootstrap stability\n(share of resamples giving the modal graph)",
         _row(L, "prop", "bootstrap_stability"),
         _row(C, "prop", "bootstrap_stability")),
        ("Self-compatibility\n(mean agreement across variable subsets)",
         _row(L, "selfcompat", None), _row(C, "selfcompat", None)),
        ("Hyperparameter invariance\n(share of grid giving the modal graph)",
         _row(L, "prop", "hyperparameter_invariance"),
         _row(C, "prop", "hyperparameter_invariance")),
        ("Acyclicity certificate\n(passes at $h(W)=0$, exactly)",
         _row(L, "exact", None), _row(C, "exact", None)),
        ("Ground-truth recovery\n(TPR against known DAGs)",
         _row(L, "tpr", None), _row(C, "tpr", None)),
    ]

    fig, ax = plt.subplots(figsize=(6.1, 3.45))
    y = np.arange(len(rows))[::-1]
    h = 0.34
    for k, (_, legacy, corrected) in enumerate(rows):
        for (val, lo, hi), off, colour in ((legacy, +h / 2, LEGACY),
                                           (corrected, -h / 2, CORRECT)):
            ax.barh(y[k] + off, val, height=h, color=colour,
                    edgecolor="white", linewidth=0.8)
            if hi > lo:                       # no bar where the value is exact
                ax.plot([lo, hi], [y[k] + off] * 2, color=INK, lw=1.0,
                        solid_capstyle="butt", zorder=3)
                for b in (lo, hi):
                    ax.plot([b, b], [y[k] + off - 0.07, y[k] + off + 0.07],
                            color=INK, lw=1.0, zorder=3)
            ax.text(max(val, hi) + 0.018, y[k] + off, f"{val:.3f}", va="center",
                    ha="left", fontsize=7.5, color=INK)

    # The dividing line: everything above consults only the estimates.
    ax.axhline(y[-1] + 0.5, color=INK, lw=0.9, linestyle=":")
    ax.text(1.20, float(np.mean(y[:-1])), "consults only\nthe solver's\nown output",
            fontsize=7.5, color=MUTED, va="center", ha="center", style="italic")
    ax.text(1.20, y[-1], "consults an\nexternal\nreference",
            fontsize=7.5, color=MUTED, va="center", ha="center", style="italic")
    ax.plot([1.09, 1.09], [y[-1] + 0.65, y[0] + 0.45], color=GRID, lw=1.2,
            solid_capstyle="butt")
    ax.plot([1.09, 1.09], [y[-1] - 0.45, y[-1] + 0.35], color=GRID, lw=1.2,
            solid_capstyle="butt")

    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlim(0, 1.34)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("Diagnostic value, with 95% interval "
                  "(higher reads as more reassuring in every row)")
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=LEGACY),
               plt.Rectangle((0, 0), 1, 1, color=CORRECT)]
    ax.legend(handles, [LABEL["legacy"], LABEL["corrected"]],
              loc="lower center", bbox_to_anchor=(0.42, -0.30), ncol=2)
    ax.set_title("On no diagnostic that consults only its own output is the "
                 "legacy solver\nthe worse of the two: three favour it, one "
                 "cannot separate them", loc="left", pad=10)
    fig.tight_layout()
    save(fig, "fig1_scorecard")


# ── Figure 2: inside the optimiser (trajectory + gradient balance) ───────
def plot_optimiser_internals():
    """Two views of the same run: what the iterate does, and why."""
    e2 = J.load_json("e2_instrument")
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(6.1, 2.6))

    # (a) the iterate
    for key in ("corrected", "legacy"):
        log = e2[key]["log"]
        axL.plot([r["outer"] for r in log], [r["W_norm"] for r in log],
                 label=LABEL[key].split(" (")[0], **STYLE[key])
    lg = e2["legacy"]["log"]
    axL.annotate(f"collapses to {lg[-1]['W_norm']:.3f}\n(initialised at 0)",
                 xy=(lg[-1]["outer"], lg[-1]["W_norm"]),
                 xytext=(-4, 34), textcoords="offset points",
                 fontsize=7, color=INK, ha="right",
                 arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    axL.set_xlabel("Outer iteration")
    axL.set_ylabel(r"$\|W\|_F$")
    axL.set_ylim(0, 1.15)
    axL.legend(loc="center left", fontsize=7.5)
    axL.set_title("(a) what the iterate does", loc="left", fontsize=8.5)

    # (b) why
    outer = [r["outer"] for r in lg]
    axR.semilogy(outer, [max(r["grad_data_norm"], 1e-12) for r in lg],
                 color=CORRECT, marker="o", ms=3.5, lw=1.6,
                 label=r"data-fit $\|\nabla \ell\|$")
    axR.semilogy(outer, [max(r["grad_acyclicity_norm"], 1e-12) for r in lg],
                 color=LEGACY, marker="s", ms=3.5, lw=1.6, linestyle="--",
                 label=r"acyclicity $\|(\rho h{+}\alpha)\nabla h\|$")
    axR.axhline(10.0, color=INK, lw=0.9, linestyle=":")
    axR.text(0.3, 16.0, "clip at norm 10", fontsize=7, color=INK, va="bottom")
    clipped = [r["outer"] for r in lg if r["clipped"]]
    axR.plot(clipped, [10.0] * len(clipped), linestyle="none", marker="v",
             ms=5, color=INK, label=f"clip active ({len(clipped)}/{len(lg)})")
    axR.set_xlabel("Outer iteration")
    axR.set_ylabel("Gradient-term norm")
    axR.legend(loc="lower right", fontsize=7)
    axR.set_title("(b) why", loc="left", fontsize=8.5)

    fig.tight_layout()
    save(fig, "fig2_optimiser_internals")


# ── Figure 4: the hyperparameter field ────────────────────────────────────
def plot_hyperparameter_field():
    """A flat field reads as robustness. Here it is a solver that does nothing."""
    e1 = J.load_json("e1_inversions")
    lams, taus = e1["lambda_grid"], e1["tau_grid"]
    fig, axes = plt.subplots(1, 2, figsize=(6.1, 2.75), sharey=True)
    fig.subplots_adjust(wspace=0.12)
    vmax = max(g["n_edges"] for s in e1["solvers"].values()
               for g in s["hyperparameter_invariance"]["grid"])
    for ax, key in zip(axes, ("legacy", "corrected")):
        grid = e1["solvers"][key]["hyperparameter_invariance"]
        M = np.zeros((len(taus), len(lams)))
        for g in grid["grid"]:
            M[taus.index(g["tau"]), lams.index(g["lambda1"])] = g["n_edges"]
        im = ax.imshow(M, origin="lower", cmap=SEQ, vmin=0, vmax=vmax,
                       aspect="auto")
        ax.set_xticks(range(len(lams)))
        ax.set_xticklabels([f"{v:g}" for v in lams], rotation=90)
        ax.set_yticks(range(len(taus)))
        ax.set_yticklabels([f"{v:g}" for v in taus])
        ax.set_xlabel(r"$\lambda_1$")
        ax.grid(False)
        # detail moves to the caption; the panel keeps only its name
        ax.set_title(LABEL[key].split(" (")[0], fontsize=8.5)
    axes[0].set_ylabel(r"$\tau$")
    cb = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.02)
    cb.set_label("Edges in the thresholded graph", fontsize=8)
    cb.outline.set_visible(False)
    save(fig, "fig4_hyperparameter_field")


# ── Figure 5: recovery, and how much of it is varsortability ─────────────
def plot_recovery():
    """Single panel: the recovery curves with and without varsortability.

    An earlier version carried a second panel plotting the TPR gap against
    varsortability. With one point per effect regime that is three points, and
    three points stating a monotone relationship are better stated in a
    sentence than drawn; the panel was dropped.
    """
    e4 = J.load_json("e4_recovery")
    ns = e4["n_grid"]
    regimes = list(e4["regimes"].keys())
    colors = {"weak": THIRD, "moderate": CORRECT, "strong": LEGACY}

    fig, ax = plt.subplots(figsize=(6.1, 2.9))
    for reg in regimes:
        raw = e4["curves"][f"corrected|{reg}|raw"]
        std = e4["curves"][f"corrected|{reg}|standardised"]
        wr = e4["regimes"][reg]
        for rows_ in (raw, std):
            m = np.array([r["mean_tpr"] for r in rows_])
            se = np.array([r.get("se_tpr", 0.0) for r in rows_])
            ax.fill_between(ns, np.clip(m - 1.96 * se, 0, 1),
                            np.clip(m + 1.96 * se, 0, 1),
                            color=colors[reg], alpha=0.13, linewidth=0)
        ax.plot(ns, [r["mean_tpr"] for r in raw], color=colors[reg],
                marker="o", ms=4, lw=1.7, linestyle="-",
                label=f"{reg} $|w| \\in$ [{wr[0]}, {wr[1]}]")
        ax.plot(ns, [r["mean_tpr"] for r in std], color=colors[reg],
                marker="s", ms=4, lw=1.4, linestyle="--", alpha=0.85)
        gap = e4["varsortability_gap_at_n77"][reg]["tpr_gap"]
        at77 = next(r for r in std if r["n"] == 77)
        # the weak label goes below its point; its solid curve runs just above
        dy = -11 if reg == "weak" else 8
        ax.annotate(f"$-${gap:.2f}", xy=(77, at77["mean_tpr"]),
                    xytext=(7, dy), textcoords="offset points",
                    fontsize=7, color=colors[reg])

    ax.axvline(77, color=INK, lw=0.9, linestyle=":")
    ax.text(74, 0.90, "$n = 77$", fontsize=7.5, color=INK, ha="right")
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(v) for v in ns])
    ax.set_xlabel("Sample size $n$")
    ax.set_ylabel("True positive rate")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.70), fontsize=7.0,
              handlelength=1.6, borderpad=0.3, labelspacing=0.3)
    ax.set_title("Solid: data as drawn.  Dashed: columns standardised.  "
                 "Labels give the gap at $n=77$", loc="left", fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig5_recovery")


# ── Figure 6: the collapse region ────────────────────────────────────────
def plot_collapse_region():
    e3 = J.load_json("e3_collapse")
    cells = e3["cells"]
    clips = [c["clip_label"] for c in cells]
    clip_order = ["0.1", "1", "10", "100", "1000", "10000", "none"]
    clip_order = [c for c in clip_order if c in set(clips)]
    lr0s = sorted({c["lr0"] for c in cells})
    inners = sorted({c["inner"] for c in cells})

    fig, axes = plt.subplots(1, len(inners), figsize=(6.1, 2.45), sharey=True)
    for ax, inner in zip(np.atleast_1d(axes), inners):
        M = np.full((len(clip_order), len(lr0s)), np.nan)
        for c in cells:
            if c["inner"] == inner:
                M[clip_order.index(c["clip_label"]),
                  lr0s.index(c["lr0"])] = c["share_collapsed"]
        im = ax.imshow(M, origin="lower", cmap=SEQ, vmin=0, vmax=1,
                       aspect="auto")
        ax.set_xticks(range(len(lr0s)))
        ax.set_xticklabels([f"{v:g}" for v in lr0s], rotation=90)
        ax.set_yticks(range(len(clip_order)))
        ax.set_yticklabels(clip_order)
        ax.set_xlabel("initial step size")
        ax.set_title(f"{inner} inner steps", fontsize=8.5)
        ax.grid(False)
        # mark the configuration that produced the published failure
        if inner == 50 and "10" in clip_order and 0.01 in lr0s:
            ax.plot([lr0s.index(0.01)], [clip_order.index("10")], marker="*",
                    ms=13, color="white", markeredgecolor=INK,
                    markeredgewidth=0.8)
            ax.annotate("the configuration\nthat shipped",
                        xy=(lr0s.index(0.01), clip_order.index("10")),
                        xytext=(-6, 22), textcoords="offset points",
                        fontsize=7, color=INK, ha="left",
                        arrowprops=dict(arrowstyle="-", color=INK, lw=0.7))
    np.atleast_1d(axes)[0].set_ylabel("gradient clip norm")
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("Share of datasets on which\nthe solver returns no edges",
                 fontsize=7.5)
    cb.outline.set_visible(False)
    fig.suptitle(f"{e3['n_cells_fully_collapsed']} of {e3['n_cells']} "
                 f"configurations collapse on every dataset",
                 x=0.085, ha="left", fontsize=9.5, y=1.04)
    save(fig, "fig6_collapse_region")


def main():
    J.print_banner("rendering figures")
    plot_scorecard()
    plot_optimiser_internals()
    plot_hyperparameter_field()
    plot_recovery()
    plot_collapse_region()
    plot_robustness()
    print("\n  all figures written to output/figures/")




# ── Figure 7: does the inversion survive a change of data? ───────────────
def plot_robustness():
    """The inversion is not a property of one dataset."""
    e5 = J.load_json("e5_robustness")
    labels = [s["label"] for s in e5["settings"]]
    panels = [("bootstrap_modal_share", "Bootstrap\nstability"),
              ("self_compatibility", "Self-\ncompatibility"),
              ("hyperparameter_modal_share", "Hyperparameter\ninvariance"),
              ("ground_truth_tpr", "Ground-truth\nrecovery")]

    fig, axes = plt.subplots(1, 4, figsize=(6.1, 2.65), sharey=True)
    y = np.arange(len(labels))[::-1]
    for ax, (key, title) in zip(axes, panels):
        for solver in ("legacy", "corrected"):
            vals = [e5["results"][f"{lab}|{solver}"][key] for lab in labels]
            ax.plot(vals, y, linestyle="none",
                    marker=STYLE[solver]["marker"], ms=6,
                    color=STYLE[solver]["color"], label=LABEL[solver])
        for k, lab in enumerate(labels):
            lo = e5["results"][f"{lab}|legacy"][key]
            hi = e5["results"][f"{lab}|corrected"][key]
            ax.plot([lo, hi], [y[k], y[k]], color=GRID, lw=1.4, zorder=0)
        ax.set_xlim(-0.05, 1.08)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_title(title, fontsize=7.5, linespacing=1.3)
        ax.grid(axis="y", visible=False)
        # the one panel where the ordering reverses
        if key == "ground_truth_tpr":
            for spine in ("bottom", "left"):
                ax.spines[spine].set_color(INK)
            ax.set_xlabel("consults truth", fontsize=7.5, color=INK)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=7.5)
    axes[0].legend(loc="lower center", bbox_to_anchor=(2.45, -0.34), ncol=2)
    # the message lives in the LaTeX caption, not in a suptitle
    fig.tight_layout(rect=(0, 0.04, 1, 1.0))
    save(fig, "fig7_robustness")


if __name__ == "__main__":
    main()
