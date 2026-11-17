"""orientation.py --- which way do the edges point?

Both conventions appear in the literature and in code. Getting it backwards
reverses every reported edge, and no stability diagnostic can detect it,
because a transposed graph is exactly as stable as the original.
"""
from __future__ import annotations

import numpy as np

from .reference import simulate_linear_sem


def check_orientation(fit, n=1000, d=6, n_edges=6, tau=0.08, seed=11):
    """Empirically determine whether ``fit`` returns W[i,j] as i->j or j->i.

    Fits on data from a known DAG and scores the estimate against the truth and
    against its transpose. The convention with the better recovery is the one
    the solver uses.

    Returns a dict with ``convention`` in {"i->j", "j->i", "undetermined"}.
    "undetermined" means neither orientation recovered the structure, which
    usually means the solver is not working at all --- run `assert_solver_valid`.
    """
    from .reference import score_recovery
    X, W_true = simulate_linear_sem(n, d, n_edges, seed=seed)
    W = np.asarray(fit(X), dtype=float)
    direct = score_recovery(W, W_true, tau)
    flipped = score_recovery(W.T, W_true, tau)

    if max(direct["tpr"], flipped["tpr"]) < 0.25:
        convention = "undetermined"
    elif direct["tpr"] >= flipped["tpr"]:
        convention = "i->j"
    else:
        convention = "j->i"
    return {"convention": convention,
            "tpr_as_given": direct["tpr"],
            "tpr_transposed": flipped["tpr"],
            "note": ("neither orientation recovers the structure; the solver "
                     "may not be converging" if convention == "undetermined"
                     else "")}
