"""jdssv_common.py --- shared setup for the JDSSV paper's experiments.

Thin adapter over the thesis revision package so the paper's scripts share one
data loader, one legacy solver and one validated solver rather than copies.

Conventions used throughout, and worth stating once:
  * every solver minimises ||X - XW||_F^2, so W[i, j] != 0 means i -> j;
  * the legacy solver is `compute_legacy_W`. It is retained only to reproduce
    the failure the paper is about, and must never be used for inference.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE_ = pathlib.Path(__file__).resolve().parent   # holds the vendored modules
if str(HERE_) not in sys.path:
    sys.path.insert(0, str(HERE_))

import common as C                      # noqa: E402
import structure_learners as SL         # noqa: E402

# The paper's own analysis runs through the released package, so the figures
# and the software contribution cannot drift apart.
_PKG = pathlib.Path(__file__).resolve().parents[1] / "dagcheck"
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))
import dagcheck as dc                   # noqa: E402

HERE = pathlib.Path(__file__).parent
RESULTS = HERE / "results"
FIGURES = HERE.parent / "output" / "figures"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

# The operating point the thesis and the companion paper both use.
LAMBDA1 = C.NOTEARS_LAMBDA1
TAU = C.NOTEARS_TAU
SEED = 42

DAGCHECK_VERSION = dc.__version__

STRUCT_VARS = list(C.STRUCT_VARS)
ASSUMED_EDGES = list(C.ASSUMED_EDGES)


def load_nepal():
    """Standardised 77 x 8 structural design matrix and its column names."""
    df_dist, _, _ = C.load_data()
    return C.struct_matrix(df_dist)


def compute_legacy_W(X, lambda1=LAMBDA1):
    """The non-convergent projected-gradient solver. Not for inference."""
    return C.notears_legacy(X, lambda1=lambda1)


def compute_corrected_W(X, lambda1=LAMBDA1):
    """Validated linear NOTEARS (augmented Lagrangian, L-BFGS-B, exact expm)."""
    return SL.notears_linear(X, lambda1=lambda1)


SOLVERS = {"legacy": compute_legacy_W, "corrected": compute_corrected_W}


def threshold_W(W, tau=TAU):
    Wt = np.asarray(W, dtype=float).copy()
    Wt[np.abs(Wt) < tau] = 0.0
    return Wt


def edge_set(W, names, tau=TAU):
    """Directed edges above tau as a set of (parent, child) pairs."""
    return {(p, c) for p, c, _ in SL.edges_from_W(threshold_W(W, tau), names)}


def save_json(name, obj):
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(_jsonable(obj), indent=1))
    print(f"  wrote results/{name}.json")
    return path


def load_json(name):
    return json.loads((RESULTS / f"{name}.json").read_text())


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, float):
        return None if not np.isfinite(o) else o
    return o


def print_banner(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def wilson_interval(k, n, z=1.96):
    """Wilson score interval for a binomial proportion.

    Used for the diagnostics that are proportions --- the share of bootstrap
    resamples or of grid points returning the modal graph. The Wilson interval
    is used rather than the normal approximation because these proportions sit
    at or near 1, where the normal interval is badly behaved and would run past
    the boundary.
    """
    import math
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))
