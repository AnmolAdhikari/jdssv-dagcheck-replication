"""Tests for dagcheck.

The decisive test is `test_gate_rejects_constant_solver`: it encodes the
failure the package exists to catch, so that if the gate ever stops catching
a constant map, the suite fails.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

import dagcheck as dc


# ── solvers used as fixtures ──────────────────────────────────────────────
def fit_empty(X, **kw):
    """The failure mode: ignores its input, returns (almost) zero."""
    d = X.shape[1]
    return np.full((d, d), 1e-3) * (1 - np.eye(d))


def fit_constant_nonzero(X, **kw):
    """A constant map that is not empty --- equally undetectable by stability."""
    d = X.shape[1]
    W = np.zeros((d, d))
    W[0, 1] = 0.9
    return W


def fit_ols_dag(X, **kw):
    """A crude but working learner: regress each variable on earlier ones in a
    variance-sorted order. Good enough to clear the gate on strong effects."""
    X = np.asarray(X, float)
    X = X - X.mean(0)
    d = X.shape[1]
    order = np.argsort(X.var(0))
    W = np.zeros((d, d))
    for k, j in enumerate(order):
        pa = order[:k]
        if len(pa) == 0:
            continue
        coef, *_ = np.linalg.lstsq(X[:, pa], X[:, j], rcond=None)
        W[pa, j] = coef
    return W


def fit_transposed(X, **kw):
    return fit_ols_dag(X).T


# ── reference machinery ───────────────────────────────────────────────────
def test_simulated_sem_satisfies_its_own_equation():
    X, W = dc.simulate_linear_sem(2000, 6, 6, seed=3)
    resid = X - X @ W
    # residuals are the noise: uncorrelated with the parents of each variable
    assert np.abs(np.corrcoef(resid.T)).max() < 1.2
    assert W.shape == (6, 6)
    assert np.allclose(np.diag(W), 0)


def test_varsortability_is_half_after_standardising():
    X, W = dc.simulate_linear_sem(2000, 8, 8, seed=5)
    assert dc.compute_varsortability(X, W) > 0.6
    Xs, Ws = dc.simulate_linear_sem(2000, 8, 8, seed=5, standardise=True)
    assert dc.compute_varsortability(Xs, Ws) == pytest.approx(0.5, abs=1e-9)


def test_score_recovery_on_perfect_and_empty():
    _, W = dc.simulate_linear_sem(200, 5, 5, seed=1)
    perfect = dc.score_recovery(W, W, tau=0.01)
    assert perfect["tpr"] == 1.0 and perfect["fdr"] == 0.0
    empty = dc.score_recovery(np.zeros_like(W), W, tau=0.01)
    assert empty["tpr"] == 0.0
    # the inversion in miniature: an empty graph has a PERFECT false discovery
    # rate, because it makes no discoveries
    assert empty["fdr"] == 0.0


# ── the gate ──────────────────────────────────────────────────────────────
def test_gate_rejects_constant_solver():
    with pytest.raises(dc.SolverGateError) as e:
        dc.assert_solver_valid(fit_empty, n=400, d=6, reps=3)
    assert "EMPTY" in str(e.value)
    # and it tells the user that an all-zero test would not have caught it
    assert "not exactly zero" in str(e.value)


def test_gate_rejects_constant_nonempty_solver():
    with pytest.raises(dc.SolverGateError):
        dc.assert_solver_valid(fit_constant_nonzero, n=400, d=6, reps=3)


def test_gate_accepts_working_solver():
    out = dc.assert_solver_valid(fit_ols_dag, n=500, d=6, reps=3, min_tpr=0.3)
    assert out["tpr"] >= 0.3


def test_gate_checks_output_shape():
    with pytest.raises(dc.SolverGateError):
        dc.assert_solver_valid(lambda X, **kw: np.zeros((3, 3)), d=6, reps=1)


def test_recovery_curve_is_monotone_ish_in_n():
    rows = dc.compute_recovery_curve(fit_ols_dag, n_grid=(50, 1000), d=6,
                                     reps=3)
    assert rows[-1]["mean_tpr"] >= rows[0]["mean_tpr"] - 0.2


def test_recovery_curve_drops_when_varsortability_removed():
    raw = dc.compute_recovery_curve(fit_ols_dag, n_grid=(500,), d=8, reps=5,
                                    standardise=False)
    std = dc.compute_recovery_curve(fit_ols_dag, n_grid=(500,), d=8, reps=5,
                                    standardise=True)
    # fit_ols_dag orders by variance, so it lives or dies by varsortability
    assert raw[0]["mean_tpr"] > std[0]["mean_tpr"]


# ── orientation ───────────────────────────────────────────────────────────
def test_orientation_detects_correct_convention():
    assert dc.check_orientation(fit_ols_dag)["convention"] == "i->j"


def test_orientation_detects_transposed_convention():
    assert dc.check_orientation(fit_transposed)["convention"] == "j->i"


def test_orientation_undetermined_for_broken_solver():
    out = dc.check_orientation(fit_empty)
    assert out["convention"] == "undetermined"
    assert "not be converging" in out["note"]


# ── collapse map ──────────────────────────────────────────────────────────
def test_collapse_map_separates_working_from_broken():
    def make_fit(broken):
        return fit_empty if broken else fit_ols_dag
    cells = dc.map_collapse_region(make_fit, {"broken": [True, False]},
                                   n=300, d=6, n_datasets=3)
    broken = next(c for c in cells if c["broken"])
    working = next(c for c in cells if not c["broken"])
    assert broken["share_collapsed"] == 1.0
    assert working["share_collapsed"] == 0.0
    assert working["mean_tpr"] > broken["mean_tpr"]


# ── trace ─────────────────────────────────────────────────────────────────
def test_trace_requires_an_instrumented_solver():
    with pytest.raises(ValueError, match="did not call the callback"):
        dc.trace_iterates(lambda X, callback=None: np.zeros((4, 4)),
                          np.random.default_rng(0).normal(size=(50, 4)))


def test_trace_collects_iterates():
    def fit(X, callback=None):
        W = np.zeros((X.shape[1], X.shape[1]))
        for t in range(4):
            W = W + 0.1
            np.fill_diagonal(W, 0)
            callback(W, {"h": 0.5 ** t})
        return W
    W, log = dc.trace_iterates(fit, np.zeros((10, 4)))
    assert len(log) == 4
    assert [r["outer"] for r in log] == [0, 1, 2, 3]
    assert log[-1]["W_norm"] > log[0]["W_norm"]
    assert "h" in log[0]
