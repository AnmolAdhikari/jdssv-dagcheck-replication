"""trace.py --- look at the optimiser instead of its output.

A solver that exposes its iterates can be inspected the way Tukey inspected
residuals. These helpers take a log of per-iteration state and draw it.

The intended workflow is that you instrument your own solver to append a dict
per outer iteration, then pass the list here. `trace_iterates` wraps the common
case where the solver accepts a callback.
"""
from __future__ import annotations

import numpy as np


def trace_iterates(fit, X, callback_kwarg="callback"):
    """Run ``fit`` on ``X``, collecting per-iteration state if it supports a callback.

    The solver is expected to call ``callback(W, info)`` once per outer
    iteration, where ``info`` is a dict that may carry ``h``, ``rho``,
    ``alpha``, ``grad_data_norm``, ``grad_acyclicity_norm`` and ``clipped``.
    Returns ``(W_final, log)``; ``log`` is a list of dicts, one per iteration,
    always containing ``outer`` and ``W_norm``.
    """
    log = []

    def _callback(W, info=None):
        row = {"outer": len(log), "W_norm": float(np.linalg.norm(W)),
               "max_abs_w": float(np.abs(W).max())}
        if info:
            row.update({k: v for k, v in info.items()})
        log.append(row)

    W = fit(X, **{callback_kwarg: _callback})
    if not log:
        raise ValueError(
            "the solver did not call the callback, so there is nothing to "
            "trace. Instrument it to call callback(W, info) once per outer "
            "iteration; see the paper's e2_instrument.py for a worked example.")
    return W, log


def plot_trajectory(logs, labels=None, ax=None):
    """Plot ||W||_F against outer iteration for one or more logs.

    A solver that collapses toward its initialisation is visible here and
    nowhere else: every statistic computed from the final estimate is
    consistent with a sparse solution.
    """
    import matplotlib.pyplot as plt
    if isinstance(logs, dict) or (logs and isinstance(logs[0], dict)):
        logs = [logs]
    labels = labels or [f"solver {k}" for k in range(len(logs))]
    ax = ax or plt.subplots(figsize=(6.1, 3.2))[1]
    for log, label in zip(logs, labels):
        ax.plot([r["outer"] for r in log], [r["W_norm"] for r in log],
                marker="o", ms=4, lw=1.8, label=label)
    ax.set_xlabel("Outer iteration")
    ax.set_ylabel(r"$\|W\|_F$")
    ax.legend(frameon=False)
    return ax


def plot_gradient_balance(log, clip=None, ax=None):
    """Plot the data-fit and acyclicity gradient norms against iteration.

    Requires ``grad_data_norm`` and ``grad_acyclicity_norm`` in the log. Draw
    the clip norm on with ``clip=`` to see whether clipping is active in the
    regime where the acyclicity term dominates, which is the configuration in
    which the data-fit magnitude stops influencing the step at all.
    """
    import matplotlib.pyplot as plt
    missing = [k for k in ("grad_data_norm", "grad_acyclicity_norm")
               if k not in log[0]]
    if missing:
        raise KeyError(f"log is missing {missing}; record them in the callback")
    ax = ax or plt.subplots(figsize=(6.1, 3.3))[1]
    outer = [r["outer"] for r in log]
    ax.semilogy(outer, [max(r["grad_data_norm"], 1e-30) for r in log],
                marker="o", ms=4, lw=1.7, label=r"data-fit $\|\nabla \ell\|$")
    ax.semilogy(outer, [max(r["grad_acyclicity_norm"], 1e-30) for r in log],
                marker="s", ms=4, lw=1.7, linestyle="--",
                label=r"acyclicity $\|(\rho h + \alpha)\nabla h\|$")
    if clip is not None:
        ax.axhline(clip, color="black", lw=1.0, linestyle=":")
    ax.set_xlabel("Outer iteration")
    ax.set_ylabel("Gradient-term norm")
    ax.legend(frameon=False)
    return ax
