"""dagcheck --- exploratory diagnostics for continuous structure-learning solvers.

This package does not learn causal structure. It wraps a solver you already
have and asks whether that solver, in your configuration, on data like yours,
works at all.

The motivating failure is documented in the accompanying paper: a solver that
returned the empty graph for every input, while bootstrap stability, subset
agreement, hyperparameter invariance and the acyclicity certificate all scored
it at or above a solver that worked. Each of those is a functional of the
solver's own output, and no such functional can detect a solver that ignores
its input. The remedy is to consult an external reference.

Minimal use, which belongs at import rather than in a cell that can be skipped:

    from dagcheck import assert_solver_valid
    assert_solver_valid(my_solver, tau=0.08, n=77, d=8)

A solver is any callable ``fit(X) -> W`` returning a d x d weighted adjacency
matrix in the convention ``W[i, j] != 0`` meaning ``i -> j``. If your solver
uses the transpose, wrap it or run `check_orientation` to find out.
"""
from .reference import simulate_linear_sem, compute_varsortability, score_recovery
from .gate import assert_solver_valid, compute_recovery_curve, SolverGateError
from .orientation import check_orientation
from .trace import trace_iterates, plot_gradient_balance, plot_trajectory
from .collapse import map_collapse_region

__version__ = "0.1.0"
__all__ = [
    "assert_solver_valid", "compute_recovery_curve", "SolverGateError",
    "check_orientation", "trace_iterates", "plot_gradient_balance",
    "plot_trajectory", "map_collapse_region",
    "simulate_linear_sem", "compute_varsortability", "score_recovery",
]
