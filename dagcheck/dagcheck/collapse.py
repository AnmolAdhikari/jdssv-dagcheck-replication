"""collapse.py --- where in configuration space does the solver stop working?

A solver that fails at one configuration usually fails on a region of them.
Mapping that region turns "we made a mistake" into a statement about which
settings are safe.
"""
from __future__ import annotations

import itertools

import numpy as np

from .reference import simulate_linear_sem, score_recovery


def map_collapse_region(make_fit, grid, n=500, d=8, n_edges=None, tau=0.08,
                        n_datasets=8, seed=1000, standardise=False):
    """Score recovery over a grid of solver configurations.

    Parameters
    ----------
    make_fit : callable
        ``make_fit(**config) -> fit``, building a solver from one configuration.
    grid : dict
        Maps parameter name to the list of values to sweep. The Cartesian
        product is taken.

    Returns a list of dicts, one per configuration, each carrying the config,
    mean TPR/FDR/SHD and ``share_collapsed`` --- the fraction of datasets on
    which the thresholded estimate is empty.
    """
    n_edges = d if n_edges is None else n_edges
    datasets = [simulate_linear_sem(n, d, n_edges, seed=seed + s,
                                    standardise=standardise)
                for s in range(n_datasets)]
    keys = list(grid)
    cells = []
    for values in itertools.product(*(grid[k] for k in keys)):
        config = dict(zip(keys, values))
        fit = make_fit(**config)
        tprs, fdrs, shds, collapsed = [], [], [], 0
        for X, W_true in datasets:
            W = np.asarray(fit(X), dtype=float)
            sc = score_recovery(W, W_true, tau)
            tprs.append(sc["tpr"])
            fdrs.append(sc["fdr"])
            shds.append(sc["shd"])
            collapsed += int(np.abs(W).max() < tau)
        cells.append({**config,
                      "mean_tpr": float(np.mean(tprs)),
                      "mean_fdr": float(np.mean(fdrs)),
                      "mean_shd": float(np.mean(shds)),
                      "share_collapsed": collapsed / len(datasets)})
    return cells
