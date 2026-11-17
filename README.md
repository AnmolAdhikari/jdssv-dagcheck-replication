# Exploratory Diagnostics for the Optimiser: replication package

Code and data for the manuscript *Exploratory Diagnostics for the Optimiser*,
submitted to the *Journal of Data Science, Statistics, and Visualisation*
(JDSSV) special issue *Exploratory Data Analysis and Visualisation: 50 Years of
John Tukey's Legacy*.

The repository contains:

- **`dagcheck`**, a small Python package of exploratory diagnostics for
  continuous structure-learning solvers such as NOTEARS. It checks whether a
  solver actually works before its output is interpreted.
- **`analysis/`**, the scripts that produce every result, figure and table in
  the paper.
- **`data/`**, the 77-district dataset the paper analyses.

## Quick start

```sh
git clone <this repository>
cd jdssv-dagcheck-replication
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e "dagcheck[plot,test]"

./run_all.sh
```

`run_all.sh` takes about ten minutes, most of it the B = 1000 bootstrap in
`e1`. It regenerates every result in `analysis/results/`, writes the figures
and tables to `output/`, and runs the package's test suite.

The results JSON files are committed, so you can run
`python analysis/make_figures.py` and `python analysis/make_tables.py` on
their own to rebuild the figures and tables without rerunning the experiments.

## Using `dagcheck`

A solver is any callable `fit(X) -> W` that returns a d x d weighted adjacency
matrix with `W[i, j] != 0` meaning `i -> j`.

```python
import dagcheck as dc

# Raises SolverGateError if the solver can't recover a known DAG at your n and d.
dc.assert_solver_valid(my_solver, tau=0.08, n=77, d=8)

# Recovery (TPR and related scores) as a function of sample size.
curve = dc.compute_recovery_curve(my_solver, n_grid=(40, 77, 150, 300, 1000))

# Is W[i, j] being read as i -> j, or as its transpose?
dc.check_orientation(my_solver)
```

`trace_iterates`, `plot_trajectory` and `plot_gradient_balance` log and plot
the optimiser's path. `map_collapse_region` maps where in hyperparameter space
the solver returns the empty graph. See the docstrings for details.

## Layout

```
dagcheck/                  the package (pip install -e dagcheck)
  dagcheck/gate.py         assert_solver_valid, compute_recovery_curve
  dagcheck/trace.py        trace_iterates, plot_gradient_balance, plot_trajectory
  dagcheck/orientation.py  check_orientation
  dagcheck/collapse.py     map_collapse_region
  dagcheck/reference.py    simulate_linear_sem, compute_varsortability, score_recovery
  tests/                   15 tests; the main one asserts the gate rejects a constant solver
analysis/
  common.py                variable set, operating point, data loader, legacy solver
  structure_learners.py    reference solvers (linear NOTEARS, DAGMA) and SEM simulator
  jdssv_common.py          shared setup: data, the two solvers, IO
  e1_inversions.py         the five diagnostics x two solvers          -> e1_inversions.json
  e2_instrument.py         iterate trajectory and gradient balance     -> e2_instrument.json
  e3_collapse.py           the collapse region in configuration space  -> e3_collapse.json
  e4_recovery.py           recovery curves and the varsortability gap  -> e4_recovery.json
  e5_robustness.py         the inversion across six DGPs               -> e5_robustness.json
  e6_threshold.py          the inversion as a function of tau          -> e6_threshold.json
  make_figures.py          the figures                                 -> output/figures/
  make_tables.py           the tables as LaTeX fragments               -> output/tables/
  results/                 saved JSON; figures and tables read these and never recompute
data/
  nepal_districts.csv      77 districts x the 8 structural variables
run_all.sh                 everything above, in order
```

## Which solver is which

`compute_legacy_W` (in `analysis/jdssv_common.py`, implemented in
`analysis/common.py`) is a projected-gradient NOTEARS solver that **does not
converge**. It is kept only to reproduce the failure the paper studies. Do not
use it for inference.

`compute_corrected_W` is linear NOTEARS (Zheng et al., 2018) solved with an
augmented Lagrangian and L-BFGS-B, using the exact matrix exponential.
`e2_instrument.py` reimplements both solvers with logging added, and checks on
every run that they match the originals to within 1e-10.

Edge orientation throughout: every solver minimises `||X - XW||_F^2`, so
`W[i, j] != 0` means `i -> j`.

## Data

`data/nepal_districts.csv` has one row per district of Nepal (77 rows) and the
eight variables used in the structural analysis: `Literacy`, `Poverty`,
`Infrastructure`, `Teacher_Quality`, `Digital_Access`, `Dropout`, `STR`
(student–teacher ratio) and `Urban_Rural_Gap`. `Literacy` is the observed
district value from the 2021 National Population and Housing Census (NPHC
2021). The other variables are microsimulated around their observed provincial
baselines, as described in the paper. The analysis standardises every column
before fitting.

## Requirements

Python 3.9 or later, with numpy, scipy, pandas, matplotlib and pytest (see
`requirements.txt`). Tested with Python 3.12.

The committed results were produced with Python 3.12. Rerunning with other
numpy/scipy versions changes the corrected solver's continuous outputs (FDR,
weight norms) in the second or third decimal place, because L-BFGS-B and BLAS
results differ slightly between builds. The input data are unaffected.
Figures and Tables 1, 2 and 4 are built only from the committed results and
match the paper exactly. Table 3 refits the solver, so with other versions its
weights can differ in the third decimal place; its edges and resample shares
do not change.

## License

MIT; see `LICENSE`.
