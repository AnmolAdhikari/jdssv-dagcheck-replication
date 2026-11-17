# dagcheck

Exploratory diagnostics for continuous structure-learning solvers such as
NOTEARS. `dagcheck` does not learn causal structure. It wraps a solver you
already have and checks whether that solver, in your configuration and on data
like yours, works at all.

```python
from dagcheck import assert_solver_valid
assert_solver_valid(my_solver, tau=0.08, n=77, d=8)
```

See the repository README for the full replication package.
