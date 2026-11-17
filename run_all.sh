#!/bin/sh
# Regenerate every result, figure and table in the paper, then check every
# numeric claim in the manuscript against the regenerated results.
# Takes about ten minutes, most of it the B = 1000 bootstrap in e1.
set -e
PY="${PYTHON:-python}"
cd "$(dirname "$0")/analysis"
$PY e1_inversions.py
$PY e2_instrument.py
$PY e3_collapse.py
$PY e4_recovery.py
$PY e5_robustness.py
$PY e6_threshold.py
$PY make_figures.py
$PY make_tables.py
$PY verify_manuscript.py
$PY -m pytest ../dagcheck/tests -q
echo "done: results in analysis/results/, figures and tables in output/"
