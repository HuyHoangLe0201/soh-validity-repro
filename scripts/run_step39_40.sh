#!/bin/bash
# Steps 39 (more seeds) and 40 (tuned transformer); torch uses 3 threads so that light analyses (1 thread) can run alongside
cd "$(dirname "$0")"
export OMP_NUM_THREADS=3 TORCH_THREADS=3
P=${PYTHON:-python}
for ds in xjtu tju matr hust; do $P -u 39_more_seeds.py $ds ../results/step39 > ../logs/step39_$ds.log 2>&1; done
for ds in xjtu tju matr; do $P -u 40_tuned_transformer.py $ds ../results/step40 > ../logs/step40_$ds.log 2>&1; done
echo done > ../logs/step40_done.flag
