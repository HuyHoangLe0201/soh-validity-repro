#!/bin/bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4
P=${PYTHON:-python}
for ds in xjtu tju matr hust; do $P -u 26_neural.py $ds ../results/step26 > ../logs/step26_$ds.log 2>&1; done
echo done > ../logs/step26_done.flag
