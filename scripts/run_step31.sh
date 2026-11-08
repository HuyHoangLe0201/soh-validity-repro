#!/bin/bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4
P=${PYTHON:-python}
for ds in xjtu tju matr hust; do $P -u 31_tuned_arch.py $ds ../results/step31 > ../logs/step31_$ds.log 2>&1; done
echo done > ../logs/step31_done.flag
