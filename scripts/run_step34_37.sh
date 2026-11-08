#!/bin/bash
# Steps 34, 09 (TJU), 36 and 37, strictly sequential (run alone: concurrent jobs oversubscribe the CPU)
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4
P=${PYTHON:-python}
$P -u 34_matr124.py ../results/step34 > ../logs/step34.log 2>&1
$P -u 09_tuned_group_inner.py tju ../results/step9 > ../logs/step9_tju.log 2>&1
for ds in xjtu tju matr hust; do $P -u 36_tuned_more.py $ds ../results/step36 > ../logs/step36_$ds.log 2>&1; done
$P -u 37_selection_extended.py ../results/step37 > ../logs/step37.log 2>&1
echo done > ../logs/step37_done.flag
