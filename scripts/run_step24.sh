#!/bin/bash
# TJU: wait for step22 (hust, matr) to finish, then run strictly sequentially
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4
P=${PYTHON:-python}
while [ ! -f ../logs/step22_done.flag ]; do sleep 30; done
$P -u 07_rank_experiment.py tju ../results/step7 > ../logs/step7_tju.log 2>&1
$P -u 20_temporal.py tju ../results/step20 > ../logs/step20_tju.log 2>&1
$P -u 22_calibration.py tju ../results/step22 > ../logs/step22_tju.log 2>&1
echo done > ../logs/step24_done.flag
