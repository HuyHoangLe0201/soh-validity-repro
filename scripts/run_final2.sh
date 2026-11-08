#!/bin/bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4
P=${PYTHON:-python}
$P -u 18_preprocessing_leakage.py ../results/step18 > ../logs/step18.log 2>&1
$P -u 16_f3_xjtu.py ../results/step16 > ../logs/step16.log 2>&1
$P -u 17_selection_optimism.py ../results/step17 > ../logs/step17.log 2>&1
$P -u 15_policy_experiment.py ../results/step15 > ../logs/step15.log 2>&1
echo done > ../logs/final2_done.flag
