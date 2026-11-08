#!/bin/bash
# strictly sequential, one OpenMP-heavy process at a time
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4
P=${PYTHON:-python}
$P -u 13_waveform_experiment.py xjtu ../results/xjtu_waveform.csv.gz ../results/step13 > ../logs/step13_xjtu.log 2>&1
for ds in xjtu matr_crate matr hust; do
  $P -u 12_tuning_repeats.py $ds ../results/step12 3 > ../logs/step12_$ds.log 2>&1
done
echo done > ../logs/final_seq_done.flag
