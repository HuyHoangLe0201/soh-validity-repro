#!/bin/bash
cd "$(dirname "$0")"
for ds in matr matr_crate hust; do
  ${PYTHON:-python} -u 07_rank_experiment.py $ds ../results/step7 > ../logs/step7_$ds.log 2>&1
done
echo ALL_DONE > ../logs/step7_done.flag
