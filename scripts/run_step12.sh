#!/bin/bash
cd "$(dirname "$0")"
for ds in xjtu matr_crate matr hust; do
  ${PYTHON:-python} -u 12_tuning_repeats.py $ds ../results/step12 3 > ../logs/step12_$ds.log 2>&1
done
echo done > ../logs/step12_done.flag
