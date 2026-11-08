#!/bin/bash
cd "$(dirname "$0")"
for ds in xjtu hust matr matr_crate; do
  ${PYTHON:-python} -u 09_tuned_group_inner.py $ds ../results/step9 > ../logs/step9_$ds.log 2>&1
done
echo done > ../logs/step9_done.flag
