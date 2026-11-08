#!/bin/bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4
P=${PYTHON:-python}
for ds in hust matr; do $P -u 22_calibration.py $ds ../results/step22 > ../logs/step22_$ds.log 2>&1; done
echo done > ../logs/step22_done.flag
