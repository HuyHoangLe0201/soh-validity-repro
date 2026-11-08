#!/bin/bash
# Step 28 (fine-tuning schedules) then step 29 (architectures, window, seeds); strictly sequential
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4
P=${PYTHON:-python}
for ds in xjtu tju matr hust; do $P -u 28_finetune.py $ds ../results/step28 > ../logs/step28_$ds.log 2>&1; done
for ds in xjtu tju matr hust; do $P -u 29_architectures.py $ds ../results/step29 > ../logs/step29_$ds.log 2>&1; done
echo done > ../logs/step29_done.flag
