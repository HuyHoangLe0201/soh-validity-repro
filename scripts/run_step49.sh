#!/bin/bash
# Step 49 (seeds 1 and 2 for the tuned variants); run alone
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 TORCH_THREADS=4
P=${PYTHON:-python}
for ds in xjtu tju hust matr; do $P -u 49_tuned_seeds.py $ds ../results/step49 > ../logs/step49_$ds.log 2>&1; done
echo done > ../logs/step49_done.flag
