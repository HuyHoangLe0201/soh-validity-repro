#!/bin/bash
# Step 48 (tuned CNN-GRU and attention GRU); run alone
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 TORCH_THREADS=4
P=${PYTHON:-python}
for ds in xjtu tju hust matr; do $P -u 48_tuned_cnngru_attn.py $ds ../results/step48 > ../logs/step48_$ds.log 2>&1; done
echo done > ../logs/step48_done.flag
