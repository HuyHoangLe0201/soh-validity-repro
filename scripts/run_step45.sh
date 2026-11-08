#!/bin/bash
# Step 40 on HUST (tuned transformer) and step 45 (tuned pooled MLP); run alone
cd "$(dirname "$0")"
export OMP_NUM_THREADS=3 TORCH_THREADS=3
P=${PYTHON:-python}
for ds in xjtu tju matr hust; do $P -u 45_tuned_mlp.py $ds ../results/step45 > ../logs/step45_$ds.log 2>&1; done
$P -u 40_tuned_transformer.py hust ../results/step40 > ../logs/step40_hust.log 2>&1
echo done > ../logs/step45_done.flag
