#!/bin/bash
# Step 20/21: temporal versus random within-cell splits (strictly sequential)
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4
P=${PYTHON:-python}
for ds in xjtu hust matr; do $P -u 20_temporal.py $ds ../results/step20 > ../logs/step20_$ds.log 2>&1; done
$P 21_temporal_analysis.py ../results/step7 ../results/step20 ../results/step21 > ../logs/step21.log 2>&1
