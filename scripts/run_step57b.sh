#!/bin/bash
cd "$(dirname "$0")"
P=${PYTHON:-python}
for ds in hust matr; do
  $P -u 57_selection_partitions.py $ds ../results/step57 >> ../logs/step57_$ds.log 2>&1
done
