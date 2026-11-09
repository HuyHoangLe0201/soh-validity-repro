#!/bin/bash
cd "$(dirname "$0")"
P=${PYTHON:-python}
for ds in xjtu tju hust matr; do
  $P -u 53_factorial_q3q5.py $ds ../results/step53 > ../logs/step53_$ds.log 2>&1
done
