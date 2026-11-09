#!/bin/bash
cd "$(dirname "$0")"
P=${PYTHON:-python}
for job in "tju cluster" "tju ctype" "tju temp" "tju rate" "xjtu cluster" "hust cluster" "matr cluster"; do
  set -- $job
  $P -u 55_alt_groupings.py $1 $2 ../results/step55 > ../logs/step55_$1_$2.log 2>&1
done
