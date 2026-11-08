#!/bin/bash
cd "$(dirname "$0")"
while [ ! -f ../logs/step40_done.flag ]; do sleep 60; done
export OMP_NUM_THREADS=4 TORCH_THREADS=4
P=${PYTHON:-python}
$P -u 43_f2_tuned.py hust ../results/hust_waveform.csv.gz ../results/step43 > ../logs/step43_hust.log 2>&1
$P -u 43_f2_tuned.py matr ../results/matr_waveform.csv.gz ../results/step43 > ../logs/step43_matr.log 2>&1
echo done > ../logs/step43_done.flag
