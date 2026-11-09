#!/bin/bash
cd "$(dirname "$0")"
P=${PYTHON:-python}
# extractor chain: xjtu after hust finished
( until [ -f ../results/hust_waveform.csv.gz ]; do sleep 10; done
  cd ../src && $P -u waveform.py xjtu "/tmp/xjtu_x/Battery Dataset" ../results/xjtu_waveform.csv.gz > ../logs/wave_xjtu.log 2>&1 ) &
# experiments
$P -u 13_waveform_experiment.py matr ../results/matr_waveform.csv.gz ../results/step13 > ../logs/step13_matr.log 2>&1
until [ -f ../results/hust_waveform.csv.gz ]; do sleep 10; done
$P -u 13_waveform_experiment.py hust ../results/hust_waveform.csv.gz ../results/step13 > ../logs/step13_hust.log 2>&1
until [ -f ../results/xjtu_waveform.csv.gz ]; do sleep 10; done
$P -u 13_waveform_experiment.py xjtu ../results/xjtu_waveform.csv.gz ../results/step13 > ../logs/step13_xjtu.log 2>&1
wait
echo done > ../logs/step13_done.flag
