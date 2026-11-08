#!/bin/bash
cd "$(dirname "$0")"
until [ -f ../results/matr_waveform.csv.gz ] && ! pgrep -f "waveform.py matr" > /dev/null; do sleep 10; done
${PYTHON:-python} -u 13_waveform_experiment.py matr ../results/matr_waveform.csv.gz ../results/step13 > ../logs/step13_matr.log 2>&1
echo done > ../logs/step13_matr_done.flag
