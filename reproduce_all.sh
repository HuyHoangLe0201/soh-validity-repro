#!/bin/bash
# Re-runs every experiment from the raw data: feature extraction, all numbered steps, the analysis scripts, and the manuscript numbers.
# Usage (paths of the unpacked raw data; see DATA.md):
#   NASA_DIR=/data/nasa MATR_DIR=/data/matr HUST_DIR=/data/hust XJTU_DIR="/data/xjtu/Battery Dataset" TJU_DIR=/data/tju PYTHON=python bash reproduce_all.sh
# Heavy steps run strictly one after the other (OpenMP).  A finished step leaves logs/done_<name>; rerunning the script skips finished steps.
# The neural-network steps (26, 28, 29, 31, 36, 38, 39, 40, 45, 48, 49) can run on a GPU machine (DEVICE=cuda; see kaggle/README.md):
# with EXTERNAL_NETWORKS=1 this script skips them and waits at a barrier until logs/kaggle_merged exists (their results copied into results/).
# Expect roughly one to two days on four CPU cores without a GPU.
set -u
ROOT=$(cd "$(dirname "$0")" && pwd); cd "$ROOT/scripts"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
P=${PYTHON:-python}; R=../results; L=../logs
: "${NASA_DIR:?}" "${MATR_DIR:?}" "${HUST_DIR:?}" "${XJTU_DIR:?}" "${TJU_DIR:?}"
mkdir -p $R $L
step() {  # step <name> <command...>
  local name=$1; shift
  if [ -f "$L/done_$name" ]; then return 0; fi
  echo "$(date '+%F %T') start $name" >> $L/driver.log
  if "$@" > "$L/$name.log" 2>&1; then touch "$L/done_$name"; echo "$(date '+%F %T') done  $name" >> $L/driver.log
  else echo "$(date '+%F %T') FAILED $name" >> $L/driver.log; fi
}
# xstep: a step that may run on another machine (kaggle/): skipped here when EXTERNAL_TABULAR=1 (its results are merged later)
xstep() { if [ -n "${EXTERNAL_TABULAR:-}" ]; then return 0; fi; step "$@"; }
# a pilot step that was started by an earlier invocation and is still running is waited for, not started twice
while pgrep -f "04_matr_protocol_pilot.py" > /dev/null; do sleep 30; done
if [ ! -f "$L/done_s4_matr" ] && [ -f $R/step4/matr_protocol_pilot.csv ] && [ -f $R/step4/matr_F1_all5models_protocol_pilot.csv ]; then touch "$L/done_s4_matr"; fi

# ---------------------------------------------------------------- A. raw data -> per-cycle tables
step s1_nasa_lobo   $P 01_reproduce_lobo.py "$NASA_DIR" $R/step1
step s2_nasa_ladder $P 02_protocol_ladder_nasa.py $R/step1/cv_features_dataset_reproduced.csv $R/step2
step s3_nasa_gap    $P 03_purged_gap_nasa.py $R/step1/cv_features_dataset_reproduced.csv $R/step3
step s4a_matr_table $P -c "import sys, os; sys.path.insert(0, '../src'); sys.path.insert(0, '.'); import importlib; m = importlib.import_module('04_matr_protocol_pilot'); from matr_loader import load_matr_summary; d = m.prepare(load_matr_summary('$MATR_DIR')); os.makedirs('$R/step4', exist_ok=True); d.to_csv('$R/step4/matr_summary_clean.csv.gz', index=False)"
step f_hust         $P -c "import sys; sys.path.insert(0, '../src'); from hust_loader import load_hust; load_hust('$HUST_DIR').to_csv('$R/hust_summary.csv.gz', index=False)"
step f_xjtu         $P ../src/xjtu_loader.py "$XJTU_DIR" $R/xjtu_features.csv.gz 2
step f_tju          $P 23_build_tju.py "$TJU_DIR" $R/tju_summary.csv.gz
step w_hust         $P -u ../src/waveform.py hust "$HUST_DIR" $R/hust_waveform.csv.gz
step w_xjtu         $P -u ../src/waveform.py xjtu "$XJTU_DIR" $R/xjtu_waveform.csv.gz
step w_matr         $P -u ../src/waveform.py matr "$MATR_DIR" $R/matr_waveform.csv.gz
[ -f $R/matr_waveform.csv.gz ] && [ -f $R/step4/matr_summary_clean.csv.gz ] && [ -f $R/tju_summary.csv.gz ] && touch $L/tables_ready
# ---------------------------------------------------------------- B. tabular models and their analyses
xstep s6_fast        $P -u 06_fast_ladder_both.py $R/step4/matr_summary_clean.csv.gz $R/hust_summary.csv.gz $R/step6
for ds in matr matr_crate hust xjtu tju; do xstep s7_$ds $P -u 07_rank_experiment.py $ds $R/step7; done
xstep s8  $P 08_rank_analysis.py $R/step7 $R/step8
for ds in xjtu hust matr matr_crate tju; do xstep s9_$ds $P -u 09_tuned_group_inner.py $ds $R/step9; done
xstep s10 $P -u 10_support_analysis.py $R/step7 $R/step9 $R/step10
xstep s11 $P 11_multiplicity.py $R/step7 $R/step11
for ds in xjtu matr_crate matr hust; do xstep s12_$ds $P -u 12_tuning_repeats.py $ds $R/step12 3; done
for ds in matr hust xjtu; do xstep s13_$ds $P -u 13_waveform_experiment.py $ds $R/${ds}_waveform.csv.gz $R/step13; done
xstep s15 $P -u 15_policy_experiment.py $R/step15
xstep s16 $P -u 16_f3_xjtu.py $R/step16
xstep s17 $P -u 17_selection_optimism.py $R/step17
xstep s18 $P -u 18_preprocessing_leakage.py $R/step18
xstep s19 $P -u 19_policy_inference.py $R $R/step19
for ds in xjtu hust matr tju; do xstep s20_$ds $P -u 20_temporal.py $ds $R/step20; done
xstep s21 $P 21_temporal_analysis.py $R/step7 $R/step20 $R/step21
for ds in hust matr xjtu tju; do xstep s22_$ds $P -u 22_calibration.py $ds $R/step22; done
xstep s25 $P 25_calibration_analysis.py $R/step22 $R/step7 $R/step25
xstep s34  $P -u 34_matr124.py $R/step34
xstep s35  $P 35_tju_matr_diagnostics.py $R/step20 $R/step35
xstep s37  $P -u 37_selection_extended.py $R/step37
xstep s41  $P 41_offset_mechanism_selfref.py $R/step41
xstep s42  $P 42_support2_multicell.py $R/step42
for ds in hust matr; do xstep s43_$ds $P -u 43_f2_tuned.py $ds $R/${ds}_waveform.csv.gz $R/step43; done
xstep s46  $P 46_shrinkage_fade.py $R/step20 $R/step46
xstep s51  $P -u 51_tju_f3_support.py $R/step51
xstep s52  $P -u 52_crate_groups_other_datasets.py $R/step52
for ds in xjtu tju hust matr; do xstep s53_$ds $P -u 53_factorial_q3q5.py $ds $R/step53; done
for job in "tju cluster" "tju ctype" "tju temp" "tju rate" "xjtu cluster" "hust cluster" "matr cluster"; do set -- $job; xstep s55_$1_$2 $P -u 55_alt_groupings.py $1 $2 $R/step55; done
xstep s56  $P 56_factorial_analysis.py $R/step53 $R/step56
for ds in xjtu tju hust matr; do xstep s57_$ds $P -u 57_selection_partitions.py $ds $R/step57; done
xstep s58  $P 58_power_groups.py $R/step15 $R/step7 $R/step58
xstep s61  $P -u 61_temporal_proportions.py $R/step7 $R/step61
# ---------------------------------------------------------------- B2. steps that stay on this machine (need the raw data or the merged results)
step s47  $P -u 47_xjtu_batch6_check.py "$XJTU_DIR" $R/step47
xstep s4_matr       $P -u 04_matr_protocol_pilot.py "$MATR_DIR" $R/step4   # pilot (about 1.6 h, not resumable inside; on Kaggle: 04 --table)
if [ -n "${EXTERNAL_TABULAR:-}" ]; then
  echo "$(date '+%F %T') waiting for the tabular results (logs/kaggle_tabular_merged)" >> $L/driver.log
  while [ ! -f $L/kaggle_tabular_merged ]; do sleep 60; done
fi
step s14 $P 14_followup_analysis.py $R $R/step14
step s60  $P 60_group_typology.py $R $R/step60
step s19b $P 19b_policy_tail.py $R $R/step19/policy_tail.csv
# ---------------------------------------------------------------- C. neural networks (GPU machine: see kaggle/README.md)
if [ -n "${EXTERNAL_NETWORKS:-}" ]; then
  echo "$(date '+%F %T') waiting for the network results (logs/kaggle_merged)" >> $L/driver.log
  while [ ! -f $L/kaggle_merged ]; do sleep 60; done
else
  for ds in xjtu tju matr hust; do step s26_$ds $P -u 26_neural.py $ds $R/step26; done
  for ds in xjtu tju matr hust; do step s28_$ds $P -u 28_finetune.py $ds $R/step28; done
  for ds in xjtu tju matr hust; do step s29_$ds $P -u 29_architectures.py $ds $R/step29; done
  for ds in xjtu tju matr hust; do step s31_$ds $P -u 31_tuned_arch.py $ds $R/step31; done
  for ds in xjtu tju matr hust; do step s36_$ds $P -u 36_tuned_more.py $ds $R/step36; done
  step s38  $P -u 38_preproc_networks.py $R/step35/preproc_networks.csv
  for ds in xjtu tju matr hust; do step s39_$ds $P -u 39_more_seeds.py $ds $R/step39; done
  for ds in xjtu tju matr hust; do step s40_$ds $P -u 40_tuned_transformer.py $ds $R/step40; done
  for ds in xjtu tju matr hust; do step s45_$ds $P -u 45_tuned_mlp.py $ds $R/step45; done
  for ds in xjtu tju hust matr; do step s48_$ds $P -u 48_tuned_cnngru_attn.py $ds $R/step48; done
  for ds in xjtu tju hust matr; do step s49_$ds $P -u 49_tuned_seeds.py $ds $R/step49; done
fi
# ---------------------------------------------------------------- D. analyses that need the network results
step s27  $P 27_neural_analysis.py $R/step7 $R/step20 $R/step22 $R/step26 $R/step27
step s30  $P 30_finetune_arch_analysis.py $R/step7 $R/step20 $R/step26 $R/step28 $R/step29 $R/step30
step s32  $P 32_extrapolation_offset.py $R/step7 $R/step20 $R/step26 $R/step29 $R/step32
step s32b $P 32b_offset_decomposition.py $R/step32/offset_decomposition.csv
step s44  $P 44_seed_summary.py $R/step44
step s50  $P 50_tuned_seed_summary.py $R/step50
step s33  $P 33_group_inference.py $R/step7 $R/step20 $R/step26 $R/step29 $R/step31 $R/step33 $R/step36
step s54  $P 54_macro_rmse.py $R/step7 $R/step20 $R/step26 $R/step29 $R/step31 $R/step54 $R/step36
step s59  $P 59_randomization_tests.py $R/step7 $R/step20 $R/step26 $R/step29 $R/step31 $R/step59 $R/step36
# ---------------------------------------------------------------- E. manuscript numbers, tables and figures
cd "$ROOT/paper"
step p_tables  $P make_tables.py
step p_figures $P make_figures3.py
echo "$(date '+%F %T') ALL STEPS FINISHED" >> $L/driver.log
