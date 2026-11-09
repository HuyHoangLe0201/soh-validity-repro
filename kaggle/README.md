# Running the reproduction on Kaggle

`run_chain.py` is the entry point of a Kaggle script kernel. Upload `src/`, `scripts/`, `run_chain.py` and the per-cycle tables (`results/*.csv.gz`, `results/step4/matr_summary_clean.csv.gz`) as one dataset, attach it to a kernel and set the chain with `os.environ["CHAIN"] = ...` at the top of the script:

| chain | machine | steps |
|---|---|---|
| A, A2, B | GPU (2x T4) | neural networks: 26, 28, 29, 38, 39 / 31, 36, 40, 45, 48, 49 (`DEVICE=cuda`) |
| C1, C2, C3, C4 | CPU (4 cores) | tabular steps (see the `CHAINS` dictionary) and the MATR pilot (`04_matr_protocol_pilot.py --table`) |

The results of a chain are packed into `/kaggle/working/results_chain_<CHAIN>.tar.gz`; extract it into the repository root. `reproduce_all.sh` with `EXTERNAL_NETWORKS=1 EXTERNAL_TABULAR=1` skips these steps and waits for `logs/kaggle_merged` and `logs/kaggle_tabular_merged`.
