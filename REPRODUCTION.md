# Re-run from the raw data (9 November 2026)

Every step was re-run from the raw datasets with `reproduce_all.sh` (features, steps 1-61, manuscript numbers). The neural-network steps ran on Kaggle GPUs (2x Tesla T4, 0.8-1.4 h per chain), the tabular steps on Kaggle CPU chains (0.8-4.8 h) and the feature extraction on a local machine. Logs: `reproduction/`; comparison with the stored results: `reproduction/reproduction_check.csv`, `reproduction/numbers_differences.csv`.

## What was reproduced exactly
- the six per-cycle tables (HUST, XJTU, TJU features; HUST, XJTU, MATR charge-curve features) and the cleaned MATR table: identical (maximum absolute difference 0);
- NASA steps 1-3: identical to 1e-15; the 30 base rows of the MATR pilot: pooled RMSE within 0.045;
- LightGBM / XGBoost / ridge pooled errors of the main experiments to the digits printed in the paper (for example the P3/P0 factors 3.5, 1.6, 3.4, 3.6), with small differences in the experiments that use random forests, the pooled MLP or a hyperparameter search (steps 9, 12, 37, 57).

## What differs, and why
- Neural networks: the stored runs seeded the order of mini-batches and the dropout masks but built the networks before `torch.manual_seed`, so the initial weights came from an unseeded generator and the stored predictions cannot be regenerated bit for bit. The code now seeds the initialization (`26_neural.py`, `_reinit`; two processes give identical predictions). The re-run (GPU) gave pooled RMSE of the GRU within 0.03 points of the stored values (median 0.002) and of the transformer within 0.87 (median 0.07; largest: TJU, temporal split, 4.22 stored against 4.93); the temporal error exceeds the held-out-group error for both networks on the same datasets as before.
- Numbers of the manuscript: 3584 entries; 2494 identical, 527 within 0.01 or 2 percent, 563 different (mostly network, tuning and group-held-out selection results). Counts that depend on the networks changed moderately: supported changes of model order between questions among the four main models (primary family) 12 instead of 8 (HUST 7, XJTU 5), 73 instead of 64 over all 14 models, 532 instead of 527 of 1113 tests significant after Holm correction; the single-family counts 469 / 52 instead of 473 / 53.
- Tabular steps on another machine (library versions, thread order): of the 313 prediction arrays of step 7, 167 are identical (differences below 1e-4); the others are mostly random forests (71), the pooled MLP (29), tuned models (55) and 12 default LightGBM arrays; for them the median per-row difference is at most 0.36 SOH points. Selection statistics that depend on the hyperparameter search vary by several percentage points (for example the maximum optimism of XGBoost on MATR in step 57: 5 and 21 percent).

The manuscript reports the stored results; `make_tables.py` regenerates all 3584 numbers from the stored predictions (release archive `soh-validity-predictions.tar.gz`).

## Defects found and fixed during the re-run
- `step19/policy_tail.csv` was produced by an unsaved snippet; `scripts/19b_policy_tail.py` now writes it (identical to 1e-16).
- `33_group_inference.py` did not create its output directory; `38` (output directory `step35`) likewise (Kaggle chain A2).
