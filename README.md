# Reproducibility package: validity threats in data-driven SOH estimation

Code, intermediate results and manuscript build scripts for *Validity Threats in Data-Driven Lithium-Ion Battery State-of-Health Estimation: A Critical Review with a Controlled Evaluation-Protocol Study* (H. H. Le, K.-A. Nguyen).

## Layout
- `src/`: dataset loaders, feature construction, evaluation harness (`harness.py`, `rankexp.py`), waveform features.
- `scripts/`: numbered experiment and analysis steps (`01`-`52`) and the `run_step*.sh` drivers. Each script states its arguments in its docstring.
- `results/`: CSV/JSON outputs of every step (`results/stepN/`) and per-cycle feature tables. Large `*.npz` prediction files (330 MB) are provided separately as the release asset `soh-validity-predictions.tar.gz`; extract it in the repository root to restore `results/**/*.npz`, which `paper/make_tables.py` needs.
- `paper/`: scripts that turn the results into the tables, figures and numbers of the manuscript (`make_tables.py`, `make_figures3.py`, `build_manuscript.py`), the LaTeX template, `references.bib`, `quotes.csv` (verbatim quotations) and `numbers.json` (all numbers cited in the text).
- `DATA.md`: where to obtain the five datasets.

## Environment
Python 3.13 (CPU only). `pip install -r requirements.txt`. The full set of experiments needs about 4 CPU cores and several days of compute; heavy steps should be run one at a time.

## Reproducing
1. Download the datasets (`DATA.md`).
2. Build the per-cycle tables with the loaders (for example `python scripts/23_build_tju.py <tju_dir> results/tju_summary.csv.gz`).
3. Run the steps in numerical order, or the drivers, for example `PYTHON=python bash scripts/run_step7.sh` (logs go to `logs/`). Step 13 additionally needs `XJTU_DIR` pointing to the unpacked XJTU data.
4. Extract the prediction archive, then regenerate the manuscript numbers and figures without rerunning the experiments, using the results provided:
```bash
cd paper
python make_tables.py
python make_figures3.py
python build_manuscript.py
```
`build_manuscript.py` fills the placeholders of `manuscript_template.tex` from `numbers.json` and writes `manuscript.tex`; compile with pdfLaTeX and BibTeX using the IEEE Access class (`ieeeaccess.cls`, not included).

## Notes
- Random seeds are fixed in each script; networks are averaged over three seeds where stated in the paper. Small numerical differences are possible across library versions.
- `quotes.csv` is verified against the full texts of the cited works, which are not redistributed; `verify_quotes.py` therefore needs those texts in `paper/sources/`.
- Code is released under the MIT licence (see `LICENSE`). Datasets keep their own licences.

## Round 5 (steps 53-61): responses to the external review
- `scripts/53_factorial_q3q5.py`, `56_factorial_analysis.py`: 2x2 factorial separation of (c) late-life coverage of the other cells and (f) early rows of the target cell (corners A-D; c in {0.6, 0.8, 1.0}); the Q3 -> Q5 gap is 98-103 % attributable to coverage for LightGBM, XGBoost and random forest on all four datasets; target rows change the error by at most 0.08 points.
- `scripts/54_macro_rmse.py`: macro-cell / macro-group RMSE; direction of the temporal-versus-held-out result unchanged (cluster-bootstrap intervals exclude 0 on all four datasets), P3/P0 factors change (MATR 3.5 -> 3.0, TJU 3.6 -> 4.1), 113 of 1204 model pairs (9 %) change order.
- `scripts/55_alt_groupings.py`, `60_group_typology.py`: alternative held-out groupings (TJU cell type / temperature / charge rate; input-space clusters for all four datasets) and the typology table; error ratio to P2 against distance ratio (Spearman 0.80 LightGBM, 0.90 ridge). HUST identifier groups have distance ratio 1.0 (cell-disjoint, not a condition shift).
- `src/inference.py`, `scripts/58_power_groups.py`, `59_randomization_tests.py`: the percentile cluster bootstrap rejects a true null in 10-16 % of draws with 6-11 groups (27 % with 3) and up to 8.5 % with 55-135 cells; p-values now come from an exact paired sign-flip randomization test (smallest p = 2^(1-U), so at least 6 units; 8 for six comparisons; 12 for 91). Steps 11, 19, 33 were re-run with it (bootstrap p-values kept in `results/_boot_inference_backup`): 1113 tests / 527 Holm-significant (previously 600), 64 supported sign changes (previously 100), 8 in the primary family of six pairs among the four main models; tabular step 11: 4 of 40 (previously 9 of 59). One global family: 473 Holm-significant, 53 sign changes.
- `scripts/57_selection_partitions.py`: selection on the reported folds over repeated random assignments of whole groups to outer folds (MATR 70 policies, HUST 10 groups, XJTU 6 batches, TJU 11 groups).
- `scripts/61_temporal_proportions.py`: temporal result for training shares 40-80 %; PT > P3 for LightGBM in 20 of 20 dataset-share combinations.
