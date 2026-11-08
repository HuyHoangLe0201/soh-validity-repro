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
