"""Kaggle entry point: runs one chain of the neural-network steps of the reproduction on the GPU, step by step.
Usage (script kernel): environment variable CHAIN = SMOKE | A | B.  The dataset that holds this folder is mounted read-only and is copied to
/kaggle/working/soh first.  A finished step leaves logs/done_<name>; the results of the chain are packed into /kaggle/working/results_chain_<CHAIN>.tar.gz.
"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

_cands = sorted(Path("/kaggle/input").rglob("run_chain.py"))
print("run_chain.py candidates:", [str(p) for p in _cands], flush=True)
SRC = _cands[0].parent
print("SRC", SRC, "contents", sorted(p.name for p in SRC.iterdir()), flush=True)
W = Path("/kaggle/working/soh")
if not W.exists():
    shutil.copytree(SRC, W)
import gzip
for p in sorted((W / "results").rglob("*.csv")):   # Kaggle unpacks .gz uploads: pack the tables again under the names the scripts expect
    with open(p, "rb") as fi, gzip.open(str(p) + ".gz", "wb") as fo:
        shutil.copyfileobj(fi, fo)
    p.unlink()
print("results:", sorted(str(p.relative_to(W)) for p in (W / "results").rglob("*") if p.is_file()), flush=True)
(W / "logs").mkdir(exist_ok=True)
os.chdir(W / "scripts")
env = dict(os.environ, DEVICE="cuda" if os.environ.get("CHAIN", "SMOKE") in ("A", "A2", "B", "SMOKE") else "cpu", OMP_NUM_THREADS="4", TORCH_THREADS="4", PYTHONUNBUFFERED="1")
R = "../results"
DS4 = ["xjtu", "tju", "matr", "hust"]
CHAINS = {
    "A": [(f"s26_{d}", ["26_neural.py", d, f"{R}/step26"]) for d in DS4]
         + [(f"s28_{d}", ["28_finetune.py", d, f"{R}/step28"]) for d in DS4]
         + [(f"s29_{d}", ["29_architectures.py", d, f"{R}/step29"]) for d in DS4]
         + [("s38", ["38_preproc_networks.py", f"{R}/step35/preproc_networks.csv"])]
         + [(f"s39_{d}", ["39_more_seeds.py", d, f"{R}/step39"]) for d in DS4],
    "B": [(f"s31_{d}", ["31_tuned_arch.py", d, f"{R}/step31"]) for d in DS4]
         + [(f"s36_{d}", ["36_tuned_more.py", d, f"{R}/step36"]) for d in DS4]
         + [(f"s40_{d}", ["40_tuned_transformer.py", d, f"{R}/step40"]) for d in DS4]
         + [(f"s45_{d}", ["45_tuned_mlp.py", d, f"{R}/step45"]) for d in DS4]
         + [(f"s48_{d}", ["48_tuned_cnngru_attn.py", d, f"{R}/step48"]) for d in ("xjtu", "tju", "hust", "matr")]
         + [(f"s49_{d}", ["49_tuned_seeds.py", d, f"{R}/step49"]) for d in ("xjtu", "tju", "hust", "matr")],
}
def _sp(name, script, *args):
    return (name, [script, *args])


_S = lambda n: f"{R}/step{n}"   # noqa: E731
_ds = lambda tag, dss, script, outdir, *extra: [(f"{tag}_{d}", [script, d, *extra, outdir]) for d in dss]  # noqa: E731
CHAINS["C1"] = (
    [("s6", ["06_fast_ladder_both.py", f"{R}/step4/matr_summary_clean.csv.gz", f"{R}/hust_summary.csv.gz", _S(6)])]
    + _ds("s7", ["matr", "matr_crate", "hust", "xjtu", "tju"], "07_rank_experiment.py", _S(7))
    + [("s8", ["08_rank_analysis.py", _S(7), _S(8)])]
    + _ds("s9", ["xjtu", "hust", "matr", "matr_crate", "tju"], "09_tuned_group_inner.py", _S(9))
    + [("s10", ["10_support_analysis.py", _S(7), _S(9), _S(10)]), ("s11", ["11_multiplicity.py", _S(7), _S(11)]),
       ("s15", ["15_policy_experiment.py", _S(15)]), ("s17", ["17_selection_optimism.py", _S(17)]),
       ("s19", ["19_policy_inference.py", R, _S(19)]), ("s37", ["37_selection_extended.py", _S(37)]),
       ("s42", ["42_support2_multicell.py", _S(42)]), ("s58", ["58_power_groups.py", _S(15), _S(7), _S(58)]),
       ("s61", ["61_temporal_proportions.py", _S(7), _S(61)]), ("s51", ["51_tju_f3_support.py", _S(51)])]
    + _ds("s20", ["xjtu", "hust", "matr", "tju"], "20_temporal.py", _S(20))
    + [("s21", ["21_temporal_analysis.py", _S(7), _S(20), _S(21)])]
    + _ds("s22", ["hust", "matr", "xjtu", "tju"], "22_calibration.py", _S(22))
    + [("s25", ["25_calibration_analysis.py", _S(22), _S(7), _S(25)]), ("s35", ["35_tju_matr_diagnostics.py", _S(20), _S(35)]),
       ("s46", ["46_shrinkage_fade.py", _S(20), _S(46)])])
CHAINS["C2"] = (
    [(f"s12_{d}", ["12_tuning_repeats.py", d, _S(12), "3"]) for d in ("xjtu", "matr_crate", "matr", "hust")]
    + [(f"s13_{d}", ["13_waveform_experiment.py", d, f"{R}/{d}_waveform.csv.gz", _S(13)]) for d in ("matr", "hust", "xjtu")]
    + [("s16", ["16_f3_xjtu.py", _S(16)]), ("s18", ["18_preprocessing_leakage.py", _S(18)]), ("s34", ["34_matr124.py", _S(34)]),
       ("s41", ["41_offset_mechanism_selfref.py", _S(41)]), ("s52", ["52_crate_groups_other_datasets.py", _S(52)])]
    + [(f"s43_{d}", ["43_f2_tuned.py", d, f"{R}/{d}_waveform.csv.gz", _S(43)]) for d in ("hust", "matr")])
CHAINS["C3"] = (
    _ds("s53", ["xjtu", "tju", "hust", "matr"], "53_factorial_q3q5.py", _S(53))
    + [(f"s55_{a}_{b}", ["55_alt_groupings.py", a, b, _S(55)]) for a, b in
       (("tju", "cluster"), ("tju", "ctype"), ("tju", "temp"), ("tju", "rate"), ("xjtu", "cluster"), ("hust", "cluster"), ("matr", "cluster"))]
    + [("s56", ["56_factorial_analysis.py", _S(53), _S(56)])]
    + _ds("s57", ["xjtu", "tju", "hust", "matr"], "57_selection_partitions.py", _S(57)))
CHAINS["C4"] = [("s4_pilot", ["04_matr_protocol_pilot.py", "-", f"{R}/step4", "--table", f"{R}/step4/matr_summary_clean.csv.gz"])]
CHAINS["A2"] = [("s38", ["38_preproc_networks.py", f"{R}/step35/preproc_networks.csv"])]
chain = os.environ.get("CHAIN", "SMOKE")
(Path(W) / "logs").mkdir(exist_ok=True)
log = open(W / "logs" / f"chain_{chain}.log", "a", buffering=1)


def say(s):
    print(s, flush=True)
    log.write(s + "\n")


if chain == "SMOKE":
    import torch
    say(f"cuda={torch.cuda.is_available()} device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None} "
        f"torch={torch.__version__} cpus={os.cpu_count()}")
    subprocess.run(["nvidia-smi"], check=False)
    code = r'''
import sys, time, importlib, numpy as np
sys.path.insert(0, "../src"); sys.path.insert(0, ".")
m26 = importlib.import_module("26_neural"); m7 = importlib.import_module("07_rank_experiment")
d, f = m7.load("tju"); R = m26.Runner(d, f); m26.STRIDE = 4; m26.EPOCHS = 3
n = len(R.d); tr = np.arange(0, n)[: int(0.6 * n)]; te = np.arange(int(0.6 * n), n)
t = time.time(); p = R.train_predict("GRU", tr, te, seed=0); a = time.time() - t
t = time.time(); p2 = R.train_predict("TRF", tr, te, seed=0); b = time.time() - t
print("DEVICE", m26.DEV, "rows", n, "GRU 3 epochs s", round(a, 1), "TRF 3 epochs s", round(b, 1),
      "rmse GRU", round(float(np.sqrt(np.mean((p - R.y0[te]) ** 2))), 4), flush=True)
'''
    for dev in ("cuda", "cpu"):
        r = subprocess.run([sys.executable, "-c", code], env=dict(env, DEVICE=dev), capture_output=True, text=True)
        say(f"[{dev}] " + r.stdout.strip() + (" ERR " + r.stderr.strip()[-800:] if r.returncode else ""))
    sys.exit(0)

t0 = time.time()
for name, args in CHAINS[chain]:
    flag = W / "logs" / f"done_{name}"
    if flag.exists():
        continue
    for arg in args:   # output directories (or the parent of an output file) must exist
        if str(arg).startswith(R):
            pa = Path(arg)
            (pa if pa.suffix == "" else pa.parent).mkdir(parents=True, exist_ok=True)
    t = time.time()
    say(f"{time.strftime('%F %T')} start {name}")
    with open(W / "logs" / f"{name}.log", "w") as fh:
        r = subprocess.run([sys.executable, "-u", *args], env=env, stdout=fh, stderr=subprocess.STDOUT)
    say(f"{time.strftime('%F %T')} {'done ' if r.returncode == 0 else 'FAILED'} {name} {time.time() - t:.0f}s")
    if r.returncode == 0:
        flag.touch()
say(f"chain {chain} finished in {(time.time() - t0) / 3600:.2f} h")
subprocess.run(["tar", "czf", f"/kaggle/working/results_chain_{chain}.tar.gz", "-C", str(W), "results", "logs"], check=False)
shutil.rmtree(W, ignore_errors=True)
