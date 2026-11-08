"""Step 23 - build per-cycle TJU table: python 23_build_tju.py <dir_with_TJU_csv_folders> results/tju_summary.csv.gz"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from tju_loader import load_tju  # noqa: E402

d = load_tju(sys.argv[1])
d.to_csv(sys.argv[2], index=False)
print(d.groupby("ctype").cell.nunique(), len(d), d.cell.nunique())
print(d.groupby(["ctype", "cond"]).cell.nunique())
