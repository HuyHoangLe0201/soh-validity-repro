"""Check that every quoted string appears verbatim (after whitespace/dash/quote/hyphenation normalisation) in its source text."""
import csv, re, sys
from pathlib import Path
D = Path(__file__).resolve().parent
def norm(s):
    s = s.replace("­", "").replace("‐", "-").replace("‑", "-").replace("–", "-").replace("—", "-")
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').replace("ﬁ", "fi").replace("ﬂ", "fl")
    s = re.sub(r"-\s*\n\s*", "-", s)   # line-break hyphenation: keep hyphen (safe for compounds), checked both ways below
    return re.sub(r"\s+", " ", s).strip().lower()
def norm2(s):  # variant joining hyphen-split words
    s = s.replace("­", "")
    s = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", s)
    return norm(s)
bad = 0
for r in csv.DictReader(open(D / "quotes.csv")):
    raw = (D / "sources" / r["source"]).read_text(errors="ignore")
    q = norm(r["quote"])
    ok = q in norm(raw) or q in norm2(raw)
    print(("OK   " if ok else "FAIL ") + r["id"], r["source"])
    bad += (not ok)
sys.exit(1 if bad else 0)
