"""Fill @@key@@ placeholders from numbers.json, check citations and tables, write manuscript.tex."""
import json, re, sys
from pathlib import Path
D = Path(__file__).resolve().parent
NUM = json.loads((D / "numbers.json").read_text())
t = (D / "manuscript_template.tex").read_text()
missing = sorted({k for k in re.findall(r"@@(\w+)@@", t) if k not in NUM})
if missing: sys.exit(f"missing numbers: {missing}")
t = re.sub(r"@@(\w+)@@", lambda m: re.sub(r"^-(?=\d)", r"\\mbox{$-$}", str(NUM[m.group(1)])), t)
(D / "manuscript.tex").write_text(t)
bib = set(re.findall(r"^@\w+\{([^,]+),", (D / "references.bib").read_text(), re.M))
cites = {c.strip() for grp in re.findall(r"\\cite\{([^}]*)\}", t) for c in grp.split(",")}
print("cited:", len(cites), "| unknown keys:", sorted(cites - bib), "| in bib but uncited:", sorted(bib - cites))
for f in re.findall(r"\\input\{([^}#]*)\}", t): assert (D / f).exists(), f
for f in re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]*)\}", t): assert (D / f).exists(), f
abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", t, re.S).group(1)
print("abstract words:", len(re.sub(r"\\\w+|[{}$]", " ", abstract).split()))
print("body words (approx):", len(re.sub(r"\\\w+\{[^}]*\}|\\\w+|[{}$~]", " ", t.split("\\maketitle")[1].split("\\bibliographystyle")[0]).split()))
