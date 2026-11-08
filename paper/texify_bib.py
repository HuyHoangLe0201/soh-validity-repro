"""Make references.bib pure ASCII (BibTeX/IEEEtran is not UTF-8 aware)."""
M = {"í": r"{\'i}", "ñ": r"{\~n}", "‑": "-", "–": "--", "É": r"{\'E}", "æ": r"{\ae}", "é": r"{\'e}", "ë": r'{\"e}', "ö": r'{\"o}', "ü": r'{\"u}', "‐": "-"}
t = open("references.bib", encoding="utf-8").read()
for k, v in M.items():
    t = t.replace(k, v)
assert all(ord(c) < 128 for c in t), sorted({c for c in t if ord(c) > 127})
open("references.bib", "w", encoding="ascii").write(t)
print("references.bib is ASCII")

# scikit-learn: cite the JMLR paper rather than the arXiv copy (Crossref returned the arXiv record)
import re
t = open("references.bib", encoding="ascii").read()
new = """@article{pedregosa2011,
  author = {Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V. and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P. and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, {\\'E}.},
  title = {Scikit-learn: Machine Learning in {P}ython},
  journal = {Journal of Machine Learning Research},
  volume = {12},
  pages = {2825--2830},
  year = {2011}
}
"""
t2 = re.sub(r"@article\{pedregosa2011,.*?\n\}\n", lambda m: new, t, flags=re.S)
if t2 != t:
    open("references.bib", "w", encoding="ascii").write(t2); print("pedregosa2011 -> JMLR")
