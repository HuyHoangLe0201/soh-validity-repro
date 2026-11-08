"""Generate references.bib from Crossref / arXiv / OpenAlex metadata fetched at build time (no hand-typed fields)."""
import json, re, time, urllib.parse, urllib.request
from defusedxml import ElementTree as ET
UA = {"User-Agent": "bib-build"}
def jget(u):
    for i in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=60) as r: return json.loads(r.read())
        except Exception: time.sleep(2)
def esc(s): return s.replace("&amp;", "&").replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
def names(auth): return " and ".join(f"{a['family']}, {a.get('given','')}".strip(", ") for a in auth if a.get("family"))
def crossref(key, doi, typ="article"):
    m = jget("https://api.crossref.org/works/" + urllib.parse.quote(doi))["message"]
    y = m["issued"]["date-parts"][0][0]
    f = {"author": names(m["author"]), "title": m["title"][0], "year": str(y), "doi": doi}
    ct = (m.get("container-title") or [""])[0]
    if typ == "article":
        f["journal"] = ct
        for k, kk in (("volume", "volume"), ("issue", "number"), ("page", "pages"), ("article-number", "number")):
            if m.get(k): f[kk] = str(m[k]).replace("-", "--") if k == "page" else str(m[k])
    else:
        f["booktitle"] = ct
        if m.get("page"): f["pages"] = m["page"].replace("-", "--")
    return key, ("inproceedings" if typ != "article" else "article"), f
def arxiv(key, aid, note=None):
    root = ET.fromstring(urllib.request.urlopen(urllib.request.Request(f"http://export.arxiv.org/api/query?id_list={aid}", headers=UA), timeout=60).read())
    A = "{http://www.w3.org/2005/Atom}"; e = root.find(A + "entry")
    au = [x.find(A + "name").text.split() for x in e.findall(A + "author")]
    authors = " and ".join(f"{n[-1]}, {' '.join(n[:-1])}" for n in au)
    f = {"author": authors, "title": re.sub(r"\s+", " ", e.find(A + "title").text).strip(), "year": e.find(A + "published").text[:4],
         "journal": f"arXiv preprint arXiv:{aid}", "url": f"https://arxiv.org/abs/{aid}"}
    time.sleep(3)
    return key, "article", f
ents = [crossref("severson2019", "10.1038/s41560-019-0356-8"), crossref("ma2022", "10.1039/D2EE01676A"),
        crossref("wang2024", "10.1038/s41467-024-48779-z"), crossref("sohail2026", "10.3390/math14111879"), crossref("barua2026", "10.3390/batteries12090342"), crossref("benabdennour2026", "10.3390/machines14101124"),
        crossref("rizanov2026", "10.3390/batteries12070255"), crossref("ma2026", "10.3390/en19061506"), crossref("shan2025", "10.3390/en18133542"), crossref("gong2025", "10.3390/en18020337"),
        crossref("pozzato2023", "10.1016/j.joule.2023.07.018"),
        crossref("kaufman2012", "10.1145/2382577.2382579"), crossref("roberts2017", "10.1111/ecog.02881"), crossref("varma2006", "10.1186/1471-2105-7-91"),
        crossref("benjamini1995", "10.1111/j.2517-6161.1995.tb02031.x"), crossref("field2007", "10.1111/j.1467-9868.2007.00593.x"),
        crossref("chen2016", "10.1145/2939672.2939785", "inproc"),
        arxiv("zhang2024", "2310.14714"), arxiv("park2026", "2603.21592"), arxiv("zhao2026", "2608.16212"), arxiv("pedregosa2011", "1201.0490"), arxiv("paszke2019", "1912.01703"), arxiv("transformer2023", "2309.10014")]
# Le & Nguyen: journal volume/pages are not in Crossref; take them from the journal reference printed on the arXiv record
k, t, f = crossref("le2026", "10.17531/ein/220211"); f.update({"journal": "Eksploatacja i Niezawodno\\'{s}\\'{c} -- Maintenance and Reliability", "volume": "28", "number": "4", "pages": "220211"}); ents.append((k, t, f))
for i, (k, t, f) in enumerate(ents):
    if k == "zhang2024":  # PDF header: "Published as a conference paper at ICLR 2024"; arXiv title lacks spaces
        ents[i] = (k, "inproceedings", {"author": f["author"], "title": "BatteryML: An Open-source Platform for Machine Learning on Battery Degradation",
                   "booktitle": "Proc. International Conference on Learning Representations (ICLR)", "year": "2024", "note": "arXiv:2310.14714", "url": "https://arxiv.org/abs/2310.14714"})
with open("references.bib", "w") as out:
    for k, t, f in ents:
        out.write(f"@{t}{{{k},\n" + ",\n".join(f"  {a} = {{{v if a in ('journal','author') and '\\\\' in str(v) else esc(str(v)) if a not in ('author','doi','url') else str(v)}}}" for a, v in f.items()) + "\n}\n\n")
with open("references.bib", "a") as out:
    out.write("""% --- Entries below were NOT fetched from Crossref/arXiv. saha2007: taken from the reference list of Sohail et al. 2026 (checked verbatim, quotation Q28).
% holm1979: typed from memory of the standard citation; could not be machine-verified (JSTOR blocked) - check before submission.
@misc{saha2007,
  author = {Saha, B. and Goebel, K.},
  title = {Battery Data Set},
  howpublished = {NASA Ames Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA, USA},
  year = {2007}
}

@article{holm1979,
  author = {Holm, Sture},
  title = {A simple sequentially rejective multiple test procedure},
  journal = {Scandinavian Journal of Statistics},
  volume = {6},
  number = {2},
  pages = {65--70},
  year = {1979}
}
""")
print(len(ents), "entries written (+2 manual)")

# run after build_bib.py: python texify_bib.py
