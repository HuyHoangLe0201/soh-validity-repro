"""Fetch and print bibliographic metadata for candidate references from Crossref / arXiv (verification, not memory)."""
import json, re, sys, time, urllib.parse, urllib.request
from defusedxml import ElementTree as ET
def get(url):
    for i in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "ref-verify (mailto:none@example.org)"}), timeout=60) as r: return r.read()
        except Exception as e: time.sleep(3)
    return None
DOIS = {
 "severson": "10.1038/s41560-019-0356-8", "ma_hust": "10.1039/D2EE01676A", "wang_xjtu": "10.1038/s41467-024-48779-z",
 "preger": "10.1149/1945-7111/abae37", "sohail": "10.3390/math14111879", "pozzato": "10.1016/j.joule.2023.07.018",
 "le": "10.17531/ein/220211", "kaufman": "10.1145/2382577.2382579", "kapoor": "10.1016/j.patter.2023.100804",
 "roberts": "10.1111/ecog.02881", "varma": "10.1186/1471-2105-7-91", "bh": "10.1111/j.2517-6161.1995.tb02031.x",
 "fieldwelsh": "10.1111/j.1467-9868.2007.00593.x", "xgboost": "10.1145/2939672.2939785", "bergmeir": "10.1016/j.ins.2011.12.028",
}
for k, d in DOIS.items():
    b = get("https://api.crossref.org/works/" + urllib.parse.quote(d))
    if not b: print(k, "NOT FOUND", d); continue
    m = json.loads(b)["message"]
    au = "; ".join(f"{a.get('family','')}, {a.get('given','')}" for a in m.get("author", [])[:4]) + (" ..." if len(m.get("author", [])) > 4 else "")
    yr = (m.get("issued", {}).get("date-parts") or [[None]])[0][0]
    print(f"{k}: {m['title'][0][:110]} | {au} | {(m.get('container-title') or [''])[0]} {m.get('volume','')}({m.get('issue','')}) {m.get('page', m.get('article-number',''))} | {yr} | {d}")
print("---- arXiv")
for a in ["2310.14714", "2502.18807", "2603.21592", "2608.16212", "2607.23482"]:
    root = ET.fromstring(get(f"http://export.arxiv.org/api/query?id_list={a}"))
    e = root.find("{http://www.w3.org/2005/Atom}entry")
    t = re.sub(r"\s+", " ", e.find("{http://www.w3.org/2005/Atom}title").text)
    au = ", ".join(x.find("{http://www.w3.org/2005/Atom}name").text for x in e.findall("{http://www.w3.org/2005/Atom}author")[:5])
    print(a, "|", t[:110], "|", au, "|", e.find("{http://www.w3.org/2005/Atom}published").text[:10])
    time.sleep(3)
print("---- title searches")
for q in ["transfer learning for battery health estimation comprehensive meta-analysis Ionics",
          "state of health estimation laboratory to real-world application data-driven vehicles Energies 2026",
          "trustworthiness transferability state of health remaining useful life estimation review uncertainty Batteries 2026",
          "benchmarking data-driven methods real-world electric vehicle battery state of health IEEE Access Oji"]:
    b = get("https://api.crossref.org/works?rows=3&select=title,author,container-title,issued,DOI,volume,page&query.bibliographic=" + urllib.parse.quote(q))
    for m in json.loads(b)["message"]["items"]:
        print(" Q:", q[:40], "->", m["title"][0][:100], "|", (m.get("container-title") or [""])[0], (m.get("issued", {}).get("date-parts") or [[None]])[0][0], m["DOI"])
