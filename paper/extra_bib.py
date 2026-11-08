"""Append the entries for the studies read in the critical reading (keys aud##, ## = id in audit/sample_60.csv) to references.bib.
Metadata come from Crossref / arXiv at build time (same helper functions as build_bib.py). Run: python extra_bib.py && python texify_bib.py"""
src = open("build_bib.py").read().split("ents = [")[0]
exec(src)
AX = {18: "2604.20438", 20: "2304.05084", 21: "2111.15348", 22: "2606.16434", 24: "2507.18320", 25: "2304.08502", 26: "2411.00233",
      28: "2308.13993", 29: "2407.16036", 30: "2607.18329", 1: "2403.05430", 13: "2509.10496", 17: "2503.23858",
      4: "2309.14125", 6: "2409.14575", 9: "2109.13448", 10: "2304.13666", 12: "1903.07672", 19: "2511.12053", 27: "2401.05474"}
DO = {31: "10.1016/j.isci.2022.104260", 32: "10.1038/s41598-025-34022-2", 36: "10.3390/s22239435", 37: "10.1038/s41598-025-93775-y",
      38: "10.1038/s41598-026-39986-3", 42: "10.3390/s26082492", 43: "10.1021/acsomega.5c03257", 44: "10.1038/s41598-024-69424-1",
      47: "10.1038/s41598-026-58379-0", 49: "10.1038/s41598-022-16692-4", 53: "10.1038/s41598-026-48906-4", 54: "10.1002/advs.202503583",
      57: "10.1021/acsomega.6c01145", 40: "10.3390/s22155536", 41: "10.3390/s25072275", 46: "10.1371/journal.pone.0354706", 52: "10.1038/s41598-025-12771-4", 60: "10.1038/s41598-025-33725-w", 45: "10.1038/s41598-020-66424-9"}
ents = [arxiv(f"aud{n:02d}", a) for n, a in AX.items()] + [crossref(f"aud{n:02d}", d) for n, d in DO.items()]
with open("references.bib", "a", encoding="utf-8") as out:
    out.write("\n% --- studies read for the critical reading (audit sample)\n")
    for k, t, f in ents:
        out.write(f"@{t}{{{k},\n" + ",\n".join(f"  {a} = {{{v if a in ('journal','author') and '\\\\' in str(v) else esc(str(v)) if a not in ('author','doi','url') else str(v)}}}" for a, v in f.items()) + "\n}\n\n")
print(len(ents), "entries appended")
