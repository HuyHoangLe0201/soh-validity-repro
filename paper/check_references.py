"""Compare every entry of references.bib with Crossref (DOI), arXiv (identifier) and, for the rest, a Crossref title search.
Writes reference_check.json; entries without issues have an empty list. Needs network access (public APIs, no key).
Known benign flags: subtitle missing in the Crossref title field (chen2016, kaufman2012), Crossref lacks volume (le2026),
names with non-ASCII letters (zhao2026), arXiv metadata with affiliation superscripts (aud13, corrected from the PDF), Art. no. kept in note."""
import re, json, sys, time, unicodedata, urllib.parse, urllib.request, difflib
from pathlib import Path
HERE = Path(__file__).resolve().parent
BIB = HERE / 'references.bib'
OUT = HERE
UA = {"User-Agent": "ref-verify/1.0 (bibliography check)"}
def get(url, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read()
        except Exception as e:
            err = e
            time.sleep(2 + 3 * i)
    return None
def parse(text):
    ents = {}
    for m in re.finditer(r'@(\w+)\{([^,]+),', text):
        i = m.end(); depth = 1; j = i
        while depth and j < len(text):
            c = text[j]
            if c == '{': depth += 1
            elif c == '}': depth -= 1
            j += 1
        body = text[i:j - 1]
        f = {}
        for fm in re.finditer(r'(\w+)\s*=\s*', body):
            k = fm.group(1).lower(); s = fm.end()
            if s >= len(body): continue
            if body[s] == '{':
                d = 1; e = s + 1
                while d and e < len(body):
                    if body[e] == '{': d += 1
                    elif body[e] == '}': d -= 1
                    e += 1
                f[k] = body[s + 1:e - 1]
            elif body[s] == '"':
                e = body.index('"', s + 1); f[k] = body[s + 1:e]
            else:
                e = re.search(r'[,\n]', body[s:]).start() + s; f[k] = body[s:e].strip()
        ents[m.group(2)] = dict(type=m.group(1).lower(), **f)
    return ents
def strip_tex(s):
    s = s or ''
    s = re.sub(r"\\['`^\"~=.uvHckrb]\{?(\w)\}?", r'\1', s)
    s = s.replace('\\&', '&').replace('\\_', '_').replace('--', '-')
    s = re.sub(r'\\[a-zA-Z]+', '', s)
    s = s.replace('{', '').replace('}', '').replace('$', '')
    return s
def norm(s):
    s = strip_tex(s)
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(c for c in s if not unicodedata.combining(c))
    return re.sub(r'[^a-z0-9]+', ' ', s.lower()).strip()
def bib_authors(a):
    out = []
    for p in re.split(r'\s+and\s+', a or ''):
        p = p.strip()
        if p.lower() in ('others', 'et al.'): out.append(('et al', '')); continue
        if ',' in p: fam, giv = [x.strip() for x in p.split(',', 1)]
        else:
            parts = p.split(); fam = parts[-1]; giv = ' '.join(parts[:-1])
        out.append((norm(fam), norm(giv)))
    return out
def sim(a, b): return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()
def cr_work(doi):
    b = get('https://api.crossref.org/works/' + urllib.parse.quote(doi))
    return json.loads(b)['message'] if b else None
def cr_search(title, author=None):
    q = 'https://api.crossref.org/works?rows=5&select=title,author,container-title,issued,DOI,volume,issue,page,type&query.bibliographic=' + urllib.parse.quote(strip_tex(title))
    b = get(q)
    return json.loads(b)['message']['items'] if b else []
def arx(aid):
    b = get('http://export.arxiv.org/api/query?id_list=' + aid)
    if not b: return None
    from xml.etree import ElementTree as ET
    ns = {'a': 'http://www.w3.org/2005/Atom', 'x': 'http://arxiv.org/schemas/atom'}
    e = ET.fromstring(b).find('a:entry', ns)
    if e is None: return None
    return dict(title=re.sub(r'\s+', ' ', e.find('a:title', ns).text).strip(),
                authors=[x.find('a:name', ns).text for x in e.findall('a:author', ns)],
                published=e.find('a:published', ns).text[:10], updated=e.find('a:updated', ns).text[:10],
                doi=(e.find('x:doi', ns).text if e.find('x:doi', ns) is not None else None),
                jref=(e.find('x:journal_ref', ns).text if e.find('x:journal_ref', ns) is not None else None))
def compare_cr(key, e, m):
    issues = []
    t = m['title'][0] if m.get('title') else ''
    ts = sim(e.get('title', ''), t)
    if ts < 0.97: issues.append(f'TITLE sim={ts:.2f}: bib="{strip_tex(e.get("title",""))}" | crossref="{t}"')
    ba = bib_authors(e.get('author', '')); ca = [(norm(a.get('family', a.get('name', ''))), norm(a.get('given', ''))) for a in m.get('author', [])]
    bl = [x for x in ba if x[0] != 'et al']
    if len(bl) != len(ca) and not any(x[0] == 'et al' for x in ba):
        issues.append(f'AUTHOR COUNT bib={len(bl)} crossref={len(ca)}')
    for i, (b, c) in enumerate(zip(bl, ca)):
        if b[0] != c[0]:
            issues.append(f'AUTHOR#{i+1} family bib="{b[0]}" crossref="{c[0]}"')
        elif b[1] and c[1] and b[1][0] != c[1][0]:
            issues.append(f'AUTHOR#{i+1} given bib="{b[1]}" crossref="{c[1]}"')
    dp = (m.get('issued', {}).get('date-parts') or [[None]])[0][0]
    pp = (m.get('published-print', {}).get('date-parts') or [[None]])[0][0]
    po = (m.get('published-online', {}).get('date-parts') or [[None]])[0][0]
    by = e.get('year')
    if by and str(by) not in {str(dp), str(pp), str(po)}: issues.append(f'YEAR bib={by} crossref issued={dp} print={pp} online={po}')
    cont = (m.get('container-title') or [''])[0]
    bj = e.get('journal') or e.get('booktitle') or ''
    if bj and cont and sim(bj, cont) < 0.6 and norm(cont) not in norm(bj) and norm(bj) not in norm(cont):
        issues.append(f'VENUE bib="{strip_tex(bj)}" crossref="{cont}"')
    if e.get('volume') and m.get('volume') and str(e['volume']) != str(m['volume']): issues.append(f'VOLUME bib={e["volume"]} crossref={m["volume"]}')
    if e.get('volume') and not m.get('volume'): issues.append(f'VOLUME bib={e["volume"]} crossref=(none)')
    if e.get('number') and m.get('issue') and str(e['number']) != str(m['issue']): issues.append(f'ISSUE bib={e["number"]} crossref={m["issue"]}')
    pg = (m.get('page') or m.get('article-number') or '')
    bp = (e.get('pages') or '').replace('--', '-').replace('–', '-')
    if bp and pg and bp.replace(' ', '') != str(pg).replace(' ', '').replace('–', '-'):
        if not (e.get('number') and str(e.get('number')) == str(m.get('article-number', '')) ): issues.append(f'PAGES bib={bp} crossref={pg}')
    if e.get('number') and e.get('type') == 'article' and not e.get('pages') and m.get('article-number') and str(e['number']) != str(m['article-number']): issues.append(f'ARTICLE-NO bib={e["number"]} crossref={m["article-number"]}')
    return issues
def main():
    ents = parse(BIB.read_text())
    res = {}
    for k, e in ents.items():
        r = dict(key=k, type=e['type'], title=strip_tex(e.get('title', '')), doi=e.get('doi'), issues=[], status='')
        doi = e.get('doi')
        url = e.get('url', '') + ' ' + e.get('journal', '') + ' ' + e.get('howpublished', '')
        am = re.search(r'arxiv\.org/abs/(\d{4}\.\d{4,5})', url) or re.search(r'arXiv:(\d{4}\.\d{4,5})', url)
        if doi:
            m = cr_work(doi)
            if not m: r['status'] = 'DOI NOT FOUND'; r['issues'].append('Crossref returned nothing for ' + doi)
            else:
                r['issues'] = compare_cr(k, e, m); r['status'] = 'crossref'
        elif am:
            a = arx(am.group(1)); time.sleep(3)
            if not a: r['status'] = 'ARXIV NOT FOUND'; r['issues'].append('arXiv API returned nothing for ' + am.group(1))
            else:
                r['status'] = 'arxiv'
                if sim(e.get('title', ''), a['title']) < 0.95: r['issues'].append(f'TITLE sim={sim(e.get("title",""), a["title"]):.2f}: bib="{strip_tex(e.get("title",""))}" | arxiv="{a["title"]}"')
                ba = [x for x in bib_authors(e.get('author', '')) if x[0] != 'et al']
                aa = [(norm(n.split()[-1]), norm(' '.join(n.split()[:-1]))) for n in a['authors']]
                if len(ba) != len(aa) and not any(x[0]=='et al' for x in bib_authors(e.get('author',''))): r['issues'].append(f'AUTHOR COUNT bib={len(ba)} arxiv={len(aa)}')
                for i, (b, c) in enumerate(zip(ba, aa)):
                    if b[0] != c[0]: r['issues'].append(f'AUTHOR#{i+1} family bib="{b[0]}" arxiv="{c[0]}"')
                yrs = {a['published'][:4], a['updated'][:4]}
                if e.get('year') and str(e['year']) not in yrs: r['issues'].append(f'YEAR bib={e["year"]} arxiv first={a["published"]} updated={a["updated"]}')
                if a['doi'] or a['jref']: r['issues'].append(f'NOTE arXiv lists published version: doi={a["doi"]} journal_ref={a["jref"]}')
        else:
            r['status'] = 'manual'
            items = cr_search(e.get('title', ''))
            for m in items[:3]:
                if sim(e.get('title', ''), m['title'][0]) > 0.9:
                    r['issues'].append(f'NOTE Crossref has a matching record: {m["DOI"]} | {m.get("container-title",[""])[0]} {m.get("volume","")} {m.get("page","")} {m["issued"]["date-parts"][0][0]}')
        res[k] = r
        print(k, r['status'], len(r['issues']), flush=True)
        time.sleep(0.4)
    (OUT / 'reference_check.json').write_text(json.dumps(res, indent=1, ensure_ascii=False))
main()
