import json, re, time, urllib.request, urllib.parse, html, xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
exec(open('/home/claude/refs/entries.py').read())
UA = {"User-Agent": "CF-SCAFFOLD-thesis-bibliography-check/1.0 (academic; python urllib)"}
def get(url, timeout=30, raw=False):
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
                return r.status, (data if raw else data.decode("utf-8", "replace")), r.geturl()
        except urllib.error.HTTPError as e:
            if e.code in (429, 503): time.sleep(3*(attempt+1)); continue
            return e.code, "", url
        except Exception as e:
            time.sleep(2); last = str(e)
    return -1, "", url
def norm(s): return re.sub(r"[^a-z0-9]+", " ", html.unescape(s or "").lower()).strip()
def head_resolve(doi):
    # do NOT follow: we only need proof the handle system resolves the DOI
    class NoRedir(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **k): return None
    op = urllib.request.build_opener(NoRedir)
    try:
        r = op.open(urllib.request.Request("https://doi.org/"+urllib.parse.quote(doi, safe="/()"), headers=UA, method="HEAD"), timeout=20)
        return r.status, r.headers.get("Location")
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Location")
    except Exception as e:
        return -1, None
def openalex(doi):
    c, t, _ = get("https://api.openalex.org/works/doi:"+urllib.parse.quote(doi, safe="/()"))
    if c != 200: return None
    j = json.loads(t); oa = j.get("open_access") or {}
    loc = (j.get("best_oa_location") or {})
    return dict(is_oa=oa.get("is_oa"), oa_status=oa.get("oa_status"), oa_url=oa.get("oa_url") or loc.get("pdf_url") or loc.get("landing_page_url"), cited_by=j.get("cited_by_count"))
def check(e):
    r = dict(key=e["key"], kind=e["kind"], id=e["id"], ok=False, notes=[])
    try:
        if e["kind"] == "doi":
            c, t, _ = get("https://api.crossref.org/works/"+urllib.parse.quote(e["id"], safe="/()"))
            meta = None
            if c == 200:
                m = json.loads(t)["message"]; r["registry"] = "Crossref"
                yr = None
                for f in ("published-print","published-online","issued","published"):
                    if m.get(f) and m[f].get("date-parts") and m[f]["date-parts"][0][0]: yr = m[f]["date-parts"][0][0]; break
                meta = dict(title=" ".join(m.get("title") or [""]), container=" ".join(m.get("container-title") or [""]),
                    year=yr, volume=m.get("volume"), issue=m.get("issue"), page=m.get("page"), article=m.get("article-number"),
                    authors=[(a.get("family") or a.get("name") or "", a.get("given","")) for a in m.get("author",[])],
                    type=m.get("type"), publisher=m.get("publisher"), isbn=m.get("ISBN"))
            else:
                c2, t2, _ = get("https://api.datacite.org/dois/"+urllib.parse.quote(e["id"].lower(), safe="/()"))
                if c2 == 200:
                    a = json.loads(t2)["data"]["attributes"]; r["registry"] = "DataCite"
                    meta = dict(title=(a.get("titles") or [{}])[0].get("title",""), container=a.get("publisher",""),
                        year=a.get("publicationYear"), volume=None, issue=None, page=None, article=None,
                        authors=[(x.get("familyName") or x.get("name",""), x.get("givenName","")) for x in a.get("creators",[])], type=a.get("types",{}).get("resourceTypeGeneral"), publisher=a.get("publisher"))
            if meta is None:
                r["notes"].append(f"DOI NOT FOUND in Crossref ({c}) or DataCite"); return r
            r["meta"] = meta
            hc, loc = head_resolve(e["id"]); r["resolver"] = hc; r["landing"] = loc
            tm = e["expect"] == "" or norm(e["expect"]) in norm(meta["title"])
            ym = meta["year"] is None or abs(int(meta["year"]) - e["year"]) <= 1
            if not tm: r["notes"].append(f"TITLE MISMATCH: registry title = {meta['title']!r}")
            if not ym: r["notes"].append(f"YEAR MISMATCH: registry {meta['year']} vs expected {e['year']}")
            if hc not in (301, 302, 303, 307, 308): r["notes"].append(f"doi.org resolver returned {hc}")
            r["oa"] = openalex(e["id"])
            r["ok"] = tm and ym and hc in (301,302,303,307,308)
        elif e["kind"] == "arxiv":
            c, t, _ = get("https://export.arxiv.org/api/query?id_list="+e["id"])
            ns = {"a":"http://www.w3.org/2005/Atom","x":"http://arxiv.org/schemas/atom"}
            root = ET.fromstring(t); ent = root.find("a:entry", ns)
            title = " ".join((ent.findtext("a:title", "", ns) or "").split())
            if not title or "error" in (ent.findtext("a:id","",ns) or "").lower():
                r["notes"].append("arXiv id not found"); return r
            pub = ent.findtext("a:published","",ns)[:4]
            jref = ent.findtext("x:journal_ref","",ns); adoi = ent.findtext("x:doi","",ns)
            authors = [" ".join(a.findtext("a:name","",ns).split()) for a in ent.findall("a:author",ns)]
            r["registry"]="arXiv"; r["meta"]=dict(title=title, year=int(pub), authors=authors, journal_ref=jref, doi=adoi)
            c3,_,_ = get("https://arxiv.org/abs/"+e["id"]); r["resolver"]=c3
            tm = e["expect"] == "" or norm(e["expect"]) in norm(title)
            if not tm: r["notes"].append(f"TITLE MISMATCH: arXiv title = {title!r}")
            if c3 != 200: r["notes"].append(f"abs page returned {c3}")
            r["ok"] = tm and c3 == 200
        elif e["kind"] == "url":
            c, t, final = get(e["id"])
            r["registry"]="direct URL"; r["resolver"]=c; r["landing"]=final
            tm = norm(e["expect"]) in norm(t)
            ttl = re.search(r"<title[^>]*>(.*?)</title>", t, re.S|re.I)
            r["meta"]=dict(title=" ".join(html.unescape(ttl.group(1)).split()) if ttl else "")
            if c != 200: r["notes"].append(f"HTTP {c}")
            if not tm: r["notes"].append("expected title text not found on page")
            r["ok"] = c == 200 and tm
        elif e["kind"] == "isbn":
            c, t, _ = get("https://openlibrary.org/isbn/"+e["id"]+".json")
            title = ""
            if c == 200:
                j = json.loads(t); title = j.get("title","") + " " + (j.get("subtitle") or ""); r["registry"]="Open Library"
                r["meta"]=dict(title=title.strip(), publishers=j.get("publishers"), publish_date=j.get("publish_date"), publish_places=j.get("publish_places"))
            else:
                c2, t2, _ = get("https://www.googleapis.com/books/v1/volumes?q=isbn:"+e["id"])
                if c2 == 200 and json.loads(t2).get("totalItems",0):
                    v = json.loads(t2)["items"][0]["volumeInfo"]; title=v.get("title",""); r["registry"]="Google Books"
                    r["meta"]=dict(title=title, publishers=[v.get("publisher")], publish_date=v.get("publishedDate"))
            r["resolver"]=c
            tm = bool(title) and norm(e["expect"]) in norm(title)
            if not title: r["notes"].append("ISBN not found in Open Library or Google Books")
            elif not tm: r["notes"].append(f"TITLE MISMATCH: {title!r}")
            r["ok"]=tm
        elif e["kind"] == "jstor":
            c, t, _ = get("https://api.openalex.org/works?filter=title.search:"+urllib.parse.quote("sequentially rejective multiple test procedure")+"&per-page=5")
            j = json.loads(t) if c==200 else {"results":[]}
            hit = [w for w in j["results"] if "holm" in json.dumps(w.get("authorships","")).lower()]
            r["registry"]="OpenAlex (JSTOR-hosted, no DOI)"
            if hit:
                w=hit[0]; b=w.get("biblio",{})
                r["meta"]=dict(title=w.get("title"), year=w.get("publication_year"), container=(w.get("primary_location") or {}).get("source",{}).get("display_name"), volume=b.get("volume"), issue=b.get("issue"), page=f"{b.get('first_page')}-{b.get('last_page')}")
            cj,_,_ = get("https://www.jstor.org/stable/"+e["id"]); r["resolver"]=cj
            r["ok"]=bool(hit)
            if cj!=200: r["notes"].append(f"jstor.org returned {cj} to automated request (normal: JSTOR blocks bots); record confirmed via OpenAlex")
    except Exception as ex:
        r["notes"].append("EXCEPTION "+repr(ex))
    return r
if __name__ == "__main__":
    with ThreadPoolExecutor(6) as ex: res = list(ex.map(check, E))
    json.dump(res, open("/home/claude/refs/verified.json","w"), indent=1, default=str)
    bad = [x for x in res if not x["ok"]]
    print(f"verified OK: {len(res)-len(bad)}/{len(res)}")
    for x in bad: print(" FAIL", x["key"], x["id"], "|", "; ".join(x["notes"]))
