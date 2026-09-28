import json, html, re, unicodedata
exec(open("/home/claude/refs/entries.py").read())
R  = {r["key"]: r for r in json.load(open("/home/claude/refs/verified.json"))}
UM = json.load(open("/home/claude/refs/url_meta.json"))
PC = json.load(open("/home/claude/refs/proceedings_check.json"))
OR = json.load(open("/home/claude/refs/openreview_check.json"))
CX = json.load(open("/home/claude/refs/crossref_extra.json"))
def U(s): return html.unescape(s or "").replace("\n"," ").strip()

# ---------- authors ----------
def init(given):
    out=[]
    for part in given.replace("."," ").split():
        out.append("-".join(p[0]+"." for p in part.split("-") if p))
    return " ".join(out)
def fmt_person(fam, giv): return (init(giv)+" "+fam).strip() if giv else fam
def split_full(name):
    if "," in name: fam, giv = [x.strip() for x in name.split(",",1)]; return fam, giv
    toks=name.split(); return toks[-1], " ".join(toks[:-1])
def ieee_authors(people):
    ppl=[fmt_person(f,g) for f,g in people]
    if len(ppl) > 6: return ppl[0]+" et al."
    if len(ppl) == 1: return ppl[0]
    if len(ppl) == 2: return ppl[0]+" and "+ppl[1]
    return ", ".join(ppl[:-1])+", and "+ppl[-1]
def bib_authors(people):
    if len(people) > 6: people = [people[0], ("others","")]   # IEEE: >6 authors -> first author et al.
    return " and ".join("others" if f=="others" else (f"{f}, {g}" if g else "{"+f+"}") for f,g in people)

AUTH_OVERRIDE = {
 "mcmahan2017fedavg":[("McMahan","H. Brendan"),("Moore","Eider"),("Ramage","Daniel"),("Hampson","Seth"),("Agüera y Arcas","Blaise")],
 "kairouz2021":[("Kairouz","Peter"),("McMahan","H. Brendan"),("Avent","Brendan"),("Bellet","Aurélien"),("Bennis","Mehdi"),("Bhagoji","Arjun Nitin"),("others","")],
 "nickparvar2021kaggle":[("Nickparvar","Masoud")],
 "sanchez2022fractal":[("Sánchez","Jacksson"),("Martín-Landrove","Miguel")],
 "serra1982":[("Serra","Jean")], "gonzalez2018dip":[("Gonzalez","Rafael C."),("Woods","Richard E.")],
 "mandelbrot1982":[("Mandelbrot","Benoit B.")], "cohen1988power":[("Cohen","Jacob")],
 "holm1979":[("Holm","Sture")],
 "li2010drlse":[("Li","Chunming"),("Xu","Chenyang"),("Gui","Changfeng"),("Fox","Martin D.")],
 "liu2001midsagittal":[("Liu","Yanxi"),("Collins","Robert T."),("Rothfus","William E.")],
 "deng2009imagenet":[("Deng","Jia"),("Dong","Wei"),("Socher","Richard"),("Li","Li-Jia"),("Li","Kai"),("Fei-Fei","Li")],
 "sattler2021cfl":[("Sattler","Felix"),("Müller","Klaus-Robert"),("Samek","Wojciech")],
}
def people_of(k):
    if k in AUTH_OVERRIDE: return AUTH_OVERRIDE[k]
    r=R[k]; m=r.get("meta",{})
    if r["kind"]=="doi": return [(U(f),U(g)) for f,g in m["authors"]]
    if r["kind"]=="arxiv": return [split_full(a) for a in m["authors"]]
    if k in UM: return [split_full(U(a)) for a in UM[k]["authors"]]
    return []

# ---------- titles ----------
TITLE_OVERRIDE = {
 "li2021fedrs":"FedRS: Federated learning with restricted softmax for label distribution non-IID data",
 "dieva2015fractalsII":"Fractals in the neurosciences, Part II: Clinical applications and future perspectives",
 "jaccard1912":"The distribution of the flora in the alpine zone",
 "cheng2017figshare":"Brain tumor dataset",
 "holm1979":"A simple sequentially rejective multiple test procedure",
 "serra1982":"Image Analysis and Mathematical Morphology","gonzalez2018dip":"Digital Image Processing",
 "mandelbrot1982":"The Fractal Geometry of Nature","cohen1988power":"Statistical Power Analysis for the Behavioral Sciences",
}
def title_of(k):
    if k in TITLE_OVERRIDE: return TITLE_OVERRIDE[k]
    r=R[k]; t=U(r["meta"].get("title") or (UM.get(k) or {}).get("title") or "")
    t=re.sub(r"\s+"," ",t).rstrip(".")
    return t

# ---------- venues for non-journal items ----------
V = {  # (venue string for IEEE, bibtex booktitle, bibtype)
 "li2022silos":("in Proc. IEEE 38th Int. Conf. Data Eng. (ICDE)","2022 IEEE 38th International Conference on Data Engineering (ICDE)","inproceedings"),
 "ronneberger2015unet":("in Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015 (Lecture Notes in Computer Science, vol. 9351)","Medical Image Computing and Computer-Assisted Intervention -- MICCAI 2015, LNCS 9351","incollection"),
 "brodersen2010balanced":("in Proc. 20th Int. Conf. Pattern Recognit. (ICPR)","2010 20th International Conference on Pattern Recognition (ICPR)","inproceedings"),
 "ismael2018statistical":("in Proc. IEEE Int. Conf. Electro/Information Technology (EIT)","2018 IEEE International Conference on Electro/Information Technology (EIT)","inproceedings"),
 "paul2017deep":("in Proc. SPIE Medical Imaging 2017: Biomedical Applications in Molecular, Structural, and Functional Imaging, vol. 10137","Proc. SPIE 10137, Medical Imaging 2017: Biomedical Applications in Molecular, Structural, and Functional Imaging","inproceedings"),
 "afshar2019capsule":("in Proc. IEEE Int. Conf. Acoust., Speech Signal Process. (ICASSP)","ICASSP 2019 - IEEE International Conference on Acoustics, Speech and Signal Processing","inproceedings"),
 "li2021moon":("in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)","2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)","inproceedings"),
 "li2021fedrs":("in Proc. 27th ACM SIGKDD Conf. Knowl. Discovery Data Mining (KDD)","Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery \\& Data Mining","inproceedings"),
 "lu2023labelmask":("in Proc. 31st ACM Int. Conf. Multimedia (MM)","Proceedings of the 31st ACM International Conference on Multimedia","inproceedings"),
 "wang2024fednlr":("in Proc. 30th ACM SIGKDD Conf. Knowl. Discovery Data Mining (KDD)","Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining","inproceedings"),
 "sheller2019brats":("in Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries (BrainLes 2018) (Lecture Notes in Computer Science, vol. 11383)","Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries, LNCS 11383","incollection"),
 "wu2018groupnorm":("in Computer Vision – ECCV 2018 (Lecture Notes in Computer Science, vol. 11217)","Computer Vision -- ECCV 2018, LNCS 11217","incollection"),
 "cui2019classbalanced":("in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)","2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)","inproceedings"),
 "he2016resnet":("in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)","2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)","inproceedings"),
 "deng2009imagenet":("in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)","2009 IEEE Conference on Computer Vision and Pattern Recognition","inproceedings"),
 "kornblith2019transfer":("in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)","2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)","inproceedings"),
 "guo2025fedvls":("in Proc. AAAI Conf. Artif. Intell.","Proceedings of the AAAI Conference on Artificial Intelligence","inproceedings"),
}
PAGE_FIX = {"wilcoxon1945":"80–83","efron1979bootstrap":"1–26","ismael2018statistical":"252–257"}
ART_FIX  = {"lakens2013effect":"863","paul2017deep":"1013710","guan2024fedmed":"110424","kapoor2023leakage":"100804","deepak2019transfer":"103345","badza2020cnn":"1999","diazpernas2021multiscale":"153","vanderwalt2014skimage":"e453","cheng2015enhanced":"e0140381","cheng2016retrieval":"e0157112"}
# URL / arXiv items: (venue_ieee, bib_booktitle_or_journal, bibtype, volume, pages, year, note)
MAN = {
 "pedregosa2011sklearn":("J. Mach. Learn. Res.","Journal of Machine Learning Research","article","12","2825–2830",2011),
 "cawley2010overfit":("J. Mach. Learn. Res.","Journal of Machine Learning Research","article","11","2079–2107",2010),
 "demsar2006":("J. Mach. Learn. Res.","Journal of Machine Learning Research","article","7","1–30",2006),
 "paszke2019pytorch":("in Adv. Neural Inf. Process. Syst. (NeurIPS)","Advances in Neural Information Processing Systems","inproceedings","32",None,2019),
 "wang2020fednova":("in Adv. Neural Inf. Process. Syst. (NeurIPS)","Advances in Neural Information Processing Systems","inproceedings","33","7611–7623",2020),
 "luo2021ccvr":("in Adv. Neural Inf. Process. Syst. (NeurIPS)","Advances in Neural Information Processing Systems","inproceedings","34","5972–5984",2021),
 "yosinski2014transferable":("in Adv. Neural Inf. Process. Syst. (NeurIPS)","Advances in Neural Information Processing Systems","inproceedings","27",None,2014),
 "raghu2019transfusion":("in Adv. Neural Inf. Process. Syst. (NeurIPS)","Advances in Neural Information Processing Systems","inproceedings","32",None,2019),
 "khosla2020supcon":("in Adv. Neural Inf. Process. Syst. (NeurIPS)","Advances in Neural Information Processing Systems","inproceedings","33","18661–18673",2020),
 "chen2022pflbench":("in Adv. Neural Inf. Process. Syst. (NeurIPS), Datasets and Benchmarks Track","Advances in Neural Information Processing Systems (Datasets and Benchmarks Track)","inproceedings","35","9344–9360",2022),
 "mcmahan2017fedavg":("in Proc. 20th Int. Conf. Artif. Intell. Statist. (AISTATS), PMLR","Proceedings of the 20th International Conference on Artificial Intelligence and Statistics, PMLR","inproceedings","54","1273–1282",2017),
 "karimireddy2020scaffold":("in Proc. 37th Int. Conf. Mach. Learn. (ICML), PMLR","Proceedings of the 37th International Conference on Machine Learning, PMLR","inproceedings","119","5132–5143",2020),
 "hsieh2020quagmire":("in Proc. 37th Int. Conf. Mach. Learn. (ICML), PMLR","Proceedings of the 37th International Conference on Machine Learning, PMLR","inproceedings","119","4387–4398",2020),
 "chen2020simclr":("in Proc. 37th Int. Conf. Mach. Learn. (ICML), PMLR","Proceedings of the 37th International Conference on Machine Learning, PMLR","inproceedings","119","1597–1607",2020),
 "zhang2022fedlc":("in Proc. 39th Int. Conf. Mach. Learn. (ICML), PMLR","Proceedings of the 39th International Conference on Machine Learning, PMLR","inproceedings","162","26311–26329",2022),
 "li2021ditto":("in Proc. 38th Int. Conf. Mach. Learn. (ICML), PMLR","Proceedings of the 38th International Conference on Machine Learning, PMLR","inproceedings","139","6357–6368",2021),
 "collins2021fedrep":("in Proc. 38th Int. Conf. Mach. Learn. (ICML), PMLR","Proceedings of the 38th International Conference on Machine Learning, PMLR","inproceedings","139","2089–2099",2021),
 "ioffe2015bn":("in Proc. 32nd Int. Conf. Mach. Learn. (ICML), PMLR","Proceedings of the 32nd International Conference on Machine Learning, PMLR","inproceedings","37","448–456",2015),
 "pascanu2013clip":("in Proc. 30th Int. Conf. Mach. Learn. (ICML), PMLR","Proceedings of the 30th International Conference on Machine Learning, PMLR","inproceedings","28","1310–1318",2013),
 "li2020fedprox":("in Proc. Mach. Learn. Syst. (MLSys)","Proceedings of Machine Learning and Systems","inproceedings","2","429–450",2020),
 "bouthillier2021variance":("in Proc. Mach. Learn. Syst. (MLSys)","Proceedings of Machine Learning and Systems","inproceedings","3","747–769",2021),
 "acar2021feddyn":("in Proc. Int. Conf. Learn. Represent. (ICLR)","International Conference on Learning Representations (ICLR)","inproceedings",None,None,2021),
 "reddi2021adaptive":("in Proc. Int. Conf. Learn. Represent. (ICLR)","International Conference on Learning Representations (ICLR)","inproceedings",None,None,2021),
 "menon2021logit":("in Proc. Int. Conf. Learn. Represent. (ICLR)","International Conference on Learning Representations (ICLR)","inproceedings",None,None,2021),
 "li2021fedbn":("in Proc. Int. Conf. Learn. Represent. (ICLR)","International Conference on Learning Representations (ICLR)","inproceedings",None,None,2021),
 "loshchilov2017sgdr":("in Proc. Int. Conf. Learn. Represent. (ICLR)","International Conference on Learning Representations (ICLR)","inproceedings",None,None,2017),
 "kumar2022distort":("in Proc. Int. Conf. Learn. Represent. (ICLR)","International Conference on Learning Representations (ICLR)","inproceedings",None,None,2022),
 "nguyen2023wheretobegin":("in Proc. Int. Conf. Learn. Represent. (ICLR)","International Conference on Learning Representations (ICLR)","inproceedings",None,None,2023),
 "chen2023pretrainfl":("in Proc. Int. Conf. Learn. Represent. (ICLR)","International Conference on Learning Representations (ICLR)","inproceedings",None,None,2023),
}
PREPRINT = {"hsu2019dirichlet","zhao2018noniid","wu2022motley","goyal2017warmup","colas2018seeds","park2025ovalp","haseeb2025mechanistic","lyu2025imbalanced"}
BOOKS = {
 "serra1982":("London, U.K.","Academic Press",1982,None,"0-12-637240-3"),
 "gonzalez2018dip":("New York, NY, USA","Pearson",2018,"4th ed.","978-0-13-335672-4"),
 "mandelbrot1982":("San Francisco, CA, USA","W. H. Freeman",1982,None,"978-0-7167-1186-5"),
 "cohen1988power":("Hillsdale, NJ, USA","Lawrence Erlbaum Associates",1988,"2nd ed.","978-0-8058-0283-2"),
 "soille2003":("Berlin, Germany","Springer",2004,"2nd ed.",None),
 "efron1993intro":("Boston, MA, USA","Springer US",1993,None,None),
}

def dash(p): return p.replace("-","–") if p else p
def link_of(k):
    e=next(x for x in E if x["key"]==k)
    if e["kind"]=="doi": return "https://doi.org/"+e["id"]
    if e["kind"]=="arxiv":
        if k in OR and OR[k]: return OR[k][1]
        if k in PC: return PC[k]["link"]
        return "https://arxiv.org/abs/"+e["id"]
    if e["kind"]=="url": return e["id"]
    if e["kind"]=="jstor": return "https://www.jstor.org/stable/"+e["id"]
    if e["kind"]=="isbn": return "https://openlibrary.org/isbn/"+e["id"]
def ieee(k):
    e=next(x for x in E if x["key"]==k); r=R[k]; m=r.get("meta",{})
    A=ieee_authors(people_of(k)); T=title_of(k)
    if k=="shannon1948":
        return f'{A}, "{T}," *Bell Syst. Tech. J.*, vol. 27, no. 3, pp. 379–423, Jul. 1948, doi: 10.1002/j.1538-7305.1948.tb01338.x; and vol. 27, no. 4, pp. 623–656, Oct. 1948, doi: 10.1002/j.1538-7305.1948.tb00917.x.'
    if k in BOOKS:
        city,pub,yr,ed,isbn=BOOKS[k]
        s=f"{A}, *{T}*"+ (f", {ed}" if ed else ".") + f" {city}: {pub}, {yr}"
        if e["kind"]=="doi": s+=f", doi: {e['id']}"
        elif isbn: s+=f", ISBN {isbn}"
        return s+"."
    if k=="holm1979":
        return f'{A}, "{T}," *Scand. J. Statist.*, vol. 6, no. 2, pp. 65–70, 1979. [Online]. Available: https://www.jstor.org/stable/4615733'
    if k=="cheng2017figshare":
        return f'{A}, "{T}," figshare, Dataset, version 5, Apr. 2017, doi: 10.6084/m9.figshare.1512427.v5.'
    if k=="nickparvar2021kaggle":
        return f'{A}, "{T}," Kaggle, Dataset, 2021, doi: 10.34740/KAGGLE/DSV/2645886.'
    if e["kind"]=="arxiv" and k in PREPRINT:
        return f'{A}, "{T}," arXiv preprint arXiv:{e["id"]}, {m["year"]}.'
    if k in MAN:
        ven,_,bt,vol,pp,yr=MAN[k]
        if bt=="article":
            return f'{A}, "{T}," *{ven}*, vol. {vol}, pp. {pp}, {yr}. [Online]. Available: {link_of(k)}'
        s=f'{A}, "{T}," {ven}'
        if vol and "PMLR" in ven: s+=f", vol. {vol}"
        elif vol: s+=f", vol. {vol}"
        s+=f", {yr}"
        if pp: s+=f", pp. {pp}"
        if e["kind"]=="arxiv": s+=f". [Online]. Available: {link_of(k)}"
        else: s+=f". [Online]. Available: {link_of(k)}"
        return s
    # DOI journal / proceedings / chapter
    yr=m.get("year"); vol=m.get("volume"); iss=m.get("issue"); pg=PAGE_FIX.get(k) or dash(m.get("page")); art=ART_FIX.get(k) or m.get("article")
    if art and pg and pg.replace("–","")==str(art): pg=None
    doi=e["id"]
    if k in V:
        ven=V[k][0]; s=f'{A}, "{T}," {ven}'+(", vol. 39, no. 16" if k=="guo2025fedvls" else "")+f", {yr}"
        if art and not pg: s+=f", art. no. {art}"
        if pg: s+=f", pp. {pg}"
        return s+f", doi: {doi}."
    J=U(m.get("container") or "").replace("®","")
    if art and str(iss)=="1": iss=None
    if iss: iss=str(iss).replace("-","–")
    s=f'{A}, "{T}," *{J}*'
    if vol: s+=f", vol. {vol}"
    if iss: s+=f", no. {iss}"
    if art and not (pg and "–" in pg): s+=f", art. no. {art}"
    elif pg: s+=f", pp. {pg}" if "–" in pg else f", p. {pg}"
    s+=f", {yr}, doi: {doi}."
    return s

import sys; sys.path.insert(0,'/home/claude/refs'); from tex_escape import tex
def bibkey_safe(s): return s.replace("&","\\&").replace("%","\\%").replace("_","\\_")
def bibtex(k):
    e=next(x for x in E if x["key"]==k); r=R[k]; m=r.get("meta",{})
    P=people_of(k); T=title_of(k); f={}
    f["author"]=bib_authors(P) if P else None
    f["title"]="{{"+bibkey_safe(T)+"}}"
    typ="article"
    if k=="shannon1948":
        f.update(journal="Bell System Technical Journal",volume="27",number="3--4",pages="379--423, 623--656",year="1948",doi="10.1002/j.1538-7305.1948.tb01338.x",url="https://doi.org/10.1002/j.1538-7305.1948.tb01338.x",note="Part II: doi:10.1002/j.1538-7305.1948.tb00917.x")
    elif k in BOOKS:
        typ="book"; city,pub,yr,ed,isbn=BOOKS[k]; f.update(publisher=pub,address=city,year=str(yr))
        if ed: f["edition"]=ed.replace(" ed.","")
        if e["kind"]=="doi": f["doi"]=e["id"]; f["url"]="https://doi.org/"+e["id"]
        if isbn: f["isbn"]=isbn
    elif k=="holm1979":
        f.update(journal="Scandinavian Journal of Statistics",volume="6",number="2",pages="65--70",year="1979",url="https://www.jstor.org/stable/4615733")
    elif k in ("cheng2017figshare","nickparvar2021kaggle"):
        typ="misc"; f.update(howpublished="figshare" if k.startswith("cheng") else "Kaggle", year="2017" if k.startswith("cheng") else "2021", doi=e["id"], url="https://doi.org/"+e["id"], note="Dataset"+(", version 5" if k.startswith("cheng") else ""))
    elif e["kind"]=="arxiv" and k in PREPRINT:
        typ="misc"; f.update(howpublished=f"arXiv preprint arXiv:{e['id']}", year=str(m["year"]), eprint=e["id"], archivePrefix="arXiv", url="https://arxiv.org/abs/"+e["id"])
    elif k in MAN:
        ven,full,bt,vol,pp,yr=MAN[k]; typ=bt
        if bt=="article": f.update(journal=full)
        else: f.update(booktitle=full)
        if vol: f["volume"]=vol
        if pp: f["pages"]=pp.replace("–","--")
        f["year"]=str(yr); f["url"]=link_of(k)
        if e["kind"]=="arxiv": f.update(eprint=e["id"], archivePrefix="arXiv")
    else:
        yr=m.get("year"); vol=m.get("volume"); iss=m.get("issue"); pg=PAGE_FIX.get(k) or m.get("page"); art=ART_FIX.get(k) or m.get("article")
        if k in V:
            typ=V[k][2]; f["booktitle"]=bibkey_safe(V[k][1]) if "\\&" not in V[k][1] else V[k][1]
        else:
            f["journal"]=bibkey_safe(U(m.get("container") or "").replace("®",""))
        if vol and typ=="article": f["volume"]=str(vol)
        if iss and typ=="article": f["number"]=str(iss)
        if pg and not (art and str(pg)==str(art)): f["pages"]=str(pg).replace("–","-").replace("-","--")
        if art: f["note"]=f"Art. no. {art}"
        f["year"]=str(yr); f["doi"]=e["id"]; f["url"]="https://doi.org/"+e["id"]
    f=[(a,(tex(b) if a not in ("url","doi","eprint") else b)) for a,b in f.items() if b]
    body=",\n".join(f"  {a:9s} = {{{b}}}" if not str(b).startswith("{") else f"  {a:9s} = {b}" for a,b in f)
    return f"@{typ}{{{k},\n{body}\n}}"

if __name__=="__main__":
    out=[]
    for e in E:
        k=e["key"]; out.append(dict(key=k, stage=e["stage"], ieee=ieee(k), bib=bibtex(k), link=link_of(k), used=e["used"]))
    json.dump(out,open("/home/claude/refs/formatted.json","w"),indent=1,ensure_ascii=False)
    for o in out: print("["+o["key"]+"] "+o["ieee"])
