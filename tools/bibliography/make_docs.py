import json, csv, re
exec(open("/home/claude/refs/entries.py").read())
F  = {f["key"]: f for f in json.load(open("/home/claude/refs/formatted.json"))}
R  = {r["key"]: r for r in json.load(open("/home/claude/refs/verified.json"))}
OR = json.load(open("/home/claude/refs/openreview_check.json"))
PC = json.load(open("/home/claude/refs/proceedings_check.json"))
STAGES = [
 ("0","Surveys, clinical context and motivation","Literature-review backbone for both pillars."),
 ("A","Datasets and evaluation protocol","Data sources, leakage-free evaluation, and the non-IID partition used in every federated stage."),
 ("B","Pre-processing, morphology and software","Everything that touches the image before a feature is computed, plus the libraries the code imports."),
 ("C","Fractal and texture descriptors (Pillar I)","The descriptor C = Ft/Fb, its components, the 16-feature set, and known estimator biases."),
 ("D","Segmentation: automatic delineation attempt (Pillar I, Stages 11-12)","Active contours, seeding by bilateral symmetry, and the overlap metrics used to score them."),
 ("E","Classification, comparators and metrics (Pillar I)","Classifier, evaluation metrics, nested cross-validation, and the published results we compare against."),
 ("F","Federated learning algorithms (Pillar II)","FedAvg, SCAFFOLD, FedProx, the restricted-softmax family, and the personalization/clustering work we tested and rejected."),
 ("G","Normalisation, optimisation and class imbalance (Pillar II)","GroupNorm vs BatchNorm, and every optimisation intervention in the eight-row ablation."),
 ("H","Transfer learning and the frozen backbone (Pillar II, Stages 7, 9, 9B)","Pre-trained ResNet-18, linear probing, contrastive learning, and prior work that bounds the frozen-backbone claim."),
 ("I","Statistical methodology (both pillars)","Every test, interval, effect size and protocol decision used in the results chapters."),
 ("J","Dataset shortcuts and robustness to segmentation (Pillar I, Stages 14-16)","The documented slice-selection shortcut in the Figshare data, the general shortcut-learning literature, and studies of how radiomic features change with the segmentation method."),
]
OPEN_HOSTS=("proceedings.mlr.press","proceedings.neurips.cc","jmlr.org","openreview.net","proceedings.mlsys.org","arxiv.org")
FREE_OVERRIDE = {  # OpenAlex pointed to unusable copies (an image file, a dead mirror, an anonymous short link); replaced with verified official copies
 "litjens2017survey":"https://arxiv.org/abs/1702.05747",
 "he2016resnet":"https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html",
 "breiman2001rf":"https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf",
}
def access(k):
    a,u=_access(k)
    return (a, FREE_OVERRIDE.get(k,u))
def _access(k):
    e=next(x for x in E if x["key"]==k); r=R[k]
    if e["kind"]=="doi":
        oa=r.get("oa") or {}
        if k=="cheng2017figshare" or k=="nickparvar2021kaggle": return ("Open data (CC BY 4.0)" if k.startswith("cheng") else "Open data (Kaggle account)"), F[k]["link"]
        if e["id"].startswith("10.1007/978-") and k in ("soille2003","efron1993intro"): 
            return ("Book (library / publisher)", None) if not oa.get("is_oa") else ("Open access", oa.get("oa_url"))
        if oa.get("is_oa"): return f"Open access ({oa.get('oa_status')})", oa.get("oa_url")
        return "Subscription (institutional access)", None
    if e["kind"]=="arxiv": return "Open access (arXiv / proceedings)", "https://arxiv.org/abs/"+e["id"]
    if e["kind"]=="url": return "Open access (official proceedings / journal site)", F[k]["link"]
    if e["kind"]=="isbn": return "Book (library copy; no free full text)", None
    if e["kind"]=="jstor": return "JSTOR (institutional access or free JSTOR reading account)", None
def verified_line(k):
    e=next(x for x in E if x["key"]==k); r=R[k]
    if e["kind"]=="doi":
        s=f"{r.get('registry')} record matched (title, year); doi.org resolver HTTP {r.get('resolver')}"
    elif e["kind"]=="arxiv":
        s=f"arXiv API record matched (title); abs page HTTP {r.get('resolver')}"
        if k in OR and OR[k]: s+=f"; peer-reviewed venue confirmed on OpenReview ({OR[k][0]})"
        elif k in PC: s+="; venue confirmed in the official proceedings listing"
        elif k in ("hsu2019dirichlet","zhao2018noniid","wu2022motley","goyal2017warmup","colas2018seeds","park2025ovalp","haseeb2025mechanistic","lyu2025imbalanced"): s+="; no peer-reviewed version found, cited as preprint"
    elif e["kind"]=="url": s=f"official page HTTP {r.get('resolver')}; title and citation meta-tags (authors, pages) read from the page"
    elif e["kind"]=="isbn": s=f"ISBN record matched in {r.get('registry')}"
    else: s="no DOI exists; identity confirmed via JSTOR id 4615733 and Mathematical Reviews MR538597"
    extra=[n for n in r.get("notes",[]) if not n.startswith("jstor.org")]
    if extra and e["kind"]!="jstor": s+=". Note: "+" ".join(extra)
    return s

# ------------------ docs/06_references.md ------------------
L=[]
L.append("# CF-SCAFFOLD: Complete Reference List\n")
L.append("**Researcher:** Zahraa Ibrahim Khaleel  |  **Supervisor:** Prof. Dr. Mohammed Sahib Mehdi  |  Al-Nahrain University, College of Science, Department of Computer Science\n")
L.append("This list replaces the earlier 32-item draft. It contains **166 references**, ordered by the stage of the pipeline where each one is used, with a direct link, the access route, the exact place and purpose of use, and how the record was verified.\n")
L.append("## How every entry was verified (September 2026)\n")
L.append("- **DOIs (122):** each DOI was looked up in Crossref (or DataCite for the two datasets); the registry title had to contain the expected title and the year had to match within one year (online-first vs issue year). The DOI was then resolved through `doi.org` to confirm it redirects to a live landing page. Open-access status and a free full-text link come from OpenAlex.")
L.append("- **arXiv items (23):** checked through the arXiv API. Where a peer-reviewed version exists, the venue was confirmed on OpenReview (ICLR) or in the official NeurIPS/MLSys proceedings listing, and the citation points to that version. Eight items have no peer-reviewed version and are cited as preprints.")
L.append("- **Proceedings/journal pages without DOI (16):** the official PMLR, NeurIPS, MLSys and JMLR pages were fetched; authors, volume and page range were read from the page's own citation meta-tags.")
L.append("- **Books (4 ISBN + 2 DOI):** ISBN matched in Open Library; publisher and place taken from that record.")
L.append("- **Holm (1979):** no DOI exists; confirmed through JSTOR stable id 4615733 and MR538597.\n")
L.append("Result: **166 / 166 verified.** (150 in September 2026; 16 added in October 2026 for the shortcut and segmentation-robustness analyses.) Corrections made during verification are listed at the end of this file and in `07_verification_log.md`.\n")
L.append("> Links resolve today. `doi.org` links are permanent by design; for subscription items use your university library proxy. The machine-readable results of the check are in `references_verification.csv`.\n")
L.append("## Contents\n")
for sid,name,_ in STAGES: L.append(f"- [{sid}. {name}](#{re.sub(r'[^a-z0-9 -]','',(sid+'. '+name).lower()).replace(' ','-')})")
L.append("- [Using this list for the literature review](#using-this-list-for-the-literature-review)\n- [Corrections found during verification](#corrections-found-during-verification)\n")
n=0; number={}
for sid,name,desc in STAGES:
    L.append(f"\n## {sid}. {name}\n\n*{desc}*\n")
    for e in [x for x in E if x["stage"]==sid]:
        n+=1; k=e["key"]; number[k]=n
        acc,freeurl=access(k)
        L.append(f"**[{n}]** {F[k]['ieee']}  ")
        L.append(f"- **Key:** `{k}`  ·  **Link:** {F[k]['link']}")
        L.append(f"- **Access:** {acc}"+(f"; free copy: {freeurl}" if freeurl and freeurl!=F[k]['link'] else ""))
        L.append(f"- **Where and how it is used:** {e['used']}")
        L.append(f"- **Verified:** {verified_line(k)}\n")
LR = {
 "Pillar I — fractal descriptor and classification":[
   ("Brain-tumour types and MRI context","louis2021who villanueva2017imaging"),
   ("Radiomics and hand-crafted features","lambin2012radiomics gillies2016radiomics zwanenburg2020ibsi vangriethuysen2017pyradiomics haralick1973glcm"),
   ("Fractal geometry and its estimation","mandelbrot1982 pentland1984fractal keller1989texture sarkar1994dbc chaudhuri1995texture lopes2009fractal smith1996fractal gonzato1998bias gonzato2000ideal"),
   ("Fractals in the brain and in brain tumours","dieva2014fractalsI dieva2015fractalsII iftekharuddin2003fractal iftekharuddin2009multimodal sanchez2022fractal battalapalli2023fd"),
   ("Lacunarity and entropy","allain1991lacunarity plotnick1996lacunarity shannon1948"),
   ("Classification on the Figshare dataset","cheng2015enhanced ismael2018statistical gumaei2019hybrid paul2017deep afshar2019capsule swati2019transfer deepak2019transfer sultan2019multi rehman2020framework badza2020cnn diazpernas2021multiscale"),
   ("Evaluation rigour and leakage","yagis2021leakage kapoor2023leakage varoquaux2022failures roberts2021pitfalls varma2006cvbias cawley2010overfit"),
   ("Segmentation and its limits","kass1988snakes chan2001acwe caselles1997gac marquezneila2014morph li2010drlse gordillo2013survey khotanlou2009symmetry liu2009symmetry ronneberger2015unet isensee2021nnunet"),
   ("Shortcuts in the Figshare data and in medical imaging","wallis2022clever vangala2026scores geirhos2020shortcut lapuschkin2019cleverhans degrave2021covid zech2018pneumonia badgeley2019hip oakden2020hidden gichoya2022race tinauer2025skull vasquez2024cleverhans"),
   ("Feature stability under automatic segmentation","tixier2019reliability haarburger2020reproducibility poirot2022robustness suter2020glioblastoma nadeem2024robust")],
 "Pillar II — federated learning under class skew":[
   ("Federated learning foundations","mcmahan2017fedavg kairouz2021 li2020challenges"),
   ("Non-IID data and client drift","zhao2018noniid hsu2019dirichlet zhu2021noniid li2022silos ye2023hfl karimireddy2020scaffold li2020fedprox wang2020fednova acar2021feddyn reddi2021adaptive li2021moon"),
   ("Label skew and missing classes","li2021fedrs zhang2022fedlc guo2025fedvls lu2023labelmask wang2024fednlr luo2021ccvr menon2021logit lyu2025imbalanced"),
   ("Normalisation under non-IID","ioffe2015bn wu2018groupnorm hsieh2020quagmire li2021fedbn casella2024norm"),
   ("Pre-training and frozen backbones in FL","nguyen2023wheretobegin chen2023pretrainfl kornblith2019transfer kumar2022distort raghu2019transfusion park2025ovalp haseeb2025mechanistic"),
   ("Personalized and clustered FL","li2021ditto collins2021fedrep wu2022motley chen2022pflbench ghosh2022ifca sattler2021cfl"),
   ("FL in medicine and brain-tumour imaging","rieke2020future kaissis2020secure sheller2019brats sheller2020fl pati2022rare dayan2021exam guan2024fedmed islam2022ensemble albalawi2024fltl")],
 "Combined review (both pillars)":[
   ("Open framing","rieke2020future kaissis2020secure guan2024fedmed litjens2017survey"),
   ("Bridge: interpretable features vs deep models under privacy constraints","gillies2016radiomics lambin2012radiomics battalapalli2023fd pati2022rare sheller2020fl"),
   ("Methodology shared by both pillars","yagis2021leakage demsar2006 wilcoxon1945 holm1979 efron1979bootstrap lakens2013effect bouthillier2021variance nosek2018prereg")],
}
L.append("\n## Using this list for the literature review\n")
L.append("Numbers refer to the list above. Each group is a paragraph-sized theme; together they cover a review of either pillar alone or of the whole thesis.\n")
for block, groups in LR.items():
    L.append(f"### {block}\n")
    for theme, keys in groups:
        ks=keys.split(); L.append(f"- **{theme}:** "+", ".join(f"[{number[k]}]" for k in ks))
    L.append("")
L.append("## Corrections found during verification\n")
for c in [
 "Kornblith et al. (CVPR 2019): pages are **2656–2666** in IEEE Xplore (the DOI version). 2661–2671 is the CVF open-access pagination and does not match the DOI.",
 "Islam et al., *Neural Processing Letters*: issue year **2023**, vol. 55, no. 4, pages **3779–3809** (not 3779–3798).",
 "MOON (Li, He, Song, CVPR 2021): pages **10708–10717**. Class-balanced loss (Cui et al., CVPR 2019): pages **9260–9269**.",
 "Gonzato et al.: the finite-size/sampling-bias paper is the **1998** one with **Marzocchi** (GJI 132(2):275–282). The **2000** paper has **Ciccotti** as third author and concerns placement/orientation bias. Both are now listed separately.",
 "Liu, *J. Biomed. Inform.* 2009 symmetry review: DOI is **10.1016/j.jbi.2009.07.003** (a DOI ending .008 points to a different paper).",
 "FL medical-imaging survey in *Pattern Recognition* 2024: first author is **H. Guan** (Guan, Yap, Bozoki, Liu), not \"Rahman\".",
 "arXiv:2502.11323 is Lyu, Zhou and Zhong, *A statistical theory of overfitting for imbalanced classification*; its described use was corrected to match what the paper actually studies.",
 "BRISC: *Scientific Data* **vol. 13, art. no. 361, 2026** (the \"vol. 12, art. 845, 2025\" form is wrong). Sánchez is the first author of the 2022 *Frontiers in Physiology* fractal paper.",
 "Eight former arXiv citations now point to their peer-reviewed versions: FedDyn, Adaptive FL, logit adjustment, FedBN (ICLR 2021); SGDR (ICLR 2017); fine-tuning distortion (ICLR 2022); Where-to-Begin and Pre-training-for-FL (ICLR 2023). pFL-Bench now cites NeurIPS 2022 Datasets and Benchmarks; Bouthillier et al. cites MLSys 2021.",
 "Mandelbrot (1982) was published in **San Francisco** by W. H. Freeman (Open Library record).",
 "The ACM record for FedRS stores only the short title \"FedRS\"; the full title is restored in the citation.",
 "A \"recent brain-tumour-specific data-leakage paper\" suggested earlier does not exist as a peer-reviewed publication and is not cited. Kapoor and Narayanan (2023) is used as the general companion to Yagis et al. (2021).",
]: L.append("- "+c)
open("/home/claude/refs/06_references.md","w").write("\n".join(L)+"\n")

# ------------------ docs/thesis_reference_list.md ------------------
T=["# Reference list for the thesis end-matter (IEEE style)\n",
   "Ready to paste into the **References** section. Entries follow the order of the pipeline stages (same numbering as `06_references.md`).\n",
   "> **If you write the thesis in LaTeX,** do not paste this list: use `paper/references.bib` with `\\bibliographystyle{IEEEtran}` and cite with `\\cite{key}`. LaTeX will number the references in the order they are first cited, which is what IEEE requires. The `.bib` file compiles with zero BibTeX warnings (checked with IEEEtran.bst).\n",
   "> **If you write it in Word,** IEEE also requires numbering by first citation. Paste this list, then reorder it to match the order in which the numbers first appear in your chapters.\n",
   "## References\n"]
for sid,_,_ in STAGES:
    for e in [x for x in E if x["stage"]==sid]:
        T.append(f"[{number[e['key']]}] {F[e['key']]['ieee'].replace('*','')}\n")
open("/home/claude/refs/thesis_reference_list.md","w").write("\n".join(T))

# ------------------ docs/references_verification.csv ------------------
with open("/home/claude/refs/references_verification.csv","w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["number","key","stage","identifier_type","identifier","registry","resolver_http","verified","open_access","free_fulltext_url","direct_link"])
    for sid,_,_ in STAGES:
        for e in [x for x in E if x["stage"]==sid]:
            k=e["key"]; r=R[k]; acc,free=access(k)
            w.writerow([number[k],k,sid,e["kind"],e["id"],r.get("registry"),r.get("resolver"),"yes" if r["ok"] else "no",acc,free or "",F[k]["link"]])
print("written; total", n)
