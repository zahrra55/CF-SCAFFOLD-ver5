"""
==================================================================
 CF-SCAFFOLD — STAGE 10B: TWO CORRECTIONS TO STAGE 10
==================================================================
 Self-contained: if the Stage 10 feature cache is present it is used,
 otherwise the features are extracted again. /kaggle/working is cleared
 when a new session starts, so a cache written by an earlier notebook
 will usually be gone. Extraction takes about six minutes on CPU.

 CORRECTION 1 - the leakage measurement in Stage 10 was confounded.
   It compared a patient-grouped split scored at PATIENT level against
   a random split scored at SLICE level. Two things differed at once.
   Majority voting over a patient's slices is an ensemble and raises
   accuracy by itself, independently of any leakage, so the reported
   -2.4 points measured neither effect cleanly.

   Here both arms are scored at slice level and only the split changes:
       random slice split      (slices of one patient on both sides)
       patient-grouped split   (no patient crosses the split)
   Their difference is the leakage inflation that Yagis et al. report
   for other cohorts.

 CORRECTION 2 - the ablation had no error bars.
   GroupKFold is deterministic, so Stage 10 gave one number per feature
   set with nothing to judge it against. A gap of 1.3 points may or may
   not be real. Each set is now evaluated over repeated shuffled
   patient-grouped splits, with paired tests between the sets that
   matter.
------------------------------------------------------------------
 Kaggle: CPU is sufficient. Attach the Figshare .mat dataset.
==================================================================
"""

import os, json, warnings
import numpy as np
warnings.filterwarnings("ignore")
try: import h5py
except Exception: os.system("pip install h5py --quiet"); import h5py
from scipy import ndimage, stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score

DATA_ROOT = "/kaggle/input"
CACHE     = "/kaggle/working/stage10_features.npz"
OUT       = "/kaggle/working/stage10b_results.json"
IMG_SIZE  = 128
N_TREES   = 300
N_REP     = 10          # repeated splits, so every number carries a spread

FEATURE_NAMES = ["Ft","Fb","C","GLCM_t","corr_t","GLCM_b","entropy_t",
                 "lacun_t","lacun_b","solidity","eccentricity","extent",
                 "circularity","edge_density","mean_grad","border_grad","size"]

def brain_mask_of(gray):
    m = gray > max(gray.mean() * 0.25, 10)
    m = ndimage.binary_fill_holes(m)
    m = ndimage.binary_opening(m, iterations=2)
    lbl, n = ndimage.label(m)
    if n >= 1:
        s = np.bincount(lbl.ravel()); s[0] = 0; m = lbl == s.argmax()
    return m.astype(np.uint8)

def boxcount_fd(gray, region, sizes=None):
    v = gray[region > 0]
    if v.size < 16: return np.nan
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-6: return np.nan
    g = np.zeros_like(gray, dtype=np.float64)
    g[region > 0] = (gray[region > 0] - lo) / (hi - lo) * 255.0
    H, W = g.shape
    if sizes is None:
        sizes = [2 ** k for k in range(1, int(np.log2(min(H, W))))][:8]
    Ns = []
    for s in sizes:
        nb = 0
        for i in range(0, H - H % s, s):
            for j in range(0, W - W % s, s):
                bm = region[i:i+s, j:j+s]
                if bm.sum() == 0: continue
                blk = g[i:i+s, j:j+s][bm > 0]
                if blk.size == 0: continue
                nb += int(np.floor(blk.max()/s) - np.floor(blk.min()/s) + 1)
        Ns.append(max(nb, 1))
    sizes_a = np.array(sizes, float); Ns = np.array(Ns, float)
    return float(np.polyfit(np.log(1.0/sizes_a), np.log(Ns), 1)[0])

def regions_of(gray, mask, min_t=16, min_b=64):
    brain = brain_mask_of(gray)
    tumor = (mask > 0) & (brain > 0)
    free  = (brain > 0) & (~tumor)
    if tumor.sum() < min_t or free.sum() < min_b: return None, None
    return tumor.astype(np.uint8), free.astype(np.uint8)

# ---------------------------------------------------------------
# The remaining fifteen features
# ---------------------------------------------------------------
def glcm_contrast_corr(patch):
    from skimage.feature import graycomatrix, graycoprops
    q = np.clip((patch/32).astype(int), 0, 7)
    m = graycomatrix(q, [1], [0], levels=8, symmetric=True, normed=True)
    return (float(graycoprops(m, "contrast")[0,0]),
            float(graycoprops(m, "correlation")[0,0]))

def shannon(v):
    h, _ = np.histogram(v, bins=32, density=True); h = h[h > 0]
    return float(-(h*np.log2(h)).sum())

def lacunarity(mask, box=8):
    H, W = mask.shape; c = []
    for i in range(0, H-box, box):
        for j in range(0, W-box, box):
            c.append(mask[i:i+box, j:j+box].sum())
    c = np.array(c, float)
    return float(1 + c.var()/(c.mean()**2)) if c.mean() > 1e-6 else 0.0

def shape_feats(mask):
    from skimage.measure import regionprops, label as sklabel
    pr = regionprops(sklabel(mask))
    if not pr: return (0.0, 0.0, 0.0, 0.0)
    p = max(pr, key=lambda r: r.area); per = p.perimeter or 1
    return (float(p.solidity), float(p.eccentricity), float(p.extent),
            float(4*np.pi*p.area/(per**2)))

def edge_feats(gray, tumor):
    gx, gy = np.gradient(gray.astype(float))
    grad = np.sqrt(gx**2 + gy**2)
    bd = ndimage.binary_dilation(tumor) & ~ndimage.binary_erosion(tumor)
    return (float((grad[tumor>0] > grad.mean()).mean()) if tumor.sum() else 0.0,
            float(grad[tumor>0].mean()) if tumor.sum() else 0.0,
            float(grad[bd>0].mean()) if bd.sum() else 0.0)

FEATURE_NAMES = ["Ft","Fb","C","GLCM_t","corr_t","GLCM_b","entropy_t",
                 "lacun_t","lacun_b","solidity","eccentricity","extent",
                 "circularity","edge_density","mean_grad","border_grad","size"]

def features_of(gray, mask):
    tumor, free = regions_of(gray, mask)
    if tumor is None: return None
    Ft = boxcount_fd(gray, tumor); Fb = boxcount_fd(gray, free)
    if not (np.isfinite(Ft) and np.isfinite(Fb) and Fb > 1e-6): return None
    gt, ct = glcm_contrast_corr((gray*(tumor>0)).astype(np.uint8))
    gb, _  = glcm_contrast_corr((gray*(free>0)).astype(np.uint8))
    sol, ecc, ext, cir = shape_feats(tumor)
    ed, mg, bg = edge_feats(gray, tumor)
    return np.array([Ft, Fb, Ft/Fb, gt, ct, gb, shannon(gray[tumor>0]),
                     lacunarity(tumor), lacunarity(free), sol, ecc, ext, cir,
                     ed, mg, bg, float(tumor.sum())], dtype=np.float64)

# ---------------------------------------------------------------
# Reading the dataset
# ---------------------------------------------------------------
def read_pid(cj):
    if "PID" not in cj: return None
    raw = np.array(cj["PID"]).ravel()
    try:
        s = "".join(chr(int(c)) for c in raw if int(c) > 0); return s.strip() or None
    except Exception: return None

def resize_pair(img, mask, size):
    from PIL import Image
    im = Image.fromarray(((img-img.min())/(img.max()-img.min()+1e-8)*255
                          ).astype(np.uint8)).resize((size,size), Image.BILINEAR)
    g = np.asarray(im).astype(np.float32)
    mk = Image.fromarray((mask*255).astype(np.uint8)).resize((size,size), Image.NEAREST)
    return g, (np.asarray(mk) > 127).astype(np.uint8)

def extract_features_from_scratch():
    files = []
    for dp, _, fns in os.walk(DATA_ROOT):
        for fn in fns:
            if fn.lower().endswith(".mat"): files.append(os.path.join(dp, fn))
    files = sorted(files)
    print(f"  found {len(files)} .mat files")
    if not files:
        raise SystemExit("No .mat files under /kaggle/input. Attach the Figshare "
                         "collection as a DATASET, not as a notebook.")
    X, Y, PID = [], [], []
    for k, p in enumerate(files):
        try:
            with h5py.File(p, "r") as f:
                cj = f["cjdata"]
                lab = int(np.array(cj["label"]).ravel()[0])
                img = np.array(cj["image"]).astype(np.float32)
                if img.ndim == 2: img = img.T
                msk = np.array(cj["tumorMask"]).astype(np.uint8)
                if msk.ndim == 2: msk = msk.T
                pid = read_pid(cj)
        except Exception:
            continue
        if pid is None: continue
        g, m = resize_pair(img, msk, IMG_SIZE)
        feat = features_of(g, m)
        if feat is None: continue
        X.append(feat); Y.append(lab-1); PID.append(pid)
        if (k+1) % 500 == 0: print(f"    processed {k+1}/{len(files)}")
    return np.array(X), np.array(Y), np.array(PID)

# ---------------------------------------------------------------
# Load, from cache if it survived the session, otherwise extract
# ---------------------------------------------------------------
if os.path.exists(CACHE):
    d = np.load(CACHE, allow_pickle=True)
    X, Y, PID = d["X"], d["Y"], d["PID"]
    print(f"  loaded cached features: {X.shape}")
else:
    print("  no cache in this session, extracting (about six minutes)")
    X, Y, PID = extract_features_from_scratch()
    np.savez_compressed(CACHE, X=X, Y=Y, PID=PID)
    print(f"  extracted and cached: {X.shape}")

idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
others = [idx[n] for n in FEATURE_NAMES if n not in ("Ft", "Fb", "C")]
print(f"  {len(X)} slices, {len(np.unique(PID))} patients, classes={np.bincount(Y)}")

results = {}

# ---------------------------------------------------------------
def slice_level_score(cols, grouped, seed):
    """Slice-level balanced accuracy. Scoring is identical in both arms;
       only the way folds are formed changes."""
    Xs = X[:, cols]
    if grouped:
        sp = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        folds = sp.split(Xs, Y, groups=PID)
    else:
        sp = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        folds = sp.split(Xs, Y)
    yt, yp = [], []
    for tr, te in folds:
        m = RandomForestClassifier(N_TREES, random_state=seed, n_jobs=-1)
        m.fit(Xs[tr], Y[tr])
        yt.extend(Y[te]); yp.extend(m.predict(Xs[te]))
    return balanced_accuracy_score(yt, yp)

def patient_level_score(cols, seed):
    """Patient-grouped split, aggregated to one prediction per patient."""
    Xs = X[:, cols]
    sp = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    yt, yp = [], []
    for tr, te in sp.split(Xs, Y, groups=PID):
        m = RandomForestClassifier(N_TREES, random_state=seed, n_jobs=-1)
        m.fit(Xs[tr], Y[tr]); pred = m.predict(Xs[te])
        for pid in np.unique(PID[te]):
            sel = PID[te] == pid
            yt.append(int(np.bincount(Y[te][sel]).argmax()))
            yp.append(int(np.bincount(pred[sel]).argmax()))
    return balanced_accuracy_score(yt, yp)

# ---------------------------------------------------------------
print("\n" + "=" * 66)
print("  CORRECTION 1 — LEAKAGE, MEASURED PROPERLY")
print("=" * 66)
print("  Both arms scored at slice level; only the split differs.\n")
full = others + [idx["Ft"], idx["Fb"], idx["C"]]
grouped_s, random_s = [], []
for s in range(N_REP):
    g = slice_level_score(full, grouped=True,  seed=s)
    r = slice_level_score(full, grouped=False, seed=s)
    grouped_s.append(g); random_s.append(r)
    print(f"    seed {s}: patient-grouped {g:.4f}   random slice {r:.4f}   "
          f"inflation {(r-g)*100:+.2f}pp")
grouped_s, random_s = np.array(grouped_s), np.array(random_s)
infl = (random_s - grouped_s) * 100
p_leak = stats.wilcoxon(random_s, grouped_s).pvalue
print(f"\n  patient-grouped : {grouped_s.mean():.4f} +/- {grouped_s.std():.4f}")
print(f"  random slice    : {random_s.mean():.4f} +/- {random_s.std():.4f}")
print(f"  inflation       : {infl.mean():+.2f} +/- {infl.std():.2f} points "
      f"(Wilcoxon p = {p_leak:.4g})")
results["leakage"] = {"grouped": grouped_s.mean(), "random": random_s.mean(),
                      "inflation_pp": infl.mean(), "p": p_leak}
print("""
  Interpretation. Yagis et al. measured 29 to 55 points on
  neurodegenerative cohorts with 2D CNNs. A hand-crafted feature set is
  far less able to memorise a particular slice than a network with
  millions of parameters, so a smaller inflation here is expected. The
  number above is the figure for this dataset and this representation,
  and it is what a published slice-level result on these data is worth
  relative to a patient-level one.
""")

# For reference, the patient-level figure with the same repeated splits
pat = np.array([patient_level_score(full, s) for s in range(N_REP)])
print(f"  For reference, patient-level with aggregation: {pat.mean():.4f} "
      f"+/- {pat.std():.4f}")
print(f"  Aggregating slices to a patient adds "
      f"{(pat.mean()-grouped_s.mean())*100:+.2f} points on top of the")
print("  grouped split. That is the ensemble effect of majority voting, and it")
print("  is a property of the protocol rather than of the features.")
results["patient_level"] = pat.mean()

# ---------------------------------------------------------------
print("\n" + "=" * 66)
print("  CORRECTION 2 — THE ABLATION, WITH ERROR BARS")
print("=" * 66)
sets = {
    "Ft, Fb separate":       [idx["Ft"], idx["Fb"]],
    "C alone":               [idx["C"]],
    "14 others, no fractal": others,
    "14 others + Ft, Fb":    others + [idx["Ft"], idx["Fb"]],
    "14 others + C":         others + [idx["C"]],
    "14 others + Ft, Fb, C": others + [idx["Ft"], idx["Fb"], idx["C"]],
}
scores = {}
for name, cols in sets.items():
    v = np.array([patient_level_score(cols, s) for s in range(N_REP)])
    scores[name] = v
    print(f"  {name:24s} {len(cols):3d} feat   {v.mean():.4f} +/- {v.std():.4f}")
results["ablation"] = {k: [v.mean(), v.std()] for k, v in scores.items()}

print("\n  Paired comparisons over the same splits:")
def compare(a, b):
    d = scores[a] - scores[b]
    p = stats.wilcoxon(scores[a], scores[b]).pvalue
    print(f"    {a:24s} vs {b:24s} {d.mean()*100:+6.2f}pp  p={p:.4f}  "
          f"wins {int((d>0).sum())}/{N_REP}")
    return d.mean(), p

print()
d_add, p_add   = compare("14 others + Ft, Fb, C", "14 others + Ft, Fb")
d_rep, p_rep   = compare("14 others + C",         "14 others + Ft, Fb")
d_frac, p_frac = compare("14 others + Ft, Fb, C", "14 others, no fractal")
results["tests"] = {"ratio_adds": [d_add, p_add],
                    "ratio_replaces": [d_rep, p_rep],
                    "fractal_family": [d_frac, p_frac]}

print("\n" + "=" * 66); print("  WHAT CAN BE CLAIMED"); print("=" * 66)
if p_frac < 0.05 and d_frac > 0:
    print(f"  The fractal family contributes {d_frac*100:+.1f} points over a")
    print(f"  feature set without it (p={p_frac:.4f}). This part is established.")
else:
    print(f"  The fractal family does not significantly improve on a feature set")
    print(f"  without it ({d_frac*100:+.1f}pp, p={p_frac:.4f}). That is a serious")
    print(f"  finding and must be reported.")
print()
if p_add < 0.05 and d_add > 0:
    print(f"  The ratio ADDS {d_add*100:+.1f} points on top of Ft and Fb")
    print(f"  (p={p_add:.4f}). It is a genuine feature, not a restatement of the")
    print(f"  two dimensions. Claim it as a supplement, never as a replacement.")
elif d_add > 0:
    print(f"  The ratio adds {d_add*100:+.1f} points but the difference is not")
    print(f"  significant (p={p_add:.4f}). Report the direction and the interval;")
    print(f"  do not claim an improvement.")
else:
    print(f"  The ratio does not add on top of Ft and Fb ({d_add*100:+.1f}pp).")
    print(f"  Its value must then rest on interpretability alone.")
print()
if d_rep < 0:
    print(f"  Replacing Ft and Fb by the ratio costs {abs(d_rep)*100:.1f} points.")
    print(f"  The ratio discards the scale of the two dimensions, and that scale")
    print(f"  carries information. The thesis must present C alongside Ft and Fb,")
    print(f"  not instead of them.")

with open(OUT, "w") as f:
    json.dump(results, f, indent=2, default=float)
print(f"\n  written to {OUT}")
print("\nDONE.\n")
