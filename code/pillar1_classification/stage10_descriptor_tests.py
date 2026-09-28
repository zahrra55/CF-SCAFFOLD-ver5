"""
==================================================================
 CF-SCAFFOLD — STAGE 10: INTERROGATING THE FRACTAL DESCRIPTOR
 Four experiments that decide what can honestly be claimed for
 C = Ft/Fb, run from one feature-extraction pass.
==================================================================
 Motivated by an adversarial prior-art review, which found the ratio
 descriptor to be unanticipated but identified four questions that a
 reviewer will ask and that we cannot currently answer.

 EXPERIMENT A — DOES THE RATIO ADD ANYTHING?           [decisive]
   Battalapalli et al. (Front. Physiol. 2023;14:1201617) already use
   tumour and non-tumour fractal dimensions as SEPARATE features. If a
   random forest given {Ft, Fb} matches one given {C}, then the ratio
   contributes nothing a classifier cannot extract by itself, and the
   descriptor is a presentational choice rather than a contribution.
   This experiment settles that, and it can settle it against us.

 EXPERIMENT B — IS THE NORMALISATION RATIONALE TRUE?
   We claim the ratio cancels acquisition effects because both regions
   come from one scan. That is asserted, never shown. Here the same
   slice is perturbed in ways that imitate scanner variation, and the
   stability of Ft is compared against that of C. If C is not steadier,
   the rationale must be withdrawn and the descriptor reframed.

 EXPERIMENT C — HOW LARGE IS SLICE-LEVEL LEAKAGE HERE?
   Yagis et al. (Sci. Rep. 2021;11:22544) measured 29 to 55 points of
   inflation on neurodegenerative MRI. No equivalent figure exists for
   this dataset, although most published results on it use slice-level
   splits. Running both protocols through one pipeline supplies it.

 EXPERIMENT D — DOES THE ESTIMATE DEPEND ON ARBITRARY CHOICES?
   Box-counting dimension is known to be biased on small regions. We
   vary the box-size range and the minimum-region thresholds and report
   how much C and the resulting accuracy move.
------------------------------------------------------------------
 Kaggle: CPU is sufficient. Attach the Figshare .mat dataset.
 Runtime: roughly 25 minutes. Auto-resume via CSV.
==================================================================
"""

import os, csv, time, warnings, json
import numpy as np
warnings.filterwarnings("ignore")
try: import h5py
except Exception: os.system("pip install h5py --quiet"); import h5py
from scipy import ndimage, stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.linear_model import LinearRegression

DATA_ROOT   = "/kaggle/input"
OUT_DIR     = "/kaggle/working"
CACHE       = f"{OUT_DIR}/stage10_features.npz"
RESULTS     = f"{OUT_DIR}/stage10_results.json"
IMG_SIZE    = 128
N_TREES     = 300
N_REPEATS   = 5          # repeat CV with different fold seeds
PERTURB_N   = 200        # slices used in the stability experiment

# ---------------------------------------------------------------
# Region separation and fractal dimension
# ---------------------------------------------------------------
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
# Data loading
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

def load_all():
    if os.path.exists(CACHE):
        d = np.load(CACHE, allow_pickle=True)
        print(f"  loaded cache: {d['X'].shape}")
        return d["X"], d["Y"], d["PID"], d["raw"], d["rawmask"]
    files = []
    for dp, _, fns in os.walk(DATA_ROOT):
        for fn in fns:
            if fn.lower().endswith(".mat"): files.append(os.path.join(dp, fn))
    files = sorted(files)
    print(f"Found {len(files)} .mat files")
    X, Y, PID, raw, rawmask = [], [], [], [], []
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
        f = features_of(g, m)
        if f is None: continue
        X.append(f); Y.append(lab-1); PID.append(pid)
        if len(raw) < PERTURB_N:
            raw.append(g); rawmask.append(m)
        if (k+1) % 500 == 0: print(f"  processed {k+1}/{len(files)}")
    X = np.array(X); Y = np.array(Y); PID = np.array(PID)
    raw = np.array(raw); rawmask = np.array(rawmask)
    np.savez_compressed(CACHE, X=X, Y=Y, PID=PID, raw=raw, rawmask=rawmask)
    print(f"  cached: {X.shape}")
    return X, Y, PID, raw, rawmask

# ---------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------
def evaluate(X, Y, PID, cols, patient_level=True, repeats=N_REPEATS):
    """Patient-level balanced accuracy, averaged over repeated CV."""
    Xs = X[:, cols]
    scores = []
    for r in range(repeats):
        if patient_level:
            splitter = GroupKFold(n_splits=5)
            folds = splitter.split(Xs, Y, groups=PID)
        else:
            splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=r)
            folds = splitter.split(Xs, Y)
        yt, yp = [], []
        for tr, te in folds:
            m = RandomForestClassifier(N_TREES, random_state=r, n_jobs=-1)
            m.fit(Xs[tr], Y[tr]); pred = m.predict(Xs[te])
            if patient_level:
                for pid in np.unique(PID[te]):
                    sel = PID[te] == pid
                    yt.append(int(np.bincount(Y[te][sel]).argmax()))
                    yp.append(int(np.bincount(pred[sel]).argmax()))
            else:
                yt.extend(Y[te]); yp.extend(pred)
        scores.append(balanced_accuracy_score(yt, yp))
        if patient_level: break   # GroupKFold is deterministic
    return float(np.mean(scores)), float(np.std(scores))

# ===============================================================
print("=" * 66); print("  LOADING AND FEATURE EXTRACTION"); print("=" * 66)
X, Y, PID, raw, rawmask = load_all()
print(f"  {len(X)} slices, {len(np.unique(PID))} patients, classes={np.bincount(Y)}")
idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
results = {}

# ---------------------------------------------------------------
print("\n" + "=" * 66)
print("  EXPERIMENT A — DOES THE RATIO ADD ANYTHING?")
print("=" * 66)
others = [idx[n] for n in FEATURE_NAMES if n not in ("Ft", "Fb", "C")]
sets = [
    ("Ft, Fb separate",              [idx["Ft"], idx["Fb"]]),
    ("C alone",                      [idx["C"]]),
    ("Ft, Fb, C together",           [idx["Ft"], idx["Fb"], idx["C"]]),
    ("14 others, no fractal",        others),
    ("14 others + Ft, Fb",           others + [idx["Ft"], idx["Fb"]]),
    ("14 others + C  (the thesis)",  others + [idx["C"]]),
    ("14 others + Ft, Fb, C",        others + [idx["Ft"], idx["Fb"], idx["C"]]),
]
tableA = {}
for name, cols in sets:
    m, _ = evaluate(X, Y, PID, cols)
    tableA[name] = m
    print(f"  {name:30s} {len(cols):3d} feat   {m:.4f}")
results["A_ablation"] = tableA

d1 = tableA["C alone"] - tableA["Ft, Fb separate"]
d2 = tableA["14 others + C  (the thesis)"] - tableA["14 others + Ft, Fb"]
print(f"\n  C alone  minus  Ft,Fb separate      : {d1:+.4f}")
print(f"  full-with-C  minus  full-with-Ft,Fb : {d2:+.4f}")
print()
if d2 > 0.005:
    print("  The ratio adds accuracy the classifier does not extract from the")
    print("  two components on its own. The descriptor earns its place.")
elif d2 < -0.005:
    print("  The two components separately do BETTER than the ratio. The ratio")
    print("  discards information. Report this and reframe the descriptor as an")
    print("  interpretable summary rather than a superior representation.")
else:
    print("  The ratio and the two components are equivalent for accuracy. The")
    print("  honest claim is then interpretability and compactness -- one named")
    print("  quantity instead of two -- not a gain in performance. State it that")
    print("  way; a reviewer will run this comparison if we do not.")

# ---------------------------------------------------------------
print("\n" + "=" * 66)
print("  EXPERIMENT B — IS THE NORMALISATION RATIONALE TRUE?")
print("=" * 66)
print("  Perturbing each slice in ways that imitate acquisition variation,")
print("  then comparing how much Ft moves against how much C moves.\n")

def perturb(g, kind, rng):
    if kind == "gain":     return np.clip(g * rng.uniform(0.7, 1.4), 0, 255)
    if kind == "gamma":    return np.clip(255*((g/255)**rng.uniform(0.7,1.4)), 0, 255)
    if kind == "noise":    return np.clip(g + rng.normal(0, 8, g.shape), 0, 255)
    if kind == "bias":
        H, W = g.shape; yy, xx = np.mgrid[0:H, 0:W]
        field = 1 + 0.3*((xx/W - 0.5)*rng.uniform(-1,1) + (yy/H - 0.5)*rng.uniform(-1,1))
        return np.clip(g*field, 0, 255)
    return g

rng = np.random.default_rng(0)
cv_ft, cv_fb, cv_c = [], [], []
n_use = min(len(raw), PERTURB_N)
for i in range(n_use):
    g, m = raw[i], rawmask[i]
    t, fr = regions_of(g, m)
    if t is None: continue
    fts, fbs, cs = [], [], []
    for kind in ["none", "gain", "gamma", "noise", "bias"]:
        gp = perturb(g, kind, rng)
        a = boxcount_fd(gp, t); b = boxcount_fd(gp, fr)
        if np.isfinite(a) and np.isfinite(b) and b > 1e-6:
            fts.append(a); fbs.append(b); cs.append(a/b)
    if len(cs) >= 4:
        cv_ft.append(np.std(fts)/abs(np.mean(fts)))
        cv_fb.append(np.std(fbs)/abs(np.mean(fbs)))
        cv_c.append(np.std(cs)/abs(np.mean(cs)))
    if (i+1) % 50 == 0: print(f"    perturbed {i+1}/{n_use}")

cv_ft, cv_fb, cv_c = map(np.array, (cv_ft, cv_fb, cv_c))
print(f"\n  slices used: {len(cv_c)}")
print(f"  coefficient of variation under perturbation (lower = steadier)")
print(f"    Ft : {cv_ft.mean():.4f}")
print(f"    Fb : {cv_fb.mean():.4f}")
print(f"    C  : {cv_c.mean():.4f}")
try:
    p_stab = stats.wilcoxon(cv_c, cv_ft).pvalue
except Exception:
    p_stab = 1.0
red = (1 - cv_c.mean()/cv_ft.mean()) * 100
print(f"\n  C is {red:+.1f}% as variable as Ft   (Wilcoxon p = {p_stab:.4g})")
results["B_stability"] = {"cv_Ft": cv_ft.mean(), "cv_Fb": cv_fb.mean(),
                          "cv_C": cv_c.mean(), "p": p_stab}
if red > 10 and p_stab < 0.05:
    print("  The normalisation does what we claimed: the ratio is materially")
    print("  steadier than the raw tumour dimension under acquisition-like")
    print("  perturbation. The rationale can stay, now with evidence.")
else:
    print("  The ratio is NOT meaningfully steadier than Ft. The stated")
    print("  rationale is not supported and must be withdrawn. Reframe C as a")
    print("  measure of tumour complexity relative to its host brain, which is")
    print("  a claim about meaning rather than about robustness.")

# ---------------------------------------------------------------
print("\n" + "=" * 66)
print("  EXPERIMENT C — HOW LARGE IS SLICE-LEVEL LEAKAGE HERE?")
print("=" * 66)
full = others + [idx["C"]]
pat, _ = evaluate(X, Y, PID, full, patient_level=True)
sli, sd = evaluate(X, Y, PID, full, patient_level=False)
print(f"  patient-grouped split : {pat:.4f}")
print(f"  slice-level split     : {sli:.4f} +/- {sd:.4f}")
print(f"  inflation             : {(sli-pat)*100:+.1f} points")
results["C_leakage"] = {"patient": pat, "slice": sli, "inflation_pp": (sli-pat)*100}
print("\n  Yagis et al. measured 29 to 55 points on neurodegenerative MRI.")
print("  The figure above is the equivalent for this dataset and this feature")
print("  set. It is what a published slice-level number on these data is worth")
print("  relative to a patient-level one.")

# ---------------------------------------------------------------
print("\n" + "=" * 66)
print("  EXPERIMENT D — DOES THE ESTIMATE DEPEND ON ARBITRARY CHOICES?")
print("=" * 66)
variants = {
    "sizes 2..128 (default)": [2,4,8,16,32,64,128],
    "sizes 2..64":            [2,4,8,16,32,64],
    "sizes 4..128":           [4,8,16,32,64,128],
    "sizes 2..32":            [2,4,8,16,32],
}
print("  effect of the box-size range on C (first 100 usable slices)\n")
base_c = None; tableD = {}
for name, sizes in variants.items():
    vals = []
    for i in range(min(100, len(raw))):
        t, fr = regions_of(raw[i], rawmask[i])
        if t is None: continue
        a = boxcount_fd(raw[i], t, sizes); b = boxcount_fd(raw[i], fr, sizes)
        if np.isfinite(a) and np.isfinite(b) and b > 1e-6: vals.append(a/b)
    vals = np.array(vals); tableD[name] = [vals.mean(), vals.std()]
    if base_c is None: base_c = vals
    n = min(len(vals), len(base_c))
    r = np.corrcoef(vals[:n], base_c[:n])[0,1] if n > 2 else float("nan")
    print(f"    {name:24s} mean C = {vals.mean():.4f} +/- {vals.std():.4f}"
          f"   corr with default = {r:.3f}")
results["D_robustness"] = tableD
print("\n  A high correlation across ranges means the descriptor ranks slices")
print("  consistently even when the estimator's window changes, which is what")
print("  matters for a feature. A shifting mean is expected and harmless.")

with open(RESULTS, "w") as f:
    json.dump(results, f, indent=2, default=float)
print(f"\n  results written to {RESULTS}")
print("\nDONE. Copy back all four experiment blocks.\n")
