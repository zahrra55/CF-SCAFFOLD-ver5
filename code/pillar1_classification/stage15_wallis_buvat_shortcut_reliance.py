"""
================================================================================
 CF-SCAFFOLD — STAGE 15: HEAD-TO-HEAD WITH WALLIS & BUVAT (2022), AND DO THE
                         TUMOUR-REGION FEATURES RIDE THE SHORTCUT?
================================================================================
 Background
   Wallis & Buvat (Medical Image Analysis 77:102368, 2022) showed that tumour
   type in this dataset can be predicted without the tumour: 0.77 accuracy from
   three non-tissue features (number of zero-intensity pixels, maximum intensity,
   slice orientation), 0.74 from the skull outline. They attributed it to the
   selection of "representative" slices. Stage 14 of this project reached 0.750
   patient-level balanced accuracy from 11 brain-mask geometry features.

 What they did not test, and this stage does
   Whether a classifier built on TUMOUR-REGION features exploits that shortcut.
   They reported that tumour-ROI radiomics and whole-slice radiomics reach the
   same accuracy (0.86), which leaves open how much of the ROI result is itself
   shortcut. Here every test slice is placed in a stratum of near-identical
   shortcut information, and balanced accuracy is measured INSIDE the strata.
   Inside a stratum the shortcut cannot separate the classes, so whatever
   accuracy remains comes from information the shortcut does not carry.

 Parts
   1  Their two reproducible features, with their preprocessing (central crop
      450/512 of the field of view, rescale so the 99th percentile = 1000):
      zero-pixel count and maximum intensity. Slice orientation is NOT stored in
      the Figshare files and is not guessed; the comparison uses 2 of their 3.
   2  One protocol for every model: patient-grouped stratified 5-fold, 5 repeats,
      patient-level balanced accuracy (plus slice-level plain accuracy, the
      metric they reported).
   3  Shortcut reliance. Primary strata: k-means (k=20) on the 4 most
      informative shortcut variables (geometry + their two features), chosen by
      random-forest importance. Secondary strata: deciles of brain area.
      retention = (within-strata accuracy - 1/3) / (overall accuracy - 1/3).

 Decision rule for the tumour-only model, primary strata (fixed before running)
   retention >= 0.75 : tumour-region features are largely robust to the shortcut
   retention <= 0.50 : they largely ride the shortcut
   otherwise         : partial
 Validated on synthetic worlds with a two-dimensional shortcut: feature set that
 rides it -> 0.34 (correct); genuine signal -> 0.98 (correct). Area deciles alone
 gave 0.54 in the first world, which is why multi-dimensional strata are primary.
--------------------------------------------------------------------------------
 Kaggle: CPU. Attach the Figshare .mat dataset. About 30 minutes; features are
 cached in /kaggle/working/stage15_cache.npz.
================================================================================
"""

import os, glob, csv, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
for _pkg, _mod in {"h5py": "h5py", "scikit-image": "skimage"}.items():
    try: __import__(_mod)
    except Exception: os.system(f"pip install {_pkg} --quiet")
import h5py
from PIL import Image
from scipy import ndimage, stats
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import regionprops, label as sklabel
from skimage.segmentation import morphological_chan_vese
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

# =============================== CONFIGURATION ================================
DATA_GLOB         = "/kaggle/input/**/*.mat"
OUT_DIR           = "/kaggle/working"
IMG_SIZE          = 128
# Pillar I / segmentation
N_FOLDS           = 5
UNET_EPOCHS       = 20
# Pillar II
N_CLIENTS         = 18
PARTITION_ALPHA   = 0.3
SEEDS             = 20
TEST_SPLIT_SEED   = 42        # the held-out patients are the same for every seed
COLLAPSE_THRESH   = 0.40
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
TYPE_NAMES        = {0: "meningioma", 1: "glioma", 2: "pituitary"}
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
#  DATA
# =============================================================================
def _read_pid(cj):
    if "PID" not in cj: return None
    try:
        s = "".join(chr(int(c)) for c in np.array(cj["PID"]).ravel() if int(c) > 0)
        return s.strip() or None
    except Exception:
        return None

def read_cjdata(path):
    """One Figshare record (MATLAB v7.3 = HDF5): image, label 1-3, mask, patient id."""
    try:
        with h5py.File(path, "r") as f:
            cj = f["cjdata"]
            label = int(np.array(cj["label"]).ravel()[0])
            image = np.array(cj["image"]).astype(np.float32)
            if image.ndim == 2: image = image.T
            mask = None
            if "tumorMask" in cj:
                mask = np.array(cj["tumorMask"]).astype(np.uint8)
                if mask.ndim == 2: mask = mask.T
            return image, label, mask, _read_pid(cj)
    except Exception:
        return None, None, None, None

def resize_pair(img, mask, size=IMG_SIZE):
    """Slice-wise min-max to 0-255, then resize image (bilinear) and mask (nearest)."""
    im = Image.fromarray(((img - img.min()) / (img.max() - img.min() + 1e-8) * 255
                          ).astype(np.uint8)).resize((size, size), Image.BILINEAR)
    g = np.asarray(im).astype(np.float32)
    mk = Image.fromarray((mask * 255).astype(np.uint8)).resize((size, size), Image.NEAREST)
    return g, (np.asarray(mk) > 127).astype(np.uint8)

def brain_mask_of(gray):
    m = gray > max(gray.mean() * 0.25, 10)
    m = ndimage.binary_fill_holes(m); m = ndimage.binary_opening(m, iterations=2)
    lbl, n = ndimage.label(m)
    if n >= 1:
        s = np.bincount(lbl.ravel()); s[0] = 0; m = lbl == s.argmax()
    return m.astype(np.uint8)

# =============================================================================
#  DESCRIPTORS
# =============================================================================
def boxcount_fd(gray, region, n_sizes=8, G=256.0):
    """Differential box counting (Sarkar & Chaudhuri, 1994) on one region.
       The region's grey levels are stretched to its own 0-255 range, so the
       value describes the region's texture, not its brightness. Box height
       h = s*G/M. Only boxes that contain region pixels are counted.   [F3]"""
    v = gray[region > 0]
    if v.size < 64: return np.nan
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-6: return np.nan
    g = np.zeros_like(gray, dtype=np.float64)
    g[region > 0] = (gray[region > 0] - lo) / (hi - lo) * (G - 1)
    H, W = g.shape; M = min(H, W)
    sizes = [2 ** k for k in range(1, int(np.log2(M)))][:n_sizes]
    r = region > 0; Ns = []
    for s in sizes:
        h = s * G / M
        Hc, Wc = H - H % s, W - W % s
        B = g[:Hc, :Wc].reshape(Hc // s, s, Wc // s, s)
        R = r[:Hc, :Wc].reshape(Hc // s, s, Wc // s, s)
        valid = R.any(axis=(1, 3))
        mx = np.where(R, B, -np.inf).max(axis=(1, 3))
        mn = np.where(R, B, np.inf).min(axis=(1, 3))
        Ns.append(max(float((np.floor(mx[valid] / h) - np.floor(mn[valid] / h) + 1).sum()), 1.0))
    return float(np.polyfit(np.log(1.0 / np.array(sizes, float)), np.log(np.array(Ns)), 1)[0])

def glcm_region(gray, region, levels=8):
    """GLCM contrast and correlation from pixel pairs inside the region only. [F2]
       Quantised to `levels` bins between the region's 1st and 99th percentiles;
       background is code 0 and all pairs touching it are discarded."""
    v = gray[region > 0]
    if v.size < 16: return 0.0, 0.0
    lo, hi = np.percentile(v, [1, 99])
    if hi - lo < 1e-6: return 0.0, 0.0
    q = np.zeros(gray.shape, dtype=np.uint8)
    q[region > 0] = 1 + np.clip(((gray[region > 0] - lo) / (hi - lo) * levels).astype(int), 0, levels - 1)
    P = graycomatrix(q, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                     levels=levels + 1, symmetric=True, normed=False).astype(np.float64)[1:, 1:]
    tot = P.sum(axis=(0, 1), keepdims=True)
    if np.any(tot == 0): return 0.0, 0.0
    P = P / tot
    corr = graycoprops(P, "correlation")
    return float(graycoprops(P, "contrast").mean()), float(np.nan_to_num(corr).mean())

def shannon_entropy(vals, bins=32):
    """Shannon entropy in bits of the region's intensity histogram.   [F1]"""
    c, _ = np.histogram(vals, bins=bins)
    p = c[c > 0] / max(c.sum(), 1)
    return float(-(p * np.log2(p)).sum())

def lacunarity(mask, box=8):
    """Box lacunarity of the binary region (a shape property: gappiness of the mask)."""
    H, W = mask.shape
    counts = mask[:H - H % box, :W - W % box].reshape(H // box, box, W // box, box).sum(axis=(1, 3)).ravel().astype(float)
    if counts.mean() < 1e-6: return 0.0
    return float(1 + counts.var() / counts.mean() ** 2)

def shape_features(mask):
    props = regionprops(sklabel(mask))
    if not props: return dict(solidity=0.0, eccentricity=0.0, extent=0.0, circularity=0.0)
    p = max(props, key=lambda r: r.area); per = p.perimeter if p.perimeter > 0 else 1.0
    return dict(solidity=float(p.solidity), eccentricity=float(p.eccentricity),
                extent=float(p.extent), circularity=float(4 * np.pi * p.area / per ** 2))

def edge_features(gray, tumour):
    gx, gy = np.gradient(gray.astype(float)); grad = np.sqrt(gx ** 2 + gy ** 2)
    border = ndimage.binary_dilation(tumour) & ~ndimage.binary_erosion(tumour)
    if tumour.sum() == 0: return dict(edge_density=0.0, mean_grad=0.0, border_grad=0.0)
    return dict(edge_density=float((grad[tumour > 0] > grad.mean()).mean()),
                mean_grad=float(grad[tumour > 0].mean()),
                border_grad=float(grad[border > 0].mean()) if border.sum() else 0.0)

FEATURE_ORDER = ["Ft", "Fb", "C",                                              # fractal family [F4]
                 "GLCM_t", "correlation_t", "GLCM_b", "Dispersion_t",           # texture, entropy
                 "lacun_t", "lacun_b", "size",                                  # lacunarity, size
                 "solidity", "eccentricity", "extent", "circularity",           # shape
                 "edge_density", "mean_grad", "border_grad"]                    # edge

def features_from(g, tumour_mask, brain):
    """17 descriptors for one slice, given a tumour delineation. None if unusable."""
    tumour = ((tumour_mask > 0) & (brain > 0)).astype(np.uint8)
    rest = ((brain > 0) & (tumour == 0)).astype(np.uint8)
    if tumour.sum() < 16 or rest.sum() < 64: return None
    Ft, Fb = boxcount_fd(g, tumour), boxcount_fd(g, rest)
    if not (np.isfinite(Ft) and np.isfinite(Fb) and Fb > 1e-6): return None
    ct, rt = glcm_region(g, tumour); cb, _ = glcm_region(g, rest)
    f = dict(Ft=Ft, Fb=Fb, C=Ft / Fb, GLCM_t=ct, correlation_t=rt, GLCM_b=cb,
             Dispersion_t=shannon_entropy(g[tumour > 0]),
             lacun_t=lacunarity(tumour), lacun_b=lacunarity(rest), size=float(tumour.sum()))
    f.update(shape_features(tumour)); f.update(edge_features(g, tumour))
    return f

def load_all():
    files = sorted(f for f in glob.glob(DATA_GLOB, recursive=True) if f.lower().endswith(".mat"))
    if not files:
        raise SystemExit("No .mat files under /kaggle/input. Attach the Figshare collection as a DATASET.")
    print(f"Found {len(files)} .mat files")
    feats, Y, PID, G, M, B = [], [], [], [], [], []
    for k, path in enumerate(files):
        img, label, mask, pid = read_cjdata(path)
        if img is None or pid is None or mask is None: continue
        g, mk = resize_pair(img, mask)
        brain = brain_mask_of(g)
        f = features_from(g, mk, brain)
        if f is None: continue
        feats.append(f); Y.append(label - 1); PID.append(pid)
        G.append(g.astype(np.uint8)); M.append(mk); B.append(brain)
        if (k + 1) % 500 == 0: print(f"  processed {k + 1}/{len(files)}")
    Y, PID = np.array(Y), np.array(PID)
    print(f"  usable slices: {len(feats)} of {len(files)} ({100 * len(feats) / len(files):.1f}%), "
          f"patients: {len(set(PID))}, class counts: {np.bincount(Y)}")
    return feats, Y, PID, np.stack(G), np.stack(M), np.stack(B)

def to_matrix(feats):
    return np.array([[f[k] for k in FEATURE_ORDER] for f in feats], dtype=np.float64)

# =============================================================================
#  CLASSIFIER AND PATIENT-LEVEL VOTE
# =============================================================================
def make_search(X_tr, Y_tr, P_tr):
    pipe = Pipeline([("sc", StandardScaler()),
                     ("rf", RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1))])
    grid = {"rf__max_depth": [None, 12], "rf__min_samples_leaf": [1, 2]}
    gs = GridSearchCV(pipe, grid, cv=list(GroupKFold(3).split(X_tr, Y_tr, P_tr)),
                      scoring="balanced_accuracy", n_jobs=-1)
    return gs.fit(X_tr, Y_tr)

def patient_vote(pred, proba, pids):
    """Majority vote per patient; ties broken by mean class probability. [F5]"""
    out = {}
    for pid in np.unique(pids):
        m = pids == pid
        cnt = np.bincount(pred[m], minlength=3)
        win = np.flatnonzero(cnt == cnt.max())
        out[pid] = int(win[0]) if len(win) == 1 else int(win[np.argmax(proba[m][:, win].mean(0))])
    return out

def patient_truth(Y, pids):
    return {pid: int(np.bincount(Y[pids == pid]).argmax()) for pid in np.unique(pids)}

from sklearn.ensemble import RandomForestRegressor
GEOM = ["brain_area", "bbox_aspect", "eccentricity_b", "solidity_b", "extent_b", "orientation_b",
        "centroid_y", "centroid_x", "lower_third", "mask_symmetry", "intensity_symmetry"]

def geometry(g, brain):
    """Slice geometry from the brain mask only (tumour included in the mask, never isolated)."""
    H, W = brain.shape
    p = max(regionprops(sklabel(brain)), key=lambda r: r.area)
    y0, x0, y1, x1 = p.bbox
    cy, cx = p.centroid
    c = int(round(cx)); half = min(c, W - c)
    if half < 4:
        msym = isym = 0.0
    else:
        L = brain[:, c - half:c].astype(float); R = brain[:, c:c + half][:, ::-1].astype(float)
        msym = float((L * R).sum() / max(np.maximum(L, R).sum(), 1))
        gl = g[:, c - half:c]; gr = g[:, c:c + half][:, ::-1]; m = (L * R) > 0
        isym = float(np.corrcoef(gl[m], gr[m])[0, 1]) if m.sum() > 50 else 0.0
    rows = np.where(brain.any(axis=1))[0]
    third = rows.min() + 2 * (rows.max() - rows.min()) // 3
    return dict(brain_area=p.area / (H * W), bbox_aspect=(y1 - y0) / max(x1 - x0, 1),
                eccentricity_b=p.eccentricity, solidity_b=p.solidity, extent_b=p.extent,
                orientation_b=abs(p.orientation), centroid_y=cy / H, centroid_x=cx / W,
                lower_third=float(brain[third:].sum() / brain.sum()),
                mask_symmetry=msym, intensity_symmetry=np.nan_to_num(isym))


from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
CACHE = f"{OUT_DIR}/stage15_cache.npz"
N_REP = 5
TUMOUR = ["Ft", "GLCM_t", "correlation_t", "Dispersion_t", "lacun_t", "size",
          "solidity", "eccentricity", "extent", "circularity", "edge_density", "mean_grad", "border_grad"]
BRAIN = ["Fb", "GLCM_b", "lacun_b"]
WB = ["wb_zero_pixels", "wb_max_intensity"]

def wallis_buvat(img):
    """Their preprocessing: central crop (450/512 of the field of view), rescale so that the
       99th percentile maps to 1000. Features: zero-intensity pixel count, maximum intensity."""
    H, W = img.shape; ch, cw = int(round(H * 450 / 512)), int(round(W * 450 / 512))
    y0, x0 = (H - ch) // 2, (W - cw) // 2
    c = img[y0:y0 + ch, x0:x0 + cw].astype(np.float64)
    p99 = np.percentile(c, 99)
    r = c / p99 * 1000.0 if p99 > 0 else c
    zero = float((c == 0).sum()) / c.size          # fraction, so 256- and 512-pixel images are comparable
    return zero, float(r.max())

def build():
    if os.path.exists(CACHE):
        d = np.load(CACHE, allow_pickle=True); print(f"  loaded cache {d['X'].shape}")
        return d["X"], d["Gm"], d["Wb"], d["Y"], d["PID"]
    files = sorted(f for f in glob.glob(DATA_GLOB, recursive=True) if f.lower().endswith(".mat"))
    if not files: raise SystemExit("No .mat files under /kaggle/input. Attach the Figshare collection as a DATASET.")
    X, Gm, Wb, Y, PID = [], [], [], [], []
    for k, path in enumerate(files):
        img, label, mask, pid = read_cjdata(path)
        if img is None or pid is None or mask is None: continue
        g, mk = resize_pair(img, mask); brain = brain_mask_of(g)
        f = features_from(g, mk, brain)
        if f is None: continue
        X.append([f[n] for n in FEATURE_ORDER]); Gm.append([geometry(g, brain)[n] for n in GEOM])
        Wb.append(wallis_buvat(img)); Y.append(label - 1); PID.append(pid)
        if (k + 1) % 500 == 0: print(f"  processed {k + 1}/{len(files)}")
    X, Gm, Wb, Y, PID = map(np.array, (X, Gm, Wb, Y, PID))
    np.savez_compressed(CACHE, X=X, Gm=Gm, Wb=Wb, Y=Y, PID=PID)
    print(f"  usable slices {len(Y)}, patients {len(set(PID))}, classes {np.bincount(Y)}")
    return X, Gm, Wb, Y, PID

def rf(): return Pipeline([("sc", StandardScaler()), ("m", RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1))])
def svm(): return Pipeline([("sc", StandardScaler()), ("m", SVC(kernel="linear", class_weight="balanced", probability=True, random_state=0))])

def evaluate(Xs, Y, PID, make, seed):
    """Out-of-fold slice predictions -> (patient balanced acc, slice plain acc, slice predictions)."""
    pred = np.zeros_like(Y); proba = np.zeros((len(Y), 3))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(Xs, Y, PID):
        m = make().fit(Xs[tr], Y[tr]); pred[te] = m.predict(Xs[te]); proba[te] = m.predict_proba(Xs[te])
    pv, pt = patient_vote(pred, proba, PID), patient_truth(Y, PID); k = sorted(pv)
    return balanced_accuracy_score([pt[p] for p in k], [pv[p] for p in k]), accuracy_score(Y, pred), pred

def stratified_bacc(y, pred, strata, min_per_class=5):
    vals, ws = [], []
    for s in np.unique(strata):
        m = strata == s
        if np.bincount(y[m], minlength=3).min() < min_per_class: continue
        vals.append(balanced_accuracy_score(y[m], pred[m])); ws.append(m.sum())
    return (float(np.average(vals, weights=ws)), len(vals), int(sum(ws))) if vals else (np.nan, 0, 0)

def retention(overall, within):
    """Undefined when the model is itself near chance (denominator ~ 0)."""
    return (within - 1 / 3) / (overall - 1 / 3) if overall >= 0.40 and np.isfinite(within) else np.nan
def fr(x): return f"{x:+10.2f}" if np.isfinite(x) else f"{'n/a':>10s}"

def boot_retention(y, pred, strata, pid, n=500, seed=0):
    rng = np.random.default_rng(seed); up = np.unique(pid); idx_of = {p: np.where(pid == p)[0] for p in up}; out = []
    for _ in range(n):
        idx = np.concatenate([idx_of[p] for p in rng.choice(up, len(up), replace=True)])
        ov = balanced_accuracy_score(y[idx], pred[idx]); w, _, _ = stratified_bacc(y[idx], pred[idx], strata[idx])
        if np.isfinite(w): out.append(retention(ov, w))
    return np.percentile(out, [2.5, 97.5]) if out else (np.nan, np.nan)

if __name__ == "__main__":
    print("=" * 76); print("  STAGE 15 — WALLIS & BUVAT HEAD-TO-HEAD, AND SHORTCUT RELIANCE"); print("=" * 76)
    X, Gm, Wb, Y, PID = build()
    col = {n: i for i, n in enumerate(FEATURE_ORDER)}
    T = X[:, [col[n] for n in TUMOUR]]; Bf = X[:, [col[n] for n in BRAIN]]
    S = np.hstack([Gm, Wb]); S_names = GEOM + WB
    print(f"  {len(Y)} slices, {len(set(PID))} patients;  zero-pixel fraction: median {np.median(Wb[:, 0]):.3f}, "
          f"slices with no exact zeros {int((Wb[:, 0] == 0).sum())}")

    print("\n" + "-" * 76); print(f"  PART 1-2 — ONE PROTOCOL FOR EVERY MODEL ({N_REP} repeats)"); print("-" * 76)
    models = [("Wallis&Buvat 2 features, linear SVM (their classifier)", Wb, svm),
              ("Wallis&Buvat 2 features, random forest", Wb, rf),
              ("brain-mask geometry, 11 features (Stage 14)", Gm, rf),
              ("all shortcut variables (geometry + W&B)", S, rf),
              ("tumour-region features only (13)", T, rf),
              ("brain-region features only (Fb, GLCM_b, lacun_b)", Bf, rf),
              ("full model, 17 features", X, rf),
              ("full model + all shortcut variables", np.hstack([X, S]), rf)]
    print(f"  {'model':58s} {'patient bal.acc':>16s} {'slice acc':>10s}")
    oof = {}
    for name, Xs, make in models:
        res = [evaluate(Xs, Y, PID, make, s) for s in range(N_REP)]
        pa = np.array([r[0] for r in res]); sa = np.array([r[1] for r in res]); oof[name] = res[0][2]
        print(f"  {name:58s} {pa.mean():.3f} +/- {pa.std():.3f}  {sa.mean():8.3f}")
    print("  Wallis & Buvat reported 0.77 (slice accuracy) with 3 features including slice orientation,")
    print("  which is not available here; their linear SVM row above uses the 2 reproducible features.")

    print("\n" + "-" * 76); print("  PART 3 — DOES EACH MODEL RIDE THE SHORTCUT? (accuracy inside shortcut strata)"); print("-" * 76)
    imp = RandomForestClassifier(300, random_state=0, n_jobs=-1).fit(S, Y).feature_importances_
    top = np.argsort(imp)[::-1][:4]
    print("  shortcut variables used for the primary strata: " + ", ".join(S_names[i] for i in top))
    primary = KMeans(20, n_init=10, random_state=0).fit_predict(StandardScaler().fit_transform(S[:, top]))
    area = Gm[:, GEOM.index("brain_area")]
    secondary = np.digitize(area, np.quantile(area, np.linspace(0, 1, 11)[1:-1]))
    print(f"  {'model':52s} {'overall':>8s} {'within k-means-20':>18s} {'retention':>10s} {'within area deciles':>20s} {'retention':>10s}")
    keep = {}
    for name, _, _ in models:
        pr = oof[name]; ov = balanced_accuracy_score(Y, pr)
        w1, n1, c1 = stratified_bacc(Y, pr, primary); w2, n2, c2 = stratified_bacc(Y, pr, secondary)
        keep[name] = (ov, w1, retention(ov, w1), w2, retention(ov, w2))
        print(f"  {name[:52]:52s} {ov:8.3f} {w1:18.3f} {fr(retention(ov, w1))} {w2:20.3f} {fr(retention(ov, w2))}")
    print(f"  (slice-level balanced accuracy; primary strata used: {n1} of 20 covering {c1} slices; "
          f"area deciles: {n2} of 10 covering {c2} slices)")
    print("  retention is n/a when a model is itself near chance (overall < 0.40).")
    if c1 < 0.5 * len(Y):
        print(f"  WARNING: the primary strata cover only {c1}/{len(Y)} slices; the verdict rests on that subset.")

    print("\n" + "=" * 76); print("  VERDICT (rule fixed before running; tumour-only model, primary strata)"); print("=" * 76)
    tname = "tumour-region features only (13)"; ov, w1, r1, w2, r2 = keep[tname]
    if not np.isfinite(r1):
        print("  NOT COMPUTABLE (tumour-only model near chance or no usable strata)."); print("\nDONE.\n"); raise SystemExit
    lo, hi = boot_retention(Y, oof[tname], primary, PID)
    print(f"  tumour-only: overall {ov:.3f}, within shortcut strata {w1:.3f}, retention {r1:.2f} [95% CI {lo:.2f}, {hi:.2f}]")
    if r1 >= 0.75:
        print("  ROBUST. Tumour-region features keep most of their accuracy when the shortcut is held fixed.")
        print("  New relative to Wallis & Buvat: an interpretable tumour-region classifier on this dataset is")
        print("  largely not exploiting the documented slice-selection shortcut.")
    elif r1 <= 0.50:
        print("  RIDES THE SHORTCUT. Much of the tumour-region accuracy disappears inside shortcut strata;")
        print("  tumour features themselves carry slice-selection information (e.g. lesion size per slice).")
    else:
        print("  PARTIAL. Part of the tumour-region accuracy is shortcut-independent, part is not.")
    for name in ("full model, 17 features", "brain-region features only (Fb, GLCM_b, lacun_b)"):
        ov, w1, r1, _, _ = keep[name]; print(f"  descriptive: {name}: retention {fr(r1).strip()}")
    print("\nDONE. Copy back everything above.\n")
