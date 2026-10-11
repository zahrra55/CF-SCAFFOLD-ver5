"""
================================================================================
 CF-SCAFFOLD — STAGE 14: IS Fb MEASURING THE TUMOUR TYPE, OR THE SLICE?
================================================================================
 Why this exists
   In the final pipeline Fb, the fractal dimension of the brain WITHOUT the
   tumour, is the strongest single descriptor (pituitary AUC 0.933, above Ft).
   It is also why classification survives automatic masks: Fb barely changes
   with the mask (r = 0.997). A property of healthy tissue should not identify
   the tumour type. One explanation: the Figshare collection contains axial,
   coronal and sagittal slices, and pituitary tumours are imaged around the
   sella, so Fb may encode the imaging plane and anatomical level rather than
   anything about the tumour.

 How it is tested
   The dataset has no plane label, so the slice's geometry is described from
   the brain mask and its left-right symmetry: area, bounding-box aspect,
   eccentricity, solidity, extent, orientation, centroid, share of brain in the
   lower third, mask symmetry and intensity symmetry. None of these looks at
   the tumour region.
     A  Can geometry alone predict the tumour type? How much of Fb does it explain?
     B  Remove from Fb the part geometry predicts (out-of-fold, no leakage) and
        measure what discrimination remains. Same for Ft and C as controls.
     D  Group slices into three geometry clusters (a proxy for plane) and
        measure Fb's discrimination WITHIN each cluster.
     E  Patient-level accuracy without Fb, and without all brain-region features.

 Decision rule (fixed before running)
   CONFOUND SUPPORTED if the pituitary AUC of Fb drops by >= 0.10 after removing
     geometry AND geometry alone reaches >= 0.60 patient-level balanced accuracy.
   CONFOUND REJECTED  if residual Fb keeps a pituitary AUC >= 0.85.
   Otherwise INCONCLUSIVE.
   Geometry is a proxy for plane, not a label; a positive result means Fb is
   entangled with slice geometry, not that plane has been measured directly.
--------------------------------------------------------------------------------
 Kaggle: CPU is enough. Attach the Figshare .mat dataset. About 25 minutes
 (features are cached in /kaggle/working/stage14_cache.npz after the first run).
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

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
CACHE = f"{OUT_DIR}/stage14_cache.npz"
N_REP = 5
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

def build():
    if os.path.exists(CACHE):
        d = np.load(CACHE, allow_pickle=True); print(f"  loaded cache {d['X'].shape}")
        return d["X"], d["Gm"], d["Y"], d["PID"]
    feats, Y, PID, G, M, B = load_all()
    X = to_matrix(feats)
    Gm = np.array([[geometry(G[i].astype(np.float32), B[i])[k] for k in GEOM] for i in range(len(Y))])
    np.savez_compressed(CACHE, X=X, Gm=Gm, Y=Y, PID=PID)
    return X, Gm, Y, PID

def dauc(v, y):
    a = roc_auc_score(y, v); return max(a, 1 - a)

def rf():
    return Pipeline([("sc", StandardScaler()),
                     ("rf", RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1))])

def patient_acc(Xs, Y, PID, seed):
    """Patient-level balanced accuracy, fixed hyper-parameters, patient-grouped stratified 5-fold."""
    pv, pt = {}, patient_truth(Y, PID)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(Xs, Y, PID):
        m = rf().fit(Xs[tr], Y[tr]); pv.update(patient_vote(m.predict(Xs[te]), m.predict_proba(Xs[te]), PID[te]))
    k = sorted(pv); return balanced_accuracy_score([pt[p] for p in k], [pv[p] for p in k])

def oof_residual(v, Gm, Y, PID):
    """v minus its out-of-fold prediction from geometry (the model never sees the slice it predicts)."""
    pred = np.zeros_like(v)
    for tr, te in GroupKFold(5).split(Gm, Y, PID):
        reg = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=0, n_jobs=-1).fit(Gm[tr], v[tr])
        pred[te] = reg.predict(Gm[te])
    r2 = 1 - ((v - pred) ** 2).sum() / ((v - v.mean()) ** 2).sum()
    return v - pred, r2

if __name__ == "__main__":
    print("=" * 70); print("  STAGE 14 — IS Fb MEASURING THE TUMOUR TYPE, OR THE SLICE?"); print("=" * 70)
    X, Gm, Y, PID = build()
    print(f"  {len(Y)} slices, {len(set(PID))} patients")
    col = {n: FEATURE_ORDER.index(n) for n in FEATURE_ORDER}

    print("\n" + "-" * 70); print("  A — GEOMETRY ALONE"); print("-" * 70)
    ga = np.array([patient_acc(Gm, Y, PID, s) for s in range(N_REP)])
    print(f"  patient-level balanced accuracy from geometry only: {ga.mean():.3f} +/- {ga.std():.3f}  (chance 0.333)")
    print("  one-vs-rest AUC of each geometry feature (direction-free):")
    print(f"  {'':20s}" + "".join(f"{TYPE_NAMES[c]:>12s}" for c in range(3)))
    for j, n in enumerate(GEOM):
        print(f"  {n:20s}" + "".join(f"{dauc(Gm[:, j], (Y == c).astype(int)):12.3f}" for c in range(3)))

    print("\n" + "-" * 70); print("  B/C — DISCRIMINATION LEFT AFTER REMOVING GEOMETRY"); print("-" * 70)
    print(f"  {'descriptor':10s} {'R2 from geometry':>17s}   AUC raw -> residual  (meningioma | glioma | pituitary)")
    resid = {}
    for n in ("Fb", "Ft", "C"):
        v = X[:, col[n]]; r, r2 = oof_residual(v, Gm, Y, PID); resid[n] = r
        raw = [dauc(v, (Y == c).astype(int)) for c in range(3)]
        res = [dauc(r, (Y == c).astype(int)) for c in range(3)]
        print(f"  {n:10s} {r2:17.3f}   " + " | ".join(f"{a:.3f}->{b:.3f}" for a, b in zip(raw, res)))

    print("\n" + "-" * 70); print("  D — WITHIN GEOMETRY CLUSTERS (proxy for imaging plane)"); print("-" * 70)
    Z = StandardScaler().fit_transform(Gm)
    cl = KMeans(3, n_init=20, random_state=0).fit_predict(Z)
    print(f"  {'cluster':8s} {'slices':>7s} " + "".join(f"{TYPE_NAMES[c]:>12s}" for c in range(3)) + "   Fb pituitary-vs-rest AUC")
    for k in range(3):
        m = cl == k; cnt = np.bincount(Y[m], minlength=3)
        yb = (Y[m] == 2).astype(int)
        auc = f"{dauc(X[m, col['Fb']], yb):.3f}" if 20 <= yb.sum() <= len(yb) - 20 else "n/a (too few of one group)"
        print(f"  {k:<8d} {m.sum():7d} " + "".join(f"{x:12d}" for x in cnt) + f"   {auc}")
    print("  If pituitary slices pile into one cluster, cluster membership alone separates them;")
    print("  the within-cluster AUC shows what Fb adds once that is held fixed.")

    print("\n" + "-" * 70); print(f"  E — CLASSIFIER WITHOUT BRAIN-REGION FEATURES ({N_REP} repeated splits)"); print("-" * 70)
    brain_feats = ["Fb", "GLCM_b", "lacun_b"]
    sets = {"all 17": FEATURE_ORDER,
            "without Fb": [f for f in FEATURE_ORDER if f != "Fb"],
            "without brain-region (Fb, GLCM_b, lacun_b)": [f for f in FEATURE_ORDER if f not in brain_feats],
            "all 17 + residual Fb instead of Fb": None}
    acc = {}
    for name, fs in sets.items():
        if fs is None:
            Xs = X.copy(); Xs[:, col["Fb"]] = resid["Fb"]
        else:
            Xs = X[:, [col[f] for f in fs]]
        acc[name] = np.array([patient_acc(Xs, Y, PID, s) for s in range(N_REP)])
        print(f"  {name:44s} {acc[name].mean():.3f} +/- {acc[name].std():.3f}")
    ref = acc["all 17"]
    for name in list(sets)[1:]:
        d = acc[name] - ref
        try: p = stats.wilcoxon(acc[name], ref).pvalue
        except Exception: p = float("nan")
        print(f"    {name:42s} {100 * d.mean():+.1f} pp vs all 17  (Wilcoxon p={p:.3f}, n={N_REP}; low power, read the size)")

    print("\n" + "=" * 70); print("  VERDICT (rule fixed before running)"); print("=" * 70)
    raw_p = dauc(X[:, col["Fb"]], (Y == 2).astype(int)); res_p = dauc(resid["Fb"], (Y == 2).astype(int))
    drop = raw_p - res_p
    print(f"  Fb pituitary AUC: raw {raw_p:.3f} -> residual {res_p:.3f} (drop {drop:+.3f}); geometry-only accuracy {ga.mean():.3f}")
    if drop >= 0.10 and ga.mean() >= 0.60:
        print("  CONFOUND SUPPORTED. Fb's discrimination is largely shared with slice geometry.")
        print("  Write Fb as a descriptor entangled with imaging plane/anatomical level, report the")
        print("  residual AUC as its geometry-independent part, and state that the robustness of")
        print("  classification to automatic masks rests partly on this entanglement.")
    elif res_p >= 0.85:
        print("  CONFOUND REJECTED. Fb keeps its discrimination after geometry is removed; it is not")
        print("  a proxy for slice plane as measured here. Report this test as evidence for Fb.")
    else:
        print("  INCONCLUSIVE. Geometry explains part of Fb but not enough to meet either rule.")
        print("  Report both AUCs and treat Fb's pituitary signal as partly geometric.")
    print("\nDONE. Copy back everything above.\n")
