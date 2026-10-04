"""
================================================================================
 CF-SCAFFOLD — STAGE 14B: DO THE FRACTAL DESCRIPTORS CARRY INFORMATION BEYOND
                          THE SIZE OF THE BRAIN SECTION?
================================================================================
 Why this exists
   Stage 14 found that slice geometry explains 84% of Fb's variance, and that
   removing it collapses Fb's pituitary AUC (0.933 -> 0.595). But geometry alone
   also predicts tumour type (0.750), and removing a class-informative variable
   strips genuine signal too: Ft, computed inside the tumour, fell as well
   (0.853 -> 0.602). Stage 14 therefore could not separate "Fb only measures
   the size of the brain section" from "Fb carries tumour information that is
   correlated with section size". The strongest geometric variable was brain
   area (pituitary AUC 0.879), which is also the variable box counting is known
   to be biased by.

 How it is tested
   No subtraction. Slices are grouped into narrow bins of brain area, and each
   descriptor is compared ONLY between tumour classes inside the same bin
   (a stratified Mann-Whitney AUC). Inside a bin, section size cannot explain a
   difference. 95% intervals resample patients, not slices.
   Controls: brain area itself must fall to about 0.5 (otherwise the matching
   failed); GLCM_t and Dispersion_t are tumour-internal, non-fractal comparisons.

 Decision rule for Fb, pituitary vs rest, 10 bins (fixed before running;
 20 bins reported as a sensitivity check)
   within-bin AUC >= 0.80 : Fb carries information beyond brain-section size
   within-bin AUC <= 0.65 : Fb's pituitary signal is mostly brain-section size
   otherwise              : partial
   The same thresholds are printed for Ft and C as descriptive results.
   Validated on two synthetic worlds in which brain area predicts the class as
   strongly as in the real data: size-only world 0.877 -> 0.521 (correct),
   size-plus-signal world 0.801 -> 0.933 (correct).
--------------------------------------------------------------------------------
 Kaggle: CPU. Reuses /kaggle/working/stage14_cache.npz if Stage 14 ran in the
 same session (about 3 minutes); otherwise re-extracts features (about 20).
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


def strata(area, n_bins):
    """Quantile bins of brain area; each slice gets a bin index."""
    edges = np.quantile(area, np.linspace(0, 1, n_bins + 1)); edges[-1] += 1e-12
    return np.clip(np.searchsorted(edges, area, side="right") - 1, 0, n_bins - 1)

MIN_EACH = 15   # minimum slices of EACH group inside a bin for that bin to count

def within_auc(v, yb, b, min_each=None):
    """AUC computed ONLY from positive/negative pairs that share an area bin, pooled over
       bins with weight n_pos*n_neg (equivalent to a stratified Mann-Whitney).
       Returns direction-free AUC and the number of usable bins."""
    min_each = MIN_EACH if min_each is None else min_each
    num = den = 0.0; used = 0
    for k in np.unique(b):
        m = b == k; y = yb[m]
        npos, nneg = int(y.sum()), int((1 - y).sum())
        if npos < min_each or nneg < min_each: continue
        a = roc_auc_score(y, v[m]); w = npos * nneg
        num += a * w; den += w; used += 1
    if den == 0: return np.nan, 0
    a = num / den
    return max(a, 1 - a), used

def within_auc_signed(v, yb, b, min_each=None):
    min_each = MIN_EACH if min_each is None else min_each
    num = den = 0.0
    for k in np.unique(b):
        m = b == k; y = yb[m]; npos, nneg = int(y.sum()), int((1 - y).sum())
        if npos < min_each or nneg < min_each: continue
        num += roc_auc_score(y, v[m]) * npos * nneg; den += npos * nneg
    return num / den if den else np.nan

def patient_bootstrap(v, yb, b, pid, n=500, seed=0):
    """95% CI of the within-bin AUC, resampling PATIENTS (slices of a patient are not independent)."""
    rng = np.random.default_rng(seed); up = np.unique(pid)
    idx_of = {p: np.where(pid == p)[0] for p in up}
    ref = within_auc_signed(v, yb, b); flip = ref < 0.5
    out = []
    for _ in range(n):
        idx = np.concatenate([idx_of[p] for p in rng.choice(up, len(up), replace=True)])
        a = within_auc_signed(v[idx], yb[idx], b[idx])
        if np.isfinite(a): out.append(1 - a if flip else a)
    return np.percentile(out, [2.5, 97.5]) if out else (np.nan, np.nan)

def balance(area, yb, b, min_each=None):
    """Mean standardized area difference (positives minus negatives) inside usable bins."""
    min_each = MIN_EACH if min_each is None else min_each
    sd = area.std(); diffs, ws = [], []
    for k in np.unique(b):
        m = b == k; y = yb[m]
        if y.sum() < min_each or (1 - y).sum() < min_each: continue
        diffs.append((area[m][y == 1].mean() - area[m][y == 0].mean()) / sd); ws.append(y.sum() * (1 - y).sum())
    return float(np.average(diffs, weights=ws)) if diffs else np.nan

def raw_auc(v, yb):
    a = roc_auc_score(yb, v); return max(a, 1 - a)

if __name__ == "__main__":
    print("=" * 74); print("  STAGE 14B — FRACTAL DESCRIPTORS WITHIN BINS OF BRAIN-SECTION AREA"); print("=" * 74)
    X, Gm, Y, PID = build()
    area = Gm[:, GEOM.index("brain_area")]
    col = {n: FEATURE_ORDER.index(n) for n in FEATURE_ORDER}
    cols = {"Fb": X[:, col["Fb"]], "Ft": X[:, col["Ft"]], "C": X[:, col["C"]],
            "GLCM_t": X[:, col["GLCM_t"]], "Dispersion_t": X[:, col["Dispersion_t"]], "brain_area (control)": area}
    print(f"  {len(Y)} slices, {len(set(PID))} patients")

    print("\n  How strongly each descriptor follows brain-section area (Pearson r):")
    for n in ("Fb", "Ft", "C"):
        r_all = np.corrcoef(cols[n], area)[0, 1]
        r_cls = [np.corrcoef(cols[n][Y == c], area[Y == c])[0, 1] for c in range(3)]
        print(f"    {n:3s} all {r_all:+.3f}   within class: " + "  ".join(f"{TYPE_NAMES[c]} {r:+.3f}" for c, r in zip(range(3), r_cls)))

    result = {}
    for c in range(3):
        yb = (Y == c).astype(int)
        print("\n" + "-" * 74); print(f"  {TYPE_NAMES[c].upper()} vs rest"); print("-" * 74)
        b10, b20 = strata(area, 10), strata(area, 20)
        fmt = lambda x: f"{x:+.3f}" if np.isfinite(x) else "n/a"
        print(f"  area imbalance inside bins (positives - negatives, in SD): "
              f"10 bins {fmt(balance(area, yb, b10))}, 20 bins {fmt(balance(area, yb, b20))}  (near 0 = matching worked)")
        print(f"  {'descriptor':22s} {'raw':>6s}   {'within 10 bins [95% CI, patients]':>36s}   {'within 20':>9s}   bins")
        for n, v in cols.items():
            a10, u10 = within_auc(v, yb, b10); a20, u20 = within_auc(v, yb, b20)
            result[(c, n)] = a10
            if np.isfinite(a10):
                lo, hi = patient_bootstrap(v, yb, b10, PID, n=500)
                ci = f"{a10:10.3f} [{lo:.3f}, {hi:.3f}]{'':12s}"
            else:
                ci = f"{'n/a (no bin has enough of both groups)':36s}"
            s20 = f"{a20:9.3f}" if np.isfinite(a20) else f"{'n/a':>9s}"
            print(f"  {n:22s} {raw_auc(v, yb):6.3f}   {ci} {s20}   {u10}/10, {u20}/20")

    print("\n" + "=" * 74); print("  VERDICT (rule fixed before running; pituitary vs rest, 10 bins)"); print("=" * 74)
    for n in ("Fb", "Ft", "C"):
        a = result[(2, n)]
        if not np.isfinite(a):
            print(f"  {n:3s} NOT COMPUTABLE: no area bin holds {MIN_EACH}+ slices of both groups"); continue
        v = ("carries information BEYOND brain-section size" if a >= 0.80 else
             "pituitary signal is MOSTLY brain-section size" if a <= 0.65 else "PARTIAL: part size, part beyond it")
        tag = "  (decision)" if n == "Fb" else "  (descriptive)"
        print(f"  {n:3s} within-bin AUC {a:.3f} -> {v}{tag}")
    ctrl = result[(2, "brain_area (control)")]
    if np.isfinite(ctrl): print(f"  control: brain area within its own bins {ctrl:.3f} "
          f"({'matching OK' if ctrl < 0.60 else 'WARNING: bins too coarse, matching incomplete'})")
    else: print("  control: not computable")
    print("\nDONE. Copy back everything above.\n")
