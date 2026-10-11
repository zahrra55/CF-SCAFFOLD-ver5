"""
================================================================================
 CF-SCAFFOLD — STAGE 19: FINAL PREREGISTERED EXTENSION OF THE DESCRIPTORS
================================================================================
 Thesis title: "Fractal-Chaos Guided Federated Learning for Brain Tumor MRI Classification"
 This is the last experimental stage. All decision rules below were fixed before running.
 Version 1.1 (after the code audit): patient-level votes in the federated arm now break ties with model
 probabilities (version 1.0 broke them towards the lowest class index), and missing values in the federated arm
 are imputed with the mean of observed values (sum / count, available from aggregated client statistics)
 instead of a pooled median. Only the Part 4 figures are affected; rerun with RUN_P1..P3 = False.

 New descriptors (each validated on synthetic cases with known answers before use)
   Fbound       box-counting fractal dimension of the tumour boundary, over box sizes from 2 px to half the
                equivalent diameter (1.0 for a smooth outline, ~1.3 for a rough one, at any size).
   GLac2_t,     gliding-box lacunarity of the INTENSITIES inside the tumour (the existing lacunarity is computed
   GLacSlope_t  on the binary mask): Lambda at box size 2 and the slope over sizes 2-8. Scale invariant.
   FuzEn_t/_b   two-dimensional fuzzy entropy (Hilal et al., 2020), m = 2, r = 0.2 std, n = 2; always defined.
   BraTS only:  fractal dimension and grayscale lacunarity of the enhancing region (label 4) and of necrosis
                (label 1), with explicit presence flags (0 when a sub-region is absent).
   Rejected during validation: roughness exponent and Higuchi dimension of the radial boundary signature
   (dominated by pixel quantisation; results reversed with tumour size).

 Parts and rules
   1  Figshare, 3 classes, 10 repeated patient-grouped splits, 218 patients (the final-pipeline set; missing
      values imputed inside each training fold).
      Block A (Fbound) and Block B (grayscale lacunarity) are each added to the 19 Stage-17 descriptors.
      RULE: a block is accepted if the gain is positive, the paired Wilcoxon p-value after Holm correction over
      the two blocks is < 0.05, and it wins in >= 7 of 10 splits.
      FuzEn2D vs SampEn2D. RULE: FuzEn2D replaces SampEn2D if it (a) is defined on at least as many slices,
      (b) is non-inferior in accuracy (mean difference >= -0.5 points), and (c) is more stable (higher
      test-retest correlation under a change of sampling seed and under small added noise).
      A tumour-only configuration (no descriptor of the brain outside the tumour) is always reported.
   2  Figshare, glioma vs non-glioma: a PROXY for malignant vs predominantly benign tumours (meningiomas and
      pituitary adenomas are mostly benign; gliomas are malignant; atypical meningiomas and low-grade gliomas
      exist). All descriptors, with radiologist masks and AFTER AUTOMATIC SEGMENTATION (U-Net localiser trained
      per fold on training patients only). Brain-mask geometry alone is the shortcut control.
      RULE: patient-level balanced accuracy with automatic masks >= 0.85 AND at least 0.10 above the control.
   3  BraTS: high-grade vs low-grade glioma (GRADE, not benign vs malignant: low-grade gliomas are malignant
      WHO grade 2 tumours). Decision threshold chosen inside each training fold (preregistered here).
      RULE: adding the sub-region block to the 19 descriptors raises balanced accuracy in >= 7 of 10 repeats
      with paired Wilcoxon p < 0.05. A presence-flags-only model is the control that tells whether the gain
      comes from fractal structure or merely from whether enhancement/necrosis is present.
   4  Federated arm (descriptive): Stage 18 design at alpha 0.3, 20 seeds, final descriptor set vs the
      Stage-17 set, slice- and patient-level balanced accuracy.
--------------------------------------------------------------------------------
 Kaggle: GPU (for the U-Net in Part 2). Attach Figshare and BraTS 2020. About 2 hours; features are cached.
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

from numpy.lib.stride_tricks import sliding_window_view
from sklearn.model_selection import StratifiedGroupKFold
RUN_A, RUN_B, RUN_C = True, True, True
N_REP          = 5
FED_SEEDS      = 20
FED_ROUNDS     = 60
N_CLIENTS      = 18
PARTITION_ALPHA = 0.3
TEST_SPLIT_SEED = 42
BRATS_GLOB     = "/kaggle/input/**/*t1ce.nii*"
BRATS_SLICES   = 5          # largest-tumour axial slices per patient
BRATS_MIN_AREA = 100        # tumour pixels at native resolution
CACHE_FIG, CACHE_BRATS = f"{OUT_DIR}/stage17_figshare.npz", f"{OUT_DIR}/stage17_brats.npz"

# ------------------------------------------------------------------ SampEn2D
def sampen2d(gray, region, m=1, r_factor=0.2, n_templates=1000, n_candidates=4000, seed=0):
    """Two-dimensional sample entropy (Silva et al., 2016) restricted to a region.
       Templates are m x m patches whose (m+1) x (m+1) extension lies inside the region;
       a match is a Chebyshev distance <= r, r = r_factor * std of the region; B counts
       matching m x m pairs, A the pairs still matching at (m+1) x (m+1); self-matches
       excluded; SampEn2D = -ln(A/B). A fixed seeded subset of positions is used
       (validated against the exact statistic: 4.016 vs 4.031 on a test texture)."""
    g = gray.astype(np.float64); r_ = region > 0; v = g[r_]
    if v.size < 50 or v.std() < 1e-9: return np.nan
    r = r_factor * v.std()
    W1 = sliding_window_view(g, (m + 1, m + 1))
    pos = np.argwhere(sliding_window_view(r_, (m + 1, m + 1)).all(axis=(2, 3)))
    if len(pos) < 20: return np.nan
    rng = np.random.default_rng(seed)
    ti = rng.choice(len(pos), min(n_templates, len(pos)), replace=False)
    ci = rng.choice(len(pos), min(n_candidates, len(pos)), replace=False)
    win = W1[pos[:, 0], pos[:, 1]]
    P1 = win.reshape(len(pos), -1); Pm = win[:, :m, :m].reshape(len(pos), -1)
    A = B = 0
    for s in range(0, len(ti), 100):
        t = ti[s:s + 100]
        dm = np.abs(Pm[t][:, None, :] - Pm[ci][None, :, :]).max(axis=2)
        d1 = np.maximum(dm, np.abs(P1[t][:, None, :] - P1[ci][None, :, :]).max(axis=2))
        ok = t[:, None] != ci[None, :]
        B += int(((dm <= r) & ok).sum()); A += int(((d1 <= r) & ok).sum())
    return float(-np.log(A / B)) if A > 0 and B > 0 else np.nan

FC_FAMILY = ["Ft", "Fb", "C", "lacun_t", "lacun_b", "SampEn_t", "SampEn_b"]
FEATURES_FC = FEATURE_ORDER + ["SampEn_t", "SampEn_b"]

def features_fc(g, mask, brain):
    f = features_from(g, mask, brain)
    if f is None: return None
    tumour = ((mask > 0) & (brain > 0)).astype(np.uint8); rest = ((brain > 0) & (tumour == 0)).astype(np.uint8)
    f["SampEn_t"] = sampen2d(g, tumour); f["SampEn_b"] = sampen2d(g, rest, n_candidates=3000)
    if not (np.isfinite(f["SampEn_t"]) and np.isfinite(f["SampEn_b"])): return None
    return f

GEOM = ["brain_area", "bbox_aspect", "eccentricity_b", "solidity_b", "extent_b", "orientation_b",
        "centroid_y", "centroid_x", "lower_third", "mask_symmetry", "intensity_symmetry"]
def geometry(g, brain):
    H, W = brain.shape
    p = max(regionprops(sklabel(brain)), key=lambda r: r.area)
    y0, x0, y1, x1 = p.bbox; cy, cx = p.centroid; c = int(round(cx)); half = min(c, W - c)
    if half < 4: msym = isym = 0.0
    else:
        L = brain[:, c - half:c].astype(float); R = brain[:, c:c + half][:, ::-1].astype(float)
        msym = float((L * R).sum() / max(np.maximum(L, R).sum(), 1)); m_ = (L * R) > 0
        isym = float(np.corrcoef(g[:, c - half:c][m_], g[:, c:c + half][:, ::-1][m_])[0, 1]) if m_.sum() > 50 else 0.0
    rows = np.where(brain.any(axis=1))[0]; third = rows.min() + 2 * (rows.max() - rows.min()) // 3
    return [p.area / (H * W), (y1 - y0) / max(x1 - x0, 1), p.eccentricity, p.solidity, p.extent, abs(p.orientation),
            cy / H, cx / W, float(brain[third:].sum() / brain.sum()), msym, float(np.nan_to_num(isym))]

# ------------------------------------------------------------------ helpers
def rf(balanced=False):
    return Pipeline([("sc", StandardScaler()), ("m", RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1,
                     class_weight="balanced" if balanced else None))])

def repeated_patient_cv(X, Y, PID, n_rep=N_REP, balanced=False):
    """Patient-level balanced accuracy and patient-level AUC (binary) over repeated stratified group splits."""
    accs, aucs = [], []
    for seed in range(n_rep):
        pred = np.zeros_like(Y); proba = np.zeros((len(Y), len(np.unique(Y))))
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, Y, PID):
            m = rf(balanced).fit(X[tr], Y[tr]); pred[te] = m.predict(X[te]); proba[te] = m.predict_proba(X[te])
        pv, pt = patient_vote(pred, proba, PID), patient_truth(Y, PID); k = sorted(pv)
        accs.append(balanced_accuracy_score([pt[p] for p in k], [pv[p] for p in k]))
        if proba.shape[1] == 2:
            pp = {p: proba[PID == p, 1].mean() for p in k}
            aucs.append(roc_auc_score([pt[p] for p in k], [pp[p] for p in k]))
    return np.array(accs), (np.array(aucs) if aucs else None)

def dauc(v, y):
    a = roc_auc_score(y, v); return max(a, 1 - a)

# ================================================================== PART A
def build_figshare():
    if os.path.exists(CACHE_FIG):
        d = np.load(CACHE_FIG, allow_pickle=True); return d["X"], d["Y"], d["PID"]
    files = sorted(f for f in glob.glob(DATA_GLOB, recursive=True) if f.lower().endswith(".mat"))
    if not files: raise SystemExit("No Figshare .mat files found under /kaggle/input.")
    X, Y, PID = [], [], []
    for k, path in enumerate(files):
        img, label, mask, pid = read_cjdata(path)
        if img is None or pid is None or mask is None: continue
        g, mk = resize_pair(img, mask); f = features_fc(g, mk, brain_mask_of(g))
        if f is None: continue
        X.append([f[n] for n in FEATURES_FC]); Y.append(label - 1); PID.append(pid)
        if (k + 1) % 500 == 0: print(f"  Figshare: processed {k + 1}/{len(files)}")
    X, Y, PID = np.array(X), np.array(Y), np.array(PID)
    np.savez_compressed(CACHE_FIG, X=X, Y=Y, PID=PID); return X, Y, PID

def part_a(X, Y, PID):
    print("\n" + "=" * 76); print("  PART A — FRACTAL-CHAOS DESCRIPTORS ON FIGSHARE (3 classes)"); print("=" * 76)
    print(f"  {len(Y)} slices, {len(set(PID))} patients, classes {np.bincount(Y)}")
    col = {n: i for i, n in enumerate(FEATURES_FC)}
    print("\n  One-vs-rest AUC of the chaos measure (direction-free):")
    for n in ("SampEn_t", "SampEn_b"):
        print(f"    {n:9s}" + "".join(f"  {TYPE_NAMES[c]} {dauc(X[:, col[n]], (Y == c).astype(int)):.3f}" for c in range(3)))
    sets = {"all 19 descriptors": FEATURES_FC,
            "fractal-chaos family only (7)": FC_FAMILY,
            "all except SampEn2D (17, final pipeline set)": FEATURE_ORDER,
            "all except the fractal-chaos family (12)": [f for f in FEATURES_FC if f not in FC_FAMILY]}
    res = {}
    print(f"\n  {'feature set':48s} patient-level balanced accuracy ({N_REP} repeated splits)")
    for name, fs in sets.items():
        a, _ = repeated_patient_cv(X[:, [col[f] for f in fs]], Y, PID); res[name] = a
        print(f"  {name:48s} {a.mean():.3f} +/- {a.std():.3f}")
    for name in list(sets)[1:]:
        d = res["all 19 descriptors"] - res[name]
        print(f"    all 19 minus [{name}]: {100 * d.mean():+.1f} pp (wins {int((d > 0).sum())}/{N_REP})")

# ================================================================== PART B
class FeatNet(nn.Module):
    def __init__(self, d, n=3):
        super().__init__(); self.f = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, n))
    def forward(self, x): return self.f(x)

def restricted_ce(logits, yb, present):
    mask = torch.full((logits.shape[1],), -1e9); mask[list(present)] = 0.0
    return F.cross_entropy(logits + mask, yb)

def _flat(m): return torch.cat([p.data.flatten() for p in m.parameters()])
def _set(m, f):
    i = 0
    for p in m.parameters(): n = p.numel(); p.data.copy_(f[i:i + n].view_as(p)); i += n

def dirichlet_partition(Y, PID, n_clients, alpha, seed):
    rng = np.random.default_rng(seed); uniq = np.unique(PID)
    pl = {p: Y[np.where(PID == p)[0][0]] for p in uniq}; cp = {c: [] for c in range(n_clients)}
    for cls in np.unique(Y):
        cps = [p for p in uniq if pl[p] == cls]; rng.shuffle(cps)
        cnt = (rng.dirichlet([alpha] * n_clients) * len(cps)).astype(int)
        if len(cps): cnt[-1] = len(cps) - cnt[:-1].sum()
        i = 0
        for c in range(n_clients):
            for _ in range(cnt[c]):
                if i < len(cps): cp[c].append(cps[i]); i += 1
    ci = {c: np.where(np.isin(PID, cp[c]))[0] for c in range(n_clients)}
    return {c: v for c, v in ci.items() if len(v) >= 4}

def fed_features(X, Y, clients, test_idx, mode, seed, lr=0.05, batch=32):
    """mode: 'fedavg' (plain cross-entropy), 'restricted', 'restricted+scaffold'.
       Feature standardisation uses only aggregated sufficient statistics (count, sum,
       sum of squares) sent by each client, so no client reveals individual descriptors."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    n = sum(len(v) for v in clients.values())
    s1 = sum(X[v].sum(0) for v in clients.values()); s2 = sum((X[v] ** 2).sum(0) for v in clients.values())
    mu = s1 / n; sd = np.sqrt(np.maximum(s2 / n - mu ** 2, 1e-12))
    Z = torch.tensor((X - mu) / sd, dtype=torch.float32); Yt = torch.tensor(Y)
    model = FeatNet(X.shape[1]); ids = list(clients); ni = {c: len(clients[c]) for c in ids}
    present = {c: set(np.unique(Y[clients[c]]).tolist()) for c in ids}
    scaffold = mode == "restricted+scaffold"
    if scaffold:
        cg = torch.zeros_like(_flat(model)); cl = {c: torch.zeros_like(cg) for c in ids}
    for _ in range(FED_ROUNDS):
        g0 = _flat(model); deltas, cdel = [], []
        for c in ids:
            _set(model, g0); opt = torch.optim.SGD(model.parameters(), lr=lr); idx = rng.permutation(clients[c]); steps = 0
            for s in range(0, len(idx), batch):
                b = idx[s:s + batch]; opt.zero_grad(); out = model(Z[b])
                loss = F.cross_entropy(out, Yt[b]) if mode == "fedavg" else restricted_ce(out, Yt[b], present[c])
                loss.backward()
                if scaffold:
                    with torch.no_grad():
                        i = 0
                        for p in model.parameters():
                            k = p.numel(); p.grad.add_((cg[i:i + k] - cl[c][i:i + k]).view_as(p)); i += k
                opt.step(); steps += 1
            yk = _flat(model)
            if scaffold:
                newc = cl[c] - cg + (g0 - yk) / (max(steps, 1) * lr); cdel.append(newc - cl[c]); cl[c] = newc
            deltas.append((ni[c], yk - g0))
        _set(model, g0 + sum(k / n * d for k, d in deltas))
        if scaffold: cg = cg + torch.stack(cdel).mean(0)
    with torch.no_grad(): pred = model(Z[test_idx]).argmax(1).numpy()
    return pred

def part_b(X, Y, PID):
    print("\n" + "=" * 76); print("  PART B — FRACTAL-CHAOS GUIDED FEDERATED LEARNING (descriptors, not images)"); print("=" * 76)
    rng = np.random.default_rng(TEST_SPLIT_SEED); uniq = np.unique(PID); rng.shuffle(uniq)
    test_p = set(uniq[:int(0.2 * len(uniq))]); te = np.where(np.isin(PID, list(test_p)))[0]
    trm = ~np.isin(PID, list(test_p)); trp = np.where(trm)[0]
    def pat_bacc(pred):
        pv = patient_vote(pred, np.eye(3)[pred], PID[te]); pt = patient_truth(Y[te], PID[te]); k = sorted(pv)
        return balanced_accuracy_score([pt[p] for p in k], [pv[p] for p in k])
    modes = ["fedavg", "restricted", "restricted+scaffold"]; acc = {m: [] for m in modes}; pacc = {m: [] for m in modes}; miss = []
    for seed in range(FED_SEEDS):
        cl = dirichlet_partition(Y[trm], PID[trm], N_CLIENTS, PARTITION_ALPHA, seed); cl = {c: trp[v] for c, v in cl.items()}
        miss.append(sum(1 for v in cl.values() if len(np.unique(Y[v])) < 3) / len(cl))
        for m in modes:
            pr = fed_features(X, Y, cl, te, m, seed)
            acc[m].append(balanced_accuracy_score(Y[te], pr)); pacc[m].append(pat_bacc(pr))
    print(f"  test patients fixed (seed {TEST_SPLIT_SEED}): {len(test_p)}; clients missing >=1 class: {100 * np.mean(miss):.0f}% on average")
    # centralised references on the same split
    mu, sd = X[trp].mean(0), X[trp].std(0) + 1e-12
    cen = []
    for seed in range(FED_SEEDS):
        torch.manual_seed(seed); net = FeatNet(X.shape[1]); opt = torch.optim.SGD(net.parameters(), lr=0.05)
        Z = torch.tensor((X - mu) / sd, dtype=torch.float32); Yt = torch.tensor(Y); r = np.random.default_rng(seed)
        for _ in range(FED_ROUNDS):
            idx = r.permutation(trp)
            for s in range(0, len(idx), 32):
                b = idx[s:s + 32]; opt.zero_grad(); F.cross_entropy(net(Z[b]), Yt[b]).backward(); opt.step()
        with torch.no_grad(): cen.append(balanced_accuracy_score(Y[te], net(Z[te]).argmax(1).numpy()))
    rfm = rf().fit(X[trp], Y[trp]); rf_acc = balanced_accuracy_score(Y[te], rfm.predict(X[te]))
    print(f"\n  {f'configuration (test slices, {FED_SEEDS} seeds)':40s} {'slice bal.acc':>14s} {'patient bal.acc':>16s} {'collapse':>9s}")
    for m in modes:
        a = np.array(acc[m]); p = np.array(pacc[m])
        print(f"  {'federated, ' + m:40s} {a.mean():7.3f} +/- {a.std():.3f}  {p.mean():8.3f} +/- {p.std():.3f}  {int((a <= 0.40).sum()):>4d}/{FED_SEEDS}")
    print(f"  {'centralised MLP (upper reference)':40s} {np.mean(cen):7.3f} +/- {np.std(cen):.3f}")
    print(f"  {'centralised random forest (reference)':40s} {rf_acc:7.3f}")
    a, b = np.array(acc["restricted"]), np.array(acc["fedavg"]); c = np.array(acc["restricted+scaffold"])
    p1 = stats.wilcoxon(a, b).pvalue if np.any(a != b) else 1.0; p2 = stats.wilcoxon(c, a).pvalue if np.any(c != a) else 1.0
    print(f"\n  restricted vs federated averaging: {100 * (a - b).mean():+.1f} pp, Wilcoxon p = {p1:.4f}, wins {int((a > b).sum())}/{FED_SEEDS}")
    print(f"  + SCAFFOLD vs restricted:          {100 * (c - a).mean():+.1f} pp, Wilcoxon p = {p2:.4f}")
    ok = p1 < 0.05 and (a - b).mean() > 0 and (a <= 0.40).sum() == 0
    print(f"  VERDICT (rule fixed before running): {'SUPPORTED' if ok else 'NOT SUPPORTED'}: the class-presence correction "
          f"{'improves' if ok else 'does not significantly improve'} federated learning on fractal-chaos descriptors.")

# ================================================================== PART C
def brats_grades():
    """Grade per subject from name_mapping.csv (BraTS 2020) or from HGG/LGG folder names."""
    grade = {}
    for csvp in glob.glob("/kaggle/input/**/name_mapping*.csv", recursive=True):
        with open(csvp) as fh:
            for r in csv.DictReader(fh):
                g = (r.get("Grade") or "").strip().upper()
                for k, v in r.items():
                    if v and "Training" in v and g in ("HGG", "LGG"): grade[v.strip()] = g
    return grade

def brain_mask_raw(raw, size=IMG_SIZE):
    """Brain mask of a skull-stripped BraTS slice, computed on the RAW intensities (background is
       exactly zero) and resized with nearest-neighbour interpolation. Computing it after slice-wise
       normalisation would let the brightest tumour pixel shift the mask boundary, so that brain
       'geometry' would carry tumour information (found and fixed during testing)."""
    m = ndimage.binary_fill_holes(raw > 0); lbl, n = ndimage.label(m)
    if n >= 1:
        s = np.bincount(lbl.ravel()); s[0] = 0; m = lbl == s.argmax()
    mk = Image.fromarray((m * 255).astype(np.uint8)).resize((size, size), Image.NEAREST)
    return (np.asarray(mk) > 127).astype(np.uint8)

def build_brats():
    if os.path.exists(CACHE_BRATS):
        d = np.load(CACHE_BRATS, allow_pickle=True); return d["X"], d["G"], d["Y"], d["PID"]
    try: import nibabel as nib
    except Exception: os.system("pip install nibabel --quiet"); import nibabel as nib
    files = sorted(glob.glob(BRATS_GLOB, recursive=True))
    if not files: raise SystemExit("No BraTS *t1ce.nii files found. Attach a BraTS dataset with HGG/LGG labels.")
    grade = brats_grades(); X, Gm, Y, PID = [], [], [], []; skipped = 0
    for k, t1 in enumerate(files):
        d = os.path.dirname(t1); sid = os.path.basename(d)
        g_ = grade.get(sid) or ("HGG" if "/HGG/" in t1 else "LGG" if "/LGG/" in t1 else None)
        seg = glob.glob(os.path.join(d, "*seg*.nii*"))
        if g_ is None or not seg: skipped += 1; continue
        vol = nib.load(t1).get_fdata().astype(np.float32); lab = nib.load(seg[0]).get_fdata()
        area = (lab > 0).sum(axis=(0, 1)); zs = [z for z in np.argsort(area)[::-1][:BRATS_SLICES] if area[z] >= BRATS_MIN_AREA]
        for z in zs:
            g, mk = resize_pair(vol[:, :, z], (lab[:, :, z] > 0).astype(np.uint8)); br = brain_mask_raw(vol[:, :, z])
            f = features_fc(g, mk, br)
            if f is None: continue
            X.append([f[n] for n in FEATURES_FC]); Gm.append(geometry(g, br)); Y.append(1 if g_ == "HGG" else 0); PID.append(sid)
        if (k + 1) % 50 == 0: print(f"  BraTS: processed {k + 1}/{len(files)} subjects")
    X, Gm, Y, PID = map(np.array, (X, Gm, Y, PID))
    print(f"  BraTS: {len(set(PID))} subjects used ({skipped} without grade or mask skipped)")
    np.savez_compressed(CACHE_BRATS, X=X, G=Gm, Y=Y, PID=PID); return X, Gm, Y, PID

def part_c(X, Gm, Y, PID):
    print("\n" + "=" * 76); print("  PART C — TRUE GRADING ON BraTS: HIGH-GRADE (HGG) vs LOW-GRADE (LGG) GLIOMA"); print("=" * 76)
    pt = patient_truth(Y, PID); n_h = sum(v == 1 for v in pt.values()); n_l = sum(v == 0 for v in pt.values())
    print(f"  {len(Y)} slices from {len(pt)} patients: {n_h} HGG, {n_l} LGG (balanced accuracy and AUC are used)")
    col = {n: i for i, n in enumerate(FEATURES_FC)}
    sets = {"fractal-chaos family (7)": X[:, [col[f] for f in FC_FAMILY]],
            "all 19 descriptors": X,
            "non-fractal descriptors (12)": X[:, [col[f] for f in FEATURES_FC if f not in FC_FAMILY]],
            "brain-mask geometry only (shortcut control)": Gm[:, :10]}   # shape and position only; intensity symmetry excluded because a one-sided tumour changes it
    res = {}
    print(f"\n  {'feature set':46s} {'patient bal.acc':>16s} {'patient AUC':>13s}")
    for name, Xs in sets.items():
        a, u = repeated_patient_cv(Xs, Y, PID, balanced=True); res[name] = (a, u)
        print(f"  {name:46s} {a.mean():.3f} +/- {a.std():.3f}  {u.mean():.3f} +/- {u.std():.3f}")
    print("\n  Single descriptors, patient-level AUC (patient mean; 95% CI by bootstrap over patients):")
    keys = sorted(pt); yy = np.array([pt[p] for p in keys]); rng = np.random.default_rng(0)
    for n in FC_FAMILY:
        v = np.array([X[PID == p, col[n]].mean() for p in keys]); a = dauc(v, yy); bs = []
        for _ in range(1000):
            i = rng.integers(0, len(keys), len(keys))
            if len(set(yy[i])) == 2: bs.append(dauc(v[i], yy[i]))
        print(f"    {n:9s} AUC {a:.3f}  [{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}]")
    fc, geo = res["fractal-chaos family (7)"][0].mean(), res["brain-mask geometry only (shortcut control)"][0].mean()
    ok = fc >= 0.70 and fc - geo >= 0.05
    print(f"\n  VERDICT (rule fixed before running): fractal-chaos family {fc:.3f}, geometry control {geo:.3f} -> "
          f"{'SUPPORTED: the descriptors carry grade information beyond slice geometry.' if ok else 'NOT SUPPORTED by the preregistered rule.'}")
    print("  Limitation: BraTS LGG cases come mostly from one collection (TCGA-LGG), so grade is partly confounded")
    print("  with acquisition site; report this alongside any grading result.")


def _block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(4, o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(4, o), nn.ReLU(inplace=True))

class TinyUNet(nn.Module):
    """Three-level U-Net (about 0.47 M parameters). Its only job is to say WHERE
       the tumour is; the boundary is drawn by the active contour afterwards."""
    def __init__(self, w=16):
        super().__init__()
        self.e1, self.e2, self.e3 = _block(1, w), _block(w, 2 * w), _block(2 * w, 4 * w)
        self.b = _block(4 * w, 8 * w)
        self.u3, self.d3 = nn.ConvTranspose2d(8 * w, 4 * w, 2, 2), _block(8 * w, 4 * w)
        self.u2, self.d2 = nn.ConvTranspose2d(4 * w, 2 * w, 2, 2), _block(4 * w, 2 * w)
        self.u1, self.d1 = nn.ConvTranspose2d(2 * w, w, 2, 2), _block(2 * w, w)
        self.out = nn.Conv2d(w, 1, 1)
    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(F.max_pool2d(e1, 2)); e3 = self.e3(F.max_pool2d(e2, 2))
        b = self.b(F.max_pool2d(e3, 2))
        d3 = self.d3(torch.cat([self.u3(b), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)

def train_unet(Xi, Mi, epochs=UNET_EPOCHS, batch=16, lr=1e-3, seed=0):
    """Xi (n,1,H,W) in [0,1]; Mi (n,H,W) radiologist masks. BCE + soft Dice; random horizontal flips."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = TinyUNet().to(DEVICE); opt = torch.optim.Adam(net.parameters(), lr=lr)
    Xt = torch.tensor(Xi, dtype=torch.float32); Mt = torch.tensor(Mi[:, None], dtype=torch.float32)
    for _ in range(epochs):
        net.train(); perm = rng.permutation(len(Xt))
        for i in range(0, len(Xt), batch):
            idx = perm[i:i + batch]; xb, mb = Xt[idx].to(DEVICE), Mt[idx].to(DEVICE)
            if rng.random() < 0.5: xb, mb = xb.flip(-1), mb.flip(-1)
            lo = net(xb); p = torch.sigmoid(lo)
            dice = 1 - (2 * (p * mb).sum((1, 2, 3)) + 1) / (p.sum((1, 2, 3)) + mb.sum((1, 2, 3)) + 1)
            loss = F.binary_cross_entropy_with_logits(lo, mb) + dice.mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return net

@torch.no_grad()
def predict_prob(net, Xi, batch=64):
    net.eval(); out = []
    for i in range(0, len(Xi), batch):
        out.append(torch.sigmoid(net(torch.tensor(Xi[i:i + batch], dtype=torch.float32).to(DEVICE))).cpu().numpy()[:, 0])
    return np.concatenate(out)

def _largest(m):
    lbl, n = ndimage.label(m)
    if n < 1: return None
    s = np.bincount(lbl.ravel()); s[0] = 0
    return (lbl == s.argmax()).astype(np.uint8)

def localise(prob, brain, thr=0.5, min_px=16):
    """U-Net probability map -> one connected region inside the brain, or None."""
    m = _largest((prob > thr) & (brain > 0))
    if m is None or m.sum() < min_px:
        if prob.max() < 0.1: return None
        m = _largest((prob > 0.5 * prob.max()) & (brain > 0))
        if m is None or m.sum() < min_px: return None
    return m

def refine_contour(g, seed, brain, margin=10, iters=40):
    """Morphological Chan-Vese (Marquez-Neila et al., 2014) started from the
       U-Net region and restricted to a box `margin` pixels around it. Keeps the
       component overlapping the seed. If the contour collapses (<0.5x) or
       escapes (>2x the seed area), the U-Net region is kept. Returns (mask, used)."""
    ys, xs = np.where(seed > 0)
    y0, y1 = max(ys.min() - margin, 0), min(ys.max() + margin + 1, seed.shape[0])
    x0, x1 = max(xs.min() - margin, 0), min(xs.max() + margin + 1, seed.shape[1])
    ls = morphological_chan_vese(g[y0:y1, x0:x1] / 255.0, num_iter=iters,
                                 init_level_set=seed[y0:y1, x0:x1], smoothing=1, lambda1=1, lambda2=1)
    ref = np.zeros_like(seed); ref[y0:y1, x0:x1] = (np.asarray(ls) > 0)
    ref = (ref > 0) & (brain > 0)
    lbl, n = ndimage.label(ref)
    if n >= 1:
        ov = np.bincount(lbl[seed > 0].ravel(), minlength=n + 1); ov[0] = 0
        ref = (lbl == ov.argmax()) if ov.max() > 0 else np.zeros_like(ref)
    a, a0 = int(ref.sum()), int(seed.sum())
    if a < 0.5 * a0 or a > 2.0 * a0: return seed.astype(np.uint8), False
    return ref.astype(np.uint8), True

def dice(a, b):
    a, b = a.astype(bool), b.astype(bool); return 2.0 * (a & b).sum() / max(a.sum() + b.sum(), 1)


# ---------------- Stage 19 descriptors ----------------
def boundary_fd(mask, min_sizes=3):
    """Box-counting fractal dimension of the tumour boundary (1-pixel contour), over box sizes from 2 px
       up to half the equivalent diameter (a range relative to tumour size)."""
    m = mask > 0
    if m.sum() < 30: return np.nan
    b = m & ~ndimage.binary_erosion(m); H, W = b.shape
    dmax = max(4, int(np.sqrt(4 * m.sum() / np.pi) / 2))
    sizes = [s for s in (2, 3, 4, 6, 8, 12, 16, 24, 32) if s <= dmax]
    if len(sizes) < min_sizes: return np.nan
    N = []
    for s in sizes:
        Hc, Wc = H - H % s, W - W % s
        N.append(b[:Hc, :Wc].reshape(Hc // s, s, Wc // s, s).any(axis=(1, 3)).sum())
    return float(-np.polyfit(np.log(sizes), np.log(N), 1)[0])


def gray_lacunarity(gray, region, sizes=(2, 3, 4, 6, 8), min_boxes=30):
    """Gliding-box lacunarity of the INTENSITIES inside the region (only boxes lying fully inside it):
       Lambda(r) = E[S^2] / E[S]^2 for box sums S. Returns Lambda(2) and the slope of log Lambda vs log r over
       the sizes that have at least min_boxes boxes. Invariant to multiplicative intensity scaling, the variation
       that matters across MRI scanners; not to additive offsets, which MRI magnitude images (physical zero = no
       signal) do not have. Subtracting a baseline was tested and rejected: it amplified noise in homogeneous
       regions and reversed the ordering of homogeneous and gappy textures."""
    g = gray.astype(np.float64); r_ = region > 0
    v = g[r_]
    if v.size < 50 or v.max() <= 0: return np.nan, np.nan
    g = np.where(r_, g, 0.0)
    I = np.pad(g.cumsum(0).cumsum(1), ((1, 0), (1, 0))); M = np.pad(r_.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    lam, used = [], []
    for s in sizes:
        S = I[s:, s:] - I[:-s, s:] - I[s:, :-s] + I[:-s, :-s]; C = M[s:, s:] - M[:-s, s:] - M[s:, :-s] + M[:-s, :-s]
        S = S[C == s * s]
        if len(S) < min_boxes or S.mean() <= 0: continue
        lam.append((S ** 2).mean() / S.mean() ** 2); used.append(s)
    if not lam: return np.nan, np.nan
    l2 = lam[0] if used[0] == 2 else np.nan
    slope = float(np.polyfit(np.log(used), np.log(lam), 1)[0]) if len(used) >= 2 else np.nan
    return float(l2), slope

def _windows(gray, region, k):
    """k x k windows of the image whose footprint lies fully inside the region, and their positions."""
    g = gray.astype(np.float64); r_ = region > 0
    W = sliding_window_view(g, (k, k)); pos = np.argwhere(sliding_window_view(r_, (k, k)).all(axis=(2, 3)))
    return g, r_, W, pos

def fuzen2d(gray, region, m=2, r_factor=0.2, n=2, n_templates=1000, n_candidates=4000, seed=0):
    """Two-dimensional fuzzy entropy (Hilal et al., 2020). Each m x m and (m+1) x (m+1) template has its own
       mean removed; similarity is the fuzzy membership exp(-(d / r)^n) of the Chebyshev distance d, with
       r = r_factor * std of the region, so the value does not depend on intensity scale. The similarity never
       reaches zero, so the measure is always defined. FuzEn2D = ln(Phi_m) - ln(Phi_{m+1}). m = 2 because mean
       removal makes a 1 x 1 template identically zero. Seeded subsets of positions, as for SampEn2D."""
    g, r_, W1, pos = _windows(gray, region, m + 1); v = g[r_]
    if v.size < 50 or v.std() < 1e-9 or len(pos) < 20: return np.nan
    r = r_factor * v.std(); rng = np.random.default_rng(seed)
    ti = rng.choice(len(pos), min(n_templates, len(pos)), replace=False); ci = rng.choice(len(pos), min(n_candidates, len(pos)), replace=False)
    win = W1[pos[:, 0], pos[:, 1]]
    P1 = win.reshape(len(pos), -1); P1 = P1 - P1.mean(1, keepdims=True)
    Pm = win[:, :m, :m].reshape(len(pos), -1); Pm = Pm - Pm.mean(1, keepdims=True)
    sm = s1 = 0.0; cnt = 0
    for s in range(0, len(ti), 100):
        t = ti[s:s + 100]; ok = t[:, None] != ci[None]
        dm = np.abs(Pm[t][:, None] - Pm[ci][None]).max(2); d1 = np.abs(P1[t][:, None] - P1[ci][None]).max(2)
        sm += np.exp(-(dm / r) ** n)[ok].sum(); s1 += np.exp(-(d1 / r) ** n)[ok].sum(); cnt += int(ok.sum())
    return float(np.log(sm / cnt) - np.log(s1 / cnt))

from sklearn.impute import SimpleImputer
from scipy.stats import spearmanr
RUN_P1, RUN_P2, RUN_P3, RUN_P4 = True, True, True, True
# Final set selected by Part 1 in the Kaggle run (FuzEn2D adopted, Block B accepted); used when RUN_P1 is False.
PRESET_FINAL = FEATURE_ORDER + ["FuzEn_t", "FuzEn_b"] + ["GLac2_t", "GLacSlope_t"]
N_REP19 = 10
CACHE_F19, CACHE_B19, CACHE_U19 = f"{OUT_DIR}/stage19_figshare.npz", f"{OUT_DIR}/stage19_brats.npz", f"{OUT_DIR}/stage19_unet.npz"
NEW = ["SampEn_t", "SampEn_b", "FuzEn_t", "FuzEn_b", "Fbound", "GLac2_t", "GLacSlope_t"]
ALL = FEATURE_ORDER + NEW
BASE19 = FEATURE_ORDER + ["SampEn_t", "SampEn_b"]
FUZ19 = FEATURE_ORDER + ["FuzEn_t", "FuzEn_b"]
BLOCK_A, BLOCK_B = ["Fbound"], ["GLac2_t", "GLacSlope_t"]
BRAIN_SIDE = {"Fb", "C", "lacun_b", "GLCM_b", "SampEn_b", "FuzEn_b"}
SUB = ["FD_enh", "GLac_enh", "FD_nec", "GLac_nec", "enh_present", "nec_present"]
FLAGS = ["enh_present", "nec_present"]

def all_desc(g, mk, brain):
    f = features_from(g, mk, brain)
    if f is None: return None
    t = ((mk > 0) & (brain > 0)).astype(np.uint8); r = ((brain > 0) & (t == 0)).astype(np.uint8)
    f["SampEn_t"] = sampen2d(g, t); f["SampEn_b"] = sampen2d(g, r, n_candidates=3000)
    f["FuzEn_t"] = fuzen2d(g, t); f["FuzEn_b"] = fuzen2d(g, r, n_candidates=3000)
    f["Fbound"] = boundary_fd(t); f["GLac2_t"], f["GLacSlope_t"] = gray_lacunarity(g, t)
    return f

def rf19(balanced=False):
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                     ("m", RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1, class_weight="balanced" if balanced else None))])

def cv_patient(X, Y, PID, n_rep=N_REP19, balanced=False):
    """Patient-level balanced accuracy (majority vote) and, for two classes, patient-level AUC (mean probability)."""
    accs, aucs = [], []
    for seed in range(n_rep):
        pred = np.zeros_like(Y); proba = np.zeros((len(Y), len(np.unique(Y))))
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, Y, PID):
            m = rf19(balanced).fit(X[tr], Y[tr]); pred[te] = m.predict(X[te]); proba[te] = m.predict_proba(X[te])
        pv, pt = patient_vote(pred, proba, PID), patient_truth(Y, PID); k = sorted(pv)
        accs.append(balanced_accuracy_score([pt[p] for p in k], [pv[p] for p in k]))
        if proba.shape[1] == 2: aucs.append(roc_auc_score([pt[p] for p in k], [proba[PID == p, 1].mean() for p in k]))
    return np.array(accs), (np.array(aucs) if aucs else None)

def cv_threshold(X, Y, PID, n_rep=N_REP19):
    """Binary; threshold chosen by inner patient-grouped cross-validation on the training patients only."""
    out = []
    for seed in range(n_rep):
        pt = patient_truth(Y, PID); yhat = {}
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, Y, PID):
            oof = np.zeros(len(tr))
            for itr, ite in StratifiedGroupKFold(5, shuffle=True, random_state=100 + seed).split(X[tr], Y[tr], PID[tr]):
                oof[ite] = rf19(True).fit(X[tr][itr], Y[tr][itr]).predict_proba(X[tr][ite])[:, 1]
            ktr = sorted(set(PID[tr])); ptr = np.array([oof[PID[tr] == p].mean() for p in ktr]); ytr = np.array([pt[p] for p in ktr])
            grid = np.linspace(0.05, 0.95, 91); thr = grid[np.argmax([balanced_accuracy_score(ytr, (ptr >= t).astype(int)) for t in grid])]
            m = rf19(True).fit(X[tr], Y[tr]); kte = sorted(set(PID[te])); pr = m.predict_proba(X[te])[:, 1]
            yhat.update({p: int(pr[PID[te] == p].mean() >= thr) for p in kte})
        k = sorted(yhat); out.append(balanced_accuracy_score([pt[p] for p in k], [yhat[p] for p in k]))
    return np.array(out)

def cols(names, pool=ALL): return [pool.index(n) for n in names]

def paired(a, b):
    d = a - b; p = stats.wilcoxon(a, b).pvalue if np.any(d != 0) else 1.0
    return d.mean(), p, int((d > 0).sum())

# ------------------------------------------------------------------ data
def build_fig19():
    if os.path.exists(CACHE_F19):
        d = np.load(CACHE_F19, allow_pickle=True); return {k: d[k] for k in d.files}
    files = sorted(f for f in glob.glob(DATA_GLOB, recursive=True) if f.lower().endswith(".mat"))
    if not files: raise SystemExit("No Figshare .mat files found under /kaggle/input.")
    X, Y, P, G, IMG, M, B, STAB = [], [], [], [], [], [], [], []
    rng = np.random.default_rng(0)
    for k, path in enumerate(files):
        img, label, mask, pid = read_cjdata(path)
        if img is None or pid is None or mask is None: continue
        g, mk = resize_pair(img, mask); brain = brain_mask_of(g); f = all_desc(g, mk, brain)
        if f is None: continue
        X.append([f[n] for n in ALL]); Y.append(label - 1); P.append(pid); G.append(geometry(g, brain)[:10])
        IMG.append(g.astype(np.uint8)); M.append(mk); B.append(brain)
        if len(STAB) < 200 and rng.random() < 0.08:          # test-retest sample for the SampEn/FuzEn stability rule
            t = ((mk > 0) & (brain > 0)).astype(np.uint8); gn = g + np.random.default_rng(k).normal(0, 1.0, g.shape)
            STAB.append([f["SampEn_t"], sampen2d(g, t, seed=1), sampen2d(gn, t), f["FuzEn_t"], fuzen2d(g, t, seed=1), fuzen2d(gn, t)])
        if (k + 1) % 500 == 0: print(f"  Figshare: {k + 1}/{len(files)}")
    d = dict(X=np.array(X), Y=np.array(Y), P=np.array(P), G=np.array(G), IMG=np.stack(IMG), M=np.stack(M), B=np.stack(B), STAB=np.array(STAB))
    np.savez_compressed(CACHE_F19, **d); return d

# ------------------------------------------------------------------ part 1
def part1(D):
    X, Y, P = D["X"], D["Y"], D["P"]
    print("\n" + "=" * 78); print("  PART 1 — FIGSHARE, 3 CLASSES: NEW DESCRIPTOR BLOCKS AND FuzEn2D vs SampEn2D"); print("=" * 78)
    print(f"  {len(Y)} slices, {len(set(P))} patients, classes {np.bincount(Y)}")
    nan = {n: int(np.isnan(X[:, ALL.index(n)]).sum()) for n in NEW}
    print("  undefined values (imputed inside training folds): " + ", ".join(f"{n} {v}" for n, v in nan.items()))
    size = X[:, ALL.index("size")]
    print("  Spearman correlation with tumour size: " + ", ".join(
        f"{n} {spearmanr(X[:, ALL.index(n)], size, nan_policy='omit')[0]:+.2f}" for n in ["Ft", "SampEn_t", "FuzEn_t", "Fbound", "GLac2_t", "GLacSlope_t"]))
    R = {}
    for name, fs in (("19 descriptors (Stage 17 set)", BASE19), ("19 + Block A (boundary FD)", BASE19 + BLOCK_A),
                     ("19 + Block B (grayscale lacunarity)", BASE19 + BLOCK_B), ("19 with FuzEn2D instead of SampEn2D", FUZ19)):
        R[name], _ = cv_patient(X[:, cols(fs)], Y, P); print(f"  {name:42s} {R[name].mean():.3f} +/- {R[name].std():.3f}")
    base = R["19 descriptors (Stage 17 set)"]
    dA, pA, wA = paired(R["19 + Block A (boundary FD)"], base); dB, pB, wB = paired(R["19 + Block B (grayscale lacunarity)"], base)
    padj = dict(zip(["A", "B"], np.minimum(1, np.maximum.accumulate(np.sort([pA, pB]) * np.array([2, 1]))[np.argsort(np.argsort([pA, pB]))])))
    acc = []
    for blk, d, w, fs in (("A", dA, wA, BLOCK_A), ("B", dB, wB, BLOCK_B)):
        ok = d > 0 and padj[blk] < 0.05 and w >= 7; acc += fs if ok else []
        print(f"  RULE Block {blk}: gain {100 * d:+.2f} pp, Holm p = {padj[blk]:.4f}, wins {w}/{N_REP19} -> {'ACCEPTED' if ok else 'not accepted'}")
    S = D["STAB"]; rs = lambda a, b: np.corrcoef(a[~np.isnan(a) & ~np.isnan(b)], b[~np.isnan(a) & ~np.isnan(b)])[0, 1]
    st = dict(se_seed=rs(S[:, 0], S[:, 1]), se_noise=rs(S[:, 0], S[:, 2]), fz_seed=rs(S[:, 3], S[:, 4]), fz_noise=rs(S[:, 3], S[:, 5]))
    dF, pF, wF = paired(R["19 with FuzEn2D instead of SampEn2D"], base)
    cov_ok = nan["FuzEn_t"] + nan["FuzEn_b"] <= nan["SampEn_t"] + nan["SampEn_b"]
    stab_ok = st["fz_seed"] > st["se_seed"] and st["fz_noise"] > st["se_noise"]
    adopt = cov_ok and dF >= -0.005 and stab_ok
    print(f"  FuzEn2D vs SampEn2D: undefined {nan['FuzEn_t'] + nan['FuzEn_b']} vs {nan['SampEn_t'] + nan['SampEn_b']}; accuracy {100 * dF:+.2f} pp "
          f"(p = {pF:.3f}); test-retest r (seed / noise): FuzEn {st['fz_seed']:.3f} / {st['fz_noise']:.3f}, SampEn {st['se_seed']:.3f} / {st['se_noise']:.3f}")
    print(f"  RULE FuzEn2D: {'ADOPTED (replaces SampEn2D)' if adopt else 'not adopted (SampEn2D kept)'}")
    final = (FUZ19 if adopt else BASE19) + acc
    ent = ["FuzEn_t", "FuzEn_b"] if adopt else ["SampEn_t", "SampEn_b"]
    fam = ["Ft", "Fb", "C", "lacun_t", "lacun_b"] + ent + [f for f in acc]
    tum = [f for f in final if f not in BRAIN_SIDE]
    for name, fs in (("FINAL descriptor set", final), ("fractal-chaos family of the final set", fam), ("tumour-only (no brain-side descriptor)", tum)):
        a, _ = cv_patient(X[:, cols(fs)], Y, P); print(f"  {name:42s} {a.mean():.3f} +/- {a.std():.3f}   ({len(fs)} descriptors)")
    return final, fam, tum

# ------------------------------------------------------------------ part 2
def part2(D, final, fam, tum):
    X, Y, P, G = D["X"], D["Y"], D["P"], D["G"]; yb = (Y == 1).astype(int)
    print("\n" + "=" * 78); print("  PART 2 — GLIOMA vs NON-GLIOMA (PROXY FOR MALIGNANT vs PREDOMINANTLY BENIGN)"); print("=" * 78)
    print(f"  {int(yb.sum())} glioma slices, {int((1 - yb).sum())} meningioma + pituitary slices")
    res = {}
    print(f"  {f'radiologist masks, {N_REP19} repeated splits':44s} {'patient bal.acc':>16s} {'patient AUC':>13s}")
    for name, Xs in (("all final descriptors", X[:, cols(final)]), ("fractal-chaos family", X[:, cols(fam)]),
                     ("tumour-only descriptors", X[:, cols(tum)]), ("C = Ft/Fb alone", X[:, cols(["C"])]),
                     ("brain-mask geometry only (shortcut control)", G)):
        a, u = cv_patient(Xs, yb, P); res[name] = a.mean(); print(f"  {name:44s} {a.mean():.3f} +/- {a.std():.3f}  {u.mean():.3f} +/- {u.std():.3f}")
    # after automatic segmentation (U-Net localiser trained per fold on training patients only)
    if os.path.exists(CACHE_U19):
        d = np.load(CACHE_U19, allow_pickle=True); XA, okA, dice_u = d["XA"], d["okA"], d["dice"]
    else:
        XA = np.full((len(Y), len(ALL)), np.nan); okA = np.zeros(len(Y), bool); dice_u = np.zeros(len(Y))
        Xi = (D["IMG"].astype(np.float32) / 255.0)[:, None]
        for k, (tr, te) in enumerate(GroupKFold(N_FOLDS).split(X, Y, P)):
            net = train_unet(Xi[tr], D["M"][tr], seed=k); prob = predict_prob(net, Xi[te])
            for j, i in enumerate(te):
                loc = localise(prob[j], D["B"][i])
                if loc is None: continue
                dice_u[i] = dice(loc, D["M"][i]); f = all_desc(D["IMG"][i].astype(np.float32), loc, D["B"][i])
                if f is not None: XA[i] = [f[n] for n in ALL]; okA[i] = True
            print(f"  U-Net fold {k + 1}/5 done")
        np.savez_compressed(CACHE_U19, XA=XA, okA=okA, dice=dice_u)
    pt = patient_truth(yb, P); pv_r, pv_a = {}, {}
    for tr, te in GroupKFold(N_FOLDS).split(X, Y, P):
        m = rf19().fit(X[tr][:, cols(final)], yb[tr]); te = te[okA[te]]
        pv_r.update(patient_vote(m.predict(X[te][:, cols(final)]), m.predict_proba(X[te][:, cols(final)]), P[te]))
        pv_a.update(patient_vote(m.predict(XA[te][:, cols(final)]), m.predict_proba(XA[te][:, cols(final)]), P[te]))
    k = sorted(set(pv_r) & set(pv_a)); r_ = balanced_accuracy_score([pt[p] for p in k], [pv_r[p] for p in k])
    a_ = balanced_accuracy_score([pt[p] for p in k], [pv_a[p] for p in k])
    print(f"\n  after automatic segmentation (U-Net median Dice {np.median(dice_u[dice_u > 0]):.3f}; {int(okA.sum())}/{len(Y)} slices usable):")
    print(f"    radiologist masks {r_:.3f}  ->  U-Net masks {a_:.3f}   ({100 * (a_ - r_):+.1f} pp, {len(k)} patients, same split)")
    geo = res["brain-mask geometry only (shortcut control)"]; ok = a_ >= 0.85 and a_ - geo >= 0.10
    print(f"  RULE: {'SUPPORTED' if ok else 'NOT SUPPORTED'}: glioma vs non-glioma after automatic segmentation {a_:.3f} "
          f"(threshold 0.85), shortcut control {geo:.3f} (needs to be at least 0.10 lower).")
    print("  Wording: 'glioma versus non-glioma, a proxy for malignant versus predominantly benign tumours'.")

# ------------------------------------------------------------------ part 3
def build_brats19():
    if os.path.exists(CACHE_B19):
        d = np.load(CACHE_B19, allow_pickle=True); return d["X"], d["S"], d["G"], d["Y"], d["P"]
    try: import nibabel as nib
    except Exception: os.system("pip install nibabel --quiet"); import nibabel as nib
    files = sorted(glob.glob(BRATS_GLOB, recursive=True)); grade = brats_grades()
    if not files: raise SystemExit("No BraTS *t1ce.nii files found.")
    X, S, G, Y, P = [], [], [], [], []
    def rs(m): return (np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)) > 127).astype(np.uint8)
    for k, t1 in enumerate(files):
        d = os.path.dirname(t1); sid = os.path.basename(d)
        g_ = grade.get(sid) or ("HGG" if "/HGG/" in t1 else "LGG" if "/LGG/" in t1 else None); seg = glob.glob(os.path.join(d, "*seg*.nii*"))
        if g_ is None or not seg: continue
        vol = nib.load(t1).get_fdata().astype(np.float32); lab = nib.load(seg[0]).get_fdata()
        area = (lab > 0).sum(axis=(0, 1)); zs = [z for z in np.argsort(area)[::-1][:BRATS_SLICES] if area[z] >= BRATS_MIN_AREA]
        for z in zs:
            g, mk = resize_pair(vol[:, :, z], (lab[:, :, z] > 0).astype(np.uint8)); br = brain_mask_raw(vol[:, :, z])
            f = all_desc(g, mk, br)
            if f is None: continue
            sub = []
            for lv in (4, 1):
                reg = rs((lab[:, :, z] == lv).astype(np.uint8)) & br
                present = reg.sum() >= 64
                fd = boxcount_fd(g, reg) if present else 0.0; gl = gray_lacunarity(g, reg)[0] if present else 0.0
                sub += [fd if np.isfinite(fd) else 0.0, gl if np.isfinite(gl) else 0.0]
            sub += [float((rs((lab[:, :, z] == 4).astype(np.uint8)) & br).sum() >= 64), float((rs((lab[:, :, z] == 1).astype(np.uint8)) & br).sum() >= 64)]
            X.append([f[n] for n in ALL]); S.append(sub); G.append(geometry(g, br)[:10]); Y.append(1 if g_ == "HGG" else 0); P.append(sid)
        if (k + 1) % 50 == 0: print(f"  BraTS: {k + 1}/{len(files)}")
    out = [np.array(v) for v in (X, S, G, Y, P)]
    np.savez_compressed(CACHE_B19, X=out[0], S=out[1], G=out[2], Y=out[3], P=out[4]); return out

def part3(final):
    X, S, G, Y, P = build_brats19()
    print("\n" + "=" * 78); print("  PART 3 — BraTS: HIGH-GRADE vs LOW-GRADE GLIOMA (GRADE, NOT BENIGN vs MALIGNANT)"); print("=" * 78)
    pt = patient_truth(Y, P); print(f"  {len(Y)} slices, {len(pt)} patients: {sum(v == 1 for v in pt.values())} HGG, {sum(v == 0 for v in pt.values())} LGG")
    print(f"  enhancing region present on {int(S[:, 4].sum())} slices, necrosis on {int(S[:, 5].sum())}")
    XS = np.hstack([X, S]); POOL = ALL + SUB; R = {}
    for name, fs in (("19 descriptors (Stage 17 set)", BASE19), ("19 + sub-region block", BASE19 + SUB), ("19 + presence flags only", BASE19 + FLAGS),
                     ("final set + sub-region block", final + SUB), ("presence flags alone", FLAGS)):
        R[name] = cv_threshold(XS[:, cols(fs, POOL)], Y, P); print(f"  {name:36s} balanced accuracy {R[name].mean():.3f} +/- {R[name].std():.3f}")
    R["geometry"] = cv_threshold(G, Y, P); print(f"  {'brain-mask geometry (control)':36s} balanced accuracy {R['geometry'].mean():.3f} +/- {R['geometry'].std():.3f}")
    d, p, w = paired(R["19 + sub-region block"], R["19 descriptors (Stage 17 set)"]); ok = d > 0 and p < 0.05 and w >= 7
    d2, p2, w2 = paired(R["19 + sub-region block"], R["19 + presence flags only"])
    print(f"  RULE: sub-region block {100 * d:+.2f} pp, p = {p:.4f}, wins {w}/{N_REP19} -> {'ACCEPTED' if ok else 'not accepted'}")
    print(f"  attribution: sub-region fractal measures vs presence flags alone, both added to the 19: {100 * d2:+.2f} pp (p = {p2:.4f}, wins {w2}/{N_REP19})")
    print("  Limitation: most low-grade cases come from one collection (TCGA-LGG); grade is partly confounded with site.")

# ------------------------------------------------------------------ part 4
def run_mlp19(Z, Y, train_sets, test_idx, seed, lr=0.05, rounds=60):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); Zt = torch.tensor(Z, dtype=torch.float32); Yt = torch.tensor(Y)
    model = FeatNet(Z.shape[1]); n = sum(len(v) for v in train_sets.values())
    for _ in range(rounds):
        g0 = _flat(model); acc = None
        for c, idx in train_sets.items():
            _set(model, g0); opt = torch.optim.SGD(model.parameters(), lr=lr)
            cc = np.bincount(Y[idx], minlength=3); w = np.array([1.0 / cc[c_] for c_ in Y[idx]]); order = rng.choice(idx, len(idx), p=w / w.sum())
            for s in range(0, len(order), 32):
                b = order[s:s + 32]; opt.zero_grad(); F.cross_entropy(model(Zt[b]), Yt[b]).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            u = _flat(model) * (len(idx) / n); acc = u if acc is None else acc + u
        _set(model, acc)
    with torch.no_grad(): return torch.softmax(model(Zt[test_idx]), 1).numpy()   # probabilities: needed for tie-breaks

def part4(D, final):
    X, Y, P = D["X"], D["Y"], D["P"]
    print("\n" + "=" * 78); print("  PART 4 — FEDERATED ARM (Stage 18 design, alpha 0.3, 20 seeds)"); print("=" * 78)
    rng = np.random.default_rng(TEST_SPLIT_SEED); uniq = np.unique(P); rng.shuffle(uniq); test_p = set(uniq[:int(0.2 * len(uniq))])
    te = np.where(np.isin(P, list(test_p)))[0]; trm = ~np.isin(P, list(test_p)); tr = np.where(trm)[0]
    for name, fs in (("19 descriptors (Stage 17 set)", BASE19), ("final descriptor set", final)):
        # missing values: mean of the observed training values = sum / count, which each hospital can send
        # without revealing individual values (a median would require pooling the raw data)
        Xs = X[:, cols(fs)]; fill = np.nansum(Xs[tr], 0) / np.maximum((~np.isnan(Xs[tr])).sum(0), 1); Xs = np.where(np.isnan(Xs), fill, Xs)
        mu, sd = Xs[tr].mean(0), Xs[tr].std(0) + 1e-12; Z = (Xs - mu) / sd
        cen, fed, fedp = [], [], []
        for seed in range(20):
            pc = run_mlp19(Z, Y, {0: tr}, te, seed).argmax(1); cen.append(balanced_accuracy_score(Y[te], pc))
            part = dirichlet_partition(Y[trm], P[trm], N_CLIENTS, 0.3, seed); cl = {c: tr[v] for c, v in part.items()}
            prf = run_mlp19(Z, Y, cl, te, seed); pf = prf.argmax(1); fed.append(balanced_accuracy_score(Y[te], pf))
            pv = patient_vote(pf, prf, P[te]); pt = patient_truth(Y[te], P[te]); k = sorted(pv)   # ties broken by model probability
            fedp.append(balanced_accuracy_score([pt[q] for q in k], [pv[q] for q in k]))
        cen, fed, fedp = map(np.array, (cen, fed, fedp))
        print(f"  {name:32s} centralised {cen.mean():.3f}  federated {fed.mean():.3f} (patient level {fedp.mean():.3f})  damage {100 * (cen - fed).mean():+.1f} pp")

if __name__ == "__main__":
    t0 = time.time()
    print("=" * 78); print("  STAGE 19 — FINAL PREREGISTERED EXTENSION"); print("=" * 78); print(f"  device: {DEVICE}")
    D = build_fig19(); final = PRESET_FINAL; fam = tum = None
    if RUN_P1: final, fam, tum = part1(D)
    if RUN_P2: part2(D, final, fam or BASE19, tum or [f for f in BASE19 if f not in BRAIN_SIDE])
    if RUN_P3: part3(final)
    if RUN_P4: part4(D, final)
    print(f"\nDONE in {(time.time() - t0) / 60:.1f} min. Copy back everything above.\n")
