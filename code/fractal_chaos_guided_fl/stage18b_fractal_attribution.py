"""
================================================================================
 CF-SCAFFOLD — STAGE 18B: IS THE ROBUSTNESS DUE TO THE FRACTAL-CHAOS FAMILY?
================================================================================
 Stage 18 showed, under a matched design, that federated learning on the 19
 descriptors loses 1.9 points to heterogeneity (alpha 0.3) against 15.2 points
 for the CNN on images. That result concerns the descriptor set as a whole. This
 stage asks whether the fractal-chaos family ON ITS OWN confers the robustness,
 which is what the thesis title asserts.

 Design: identical to Stage 18 (same recipe, same hospitals, same test patients,
 20 seeds, alpha 0.3 primary, 0.1 and 1.0 descriptive), with the MLP trained on:
   fractal-chaos family (7): Ft, Fb, C, lacunarity (tumour, brain), SampEn2D (tumour, brain)
   other descriptors (12):  texture, entropy, shape, edge, size
   all 19:                  reproduces the Stage 18 descriptor arm (consistency check)

 Absolute damage alone would favour a weaker model, which has less accuracy to
 lose. Relative damage = damage / (centralised accuracy - 1/3) is therefore used
 as well. Stage 18 CNN reference at alpha 0.3: damage 15.2 points, centralised
 0.736, relative damage 0.152 / (0.736 - 0.333) = 0.377.

 Rule (fixed before running, alpha 0.3): "fractal-chaos guided" is SUPPORTED if
 the fractal-chaos family alone has (1) absolute damage <= 10.2 points (at least
 5 below the CNN) AND (2) relative damage < 0.377. The comparison with the other
 12 descriptors is reported for attribution and does not enter the verdict.
--------------------------------------------------------------------------------
 Kaggle: CPU is enough. Attach Figshare. Reuses /kaggle/working/stage17_figshare.npz
 if present. About 30-45 minutes; resumes after an interruption.
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


ALPHAS, PRIMARY_ALPHA, SEEDS18, ROUNDS18 = [0.3, 0.1, 1.0], 0.3, 20, 60
CNN_REF = dict(damage=0.152, central=0.736)
LOG18B = f"{OUT_DIR}/stage18b_log.csv"
SUBSETS = {"fractal-chaos family (7)": FC_FAMILY,
           "other descriptors (12)": [f for f in FEATURES_FC if f not in FC_FAMILY],
           "all 19": FEATURES_FC}

def partition_set_218():
    """Patient IDs and labels of the 218-patient set of the final pipeline (no images kept)."""
    files = sorted(f for f in glob.glob(DATA_GLOB, recursive=True) if f.lower().endswith(".mat"))
    P, Y = [], []
    for k, path in enumerate(files):
        img, label, mask, pid = read_cjdata(path)
        if img is None or pid is None or mask is None: continue
        g, mk = resize_pair(img, mask)
        if features_from(g, mk, brain_mask_of(g)) is None: continue
        P.append(pid); Y.append(label - 1)
        if (k + 1) % 500 == 0: print(f"  partition set: {k + 1}/{len(files)}")
    return np.array(P), np.array(Y)

def balanced_order(y, idx, rng):
    cc = np.bincount(y[idx], minlength=3); w = np.array([1.0 / cc[c] for c in y[idx]]); w /= w.sum()
    return rng.choice(idx, size=len(idx), replace=True, p=w)

def run_mlp(Z, Y, train_sets, test_idx, seed, lr=0.05):
    """Same recipe as Stage 18 (balanced sampling, clipping 1.0, plain cross-entropy, weighted averaging)."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    Zt = torch.tensor(Z, dtype=torch.float32); Yt = torch.tensor(Y); model = FeatNet(Z.shape[1])
    n = sum(len(v) for v in train_sets.values())
    for _ in range(ROUNDS18):
        g0 = _flat(model); acc = None
        for c, idx in train_sets.items():
            _set(model, g0); opt = torch.optim.SGD(model.parameters(), lr=lr); order = balanced_order(Y, idx, rng)
            for s in range(0, len(order), 32):
                b = order[s:s + 32]; opt.zero_grad(); F.cross_entropy(model(Zt[b]), Yt[b]).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            u = _flat(model) * (len(idx) / n); acc = u if acc is None else acc + u
        _set(model, acc)
    with torch.no_grad(): return balanced_accuracy_score(Y[test_idx], model(Zt[test_idx]).argmax(1).numpy())

def done_rows():
    if not os.path.exists(LOG18B): return {}
    with open(LOG18B) as f: return {(r["subset"], r["setting"], int(r["seed"])): float(r["acc"]) for r in csv.DictReader(f)}

def log_row(subset, setting, seed, acc):
    new = not os.path.exists(LOG18B)
    with open(LOG18B, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["subset", "setting", "seed", "acc"])
        if new: w.writeheader()
        w.writerow(dict(subset=subset, setting=setting, seed=seed, acc=round(acc, 5)))

if __name__ == "__main__":
    t0 = time.time()
    print("=" * 76); print("  STAGE 18B — IS THE ROBUSTNESS DUE TO THE FRACTAL-CHAOS FAMILY?"); print("=" * 76)
    P218, Y218 = partition_set_218(); Xd, Yd, Pd = build_figshare()
    rng = np.random.default_rng(TEST_SPLIT_SEED); uniq = np.unique(P218); rng.shuffle(uniq)
    test_p = set(uniq[:int(0.2 * len(uniq))]); trmC = ~np.isin(P218, list(test_p)); trC = np.where(trmC)[0]
    teD = np.where(np.isin(Pd, list(test_p)))[0]; trD = np.where(~np.isin(Pd, list(test_p)))[0]
    col = {n: i for i, n in enumerate(FEATURES_FC)}
    print(f"  test patients {len(test_p)}; descriptor slices {len(teD)} test / {len(trD)} train")
    done = done_rows()
    parts = {}
    for alpha in ALPHAS:
        for seed in range(SEEDS18):
            part = dirichlet_partition(Y218[trmC], P218[trmC], N_CLIENTS, alpha, seed)
            pids = {c: set(P218[trC[v]]) for c, v in part.items()}
            cl = {c: trD[np.isin(Pd[trD], list(p))] for c, p in pids.items()}
            parts[(alpha, seed)] = {c: v for c, v in cl.items() if len(v) > 0}
    for name, fs in SUBSETS.items():
        X = Xd[:, [col[f] for f in fs]]
        mu = X[trD].mean(0); sd = np.sqrt(np.maximum((X[trD] ** 2).mean(0) - mu ** 2, 1e-12)); Z = (X - mu) / sd
        for seed in range(SEEDS18):
            if (name, "central", seed) not in done: log_row(name, "central", seed, run_mlp(Z, Yd, {0: trD}, teD, seed))
            for alpha in ALPHAS:
                if (name, f"fed_alpha{alpha}", seed) not in done:
                    log_row(name, f"fed_alpha{alpha}", seed, run_mlp(Z, Yd, parts[(alpha, seed)], teD, seed))
        print(f"  {name}: done ({(time.time() - t0) / 60:.0f} min)")

    rows = done_rows(); get = lambda s, k: np.array([rows[(s, k, i)] for i in range(SEEDS18)])
    print("\n" + "=" * 76); print(f"  RESULTS (slice-level balanced accuracy, {SEEDS18} seeds, same hospitals and test patients)"); print("=" * 76)
    cnn_rel = CNN_REF["damage"] / (CNN_REF["central"] - 1 / 3)
    print(f"  CNN reference (Stage 18, alpha 0.3): centralised {CNN_REF['central']:.3f}, damage {100 * CNN_REF['damage']:.1f}pp, relative damage {cnn_rel:.3f}")
    dmg = {}
    for name in SUBSETS:
        c = get(name, "central")
        print(f"\n  {name}: centralised {c.mean():.3f} +/- {c.std():.3f}")
        for alpha in sorted(ALPHAS):
            f = get(name, f"fed_alpha{alpha}"); d = c - f; rel = d.mean() / (c.mean() - 1 / 3); dmg[(name, alpha)] = d
            tag = "   <- primary" if alpha == PRIMARY_ALPHA else ""
            print(f"    alpha {alpha:<4}: federated {f.mean():.3f} +/- {f.std():.3f}   damage {100 * d.mean():+5.1f}pp   relative damage {rel:.3f}{tag}")
    a, b = dmg[("fractal-chaos family (7)", PRIMARY_ALPHA)], dmg[("other descriptors (12)", PRIMARY_ALPHA)]
    p = stats.wilcoxon(a, b).pvalue if np.any(a != b) else 1.0
    print(f"\n  attribution, alpha {PRIMARY_ALPHA}: damage fractal-chaos {100 * a.mean():+.1f}pp vs other descriptors {100 * b.mean():+.1f}pp "
          f"(paired Wilcoxon p = {p:.4f}, two-sided)")
    c = get("fractal-chaos family (7)", "central"); rel = a.mean() / (c.mean() - 1 / 3)
    ok = a.mean() <= CNN_REF["damage"] - 0.05 and rel < cnn_rel
    print(f"\n  VERDICT (rule fixed before running, alpha {PRIMARY_ALPHA}): {'SUPPORTED' if ok else 'NOT SUPPORTED'}: fractal-chaos family alone "
          f"damage {100 * a.mean():.1f}pp (limit 10.2), relative damage {rel:.3f} (CNN {cnn_rel:.3f}).")
    print("  Consistency check: the 'all 19' arm should reproduce Stage 18 (centralised 0.876, alpha 0.3 federated 0.857).")
    print(f"\nDONE in {(time.time() - t0) / 60:.1f} min. Copy back everything above.\n")
