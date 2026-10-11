"""
================================================================================
 CF-SCAFFOLD — STAGE 17B: THREE CHECKS ON STAGE 17
================================================================================
 B'  Fractal-chaos guided federated learning on EXACTLY the test patients of the
     CNN experiments (seed 42 over the 218 patients of the final pipeline), and a
     centralised CNN trained on the same split, so that the heterogeneity damage
     of the two representations is measured on identical patients.
     Federated CNN figures on this split come from the final pipeline run:
     federated averaging 0.554 and with the class-presence correction 0.659
     (20 seeds each). Descriptor runs use two recipes: the CNN baseline recipe
     (20 rounds x 2 local epochs) and the Stage 17 recipe (60 rounds x 1 epoch).
     Rule (fixed before running): the descriptor model's gap to its centralised
     reference is at least 5 points smaller than the CNN's gap to its own.
     Caveat stated in advance: descriptors use the radiologist's tumour mask,
     the CNN uses the image only.

 C'  BraTS grading with a decision threshold chosen INSIDE each training fold
     (inner cross-validation on the training patients only), to separate the
     ranking ability shown by the AUC from the default 0.5 threshold.
     EXPLORATORY: this analysis was decided after seeing Stage 17, and the
     preregistered verdict of Stage 17 (not supported) stands.

 D   Is the Figshare brain-geometry shortcut (Stage 14, 0.750) an artefact of
     slice-wise normalisation? The brain mask is recomputed after normalising
     each slice by the 99th percentile of its NON-tumour pixels, so the tumour
     cannot influence the mask. Rule (fixed before running): if geometry-only
     accuracy stays within 0.03 of the value obtained with the pipeline
     normalisation, the shortcut is not a normalisation artefact.
--------------------------------------------------------------------------------
 Kaggle: attach Figshare and the BraTS dataset used in Stage 17. GPU recommended
 for the centralised CNN (CPU works, slower). Reuses the Stage 17 caches in
 /kaggle/working when present.
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

def fed_features(X, Y, clients, test_idx, mode, seed, lr=0.05, batch=32, rounds=None, local_epochs=1):
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
    for _ in range(FED_ROUNDS if rounds is None else rounds):
        g0 = _flat(model); deltas, cdel = [], []
        for c in ids:
            _set(model, g0); opt = torch.optim.SGD(model.parameters(), lr=lr); steps = 0
            idx = np.concatenate([rng.permutation(clients[c]) for _ in range(local_epochs)])   # local_epochs passes over the client data
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


CNN_FED = {"federated averaging (final pipeline)": 0.554, "with class-presence correction (final pipeline)": 0.659}
CNN_SEEDS, CNN_EPOCHS = 5, 30

class SmallNet(nn.Module):
    """Same architecture as the federated CNN of the final pipeline (GroupNorm)."""
    def __init__(self, n=3):
        super().__init__()
        self.c1 = nn.Conv2d(1, 16, 3, padding=1); self.n1 = nn.GroupNorm(4, 16)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1); self.n2 = nn.GroupNorm(4, 32)
        self.c3 = nn.Conv2d(32, 64, 3, padding=1); self.n3 = nn.GroupNorm(8, 64)
        self.fc = nn.Linear(64, n)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.n1(self.c1(x))), 2); x = F.max_pool2d(F.relu(self.n2(self.c2(x))), 2)
        x = F.max_pool2d(F.relu(self.n3(self.c3(x))), 2); return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))

def figshare_pass():
    """One pass over the .mat files: the 218-patient set of the final pipeline (features_from not None),
       images for the centralised CNN, and brain geometry under two normalisations."""
    files = sorted(f for f in glob.glob(DATA_GLOB, recursive=True) if f.lower().endswith(".mat"))
    PIDs, Ys, IMG, GA, GB = [], [], [], [], []
    for k, path in enumerate(files):
        img, label, mask, pid = read_cjdata(path)
        if img is None or pid is None or mask is None: continue
        g, mk = resize_pair(img, mask); brain = brain_mask_of(g)
        if features_from(g, mk, brain) is None: continue
        # tumour-independent normalisation: scale by the 99th percentile of non-tumour pixels
        nt = img[mask == 0]; p99 = np.percentile(nt, 99) if nt.size else img.max()
        gb = np.clip(img / max(p99, 1e-6) * 255.0, 0, 255)
        gb = np.asarray(Image.fromarray(gb.astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)).astype(np.float32)
        PIDs.append(pid); Ys.append(label - 1); IMG.append(g.astype(np.uint8))
        GA.append(geometry(g, brain)); GB.append(geometry(gb, brain_mask_of(gb)))
        if (k + 1) % 500 == 0: print(f"  Figshare pass: {k + 1}/{len(files)}")
    return np.array(PIDs), np.array(Ys), np.stack(IMG), np.array(GA), np.array(GB)

def test_patients_of(pids_218):
    rng = np.random.default_rng(TEST_SPLIT_SEED); uniq = np.unique(pids_218); rng.shuffle(uniq)
    return set(uniq[:int(0.2 * len(uniq))])

def part_b2(Xd, Yd, Pd, P218, Y218, IMG):
    print("\n" + "=" * 76); print("  PART B' — SAME TEST PATIENTS AS THE CNN EXPERIMENTS"); print("=" * 76)
    test_p = test_patients_of(P218)
    te = np.where(np.isin(Pd, list(test_p)))[0]; trm = ~np.isin(Pd, list(test_p)); trp = np.where(trm)[0]
    print(f"  CNN split: {len(test_p)} test patients; present in the descriptor data: {len(set(Pd[te]))} "
          f"({len(te)} test slices with descriptors)")
    # centralised CNN on the identical split (all slices of the 218-patient set)
    teI = np.where(np.isin(P218, list(test_p)))[0]; trI = np.where(~np.isin(P218, list(test_p)))[0]
    Xi = torch.tensor((IMG.astype(np.float32) / 255.0)[:, None]); Yi = torch.tensor(Y218); cnn = []
    for seed in range(CNN_SEEDS):
        torch.manual_seed(seed); r = np.random.default_rng(seed); net = SmallNet().to(DEVICE)
        opt = torch.optim.SGD(net.parameters(), lr=0.01)
        for _ in range(CNN_EPOCHS):
            net.train(); idx = r.permutation(trI)
            for s in range(0, len(idx), 32):
                b = idx[s:s + 32]; opt.zero_grad()
                F.cross_entropy(net(Xi[b].to(DEVICE)), Yi[b].to(DEVICE)).backward(); opt.step()
        net.eval()
        with torch.no_grad(): pr = torch.cat([net(Xi[teI[s:s + 256]].to(DEVICE)).argmax(1).cpu() for s in range(0, len(teI), 256)]).numpy()
        cnn.append(balanced_accuracy_score(Y218[teI], pr)); print(f"  centralised CNN seed {seed}: {cnn[-1]:.3f}")
    cnn_c = float(np.mean(cnn))
    # descriptor runs, two recipes
    res = {}
    for recipe, (rounds, epochs) in {"CNN baseline recipe (20 rounds x 2 epochs)": (20, 2),
                                     "Stage 17 recipe (60 rounds x 1 epoch)": (60, 1)}.items():
        for mode in ("fedavg", "restricted"):
            accs = []
            for seed in range(FED_SEEDS):
                cl = dirichlet_partition(Yd[trm], Pd[trm], N_CLIENTS, PARTITION_ALPHA, seed); cl = {c: trp[v] for c, v in cl.items()}
                accs.append(balanced_accuracy_score(Yd[te], fed_features(Xd, Yd, cl, te, mode, seed, rounds=rounds, local_epochs=epochs)))
            res[(recipe, mode)] = np.array(accs)
    mu, sd = Xd[trp].mean(0), Xd[trp].std(0) + 1e-12; cen = []
    for seed in range(FED_SEEDS):
        torch.manual_seed(seed); net = FeatNet(Xd.shape[1]); opt = torch.optim.SGD(net.parameters(), lr=0.05)
        Z = torch.tensor((Xd - mu) / sd, dtype=torch.float32); Yt = torch.tensor(Yd); r = np.random.default_rng(seed)
        for _ in range(60):
            idx = r.permutation(trp)
            for s in range(0, len(idx), 32): b = idx[s:s + 32]; opt.zero_grad(); F.cross_entropy(net(Z[b]), Yt[b]).backward(); opt.step()
        with torch.no_grad(): cen.append(balanced_accuracy_score(Yd[te], net(Z[te]).argmax(1).numpy()))
    desc_c = float(np.mean(cen))
    print(f"\n  {'model':68s} {'slice bal.acc':>13s} {'damage vs centralised':>22s}")
    print(f"  {'CNN on images, centralised (this run)':68s} {cnn_c:13.3f}")
    for k, v in CNN_FED.items(): print(f"  {'CNN, ' + k:68s} {v:13.3f} {100 * (cnn_c - v):+20.1f}pp")
    print(f"  {'descriptor MLP, centralised':68s} {desc_c:13.3f}")
    for (recipe, mode), a in res.items():
        print(f"  {'descriptors, ' + mode + ', ' + recipe:68s} {a.mean():13.3f} {100 * (desc_c - a.mean()):+20.1f}pp   collapse {int((a <= 0.40).sum())}/{FED_SEEDS}")
    cnn_dmg = cnn_c - CNN_FED["federated averaging (final pipeline)"]                       # damage = centralised - federated
    desc_dmg = desc_c - res[("CNN baseline recipe (20 rounds x 2 epochs)", "fedavg")].mean()
    ok = (cnn_dmg - desc_dmg) >= 0.05
    print(f"\n  damage of federated averaging under the same recipe: CNN {100 * cnn_dmg:+.1f}pp, descriptors {100 * desc_dmg:+.1f}pp")
    print("  (positive damage = federated training is worse than centralised training)")
    print(f"  VERDICT (rule fixed before running): {'SUPPORTED' if ok else 'NOT SUPPORTED'}: fractal-chaos descriptors "
          f"{'reduce' if ok else 'do not reduce by 5 points or more'} the damage of missing classes to federated training.")
    print("  Caveat: descriptors are computed from the radiologist's tumour mask; the CNN sees the image only.")

def part_c2(X, Gm, Y, PID):
    print("\n" + "=" * 76); print("  PART C' — BraTS GRADING WITH A THRESHOLD CHOSEN INSIDE TRAINING FOLDS (EXPLORATORY)"); print("=" * 76)
    col = {n: i for i, n in enumerate(FEATURES_FC)}
    sets = {"fractal-chaos family (7)": X[:, [col[f] for f in FC_FAMILY]], "all 19 descriptors": X,
            "non-fractal descriptors (12)": X[:, [col[f] for f in FEATURES_FC if f not in FC_FAMILY]],
            "brain-mask geometry only (shortcut control)": Gm[:, :10]}
    def pmean(proba, pid, keys): return np.array([proba[pid == p].mean() for p in keys])
    print(f"  {'feature set':46s} {'bal.acc':>8s} {'sensitivity':>12s} {'specificity':>12s}   (patient level, {N_REP} repeats)")
    for name, Xs in sets.items():
        ba, se, sp = [], [], []
        for seed in range(N_REP):
            pt = patient_truth(Y, PID); yhat = {}
            for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(Xs, Y, PID):
                oof = np.zeros(len(tr))                       # inner cross-validation on training patients only
                for itr, ite in StratifiedGroupKFold(5, shuffle=True, random_state=100 + seed).split(Xs[tr], Y[tr], PID[tr]):
                    oof[ite] = rf(True).fit(Xs[tr][itr], Y[tr][itr]).predict_proba(Xs[tr][ite])[:, 1]
                ktr = sorted(set(PID[tr])); ptr = pmean(oof, PID[tr], ktr); ytr = np.array([pt[p] for p in ktr])
                grid = np.linspace(0.05, 0.95, 91)
                thr = grid[np.argmax([balanced_accuracy_score(ytr, (ptr >= t).astype(int)) for t in grid])]
                m = rf(True).fit(Xs[tr], Y[tr]); kte = sorted(set(PID[te]))
                pte = pmean(m.predict_proba(Xs[te])[:, 1], PID[te], kte)
                yhat.update({p: int(v >= thr) for p, v in zip(kte, pte)})
            k = sorted(yhat); yt = np.array([pt[p] for p in k]); yp = np.array([yhat[p] for p in k])
            ba.append(balanced_accuracy_score(yt, yp)); se.append((yp[yt == 1] == 1).mean()); sp.append((yp[yt == 0] == 0).mean())
        print(f"  {name:46s} {np.mean(ba):8.3f} {np.mean(se):12.3f} {np.mean(sp):12.3f}")
    print("  Sensitivity = high-grade recall, specificity = low-grade recall. Exploratory: the preregistered")
    print("  Stage 17 verdict (default threshold, not supported) is unchanged by this analysis.")

def part_d(P218, Y218, GA, GB):
    print("\n" + "=" * 76); print("  PART D — IS THE FIGSHARE GEOMETRY SHORTCUT A NORMALISATION ARTEFACT?"); print("=" * 76)
    r = np.corrcoef(GA[:, 0], GB[:, 0])[0, 1]
    print(f"  brain area under the two normalisations: Pearson r = {r:.3f}")
    out = {}
    for name, G in (("pipeline normalisation (slice min-max)", GA), ("tumour-independent normalisation", GB)):
        for sub, cols in (("11 variables", slice(0, 11)), ("10 shape variables", slice(0, 10))):
            a, _ = repeated_patient_cv(G[:, cols], Y218, P218); out[(name, sub)] = a.mean()
            print(f"  {name:40s} {sub:18s} patient bal.acc {a.mean():.3f} +/- {a.std():.3f}")
    d = abs(out[("pipeline normalisation (slice min-max)", "11 variables")] - out[("tumour-independent normalisation", "11 variables")])
    print(f"\n  VERDICT (rule fixed before running): difference {d:.3f} -> "
          f"{'NOT an artefact: the shortcut survives tumour-independent normalisation.' if d <= 0.03 else 'PARTLY an artefact: the Stage 14 figure must be revised.'}")

if __name__ == "__main__":
    t0 = time.time()
    print("=" * 76); print("  STAGE 17B — SAME-SPLIT FEDERATED COMPARISON, GRADING THRESHOLD, MASK CHECK"); print("=" * 76)
    print(f"  device: {DEVICE}")
    P218, Y218, IMG, GA, GB = figshare_pass()
    print(f"  final-pipeline set: {len(Y218)} slices, {len(set(P218))} patients")
    Xd, Yd, Pd = build_figshare()
    part_b2(Xd, Yd, Pd, P218, Y218, IMG)
    Xb, Gb, Yb, Pb = build_brats(); part_c2(Xb, Gb, Yb, Pb)
    part_d(P218, Y218, GA, GB)
    print(f"\nDONE in {(time.time() - t0) / 60:.1f} min. Copy back everything above.\n")
