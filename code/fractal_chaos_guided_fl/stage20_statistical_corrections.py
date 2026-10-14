"""
================================================================================
 CF-SCAFFOLD — STAGE 20: STATISTICAL CORRECTION AND RECTIFICATION (FINAL)
================================================================================
 Thesis: "Fractal-Chaos Guided Federated Learning for Brain Tumor MRI Classification"
 Written in response to an external code review of Stage 19. Decision rules below were fixed before this run;
 they are not an external preregistration (the same data were analysed in earlier stages).

 What changes relative to Stages 10B-19
   * Statistics. Repeated cross-validation folds share patients, so Wilcoxon tests over repeats give p-values
     that are too small. All descriptor comparisons now use the corrected repeated k-fold t-test (Nadeau and
     Bengio, 2003; Bouckaert and Frank, 2004) on per-fold patient-level balanced accuracy, with 95% CIs.
     Federated comparisons add a bootstrap over TEST PATIENTS, so that uncertainty in the test sample is included,
     not only the variation between training seeds.
   * Selection. The final descriptor set is re-estimated by NESTED selection (the choice among the candidate sets
     is made inside each outer training fold), giving an estimate free of selection optimism.
   * Strict tumour-only set: descriptors that use pixels outside the tumour or mostly encode tumour size
     (edge_density, border_grad, binary-mask lacunarity of the tumour) are excluded; mean_grad is recomputed on
     the tumour interior.
   * Glioma vs non-glioma after automatic segmentation: intention-to-treat (a patient whose slices all fail is
     counted as an error), Dice over all slices including failures, all arms on the same split, bootstrap CIs.
   * BraTS: segmentation files found case-insensitively (recovers the case whose file is named "...Segm.nii");
     label 1 is NCR/NET (necrotic and non-enhancing tumour core), not necrosis alone; the tumour mask is the whole
     tumour including oedema; an absent sub-region (value 0 + flag) is kept distinct from a failed computation (NaN).
   * Stage 18 rerun with predictions saved, for the patient-bootstrap CI of the heterogeneity damage gap.
   * Terminology: on Figshare (not skull-stripped) the "brain" mask is the HEAD region; Fb = fractal dimension of
     the head region excluding the tumour. On BraTS (skull-stripped) it is the brain.
   * Engineering: caches carry a version signature and are rejected on mismatch; the final set chosen in Part 1 is
     saved to a file and read by the later parts (no hand-set preset); U-Net training is deterministic; library
     versions are printed; no global warning suppression.

 Decision rules (fixed before this run)
   P1  First, FuzEn2D replaces SampEn2D if it is defined on at least as many slices AND the lower bound of the 95% CI
       of its accuracy difference is >= -0.5 points (non-inferiority). Then each block is tested on the chosen base and
       accepted if the corrected t-test gives p < 0.05 after Holm correction over the two blocks, with a positive gain.
   P2  Glioma vs non-glioma after automatic segmentation (intention-to-treat): lower 95% bootstrap bound >= 0.85,
       and lower bound of the difference to the geometry control >= 0.10.
   P3  BraTS: the sub-region block is accepted if the corrected t-test against the 19 descriptors gives p < 0.05
       with a positive gain.
   P4  Heterogeneity damage at alpha 0.3 with the 19 descriptors of Stage 18: CNN damage minus descriptor damage
       >= 5 points AND the lower bound of the patient-bootstrap 95% CI > 0. The final descriptor set is also reported.
--------------------------------------------------------------------------------
 Kaggle: GPU (one T4 is used). Attach the Figshare dataset and the BraTS 2020 training data. About 5-6 hours;
 every part caches its work and resumes after an interruption.
================================================================================
"""
import os, glob, csv, json, time, hashlib, inspect, re, warnings
import numpy as np
for _pkg, _mod in {"h5py": "h5py", "scikit-image": "skimage", "nibabel": "nibabel"}.items():
    try: __import__(_mod)
    except Exception: os.system(f"pip install {_pkg} --quiet")
import h5py
from PIL import Image
from scipy import ndimage, stats
from scipy.stats import spearmanr
from numpy.lib.stride_tricks import sliding_window_view
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import regionprops, label as sklabel, find_contours
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
import torch, torch.nn as nn, torch.nn.functional as F
warnings.filterwarnings("once", category=UserWarning)          # show each distinct warning once, never silence them

# =============================== CONFIGURATION ================================
INPUT_ROOT  = "/kaggle/input"
DATA_GLOB   = f"{INPUT_ROOT}/**/*.mat"
BRATS_GLOB  = f"{INPUT_ROOT}/**/*t1ce.nii*"
OUT_DIR     = "/kaggle/working"
RUN_P1, RUN_P2, RUN_P3, RUN_P4 = True, True, True, True
IMG_SIZE, N_FOLDS, UNET_EPOCHS = 128, 5, 20
N_REP, NESTED_REP, N_BOOT = 10, 3, 2000
N_CLIENTS, PRIMARY_ALPHA, ALPHAS, FED_SEEDS, FED_ROUNDS, TEST_SPLIT_SEED = 18, 0.3, [0.1, 0.3, 1.0], 20, 60, 42
BRATS_SLICES, BRATS_MIN_AREA = 5, 100
TEST_TRAIN_RATIO = 1.0 / (N_FOLDS - 1)                          # n_test / n_train in patient units for k-fold CV
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TYPE_NAMES = {0: "meningioma", 1: "glioma", 2: "pituitary"}
os.makedirs(OUT_DIR, exist_ok=True)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


# ======================= VALIDATED FUNCTIONS (verbatim from Stage 19) =======================
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

def _windows(gray, region, k):
    """k x k windows of the image whose footprint lies fully inside the region, and their positions."""
    g = gray.astype(np.float64); r_ = region > 0
    W = sliding_window_view(g, (k, k)); pos = np.argwhere(sliding_window_view(r_, (k, k)).all(axis=(2, 3)))
    return g, r_, W, pos

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

def brats_grades():
    """Grade per subject from name_mapping.csv (BraTS 2020) or from HGG/LGG folder names."""
    grade = {}
    for csvp in glob.glob(f"{INPUT_ROOT}/**/name_mapping*.csv", recursive=True):
        with open(csvp) as fh:
            for r in csv.DictReader(fh):
                g = (r.get("Grade") or "").strip().upper()
                for k, v in r.items():
                    if v and "Training" in v and g in ("HGG", "LGG"): grade[v.strip()] = g
    return grade

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

def dice(a, b):
    a, b = a.astype(bool), b.astype(bool); return 2.0 * (a & b).sum() / max(a.sum() + b.sum(), 1)

class FeatNet(nn.Module):
    def __init__(self, d, n=3):
        super().__init__(); self.f = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, n))
    def forward(self, x): return self.f(x)

def _flat(m): return torch.cat([p.data.flatten() for p in m.parameters()])

def _set(m, f):
    i = 0
    for p in m.parameters(): n = p.numel(); p.data.copy_(f[i:i + n].view_as(p)); i += n


# ============================ DESCRIPTOR SETS AND CACHING ============================
NEW = ["SampEn_t", "SampEn_b", "FuzEn_t", "FuzEn_b", "Fbound", "GLac2_t", "GLacSlope_t", "mean_grad_in"]
ALL = FEATURE_ORDER + NEW
BASE19 = FEATURE_ORDER + ["SampEn_t", "SampEn_b"]
FUZ19 = FEATURE_ORDER + ["FuzEn_t", "FuzEn_b"]
BLOCK_A, BLOCK_B = ["Fbound"], ["GLac2_t", "GLacSlope_t"]
HEAD_SIDE = {"Fb", "C", "lacun_b", "GLCM_b", "SampEn_b", "FuzEn_b"}          # computed on the head/brain outside the tumour
NOT_STRICT = {"edge_density", "border_grad", "lacun_t", "mean_grad"}          # use outside pixels, or mostly encode size
SUB = ["FD_enh", "GLac_enh", "FD_ncrnet", "GLac_ncrnet", "enh_present", "ncrnet_present"]
FLAGS = ["enh_present", "ncrnet_present"]

def all_desc(g, mk, head, need=None):
    """All descriptors for one slice. need = set of names to compute (None = all); the rest are NaN."""
    f = features_from(g, mk, head)
    if f is None: return None
    t = ((mk > 0) & (head > 0)).astype(np.uint8); r = ((head > 0) & (t == 0)).astype(np.uint8)
    want = lambda n: need is None or n in need
    f["SampEn_t"] = sampen2d(g, t) if want("SampEn_t") else np.nan
    f["SampEn_b"] = sampen2d(g, r, n_candidates=3000) if want("SampEn_b") else np.nan
    f["FuzEn_t"] = fuzen2d(g, t) if want("FuzEn_t") else np.nan
    f["FuzEn_b"] = fuzen2d(g, r, n_candidates=3000) if want("FuzEn_b") else np.nan
    f["Fbound"] = boundary_fd(t) if want("Fbound") else np.nan
    f["GLac2_t"], f["GLacSlope_t"] = gray_lacunarity(g, t) if (want("GLac2_t") or want("GLacSlope_t")) else (np.nan, np.nan)
    inner = ndimage.binary_erosion(t, iterations=1)                        # interior: gradients use tumour pixels only
    gx, gy = np.gradient(g.astype(float)); grad = np.sqrt(gx ** 2 + gy ** 2)
    f["mean_grad_in"] = float(grad[inner].mean()) if inner.sum() >= 4 else np.nan
    return f

def cols(names, pool=ALL): return [pool.index(n) for n in names]

def signature(extra=""):
    """Version signature: source of every descriptor function + the column order + parameters."""
    fns = [read_cjdata, resize_pair, brain_mask_of, brain_mask_raw, boxcount_fd, glcm_region, shannon_entropy, lacunarity,
           shape_features, edge_features, features_from, sampen2d, fuzen2d, boundary_fd, gray_lacunarity, geometry, all_desc]
    src = "".join(inspect.getsource(f) for f in fns) + "|".join(ALL) + f"|{IMG_SIZE}|{BRATS_SLICES}|{BRATS_MIN_AREA}|" + extra
    return hashlib.md5(src.encode()).hexdigest()

def load_cache(path, sig):
    if not os.path.exists(path): return None
    d = np.load(path, allow_pickle=True)
    if str(d["signature"]) != sig:
        print(f"  cache {os.path.basename(path)} was built by different code: rebuilding"); return None
    return {k: d[k] for k in d.files}

def save_cache(path, sig, **arrays): np.savez_compressed(path, signature=sig, **arrays)

def seg_file(d):
    """Segmentation file of a BraTS case, matched case-insensitively (one 2020 case is named '...Segm.nii')."""
    c = [f for f in os.listdir(d) if re.search("seg", f, re.I) and f.lower().endswith((".nii", ".nii.gz"))]
    return os.path.join(d, sorted(c)[0]) if c else None


# ================================ STATISTICS ================================
def corrected_t(diff_folds, ratio=TEST_TRAIN_RATIO):
    """Corrected repeated k-fold CV t-test (Nadeau and Bengio, 2003; Bouckaert and Frank, 2004) on per-fold
       differences; variance (1/J + n_test/n_train) s^2 instead of s^2 / J. Returns mean, two-sided p, 95% CI."""
    d = np.asarray(diff_folds, float).ravel(); J = len(d); m = d.mean(); s2 = d.var(ddof=1)
    if s2 == 0: return float(m), (1.0 if m == 0 else 0.0), (float(m), float(m))
    se = np.sqrt((1.0 / J + ratio) * s2); t = m / se
    p = 2 * stats.t.sf(abs(t), J - 1); h = stats.t.ppf(0.975, J - 1) * se
    return float(m), float(p), (float(m - h), float(m + h))

def holm(pvals):
    p = np.asarray(pvals, float); order = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order): run = max(run, (m - rank) * p[i]); adj[i] = min(1.0, run)
    return adj

def fmt_test(name, m, p, ci, unit=100):
    return f"  {name:58s} {unit * m:+6.2f} pp  [{unit * ci[0]:+.2f}, {unit * ci[1]:+.2f}]  corrected p = {p:.4f}"

# ================================ CLASSIFIERS ================================
def rf(balanced=False):
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                     ("m", RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1, class_weight="balanced" if balanced else None))])

def fold_bacc(Yte, Pte, pred, proba):
    pv, pt = patient_vote(pred, proba, Pte), patient_truth(Yte, Pte); k = sorted(pv)
    return balanced_accuracy_score([pt[p] for p in k], [pv[p] for p in k])

def cv_per_fold(X, Y, P, sets, n_rep=N_REP, balanced=False):
    """Paired evaluation: every set on the same splits. Returns {name: array (n_rep x k) of per-fold patient-level
       balanced accuracy} and {name: array of per-repeat pooled balanced accuracy}."""
    folds = {n: np.zeros((n_rep, N_FOLDS)) for n in sets}; pooled = {n: [] for n in sets}
    for r in range(n_rep):
        allp = {n: (np.zeros_like(Y), np.zeros((len(Y), len(np.unique(Y))))) for n in sets}
        for k, (tr, te) in enumerate(StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=r).split(X, Y, P)):
            for n, c in sets.items():
                m = rf(balanced).fit(X[tr][:, c], Y[tr]); pr = m.predict(X[te][:, c]); pb = m.predict_proba(X[te][:, c])
                folds[n][r, k] = fold_bacc(Y[te], P[te], pr, pb); allp[n][0][te] = pr; allp[n][1][te] = pb
        for n in sets: pooled[n].append(fold_bacc(Y, P, allp[n][0], allp[n][1]))
    return folds, {n: np.array(v) for n, v in pooled.items()}

def cv_threshold_per_fold(X, Y, P, sets, n_rep=N_REP):
    """Binary; threshold chosen by inner patient-grouped CV on each outer training set. Per-fold patient-level
       balanced accuracy (patient = mean slice probability vs threshold)."""
    folds = {n: np.zeros((n_rep, N_FOLDS)) for n in sets}
    for r in range(n_rep):
        for k, (tr, te) in enumerate(StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=r).split(X, Y, P)):
            pt = patient_truth(Y, P)
            for n, c in sets.items():
                oof = np.zeros(len(tr))
                for itr, ite in StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=100 + r).split(X[tr], Y[tr], P[tr]):
                    oof[ite] = rf(True).fit(X[tr][itr][:, c], Y[tr][itr]).predict_proba(X[tr][ite][:, c])[:, 1]
                ktr = sorted(set(P[tr])); ptr = np.array([oof[P[tr] == q].mean() for q in ktr]); ytr = np.array([pt[q] for q in ktr])
                grid = np.linspace(0.05, 0.95, 91); thr = grid[np.argmax([balanced_accuracy_score(ytr, (ptr >= t).astype(int)) for t in grid])]
                m = rf(True).fit(X[tr][:, c], Y[tr]); pr = m.predict_proba(X[te][:, c])[:, 1]; kte = sorted(set(P[te]))
                folds[n][r, k] = balanced_accuracy_score([pt[q] for q in kte], [int(pr[P[te] == q].mean() >= thr) for q in kte])
    return folds

def nested_selection(X, Y, P, candidates, n_rep=NESTED_REP):
    """Outer folds: choose the candidate set with the best mean inner-CV patient-level balanced accuracy on the outer
       training patients only, refit it, and score it on the outer test patients."""
    scores, chosen = [], []
    for r in range(n_rep):
        for tr, te in StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=r).split(X, Y, P):
            inner = {n: [] for n in candidates}
            for itr, ite in StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=1000 + r).split(X[tr], Y[tr], P[tr]):
                for n, c in candidates.items():
                    m = rf().fit(X[tr][itr][:, c], Y[tr][itr])
                    inner[n].append(fold_bacc(Y[tr][ite], P[tr][ite], m.predict(X[tr][ite][:, c]), m.predict_proba(X[tr][ite][:, c])))
            best = max(candidates, key=lambda n: np.mean(inner[n])); c = candidates[best]
            m = rf().fit(X[tr][:, c], Y[tr]); scores.append(fold_bacc(Y[te], P[te], m.predict(X[te][:, c]), m.predict_proba(X[te][:, c])))
            chosen.append(best)
    return np.array(scores), chosen


# ================================== DATA ==================================
def build_figshare20():
    sig = signature("figshare"); path = f"{OUT_DIR}/stage20_figshare.npz"
    d = load_cache(path, sig)
    if d is not None: return d
    files = sorted(f for f in glob.glob(DATA_GLOB, recursive=True) if f.lower().endswith(".mat"))
    if not files: raise SystemExit(f"No Figshare .mat files under {INPUT_ROOT}. Attach the dataset.")
    X, Y, P, G, IMG, M, H = [], [], [], [], [], [], []; skipped = 0
    for k, path_ in enumerate(files):
        img, label, mask, pid = read_cjdata(path_)
        if img is None or pid is None or mask is None: skipped += 1; continue
        g, mk = resize_pair(img, mask); head = brain_mask_of(g); f = all_desc(g, mk, head)
        if f is None: skipped += 1; continue
        X.append([f[n] for n in ALL]); Y.append(label - 1); P.append(pid); G.append(geometry(g, head)[:10])
        IMG.append(g.astype(np.uint8)); M.append(mk); H.append(head)
        if (k + 1) % 500 == 0: print(f"  Figshare: {k + 1}/{len(files)}")
    print(f"  Figshare: {len(Y)} slices kept, {skipped} skipped (unreadable or tumour/head region too small)")
    d = dict(X=np.array(X), Y=np.array(Y), P=np.array(P), G=np.array(G), IMG=np.stack(IMG), M=np.stack(M), H=np.stack(H))
    save_cache(path, sig, **d); return d

def derive_sets(final):
    ent = [f for f in final if f.startswith(("FuzEn", "SampEn"))]
    fam = ["Ft", "Fb", "C", "lacun_t", "lacun_b"] + ent + [f for f in final if f in BLOCK_A + BLOCK_B]
    tum = [f for f in final if f not in HEAD_SIDE]
    strict = [f for f in tum if f not in NOT_STRICT] + ["mean_grad_in"]
    return fam, tum, strict

# ================================= PART 1 =================================
def part1(D):
    X, Y, P = D["X"], D["Y"], D["P"]
    print("\n" + "=" * 80); print("  PART 1 — FIGSHARE, 3 CLASSES (corrected repeated-CV t-test, nested selection)"); print("=" * 80)
    print(f"  {len(Y)} slices, {len(set(P))} patients, classes {np.bincount(Y)}; {N_REP} x {N_FOLDS}-fold patient-grouped CV")
    nan = {n: int(np.isnan(X[:, ALL.index(n)]).sum()) for n in NEW}
    print("  undefined values (imputed inside training folds): " + ", ".join(f"{n} {v}" for n, v in nan.items()))
    size = X[:, ALL.index("size")]
    print("  Spearman correlation with tumour size: " + ", ".join(f"{n} {spearmanr(X[:, ALL.index(n)], size, nan_policy='omit')[0]:+.2f}"
          for n in ["Ft", "lacun_t", "FuzEn_t", "GLac2_t", "GLacSlope_t", "Fbound", "mean_grad_in"]))
    S = {f"{e}{b}": base + blk for e, base in (("BASE19", BASE19), ("FUZ19", FUZ19))
         for b, blk in (("", []), ("+A", BLOCK_A), ("+B", BLOCK_B), ("+A+B", BLOCK_A + BLOCK_B))}
    folds, pooled = cv_per_fold(X, Y, P, {n: cols(s) for n, s in S.items()})
    print(f"\n  {'set':16s} {'pooled patient bal.acc (mean +/- sd over repeats)':>50s}")
    for n in S: print(f"  {n:16s} {pooled[n].mean():.3f} +/- {pooled[n].std():.3f}")
    print("\n  Decisions (rules fixed before this run):")
    m, p, ci = corrected_t(folds["FUZ19"] - folds["BASE19"])
    cov_ok = nan["FuzEn_t"] + nan["FuzEn_b"] <= nan["SampEn_t"] + nan["SampEn_b"]; fz = cov_ok and ci[0] >= -0.005
    print(fmt_test("FuzEn2D instead of SampEn2D", m, p, ci))
    print(f"    undefined {nan['FuzEn_t'] + nan['FuzEn_b']} vs {nan['SampEn_t'] + nan['SampEn_b']}; non-inferiority needs CI lower bound >= -0.50 pp"
          f" -> {'ADOPTED' if fz else 'not adopted'}")
    bname = "FUZ19" if fz else "BASE19"
    tA = corrected_t(folds[bname + "+A"] - folds[bname]); tB = corrected_t(folds[bname + "+B"] - folds[bname])
    adj = holm([tA[1], tB[1]]); acc = []
    for blk, t, pa, fs in (("Block A (boundary FD)", tA, adj[0], BLOCK_A), ("Block B (grayscale lacunarity)", tB, adj[1], BLOCK_B)):
        ok = t[0] > 0 and pa < 0.05; acc += fs if ok else []
        print(fmt_test(f"{blk} on {bname}", *t) + f"  Holm p = {pa:.4f} -> {'ACCEPTED' if ok else 'not accepted'}")
    final = (FUZ19 if fz else BASE19) + acc
    fname = bname + ("+A" if BLOCK_A[0] in acc else "") + ("+B" if BLOCK_B[0] in acc else "")
    fam, tum, strict = derive_sets(final)
    rest = [f for f in final if f not in fam]
    folds2, pooled2 = cv_per_fold(X, Y, P, {"family": cols(fam), "tumour-only": cols(tum), "strict tumour-only": cols(strict), "final minus family": cols(rest)})
    print(f"\n  FINAL set = {fname} ({len(final)} descriptors): {pooled[fname].mean():.3f} +/- {pooled[fname].std():.3f}")
    for n, k in (("fractal-chaos family", "family"), ("tumour-only", "tumour-only"), ("strict tumour-only", "strict tumour-only"), ("final minus family", "final minus family")):
        print(f"  {n:24s} {pooled2[k].mean():.3f} +/- {pooled2[k].std():.3f}")
    print(fmt_test("final vs the Stage-17 set (BASE19)", *corrected_t(folds[fname] - folds["BASE19"])))
    print(fmt_test("contribution of the fractal-chaos family (final minus it)", *corrected_t(folds[fname] - folds2["final minus family"])))
    print(fmt_test("strict tumour-only vs tumour-only", *corrected_t(folds2["strict tumour-only"] - folds2["tumour-only"])))
    ns, chosen = nested_selection(X, Y, P, {n: cols(s) for n, s in S.items()})
    ref = folds[fname][:NESTED_REP].ravel()
    print(f"\n  NESTED selection among the 8 candidate sets ({NESTED_REP} x {N_FOLDS} outer folds): per-fold patient bal.acc "
          f"{ns.mean():.3f} +/- {ns.std():.3f}")
    print(f"    same outer folds, final set chosen on all data: {ref.mean():.3f}  -> selection optimism {100 * (ref.mean() - ns.mean()):+.2f} pp")
    print("    sets chosen in the outer folds: " + ", ".join(f"{n} {chosen.count(n)}" for n in S if chosen.count(n)))
    info = dict(final=final, fam=fam, tum=tum, strict=strict, final_name=fname)
    json.dump(info, open(f"{OUT_DIR}/stage20_final_set.json", "w"), indent=1); return info

def load_final():
    p = f"{OUT_DIR}/stage20_final_set.json"
    if not os.path.exists(p): raise SystemExit("stage20_final_set.json not found: run Part 1 first (no hand-set preset is used).")
    return json.load(open(p))

def boot_ci(v): return np.percentile(v, [2.5, 97.5])

# ================================= PART 2 =================================
def part2(D, info):
    X, Y, P, G = D["X"], D["Y"], D["P"], D["G"]; yb = (Y == 1).astype(int); final = info["final"]
    print("\n" + "=" * 80); print("  PART 2 — GLIOMA vs NON-GLIOMA (PROXY FOR MALIGNANT vs PREDOMINANTLY BENIGN)"); print("=" * 80)
    print(f"  {int(yb.sum())} glioma slices, {int((1 - yb).sum())} meningioma + pituitary slices; final set {info['final_name']}")
    sets = {"final descriptors": cols(final), "fractal-chaos family": cols(info["fam"]), "strict tumour-only": cols(info["strict"])}
    f_r, p_r = cv_per_fold(X, yb, P, sets); f_g, p_g = cv_per_fold(G, yb, P, {"geometry": list(range(G.shape[1]))})
    print(f"  radiologist masks, {N_REP} x {N_FOLDS}-fold: " + "; ".join(f"{n} {p_r[n].mean():.3f}" for n in sets) + f"; geometry control {p_g['geometry'].mean():.3f}")
    sig = signature("unet|" + "|".join(final) + f"|{UNET_EPOCHS}"); upath = f"{OUT_DIR}/stage20_unet.npz"; u = load_cache(upath, sig)
    split = list(GroupKFold(N_FOLDS).split(X, Y, P))
    if u is None:
        XA = np.full((len(Y), len(ALL)), np.nan); okA = np.zeros(len(Y), bool); dice_all = np.zeros(len(Y)); need = set(final)
        Xi = (D["IMG"].astype(np.float32) / 255.0)[:, None]
        for k, (tr, te) in enumerate(split):
            net = train_unet(Xi[tr], D["M"][tr], seed=k); prob = predict_prob(net, Xi[te])
            for j, i in enumerate(te):
                loc = localise(prob[j], D["H"][i])
                if loc is None: continue                                   # failure: Dice 0, no features
                dice_all[i] = dice(loc, D["M"][i]); f = all_desc(D["IMG"][i].astype(np.float32), loc, D["H"][i], need=need)
                if f is not None: XA[i] = [f[n] for n in ALL]; okA[i] = True
            print(f"  U-Net fold {k + 1}/{N_FOLDS} done")
        save_cache(upath, sig, XA=XA, okA=okA, dice_all=dice_all); u = dict(XA=XA, okA=okA, dice_all=dice_all)
    XA, okA, dice_all = u["XA"], u["okA"], u["dice_all"]
    print(f"  U-Net Dice over ALL {len(Y)} slices (failures = 0): mean {dice_all.mean():.3f}, median {np.median(dice_all):.3f}, "
          f"> 0.7 on {100 * (dice_all > 0.7).mean():.1f}%; usable automatic features on {int(okA.sum())} slices")
    pt = patient_truth(yb, P); arms = {"radiologist": {}, "U-Net (intention-to-treat)": {}, "geometry control": {}, "fractal-chaos family": {}, "strict tumour-only": {}}
    n_fail_pat = 0
    for tr, te in split:
        for arm, Xm, c in (("radiologist", X, cols(final)), ("geometry control", G, list(range(G.shape[1]))),
                           ("fractal-chaos family", X, cols(info["fam"])), ("strict tumour-only", X, cols(info["strict"]))):
            m = rf().fit(Xm[tr][:, c], yb[tr]); arms[arm].update(patient_vote(m.predict(Xm[te][:, c]), m.predict_proba(Xm[te][:, c]), P[te]))
        m = rf().fit(X[tr][:, cols(final)], yb[tr]); ok = te[okA[te]]
        pv = patient_vote(m.predict(XA[ok][:, cols(final)]), m.predict_proba(XA[ok][:, cols(final)]), P[ok]) if len(ok) else {}
        for q in set(P[te]):
            if q not in pv: pv[q] = 1 - pt[q]; n_fail_pat += 1           # every slice failed: counted as an error
        arms["U-Net (intention-to-treat)"].update(pv)
    keys = sorted(pt); yt = np.array([pt[q] for q in keys]); rng = np.random.default_rng(0)
    pred = {a: np.array([arms[a][q] for q in keys]) for a in arms}
    point = {a: balanced_accuracy_score(yt, pred[a]) for a in arms}; boot = {a: [] for a in arms}; diff = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(keys), len(keys))
        if len(set(yt[i])) < 2: continue
        for a in arms: boot[a].append(balanced_accuracy_score(yt[i], pred[a][i]))
        diff.append(boot["U-Net (intention-to-treat)"][-1] - boot["geometry control"][-1])
    print(f"\n  same split for every arm ({len(keys)} patients; {n_fail_pat} patients with no usable automatic slice, counted as errors):")
    for a in arms:
        lo, hi = boot_ci(boot[a]); print(f"    {a:30s} {point[a]:.3f}  [{lo:.3f}, {hi:.3f}]")
    dl, dh = boot_ci(diff); ul = boot_ci(boot["U-Net (intention-to-treat)"])[0]
    ok = ul >= 0.85 and dl >= 0.10
    print(f"    U-Net minus geometry control: {point['U-Net (intention-to-treat)'] - point['geometry control']:+.3f}  [{dl:+.3f}, {dh:+.3f}]")
    print(f"  RULE P2: {'SUPPORTED' if ok else 'NOT SUPPORTED'} (lower bound {ul:.3f} vs 0.85; difference lower bound {dl:+.3f} vs +0.10)")
    print("  Wording: 'glioma versus non-glioma, a proxy for malignant versus predominantly benign tumours'.")


# ================================= PART 3 =================================
def build_brats20():
    sig = signature("brats"); path = f"{OUT_DIR}/stage20_brats.npz"; d = load_cache(path, sig)
    if d is not None: return d
    import nibabel as nib
    files = sorted(glob.glob(BRATS_GLOB, recursive=True)); grade = brats_grades()
    if not files: raise SystemExit(f"No BraTS *t1ce.nii files under {INPUT_ROOT}.")
    X, S, G, Y, P = [], [], [], [], []; no_grade = no_seg = nonstd = 0
    def rs(m): return (np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)) > 127).astype(np.uint8)
    for k, t1 in enumerate(files):
        d_ = os.path.dirname(t1); sid = os.path.basename(d_)
        g_ = grade.get(sid) or ("HGG" if "/HGG/" in t1 else "LGG" if "/LGG/" in t1 else None)
        if g_ is None: no_grade += 1; continue
        sf = seg_file(d_)
        if sf is None: no_seg += 1; continue
        nonstd += "seg" not in os.path.basename(sf)
        vol = nib.load(t1).get_fdata().astype(np.float32); lab = nib.load(sf).get_fdata()
        area = (lab > 0).sum(axis=(0, 1)); zs = [z for z in np.argsort(area)[::-1][:BRATS_SLICES] if area[z] >= BRATS_MIN_AREA]
        for z in zs:
            g, mk = resize_pair(vol[:, :, z], (lab[:, :, z] > 0).astype(np.uint8))   # whole tumour, including oedema
            br = brain_mask_raw(vol[:, :, z]); f = all_desc(g, mk, br)
            if f is None: continue
            sub, flags = [], []
            for lv in (4, 1):                                                   # 4 = enhancing tumour, 1 = NCR/NET
                reg = rs((lab[:, :, z] == lv).astype(np.uint8)) & br; present = reg.sum() >= 64
                fd = boxcount_fd(g, reg) if present else 0.0                    # absent -> 0 (+ flag); failed -> NaN (imputed)
                gl = gray_lacunarity(g, reg)[0] if present else 0.0
                sub += [fd, gl]; flags.append(float(present))
            X.append([f[n] for n in ALL]); S.append(sub + flags); G.append(geometry(g, br)[:10]); Y.append(1 if g_ == "HGG" else 0); P.append(sid)
        if (k + 1) % 50 == 0: print(f"  BraTS: {k + 1}/{len(files)}")
    print(f"  BraTS: {len(set(P))} subjects used; skipped {no_grade} without grade (validation cases), {no_seg} without segmentation; "
          f"{nonstd} segmentation file(s) with a non-standard name recovered")
    d = dict(X=np.array(X), S=np.array(S, dtype=float), G=np.array(G), Y=np.array(Y), P=np.array(P)); save_cache(path, sig, **d); return d

def part3(info):
    B = build_brats20(); X, S, G, Y, P = B["X"], B["S"], B["G"], B["Y"], B["P"]; final = info["final"]
    print("\n" + "=" * 80); print("  PART 3 — BraTS: HIGH-GRADE vs LOW-GRADE GLIOMA (GRADE, not benign vs malignant)"); print("=" * 80)
    pt = patient_truth(Y, P); print(f"  {len(Y)} slices, {len(pt)} patients: {sum(v == 1 for v in pt.values())} HGG, {sum(v == 0 for v in pt.values())} LGG; "
          f"enhancing region on {int(S[:, 4].sum())} slices, NCR/NET on {int(S[:, 5].sum())}; failed sub-region computations: {int(np.isnan(S[:, :4]).sum())}")
    print("  Tumour = whole tumour including oedema; sub-regions and slice selection use the EXPERT segmentation (upper bound).")
    XS = np.hstack([X, S]); POOL = ALL + SUB
    sets = {"BASE19": cols(BASE19, POOL), "BASE19+SUB": cols(BASE19 + SUB, POOL), "BASE19+FLAGS": cols(BASE19 + FLAGS, POOL),
            "FLAGS": cols(FLAGS, POOL), "FINAL": cols(final, POOL), "FINAL+SUB": cols(final + SUB, POOL)}
    fo = cv_threshold_per_fold(XS, Y, P, sets); fg = cv_threshold_per_fold(G, Y, P, {"geometry": list(range(G.shape[1]))})
    for n in sets: print(f"  {n:14s} per-fold balanced accuracy {fo[n].mean():.3f} +/- {fo[n].std():.3f}")
    print(f"  {'geometry':14s} per-fold balanced accuracy {fg['geometry'].mean():.3f} +/- {fg['geometry'].std():.3f}")
    t = corrected_t(fo["BASE19+SUB"] - fo["BASE19"]); ok = t[0] > 0 and t[1] < 0.05
    print(fmt_test("sub-region block added to BASE19", *t) + f" -> {'ACCEPTED' if ok else 'not accepted'}")
    print(fmt_test("sub-region fractal measures vs presence flags alone", *corrected_t(fo["BASE19+SUB"] - fo["BASE19+FLAGS"])))
    print(fmt_test("sub-region block added to the final set (descriptive)", *corrected_t(fo["FINAL+SUB"] - fo["FINAL"])))
    print("  Limitation: most low-grade cases come from one collection (TCGA-LGG); grade is partly confounded with site.")

# ================================= PART 4 =================================
class SmallNet(nn.Module):
    """GroupNorm CNN of the federated experiments (Stages 1-18)."""
    def __init__(self, n=3):
        super().__init__()
        self.c1 = nn.Conv2d(1, 16, 3, padding=1); self.n1 = nn.GroupNorm(4, 16)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1); self.n2 = nn.GroupNorm(4, 32)
        self.c3 = nn.Conv2d(32, 64, 3, padding=1); self.n3 = nn.GroupNorm(8, 64)
        self.fc = nn.Linear(64, n)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.n1(self.c1(x))), 2); x = F.max_pool2d(F.relu(self.n2(self.c2(x))), 2)
        x = F.max_pool2d(F.relu(self.n3(self.c3(x))), 2); return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))

def train_and_predict(kind, Xr, Y, train_sets, test_idx, seed):
    """Stage 18 recipe for both centralised ({0: all training slices}) and federated training: class-balanced sampling,
       gradient clipping at 1.0, plain cross-entropy, sample-weighted federated averaging, FED_ROUNDS rounds x 1 epoch.
       Returns class probabilities on the test slices."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed); lr = 0.01 if kind == "cnn" else 0.05
    Xt = torch.tensor(Xr, dtype=torch.float32); Yt = torch.tensor(Y)
    model = (SmallNet() if kind == "cnn" else FeatNet(Xr.shape[1])).to(DEVICE); n = sum(len(v) for v in train_sets.values())
    for _ in range(FED_ROUNDS):
        g0 = _flat(model); acc = None
        for c, idx in train_sets.items():
            _set(model, g0); opt = torch.optim.SGD(model.parameters(), lr=lr); model.train()
            cc = np.bincount(Y[idx], minlength=3); w = np.array([1.0 / cc[c_] for c_ in Y[idx]]); order = rng.choice(idx, len(idx), p=w / w.sum())
            for s in range(0, len(order), 32):
                b = order[s:s + 32]; opt.zero_grad(); F.cross_entropy(model(Xt[b].to(DEVICE)), Yt[b].to(DEVICE)).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            u = _flat(model) * (len(idx) / n); acc = u if acc is None else acc + u
        _set(model, acc)
    model.eval()
    with torch.no_grad():
        return torch.cat([torch.softmax(model(Xt[test_idx[s:s + 512]].to(DEVICE)), 1).cpu() for s in range(0, len(test_idx), 512)]).numpy()

def part4(D, info):
    X, Y, P = D["X"], D["Y"], D["P"]
    print("\n" + "=" * 80); print("  PART 4 — STAGE 18 RERUN: HETEROGENEITY DAMAGE WITH A BOOTSTRAP OVER TEST PATIENTS"); print("=" * 80)
    rng = np.random.default_rng(TEST_SPLIT_SEED); uniq = np.unique(P); rng.shuffle(uniq); test_p = set(uniq[:int(0.2 * len(uniq))])
    te = np.where(np.isin(P, list(test_p)))[0]; trm = ~np.isin(P, list(test_p)); tr = np.where(trm)[0]
    def zfeat(names):
        Xs = X[:, cols(names)]; fill = np.nansum(Xs[tr], 0) / np.maximum((~np.isnan(Xs[tr])).sum(0), 1)
        Xs = np.where(np.isnan(Xs), fill, Xs); mu, sd = Xs[tr].mean(0), Xs[tr].std(0) + 1e-12; return (Xs - mu) / sd
    reps = {"cnn": ("cnn", (D["IMG"].astype(np.float32) / 255.0)[:, None]), "desc19": ("mlp", zfeat(BASE19)), "descfinal": ("mlp", zfeat(info["final"]))}
    rdir = f"{OUT_DIR}/stage20_runs"; os.makedirs(rdir, exist_ok=True)
    parts, dropped = {}, []
    for a in ALPHAS:
        for s in range(FED_SEEDS):
            cl = dirichlet_partition(Y[trm], P[trm], N_CLIENTS, a, s); parts[(a, s)] = {c: tr[v] for c, v in cl.items()}
            dropped.append(len(tr) - sum(len(v) for v in cl.values()))
    print(f"  test patients {len(test_p)} ({len(te)} slices); training slices dropped by the partition (clients < 4 slices): "
          f"mean {np.mean(dropped):.1f}, max {max(dropped)} of {len(tr)}")
    settings = ["central"] + [f"fed{a}" for a in ALPHAS]
    for rep, (kind, Xr) in reps.items():
        for st in settings:
            for s in range(FED_SEEDS):
                f = f"{rdir}/{rep}__{st}__{s}.npy"
                if os.path.exists(f): continue
                ts = {0: tr} if st == "central" else parts[(float(st[3:]), s)]
                np.save(f, train_and_predict(kind, Xr, Y, ts, te, s))
        print(f"  {rep}: all runs done")
    yt = Y[te]; pids = P[te]; up = np.unique(pids); pidx = {q: np.where(pids == q)[0] for q in up}
    prob = {(rep, st, s): np.load(f"{rdir}/{rep}__{st}__{s}.npy") for rep in reps for st in settings for s in range(FED_SEEDS)}
    acc = {k: balanced_accuracy_score(yt, v.argmax(1)) for k, v in prob.items()}
    print(f"\n  {'representation':12s} {'centralised':>12s}" + "".join(f"{'fed a=' + str(a):>12s}{'damage':>8s}" for a in ALPHAS))
    for rep in reps:
        c = np.mean([acc[(rep, 'central', s)] for s in range(FED_SEEDS)]); line = f"  {rep:12s} {c:12.3f}"
        for a in ALPHAS:
            fm = np.mean([acc[(rep, f'fed{a}', s)] for s in range(FED_SEEDS)]); line += f"{fm:12.3f}{100 * (c - fm):+7.1f}"
        print(line)
    # bootstrap over test patients: per-patient correct counts per class, summed with multinomial weights
    T = np.array([[np.sum(yt[pidx[q]] == c) for c in range(3)] for q in up], float)
    Cc = {k: np.array([[np.sum((v.argmax(1)[pidx[q]] == c) & (yt[pidx[q]] == c)) for c in range(3)] for q in up], float) for k, v in prob.items()}
    rb = np.random.default_rng(1); W = np.stack([np.bincount(rb.integers(0, len(up), len(up)), minlength=len(up)) for _ in range(N_BOOT)]).astype(float)
    tot = W @ T; valid = (tot > 0)
    def bacc_boot(k):
        rec = np.where(valid, (W @ Cc[k]) / np.where(valid, tot, 1), np.nan)   # recall per class; NaN if the class is absent
        return np.nanmean(rec, axis=1)                                     # balanced accuracy per bootstrap replicate
    mean_set = lambda rep, st: np.mean([bacc_boot((rep, st, s)) for s in range(FED_SEEDS)], 0)
    print("\n  damage gap (CNN damage minus descriptor damage), point estimate and 95% bootstrap CI over test patients:")
    verdict = None
    for drep in ("desc19", "descfinal"):
        for a in ALPHAS:
            dc = mean_set("cnn", "central") - mean_set("cnn", f"fed{a}"); dd = mean_set(drep, "central") - mean_set(drep, f"fed{a}")
            gap = dc - dd; pt_ = np.mean([acc[("cnn", "central", s)] - acc[("cnn", f"fed{a}", s)] - acc[(drep, "central", s)] + acc[(drep, f"fed{a}", s)] for s in range(FED_SEEDS)])
            lo, hi = boot_ci(gap); seeds = np.array([acc[("cnn", "central", s)] - acc[("cnn", f"fed{a}", s)] - acc[(drep, "central", s)] + acc[(drep, f"fed{a}", s)] for s in range(FED_SEEDS)])
            pw = stats.wilcoxon(seeds).pvalue if np.any(seeds != 0) else 1.0
            print(f"    {drep:10s} alpha {a:<4}: {100 * pt_:+6.1f} pp  [{100 * lo:+.1f}, {100 * hi:+.1f}]  (seed-level Wilcoxon p = {pw:.4f}, conditional on the test set)")
            if drep == "desc19" and a == PRIMARY_ALPHA: verdict = (pt_ >= 0.05 and lo > 0, pt_, lo)
    ok, g_, lo_ = verdict
    print(f"  RULE P4 (19 descriptors, alpha {PRIMARY_ALPHA}): {'SUPPORTED' if ok else 'NOT SUPPORTED'} (gap {100 * g_:+.1f} pp, CI lower bound {100 * lo_:+.1f} pp)")
    print("  Caveat: descriptors use the radiologist's tumour mask; the CNN sees the image only.")

# ================================== MAIN ==================================
if __name__ == "__main__":
    t0 = time.time()
    print("=" * 80); print("  STAGE 20 — STATISTICAL CORRECTION AND RECTIFICATION"); print("=" * 80)
    import sklearn, scipy, skimage, PIL
    print(f"  device {DEVICE}; numpy {np.__version__}, torch {torch.__version__}, scikit-learn {sklearn.__version__}, scipy {scipy.__version__}, "
          f"scikit-image {skimage.__version__}, Pillow {PIL.__version__}")
    D = build_figshare20() if (RUN_P1 or RUN_P2 or RUN_P4) else None
    info = part1(D) if RUN_P1 else None
    if RUN_P2: part2(D, info or load_final())
    if RUN_P3: part3(info or load_final())
    if RUN_P4: part4(D, info or load_final())
    print(f"\nDONE in {(time.time() - t0) / 60:.1f} min. Copy back everything above.\n")
