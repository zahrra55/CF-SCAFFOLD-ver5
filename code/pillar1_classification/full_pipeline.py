"""
================================================================================
 CF-SCAFFOLD — Brain tumour MRI: interpretable classification and federated
 learning under missing classes.            FINAL VERSION FOR SUPERVISOR REVIEW
================================================================================
 Researcher : Zahraa Ibrahim Khaleel
 Supervisor : Prof. Dr. Mohammed Sahib Mehdi
 Department of Computer Science, College of Science, Al-Nahrain University
 Dataset    : Figshare brain tumour MRI (Cheng et al., 2015; 3064 T1-CE slices,
              233 patients, 3 tumour types, radiologist masks, patient IDs).
 Platform   : Kaggle, GPU. Attach the Figshare .mat collection as a dataset.
--------------------------------------------------------------------------------
 WHAT THE FILE DOES — four stages, each behind a switch below

 1. PILLAR I  (RUN_PILLAR_1) — classification from interpretable descriptors.
    Regions: tumour (radiologist mask) and brain-without-tumour. Features: the
    fractal family Ft, Fb and their ratio C = Ft/Fb, plus 14 texture, entropy,
    lacunarity, shape and edge features (17 in total). Random forest, nested
    patient-grouped cross-validation, slice and patient-level balanced accuracy.
    C is reported as an interpretable member of the fractal family: over ten
    repeated splits it adds nothing on top of Ft and Fb, while the family as a
    whole adds about +1.7 points (see docs/08 in the repository).

 2. AUTOMATIC DELINEATION (RUN_SEGMENTATION) — can the radiologist mask be
    replaced? A small U-Net says WHERE the tumour is; a morphological Chan-Vese
    active contour, restricted to a box around that location, draws the
    boundary. The U-Net for each fold is trained only on that fold's training
    patients, and the classifier trained on radiologist-mask features is then
    applied to features from the automatic masks of the unseen test patients.
    This answers the practical question directly: what does removing the
    radiologist cost? (Earlier seeding from the brightest region failed,
    median Dice 0.000, because it did not find the tumour.)

 3. PILLAR II (RUN_PILLAR_2) — federated learning under class skew.
    18 simulated hospitals, Dirichlet(0.3). Most clients miss at least one
    tumour type, and naive training collapses. The class-presence-guided
    restricted softmax removes the gradient on absent-class logits. Eight-row
    ablation over 20 seeds; rows 1-7 aggregate with weighted FedAvg, row 8 adds
    real SCAFFOLD control variates (Karimireddy et al., ICML 2020).

 4. LEAKAGE GUARD (RUN_LEAKAGE_GUARD) — shows why per-patient personalisation
    is invalid on this dataset (an image-blind predictor scores ~1.0).
--------------------------------------------------------------------------------
 CHANGES IN THIS VERSION (all verified before release; see docs/07-08)

  [Pillar I — these change the Pillar I numbers; earlier figures such as 0.919
   were produced with the uncorrected features and are superseded]
   F1  Shannon entropy now uses probabilities. The earlier histogram used
       density=True, so the value depended on the intensity range instead of
       the shape of the distribution.
   F2  GLCM texture is computed from pixel pairs inside the region only, with
       region-relative quantisation (1st-99th percentile, 8 levels, 4 angles).
       Earlier it was computed on the whole image with the background set to
       zero, so it mainly measured tumour size and border.
   F3  Differential box counting uses the box height of Sarkar & Chaudhuri
       (1994), h = s*G/M; earlier h = s. Vectorised (identical output to the
       loop version, about four times faster).
   F4  Ft is now a feature (17 features). The earlier set contained Fb and C
       but not Ft, which did not match how the descriptor is presented.
   F5  Patient-level vote: ties are broken by mean predicted probability
       instead of always going to the lowest class index.
   F8  Lacunarity counts the last row and column of boxes, which the earlier
       loop skipped (minor).
  [Pillar II]
   F6  SCAFFOLD global control variate: plain mean of the participating
       clients' control-variate changes, as in the paper. Earlier it was
       sample-weighted and scaled by 17/18 although all existing clients train
       every round. Only row 8 is affected; rows 1-7 are unchanged.
   F7  Documentation: the train/test patient split is FIXED across the 20 seeds
       (TEST_SPLIT_SEED); seeds vary the client partition, initialisation and
       batch order. Earlier text wrongly said the seed also changed the split.
  [New]
   N1  Stage 2, automatic delineation by U-Net localiser + active contour.
--------------------------------------------------------------------------------
 Statistics: paired Wilcoxon, Cohen's d (paired), 10,000-resample bootstrap CI.
 GroupNorm in every trained network. Results stream to /kaggle/working/*.csv;
 Pillar II resumes automatically after an interrupted session.
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
RUN_PILLAR_1      = True      # interpretable classification (about 10 min)
RUN_SEGMENTATION  = True      # U-Net localiser + active contour (about 20 min on GPU)
RUN_PILLAR_2      = True      # federated ablation, 8 configs x 20 seeds (several hours; resumable)
RUN_LEAKAGE_GUARD = True      # protocol check (seconds)
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

# =============================================================================
#  STAGE 1 — PILLAR I: CLASSIFICATION WITH RADIOLOGIST MASKS
# =============================================================================
def run_classification(feats, Y, PID):
    print("\n" + "=" * 70); print("  STAGE 1 — PILLAR I: CLASSIFICATION (radiologist masks)"); print("=" * 70)
    X = to_matrix(feats)
    folds, slice_acc, pv, pt = [], [], {}, patient_truth(Y, PID)
    for k, (tr, te) in enumerate(GroupKFold(N_FOLDS).split(X, Y, PID)):
        gs = make_search(X[tr], Y[tr], PID[tr])
        pred, proba = gs.predict(X[te]), gs.predict_proba(X[te])
        slice_acc.append(balanced_accuracy_score(Y[te], pred))
        pv.update(patient_vote(pred, proba, PID[te]))
        folds.append((tr, te, gs.best_estimator_))
        print(f"  fold {k + 1}: slice balanced accuracy {slice_acc[-1]:.3f}   best {gs.best_params_}")
    keys = sorted(pv)
    pa = balanced_accuracy_score([pt[p] for p in keys], [pv[p] for p in keys])
    print(f"\n  slice-level balanced accuracy   : {np.mean(slice_acc):.3f} +/- {np.std(slice_acc):.3f}")
    print(f"  PATIENT-level balanced accuracy : {pa:.3f}   ({len(keys)} patients, majority vote)")
    print("  (Patient and slice levels are not comparable: voting alone adds several points.)")

    print("\n  One-vs-rest AUC of each fractal descriptor on its own:")
    print(f"  {'':6s}" + "".join(f"{TYPE_NAMES[c]:>13s}" for c in range(3)))
    for name in ("Ft", "Fb", "C"):
        v = X[:, FEATURE_ORDER.index(name)]; row = []
        for c in range(3):
            a = roc_auc_score((Y == c).astype(int), v); row.append(max(a, 1 - a))
        print(f"  {name:6s}" + "".join(f"{a:13.3f}" for a in row))
    print("  (Direction-free AUC, max(AUC, 1-AUC); 0.5 = no separation.)")

    imp = np.mean([f[2].named_steps["rf"].feature_importances_ for f in folds], axis=0)
    order = np.argsort(imp)[::-1]
    print("\n  Random-forest importance (mean over folds), top 8:")
    for i in order[:8]: print(f"    {FEATURE_ORDER[i]:14s} {imp[i]:.3f}")

    with open(f"{OUT_DIR}/pillar1_results.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["metric", "value"])
        w.writerow(["slice_balanced_accuracy_mean", round(float(np.mean(slice_acc)), 4)])
        w.writerow(["patient_balanced_accuracy", round(float(pa), 4)])
        w.writerow(["n_slices", len(Y)]); w.writerow(["n_patients", len(keys)])
    return folds, pa

# =============================================================================
#  STAGE 2 — AUTOMATIC DELINEATION: U-NET LOCALISER + ACTIVE CONTOUR
# =============================================================================
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

def run_segmentation(feats, Y, PID, G, M, B, folds=None):
    print("\n" + "=" * 70); print("  STAGE 2 — AUTOMATIC DELINEATION: U-Net localiser + active contour"); print("=" * 70)
    X = to_matrix(feats); Xi = (G.astype(np.float32) / 255.0)[:, None]
    if folds is None:
        folds = []
        for tr, te in GroupKFold(N_FOLDS).split(X, Y, PID):
            folds.append((tr, te, make_search(X[tr], Y[tr], PID[tr]).best_estimator_))
    rows = []; F_auto = {"unet": {}, "unet+contour": {}}
    for k, (tr, te, clf) in enumerate(folds):
        t0 = time.time()
        net = train_unet(Xi[tr], M[tr], seed=k)      # training patients of this fold only
        prob = predict_prob(net, Xi[te])
        for j, i in enumerate(te):
            loc = localise(prob[j], B[i])
            if loc is None:
                rows.append(dict(i=i, pid=PID[i], y=Y[i], found=0, dice_unet=0.0, dice_ref=0.0, contour_used=0)); continue
            ref, used = refine_contour(G[i].astype(np.float32), loc, B[i])
            rows.append(dict(i=i, pid=PID[i], y=Y[i], found=1, dice_unet=dice(loc, M[i]),
                             dice_ref=dice(ref, M[i]), contour_used=int(used)))
            for name, msk in (("unet", loc), ("unet+contour", ref)):
                f = features_from(G[i].astype(np.float32), msk, B[i])
                if f is not None: F_auto[name][i] = f
        print(f"  fold {k + 1}: U-Net trained on {len(tr)} slices, {len(te)} test slices, {time.time() - t0:.0f}s")

    d_u = np.array([r["dice_unet"] for r in rows]); d_r = np.array([r["dice_ref"] for r in rows])
    found = np.array([r["found"] for r in rows]); used = np.array([r["contour_used"] for r in rows])
    print(f"\n  Agreement with the radiologist mask (all {len(rows)} slices; not found = Dice 0):")
    print(f"    {'method':24s} {'mean':>7s} {'median':>7s} {'>0.7':>7s}")
    for name, d in (("U-Net only", d_u), ("U-Net + active contour", d_r)):
        print(f"    {name:24s} {d.mean():7.3f} {np.median(d):7.3f} {100 * (d > 0.7).mean():6.1f}%")
    print(f"    tumour found on {found.sum()}/{len(rows)} slices; contour accepted on "
          f"{used.sum()}/{found.sum()} (otherwise the U-Net region was kept)")
    print("    reference, earlier brightest-region seed + contour: mean 0.034, median 0.000")
    try: p_dice = stats.wilcoxon(d_r[found == 1], d_u[found == 1]).pvalue
    except Exception: p_dice = float("nan")
    print(f"    does the contour improve on the U-Net region? mean diff "
          f"{(d_r - d_u)[found == 1].mean():+.3f}, Wilcoxon p = {p_dice:.3g}")

    print("\n  What removing the radiologist costs (classifier trained on radiologist-mask")
    print("  features of the training patients, applied to the unseen test patients):")
    pt = patient_truth(Y, PID); res = {}
    common = sorted(set(F_auto["unet"]) & set(F_auto["unet+contour"]))
    common = np.array(common, dtype=int)
    for name in ("radiologist", "unet", "unet+contour"):
        pv = {}
        for tr, te, clf in folds:
            idx = np.intersect1d(te, common)
            if len(idx) == 0: continue
            Xt = X[idx] if name == "radiologist" else to_matrix([F_auto[name][i] for i in idx])
            pv.update(patient_vote(clf.predict(Xt), clf.predict_proba(Xt), PID[idx]))
        keys = sorted(pv); res[name] = balanced_accuracy_score([pt[p] for p in keys], [pv[p] for p in keys])
    print(f"    slices with usable automatic features: {len(common)}/{len(Y)} "
          f"({len(set(PID[common]))} patients); the same subset is used for all three rows")
    print(f"    {'masks at test time':26s} patient-level balanced accuracy")
    for name, lab in (("radiologist", "radiologist (reference)"), ("unet", "U-Net only"), ("unet+contour", "U-Net + active contour")):
        print(f"    {lab:26s} {res[name]:.3f}   ({100 * (res[name] - res['radiologist']):+.1f} pp)")

    print("\n  How much each fractal descriptor moves when the mask is automatic")
    print("  (Pearson r, radiologist vs U-Net + contour, same slices):")
    for nm in ("Ft", "Fb", "C"):
        a = X[common, FEATURE_ORDER.index(nm)]
        b = np.array([F_auto["unet+contour"][i][nm] for i in common])
        print(f"    {nm:3s} r = {np.corrcoef(a, b)[0, 1]:.3f}")

    with open(f"{OUT_DIR}/segmentation_per_slice.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    return res

# =============================================================================
#  STAGE 3 — PILLAR II: FEDERATED LEARNING UNDER CLASS SKEW
# =============================================================================
class SmallNet(nn.Module):
    """Small CNN with GroupNorm (BatchNorm statistics are unreliable across non-IID clients)."""
    def __init__(self, n=3):
        super().__init__()
        self.c1 = nn.Conv2d(1, 16, 3, padding=1); self.n1 = nn.GroupNorm(4, 16)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1); self.n2 = nn.GroupNorm(4, 32)
        self.c3 = nn.Conv2d(32, 64, 3, padding=1); self.n3 = nn.GroupNorm(8, 64)
        self.fc = nn.Linear(64, n)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.n1(self.c1(x))), 2)
        x = F.max_pool2d(F.relu(self.n2(self.c2(x))), 2)
        x = F.max_pool2d(F.relu(self.n3(self.c3(x))), 2)
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))

def _flat(m): return torch.cat([p.data.flatten() for p in m.parameters()])
def _set(m, f):
    i = 0
    for p in m.parameters():
        n = p.numel(); p.data.copy_(f[i:i + n].view_as(p)); i += n

def restricted_ce(logits, yb, present):
    """Class-presence-guided restricted softmax: the logits of classes absent from
       this client are set to -1e9 before the softmax, which removes their gradient.
       For an absent class c the ordinary gradient is dL/dz_c = p_c > 0 at every step,
       pushing its weights down with no opposing signal; masking removes that term.
       The mask uses only which classes a client holds, not C, Ft or Fb."""
    mask = torch.full((logits.shape[1],), -1e9, device=logits.device)
    for c in present: mask[c] = 0.0
    return F.cross_entropy(logits + mask, yb)

def _balanced_loader(X, Y, idx, batch):
    ds = TensorDataset(torch.tensor(X[idx]), torch.tensor(Y[idx]))
    y = Y[idx]; cc = np.bincount(y, minlength=3)
    w = np.array([1.0 / cc[c] if cc[c] > 0 else 0.0 for c in y])
    return DataLoader(ds, batch_size=batch,
                      sampler=WeightedRandomSampler(torch.tensor(w, dtype=torch.double), len(w), replacement=True))

def federated_train(X, Y, client_idx, cfg, seed, test_idx):
    """One federated run. Aggregation is weighted FedAvg; cfg['scaffold']=True adds
       SCAFFOLD control variates (Karimireddy et al., 2020, Option II):
         local step:  g <- g + (c - c_i)
         after K steps: c_i <- c_i - c + (x - y_i) / (K * lr)
         server:      c <- c + mean_i(delta c_i)     (all existing clients take part) [F6]
       With scaffold=False the code path is the one used for rows 1-7."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = SmallNet().to(DEVICE)
    clients = list(client_idx); ni = {c: len(client_idx[c]) for c in clients}
    use_scaffold = cfg.get("scaffold", False)
    if use_scaffold:
        c_global = torch.zeros_like(_flat(model))
        c_local = {c: torch.zeros_like(_flat(model)) for c in clients}   # persists across rounds
    present = {c: list(np.unique(Y[client_idx[c]])) for c in clients}
    for rd in range(cfg["rounds"]):
        gflat = _flat(model); gparams = [p.data.clone() for p in model.parameters()]
        lr = cfg["peak_lr"]
        if cfg["warmup_cosine"]:
            lr = (cfg["peak_lr"] * (rd + 1) / 2 if rd < 2
                  else cfg["peak_lr"] * 0.5 * (1 + np.cos(np.pi * (rd - 2) / max(1, cfg["rounds"] - 2))))
        deltas, c_deltas = [], []
        for c in clients:
            _set(model, gflat)
            loader = (_balanced_loader(X, Y, client_idx[c], 32) if cfg["balanced"]
                      else DataLoader(TensorDataset(torch.tensor(X[client_idx[c]]), torch.tensor(Y[client_idx[c]])),
                                      batch_size=32, shuffle=True))
            opt = torch.optim.SGD(model.parameters(), lr=lr); model.train(); steps = 0
            for _ in range(cfg["local_epochs"]):
                for xb, yb in loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE); opt.zero_grad()
                    logits = model(xb)
                    loss = restricted_ce(logits, yb, present[c]) if cfg["restricted_softmax"] else F.cross_entropy(logits, yb)
                    if cfg["fedprox_mu"] > 0:
                        loss = loss + cfg["fedprox_mu"] / 2 * sum(((p - gp) ** 2).sum() for p, gp in zip(model.parameters(), gparams))
                    loss.backward()
                    if use_scaffold:
                        with torch.no_grad():
                            i = 0
                            for p in model.parameters():
                                n = p.numel(); p.grad.add_((c_global[i:i + n] - c_local[c][i:i + n]).view_as(p)); i += n
                    if cfg["grad_clip"]: torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step(); steps += 1
            y_k = _flat(model)
            if use_scaffold:
                new_c = c_local[c] - c_global + (gflat - y_k) / (max(steps, 1) * lr)
                c_deltas.append(new_c - c_local[c]); c_local[c] = new_c
            deltas.append((ni[c], y_k - gflat))
        tot = sum(n for n, _ in deltas)
        _set(model, gflat + sum(n / tot * d for n, d in deltas))
        if use_scaffold: c_global = c_global + torch.stack(c_deltas).mean(0)
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X[test_idx]).to(DEVICE)).argmax(1).cpu().numpy()
    return balanced_accuracy_score(Y[test_idx], pred)

def dirichlet_partition(Y, PID, n_clients, alpha, seed):
    """Patient-level Dirichlet partition (Hsu et al., 2019): every slice of a patient
       goes to the same client. Clients with fewer than 4 slices are dropped."""
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

_BASE = dict(rounds=20, local_epochs=2, peak_lr=0.01, grad_clip=False, warmup_cosine=False,
             fedprox_mu=0.0, balanced=False, restricted_softmax=False, scaffold=False)
def _cfg(**kw): c = dict(_BASE); c.update(kw); return c
ABLATION = {
    "baseline":                     _cfg(),
    "+grad_clip":                   _cfg(grad_clip=True),
    "+warmup_cosine":               _cfg(grad_clip=True, warmup_cosine=True, peak_lr=0.003),
    "+more_rounds":                 _cfg(grad_clip=True, warmup_cosine=True, peak_lr=0.003, rounds=60, local_epochs=1),
    "+balanced_sampling":           _cfg(grad_clip=True, warmup_cosine=True, peak_lr=0.003, rounds=60, local_epochs=1, balanced=True),
    "+restricted_softmax":          _cfg(grad_clip=True, rounds=60, local_epochs=1, balanced=True, restricted_softmax=True),
    "+restricted_softmax+FedProx":  _cfg(grad_clip=True, rounds=60, local_epochs=1, balanced=True, restricted_softmax=True, fedprox_mu=0.01),
    "+restricted_softmax+SCAFFOLD": _cfg(grad_clip=True, rounds=60, local_epochs=1, balanced=True, restricted_softmax=True, scaffold=True),
}

def _cohend(a, b): d = np.asarray(a) - np.asarray(b); return d.mean() / (d.std(ddof=1) + 1e-12)
def _boot(a, b, n=10000):
    d = np.asarray(a) - np.asarray(b); rng = np.random.default_rng(0)
    return np.percentile(d[rng.integers(0, len(d), (n, len(d)))].mean(1), [2.5, 97.5])
def _paired(label, a, b):
    p = stats.wilcoxon(a, b).pvalue if np.any(a != b) else 1.0
    lo, hi = _boot(a, b)
    print(f"    {label:42s} {100 * (a.mean() - b.mean()):+6.2f} pp   p={p:.4f}   d={_cohend(a, b):5.2f}"
          f"   95% CI [{100 * lo:+.1f}, {100 * hi:+.1f}]   wins {int((a > b).sum())}/{len(a)}")

def run_federated(X, Y, PID):
    print("\n" + "=" * 70); print("  STAGE 3 — PILLAR II: FEDERATED LEARNING UNDER CLASS SKEW"); print("=" * 70)
    rng = np.random.default_rng(TEST_SPLIT_SEED); uniq = np.unique(PID); rng.shuffle(uniq)
    test_pids = set(uniq[:int(0.2 * len(uniq))])
    test_idx = np.where(np.isin(PID, list(test_pids)))[0]
    train_mask = ~np.isin(PID, list(test_pids)); train_pos = np.where(train_mask)[0]
    print(f"  held-out test patients: {len(test_pids)} ({len(test_idx)} slices), fixed for all seeds [F7]")
    miss, n_cl = [], []
    for s in range(SEEDS):
        ci = dirichlet_partition(Y[train_mask], PID[train_mask], N_CLIENTS, PARTITION_ALPHA, s)
        miss.append(sum(1 for v in ci.values() if len(np.unique(Y[train_pos[v]])) < 3)); n_cl.append(len(ci))
    print(f"  clients missing >=1 tumour type: {np.mean(miss):.1f} of {np.mean(n_cl):.1f} on average over {SEEDS} seeds\n")
    csv_path = f"{OUT_DIR}/stability_log.csv"
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f: csv.DictWriter(f, fieldnames=["config", "seed", "acc", "t"]).writeheader()
    def done():
        with open(csv_path) as f: return {(r["config"], int(r["seed"])) for r in csv.DictReader(f)}
    finished = done()
    for name, cfg in ABLATION.items():
        for seed in range(SEEDS):
            if (name, seed) in finished: continue
            ci = dirichlet_partition(Y[train_mask], PID[train_mask], N_CLIENTS, PARTITION_ALPHA, seed)
            ci = {c: train_pos[v] for c, v in ci.items()}
            acc = federated_train(X, Y, ci, cfg, seed, test_idx)
            with open(csv_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=["config", "seed", "acc", "t"]).writerow(
                    dict(config=name, seed=seed, acc=round(acc, 4), t=round(time.time())))
        print(f"  finished: {name}")
    rows = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f): rows.setdefault(r["config"], {})[int(r["seed"])] = float(r["acc"])
    print(f"\n  {'configuration':32s} {'mean':>7s} {'std':>7s} {'collapse':>9s}")
    for name in ABLATION:
        a = np.array(list(rows.get(name, {}).values()))
        if len(a): print(f"  {name:32s} {a.mean():7.3f} {a.std():7.3f} {int((a <= COLLAPSE_THRESH).sum()):>6d}/{len(a)}")
    def pair(x, y):
        s = sorted(set(rows.get(x, {})) & set(rows.get(y, {})))
        return np.array([rows[x][k] for k in s]), np.array([rows[y][k] for k in s])
    print("\n  Paired comparisons over identical seeds:")
    _paired("restricted softmax  vs  baseline", *pair("+restricted_softmax", "baseline"))
    _paired("+ FedProx           vs  restricted softmax", *pair("+restricted_softmax+FedProx", "+restricted_softmax"))
    _paired("+ SCAFFOLD          vs  restricted softmax", *pair("+restricted_softmax+SCAFFOLD", "+restricted_softmax"))
    print("  Reading: the first line is the contribution; the last two test whether a drift")
    print("  correction on top of it adds anything (earlier runs: it did not).")

# =============================================================================
#  STAGE 4 — LEAKAGE GUARD
# =============================================================================
def leakage_guard(Y, PID):
    print("\n" + "=" * 70); print("  STAGE 4 — LEAKAGE GUARD (per-patient personalisation)"); print("=" * 70)
    rng = np.random.default_rng(0); uniq = np.unique(PID); rng.shuffle(uniq)
    correct = total = 0
    for pid in uniq[:int(0.2 * len(uniq))]:
        idx = np.where(PID == pid)[0]; rng.shuffle(idx)
        ns = max(1, len(idx) // 2); sup, qry = idx[:ns], idx[ns:]
        if len(qry) == 0: continue
        correct += (Y[qry] == np.bincount(Y[sup]).argmax()).sum(); total += len(qry)
    print(f"  image-blind 'copy the support label' predictor: accuracy {correct / max(total, 1):.3f}")
    print("  Each patient has one tumour type, so a support/query split within a patient leaks")
    print("  the label. Per-patient fine-tuning results are therefore not reported.")

# =============================================================================
#  MAIN
# =============================================================================
if __name__ == "__main__":
    t_start = time.time()
    print("=" * 70); print("  CF-SCAFFOLD — FINAL PIPELINE"); print("=" * 70)
    print(f"  device: {DEVICE}")
    feats, Y, PID, G, M, B = load_all()
    folds = None
    if RUN_PILLAR_1: folds, _ = run_classification(feats, Y, PID)
    if RUN_SEGMENTATION: run_segmentation(feats, Y, PID, G, M, B, folds)
    if RUN_PILLAR_2: run_federated((G.astype(np.float32) / 255.0)[:, None], Y, PID)
    if RUN_LEAKAGE_GUARD: leakage_guard(Y, PID)
    print("\n" + "=" * 70)
    print(f"  DONE in {(time.time() - t_start) / 60:.1f} min. CSV outputs in {OUT_DIR}:")
    print("    pillar1_results.csv, segmentation_per_slice.csv, stability_log.csv")
    print("=" * 70)
