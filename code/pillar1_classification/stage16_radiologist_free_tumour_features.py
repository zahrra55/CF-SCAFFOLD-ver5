"""
================================================================================
 CF-SCAFFOLD — STAGE 16: DOES RADIOLOGIST-FREE CLASSIFICATION HOLD WITH THE
                         SHORTCUT-ROBUST TUMOUR-REGION FEATURES?
================================================================================
 Why this exists
   The final pipeline found that replacing radiologist masks with U-Net masks
   costs only 0.2 points (0.903 -> 0.902), but with the full 17-feature model.
   Stage 15 then showed that the brain-region features in that model (Fb,
   GLCM_b, lacun_b) partly ride the dataset's slice-selection shortcut
   (retention 0.65), while the 13 tumour-region features are far less dependent
   on it (retention 0.97). Fb is also almost unchanged by the mask (r = 0.997),
   so the small cost may have come from the features that ride the shortcut.
   This stage asks whether the tumour-region features alone survive U-Net masks.

 Design (fair by construction)
   One U-Net per fold, trained on that fold's training patients only; the SAME
   automatic masks feed both feature sets. Each classifier is trained on
   radiologist-mask features of the training patients and applied to the unseen
   test patients twice: with radiologist-mask features (reference) and with
   U-Net-mask features. U-Net only: the active-contour refinement was rejected
   in the final pipeline run (it lowered Dice).
   The cost of automation and its 95% interval come from a paired bootstrap over
   PATIENTS on the same predictions.

 Decision rule for the tumour-only model (fixed before running)
   cost >= -2 points        : radiologist-free classification holds with
                              shortcut-robust features
   cost <= -5 points        : the earlier radiologist-free result depended on
                              the brain-region features
   otherwise                : partial
 Control: the full 17-feature model is run on the same masks; it must reproduce
 the earlier small cost (about -0.2 points), otherwise this run differs from the
 final pipeline and the comparison is not valid.
--------------------------------------------------------------------------------
 Kaggle: GPU. Attach the Figshare .mat dataset. About 25 minutes.
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


TUMOUR = ["Ft", "GLCM_t", "correlation_t", "Dispersion_t", "lacun_t", "size",
          "solidity", "eccentricity", "extent", "circularity", "edge_density", "mean_grad", "border_grad"]
SETS = {"tumour-region only (13)": TUMOUR, "full model (17)": FEATURE_ORDER}

def bal_acc_from(pt, pv, keys): return balanced_accuracy_score([pt[k] for k in keys], [pv[k] for k in keys])

def paired_boot(pt, pv_ref, pv_auto, keys, n=2000, seed=0):
    """95% CI of (auto - reference) patient-level balanced accuracy, resampling patients."""
    rng = np.random.default_rng(seed); keys = np.array(keys); out = []
    for _ in range(n):
        k = rng.choice(keys, len(keys), replace=True)
        y = [pt[x] for x in k]
        if len(set(y)) < 3: continue
        out.append(balanced_accuracy_score(y, [pv_auto[x] for x in k]) - balanced_accuracy_score(y, [pv_ref[x] for x in k]))
    return np.percentile(out, [2.5, 97.5])

if __name__ == "__main__":
    t0 = time.time()
    print("=" * 74); print("  STAGE 16 — RADIOLOGIST-FREE CLASSIFICATION WITH TUMOUR-REGION FEATURES"); print("=" * 74)
    print(f"  device: {DEVICE}")
    feats, Y, PID, G, M, B = load_all()
    X = to_matrix(feats); col = {n: i for i, n in enumerate(FEATURE_ORDER)}
    Xi = (G.astype(np.float32) / 255.0)[:, None]
    auto_feats, dice_u = {}, np.zeros(len(Y))
    preds = {name: {"ref": {}, "auto": {}} for name in SETS}
    for k, (tr, te) in enumerate(GroupKFold(N_FOLDS).split(X, Y, PID)):
        t1 = time.time()
        net = train_unet(Xi[tr], M[tr], seed=k)                      # training patients of this fold only
        prob = predict_prob(net, Xi[te])
        for j, i in enumerate(te):
            loc = localise(prob[j], B[i])
            if loc is None: continue
            dice_u[i] = dice(loc, M[i])
            f = features_from(G[i].astype(np.float32), loc, B[i])
            if f is not None: auto_feats[i] = f
        te_ok = np.array([i for i in te if i in auto_feats], dtype=int)
        Xa = to_matrix([auto_feats[i] for i in te_ok])
        for name, fs in SETS.items():
            idx = [col[n] for n in fs]
            clf = make_search(X[tr][:, idx], Y[tr], PID[tr]).best_estimator_
            preds[name]["ref"].update(patient_vote(clf.predict(X[te_ok][:, idx]), clf.predict_proba(X[te_ok][:, idx]), PID[te_ok]))
            preds[name]["auto"].update(patient_vote(clf.predict(Xa[:, idx]), clf.predict_proba(Xa[:, idx]), PID[te_ok]))
        print(f"  fold {k + 1}: U-Net on {len(tr)} slices, {len(te_ok)}/{len(te)} test slices with usable automatic features, {time.time() - t1:.0f}s")

    pt = patient_truth(Y, PID)
    print(f"\n  U-Net Dice vs radiologist: mean {dice_u.mean():.3f}, median {np.median(dice_u):.3f}  "
          f"(final pipeline run: 0.711 / 0.820)")
    print(f"  slices with usable automatic features: {len(auto_feats)}/{len(Y)}")

    print("\n" + "-" * 74); print("  COST OF REMOVING THE RADIOLOGIST (patient-level balanced accuracy)"); print("-" * 74)
    print(f"  {'feature set':26s} {'radiologist':>12s} {'U-Net':>8s} {'cost':>9s}   95% CI of cost (paired, patients)")
    res = {}
    for name in SETS:
        keys = sorted(set(preds[name]["ref"]) & set(preds[name]["auto"]))
        r = bal_acc_from(pt, preds[name]["ref"], keys); a = bal_acc_from(pt, preds[name]["auto"], keys)
        lo, hi = paired_boot(pt, preds[name]["ref"], preds[name]["auto"], keys)
        res[name] = (r, a, a - r, lo, hi)
        print(f"  {name:26s} {r:12.3f} {a:8.3f} {100 * (a - r):+8.1f}pp   [{100 * lo:+.1f}, {100 * hi:+.1f}]   ({len(keys)} patients)")

    print("\n  Which tumour-region features move when the mask is automatic? (Pearson r, same slices)")
    ok = np.array(sorted(auto_feats), dtype=int)
    for n in TUMOUR + ["Fb"]:
        a = X[ok, col[n]]; b = np.array([auto_feats[i][n] for i in ok])
        r = np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else float("nan")
        print(f"    {n:14s} r = {r:.3f}")

    print("\n" + "=" * 74); print("  VERDICT (rule fixed before running)"); print("=" * 74)
    fr, fa, fc, flo, fhi = res["full model (17)"]
    ctrl_ok = fc > -0.02
    print(f"  control, full model: cost {100 * fc:+.1f}pp (earlier -0.2pp) -> "
          f"{'reproduced, comparison valid' if ctrl_ok else 'NOT reproduced: this run differs from the final pipeline; do not interpret the verdict'}")
    r, a, c, lo, hi = res["tumour-region only (13)"]
    if c >= -0.02:
        v = "HOLDS: radiologist-free classification works with the shortcut-robust tumour-region features."
    elif c <= -0.05:
        v = "DOES NOT HOLD: the earlier radiologist-free result depended on the brain-region features."
    else:
        v = "PARTIAL: tumour-region features lose some accuracy with automatic masks."
    print(f"  tumour-region only: {r:.3f} -> {a:.3f}, cost {100 * c:+.1f}pp [{100 * lo:+.1f}, {100 * hi:+.1f}]")
    print(f"  {v}" if ctrl_ok else "  VERDICT WITHHELD: the control failed, so this run cannot be compared with the final pipeline.")
    print(f"\nDONE in {(time.time() - t0) / 60:.1f} min. Copy back everything above.\n")
