"""
==================================================================
 CF-SCAFFOLD — STAGE 11: ACTIVE CONTOUR SEGMENTATION
 Replacing the expert mask with an automatic one, and asking what
 that does to the fractal descriptor.
==================================================================
 WHY THIS MATTERS
   Every result so far used the expert tumour masks distributed with
   the dataset. That was deliberate: it isolates the descriptor from
   segmentation error. But it also means the method as evaluated cannot
   run without a radiologist, which a reviewer has already identified
   as its main practical limitation.

   This stage supplies the automatic delineation and measures the cost.

 THE SEGMENTATION
   A seeded morphological active contour. The seed comes from the
   bright region inside the estimated brain; the contour then evolves
   under a Chan-Vese or geodesic energy. Unseeded initialisation
   (checkerboard) was tested and fails outright, so the seed is not
   optional. The contour evolution earns its place: on a controlled
   test it improved Dice by about 0.10 over the seed alone.

 THE QUESTION WORTH ASKING
   Segmentation error moves the region boundary, which moves Ft and Fb.
   If the error is systematic - the contour consistently a little tight
   or a little loose - then both dimensions shift in the same direction
   and their RATIO may cancel part of it.

   That would be an argument for C that we have not tested and that
   does not depend on accuracy: robustness to imperfect delineation,
   which is exactly the condition under which the method would be used.
   Experiment C below tests it directly, and can refute it.
------------------------------------------------------------------
 Kaggle: CPU. Attach the Figshare .mat dataset. About 30 minutes.
==================================================================
"""

import os, json, warnings
import numpy as np
warnings.filterwarnings("ignore")
try: import h5py
except Exception: os.system("pip install h5py --quiet"); import h5py
from scipy import ndimage, stats
from skimage.segmentation import (checkerboard_level_set, disk_level_set,
                                  morphological_chan_vese,
                                  morphological_geodesic_active_contour,
                                  inverse_gaussian_gradient)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score

DATA_ROOT = "/kaggle/input"
OUT_DIR   = "/kaggle/working"
CACHE     = f"{OUT_DIR}/stage11_cache.npz"
RESULTS   = f"{OUT_DIR}/stage11_results.json"
IMG_SIZE  = 128
N_TREES   = 300
N_REP     = 10
METHOD    = "chanvese"      # "chanvese" or "geodesic"

# ---------------------------------------------------------------
# Region estimation
# ---------------------------------------------------------------
def brain_mask_of(gray):
    m = gray > max(gray.mean() * 0.25, 10)
    m = ndimage.binary_fill_holes(m)
    m = ndimage.binary_opening(m, iterations=2)
    lbl, n = ndimage.label(m)
    if n >= 1:
        s = np.bincount(lbl.ravel()); s[0] = 0; m = lbl == s.argmax()
    return m.astype(np.uint8)

def largest_component(mask):
    lbl, n = ndimage.label(mask)
    if n < 1: return mask.astype(np.uint8)
    s = np.bincount(lbl.ravel()); s[0] = 0
    return (lbl == s.argmax()).astype(np.uint8)

def seed_from_intensity(gray, brain, pct=90):
    """Initial guess: the brightest tissue inside the brain. Contrast-enhanced
       tumour is typically hyperintense on T1-CE, which is what this exploits."""
    inside = gray * brain
    vals = inside[brain > 0]
    if vals.size < 32: return None
    thr = np.percentile(vals, pct)
    seed = ((inside > thr) & (brain > 0)).astype(np.uint8)
    seed = ndimage.binary_opening(seed, iterations=1).astype(np.uint8)
    if seed.sum() < 8: return None
    return largest_component(seed)

def segment_active_contour(gray, method=METHOD, iters=60):
    """Automatic tumour delineation. Returns None when no plausible seed
       exists, which is itself informative and is counted."""
    brain = brain_mask_of(gray)
    seed = seed_from_intensity(gray, brain)
    if seed is None: return None, brain, None
    g = gray / 255.0
    if method == "geodesic":
        gimg = inverse_gaussian_gradient(g)
        ls = morphological_geodesic_active_contour(
            gimg, num_iter=iters, init_level_set=seed,
            smoothing=2, balloon=1, threshold=0.7)
    else:
        ls = morphological_chan_vese(
            g, num_iter=iters, init_level_set=seed,
            smoothing=2, lambda1=1, lambda2=2)
    auto = (np.asarray(ls) > 0).astype(np.uint8) * brain
    if auto.sum() < 8: return None, brain, seed
    auto = largest_component(auto)
    # a contour that has swallowed most of the brain has escaped; reject it
    if auto.sum() > 0.6 * max(brain.sum(), 1): return None, brain, seed
    return auto, brain, seed

# ---------------------------------------------------------------
# Fractal dimension
# ---------------------------------------------------------------
def boxcount_fd(gray, region):
    v = gray[region > 0]
    if v.size < 16: return np.nan
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-6: return np.nan
    g = np.zeros_like(gray, dtype=np.float64)
    g[region > 0] = (gray[region > 0] - lo) / (hi - lo) * 255.0
    H, W = g.shape
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
    sa = np.array(sizes, float); Ns = np.array(Ns, float)
    return float(np.polyfit(np.log(1.0/sa), np.log(Ns), 1)[0])

def fractal_triple(gray, tumour, brain):
    """Return (Ft, Fb, C) for a given delineation, or None."""
    t = (tumour > 0) & (brain > 0)
    f = (brain > 0) & (~t)
    if t.sum() < 16 or f.sum() < 64: return None
    Ft = boxcount_fd(gray, t.astype(np.uint8))
    Fb = boxcount_fd(gray, f.astype(np.uint8))
    if not (np.isfinite(Ft) and np.isfinite(Fb) and Fb > 1e-6): return None
    return Ft, Fb, Ft / Fb

def dice(a, b):
    a = a.astype(bool); b = b.astype(bool)
    return 2.0 * (a & b).sum() / max(a.sum() + b.sum(), 1)

def iou(a, b):
    a = a.astype(bool); b = b.astype(bool)
    return (a & b).sum() / max((a | b).sum(), 1)

# ---------------------------------------------------------------
# Data
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

def build():
    if os.path.exists(CACHE):
        d = np.load(CACHE, allow_pickle=True)
        print(f"  loaded cache: {d['exp'].shape}")
        return (d["exp"], d["auto"], d["Y"], d["PID"], d["dice"], d["iou"],
                int(d["n_fail"]))
    files = []
    for dp, _, fns in os.walk(DATA_ROOT):
        for fn in fns:
            if fn.lower().endswith(".mat"): files.append(os.path.join(dp, fn))
    files = sorted(files)
    print(f"  found {len(files)} .mat files")
    if not files:
        raise SystemExit("No .mat files under /kaggle/input. Attach the Figshare "
                         "collection as a DATASET, not as a notebook.")
    exp, auto, Y, PID, dices, ious = [], [], [], [], [], []
    n_fail = 0
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
        g, m_exp = resize_pair(img, msk, IMG_SIZE)
        m_auto, brain, _ = segment_active_contour(g)
        if m_auto is None:
            n_fail += 1
            continue
        te = fractal_triple(g, m_exp, brain)
        ta = fractal_triple(g, m_auto, brain)
        if te is None or ta is None:
            n_fail += 1
            continue
        exp.append(te); auto.append(ta); Y.append(lab-1); PID.append(pid)
        dices.append(dice(m_auto, m_exp)); ious.append(iou(m_auto, m_exp))
        if (k+1) % 300 == 0:
            print(f"    processed {k+1}/{len(files)}  "
                  f"(mean Dice so far {np.mean(dices):.3f})")
    exp = np.array(exp); auto = np.array(auto)
    Y = np.array(Y); PID = np.array(PID)
    dices = np.array(dices); ious = np.array(ious)
    np.savez_compressed(CACHE, exp=exp, auto=auto, Y=Y, PID=PID,
                        dice=dices, iou=ious, n_fail=n_fail)
    print(f"  cached: {exp.shape}")
    return exp, auto, Y, PID, dices, ious, n_fail


# ---------------------------------------------------------------
# Which initialisation finds the tumour?
# ---------------------------------------------------------------
def init_variants(gray, brain, expert_mask):
    """Four ways to start the contour, from no prior to full expert input.

    The evolution rule is the same in every case, so any difference is
    attributable to where the contour began. The last variant uses the expert
    mask and is therefore not usable in practice; it is included as a ceiling,
    and the gap to the third variant is the price of removing the expert."""
    from skimage.segmentation import checkerboard_level_set, disk_level_set
    out = {}
    out["checkerboard"] = checkerboard_level_set(gray.shape, 6)
    out["centre disk"]  = disk_level_set(gray.shape,
                                         center=(gray.shape[0]//2, gray.shape[1]//2),
                                         radius=25)
    seed = seed_from_intensity(gray, brain)
    out["intensity seed"] = seed if seed is not None else None
    if expert_mask is not None and expert_mask.sum() > 0:
        cy, cx = ndimage.center_of_mass(expert_mask > 0)
        out["expert centroid"] = disk_level_set(gray.shape,
                                                center=(int(cy), int(cx)), radius=8)
    else:
        out["expert centroid"] = None
    return out

def evolve(gray, brain, init):
    if init is None: return None
    ls = morphological_chan_vese(gray/255.0, num_iter=60, init_level_set=init,
                                 smoothing=2, lambda1=1, lambda2=2)
    a = (np.asarray(ls) > 0).astype(np.uint8) * brain
    if a.sum() < 8: return None
    a = largest_component(a)
    if a.sum() > 0.6*max(brain.sum(), 1): return None
    return a

def compare_initialisations(n=300):
    """Report agreement with the expert mask for each initialisation."""
    files = []
    for dp, _, fns in os.walk(DATA_ROOT):
        for fn in fns:
            if fn.lower().endswith(".mat"): files.append(os.path.join(dp, fn))
    files = sorted(files)[:n]
    names = ["checkerboard", "centre disk", "intensity seed", "expert centroid"]
    acc = {k: {"dice": [], "fail": 0} for k in names}
    for k, p in enumerate(files):
        try:
            with h5py.File(p, "r") as f:
                cj = f["cjdata"]
                img = np.array(cj["image"]).astype(np.float32)
                if img.ndim == 2: img = img.T
                msk = np.array(cj["tumorMask"]).astype(np.uint8)
                if msk.ndim == 2: msk = msk.T
        except Exception:
            continue
        g, m = resize_pair(img, msk, IMG_SIZE)
        brain = brain_mask_of(g)
        inits = init_variants(g, brain, m)
        for nm in names:
            a = evolve(g, brain, inits[nm])
            if a is None: acc[nm]["fail"] += 1
            else: acc[nm]["dice"].append(dice(a, m))
        if (k+1) % 100 == 0: print(f"    compared {k+1}/{len(files)}")
    return acc, len(files)

print("=" * 68); print("  WHICH INITIALISATION FINDS THE TUMOUR?"); print("=" * 68)
print("  Same contour evolution in every case; only the starting point differs.\n")
acc, n_tried = compare_initialisations()
print(f"\n  {'initialisation':18s} {'Dice':>18} {'failed':>10}")
for nm in ["checkerboard", "centre disk", "intensity seed", "expert centroid"]:
    d = np.array(acc[nm]["dice"])
    if len(d) == 0:
        print(f"  {nm:18s} {'--':>18} {acc[nm]['fail']:>7}/{n_tried}")
    else:
        print(f"  {nm:18s} {d.mean():.3f} +/- {d.std():.3f}      "
              f"{acc[nm]['fail']:>7}/{n_tried}")
d_auto = np.array(acc["intensity seed"]["dice"])
d_exp  = np.array(acc["expert centroid"]["dice"])
if len(d_auto) and len(d_exp):
    print(f"\n  price of removing the expert: "
          f"{d_exp.mean() - d_auto.mean():+.3f} Dice")
    print("  The intensity-seed pipeline uses no expert information at any stage,")
    print("  so it is the one a deployed system could run. The two contour-free")
    print("  initialisations above it are reported to show that the choice of")
    print("  starting point, not the contour evolution, is what makes this work.")
print()

# ===============================================================
print("=" * 68); print("  SEGMENTING AND MEASURING"); print("=" * 68)
exp, auto, Y, PID, dices, ious, n_fail = build()
print(f"  usable slices {len(exp)}, patients {len(np.unique(PID))}, "
      f"rejected {n_fail}")
results = {"n_slices": int(len(exp)), "n_rejected": int(n_fail)}

# ---------------------------------------------------------------
print("\n" + "=" * 68)
print("  A — HOW GOOD IS THE AUTOMATIC SEGMENTATION?")
print("=" * 68)
print(f"  Dice  : {dices.mean():.3f} +/- {dices.std():.3f}   "
      f"median {np.median(dices):.3f}")
print(f"  IoU   : {ious.mean():.3f} +/- {ious.std():.3f}")
for q in [10, 25, 50, 75, 90]:
    print(f"    {q:2d}th percentile Dice : {np.percentile(dices, q):.3f}")
good = (dices > 0.7).mean() * 100
print(f"\n  slices with Dice above 0.7 : {good:.1f}%")
print(f"  slices the contour refused  : {n_fail} "
      f"({100*n_fail/max(len(exp)+n_fail,1):.1f}% of the dataset)")
results["segmentation"] = {"dice_mean": dices.mean(), "dice_std": dices.std(),
                           "iou_mean": ious.mean(), "pct_above_0.7": good}

# ---------------------------------------------------------------
print("\n" + "=" * 68)
print("  B — WHAT DOES SEGMENTATION ERROR DO TO EACH QUANTITY?")
print("=" * 68)
names = ["Ft", "Fb", "C"]
print(f"  {'quantity':10s} {'expert':>16s} {'automatic':>16s} "
      f"{'corr':>7s} {'rel.err':>9s}")
robust = {}
for i, nm in enumerate(names):
    e, a = exp[:, i], auto[:, i]
    r = np.corrcoef(e, a)[0, 1]
    rel = np.mean(np.abs(a - e) / np.maximum(np.abs(e), 1e-9)) * 100
    robust[nm] = {"corr": float(r), "rel_err_pct": float(rel)}
    print(f"  {nm:10s} {e.mean():8.4f}±{e.std():.4f} "
          f"{a.mean():8.4f}±{a.std():.4f} {r:7.3f} {rel:8.2f}%")
results["robustness"] = robust

print("\n  The question this stage was built to answer:")
rc, rf = robust["C"]["rel_err_pct"], robust["Ft"]["rel_err_pct"]
print(f"    relative error under automatic segmentation")
print(f"      Ft : {rf:.2f}%")
print(f"      Fb : {robust['Fb']['rel_err_pct']:.2f}%")
print(f"      C  : {rc:.2f}%")
err_ft = np.abs(auto[:,0]-exp[:,0])/np.maximum(np.abs(exp[:,0]),1e-9)
err_c  = np.abs(auto[:,2]-exp[:,2])/np.maximum(np.abs(exp[:,2]),1e-9)
try: p_rob = stats.wilcoxon(err_c, err_ft).pvalue
except Exception: p_rob = 1.0
print(f"    C versus Ft : {(1-rc/rf)*100:+.1f}% relative change, "
      f"Wilcoxon p = {p_rob:.4g}")
results["robustness"]["C_vs_Ft_p"] = float(p_rob)
print()
if rc < rf and p_rob < 0.05:
    print("  The ratio survives imperfect delineation better than the tumour")
    print("  dimension alone. Boundary error shifts Ft and Fb together and the")
    print("  quotient removes part of that shift. This is an argument for C")
    print("  that does not rest on accuracy, and it is the condition under")
    print("  which the method would actually be used.")
else:
    print("  The ratio is NOT more robust to segmentation error than Ft. The")
    print("  hoped-for cancellation does not occur, and this line of defence")
    print("  for the descriptor closes. Report it.")

# ---------------------------------------------------------------
print("\n" + "=" * 68)
print("  C — DOES CLASSIFICATION SURVIVE WITHOUT THE RADIOLOGIST?")
print("=" * 68)
def score(F, cols, seed):
    Xs = F[:, cols]
    sp = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    yt, yp = [], []
    for tr, te in sp.split(Xs, Y, groups=PID):
        m = RandomForestClassifier(N_TREES, random_state=seed, n_jobs=-1)
        m.fit(Xs[tr], Y[tr]); pred = m.predict(Xs[te])
        for pid in np.unique(PID[te]):
            sel = PID[te] == pid
            yt.append(int(np.bincount(Y[te][sel]).argmax()))
            yp.append(int(np.bincount(pred[sel]).argmax()))
    return balanced_accuracy_score(yt, yp)

sets = {"Ft, Fb": [0, 1], "C alone": [2], "Ft, Fb, C": [0, 1, 2]}
print(f"  {'features':12s} {'expert masks':>18s} {'automatic masks':>18s} {'cost':>8s}")
tableC = {}
for nm, cols in sets.items():
    se = np.array([score(exp, cols, s) for s in range(N_REP)])
    sa = np.array([score(auto, cols, s) for s in range(N_REP)])
    tableC[nm] = {"expert": [se.mean(), se.std()], "auto": [sa.mean(), sa.std()]}
    print(f"  {nm:12s} {se.mean():8.4f}±{se.std():.4f} "
          f"{sa.mean():8.4f}±{sa.std():.4f} {(sa.mean()-se.mean())*100:+7.1f}pp")
results["classification"] = tableC

print("\n  Note that these use the fractal features ONLY, three at most, so the")
print("  absolute values are far below the full sixteen-feature model. What")
print("  matters here is the gap between the two columns: it is the price of")
print("  removing the radiologist from the pipeline.")

# ---------------------------------------------------------------
print("\n" + "=" * 68)
print("  D — DOES THE ABLATION VERDICT CHANGE?")
print("=" * 68)
for label, F in [("expert masks", exp), ("automatic masks", auto)]:
    a = np.array([score(F, [0, 1], s) for s in range(N_REP)])
    b = np.array([score(F, [0, 1, 2], s) for s in range(N_REP)])
    d = b - a
    try: p = stats.wilcoxon(b, a).pvalue
    except Exception: p = 1.0
    print(f"  {label:16s}  C added to Ft,Fb : {d.mean()*100:+.2f}pp  "
          f"p={p:.4f}  wins {int((d>0).sum())}/{N_REP}")
    results.setdefault("ablation", {})[label] = [d.mean(), p]

print("""
  Under expert masks the ratio added nothing (Stage 10B). If it adds
  something once the delineation is imperfect, that is a finding: the
  ratio would be earning its place precisely where the boundary is
  uncertain. If it still adds nothing, the descriptor's case rests on
  interpretability alone and the thesis should say so plainly.
""")

with open(RESULTS, "w") as f:
    json.dump(results, f, indent=2, default=float)
print(f"  written to {RESULTS}")
print("\nDONE. Copy back all four blocks.\n")
