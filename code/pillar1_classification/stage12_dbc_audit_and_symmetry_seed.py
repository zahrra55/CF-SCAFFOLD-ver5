"""
==================================================================
 CF-SCAFFOLD — STAGE 12: TWO DECISIVE TESTS
==================================================================
 PART A — Does fixing two code defects collapse the reported effect?

   A colleague's script reported an extremely clean separation of the
   three tumour types by C = Ft/Fb (ANOVA p = 4e-272). Inspection found
   two defects:

     BUG 1 (brain mask): the script casts RAW MRI intensity directly to
     uint8 before thresholding: `gray_mri.astype(np.uint8)`. Figshare
     T1-CE values commonly run into the hundreds or low thousands, so
     this truncates by wraparound (value mod 256), corrupting the brain
     mask in a way that depends on each file's own raw intensity scale.

     BUG 2 (normalisation): the script sets the background to zero
     with `np.where(mask, img, 0)` and then rescales using the ARRAY-
     WIDE min and max, which is always [0, region_max] because of the
     zero fill. The intended per-region min/max normalisation (which
     our own pipeline uses and separately validated as stable under
     acquisition-like perturbation) is not what this computes.

   This stage runs BOTH the buggy version, reproduced faithfully, and a
   fixed version, on the SAME slices, and reports whether the ANOVA
   separation survives the fix. If it collapses, the original result was
   an artefact of the two defects, not a discovery about tumour biology.

 PART B — Symmetry-based seeding for automatic delineation

   Stage 11 found that a seeded active contour reaches Dice 0.66 from a
   good seed but only 0.05 from "brightest region in the brain", because
   many tumours are not the brightest structure present (vessels, choroid
   plexus, enhancing normal tissue compete, and some tumours are not
   hyperintense at all).

   Bilateral asymmetry is a different and more targeted cue: healthy
   brain tissue is approximately left-right symmetric, and a tumour
   breaks that symmetry regardless of whether it is brighter or darker
   than its surroundings. The seed is the connected region of highest
   mean asymmetry, not the brightest region.

   A first implementation of this failed in initial testing (a synthetic
   check found the seed nearly empty). Two errors were found and are
   fixed here: a fixed percentile threshold sits too close to the
   background level when the lesion is a small fraction of the brain, so
   an Otsu threshold on the (smoothed) asymmetry map is used instead; and
   selecting the largest connected component picks up noise rather than
   the lesion, so the component with the highest MEAN asymmetry is
   selected instead. Both fixes were verified on synthetic bright and
   dark lesions before being included here.
------------------------------------------------------------------
 Kaggle: CPU is sufficient. Attach the Figshare .mat dataset.
 Runtime: roughly 25 minutes.
==================================================================
"""

import os, json, warnings
import numpy as np
warnings.filterwarnings("ignore")
try: import h5py
except Exception: os.system("pip install h5py --quiet"); import h5py
try: import cv2
except Exception: os.system("pip install opencv-python-headless --quiet"); import cv2
from scipy import ndimage, stats
from skimage.filters import threshold_otsu
from skimage.segmentation import morphological_chan_vese

DATA_ROOT = "/kaggle/input"
OUT_DIR   = "/kaggle/working"
CACHE     = f"{OUT_DIR}/stage12_cache.npz"
RESULTS   = f"{OUT_DIR}/stage12_results.json"

# =================================================================
# PART A — reproduce the two bugs faithfully, then fix them
# =================================================================
def differential_box_counting(gray_image, mask=None, min_box=2, max_box=None,
                              num_scales=10):
    """Verbatim reproduction of the colleague's DBC, unchanged, so that
       BUG 2 (array-wide zero-floor normalisation) is faithfully present
       when this function is called on a zero-filled region."""
    img = np.array(gray_image, dtype=np.float64)
    if img.max() != img.min():
        img = 255.0 * (img - img.min()) / (img.max() - img.min())
    H, W = img.shape; M = min(H, W); img = img[:M, :M]
    mask = (mask[:M, :M] > 0) if mask is not None else np.ones((M, M), bool)
    if max_box is None: max_box = M // 4
    box_sizes = np.unique(np.logspace(np.log10(min_box), np.log10(max_box),
                                      num=num_scales, dtype=int))
    N_r, r_vals, G = [], [], 256.0
    for s in box_sizes:
        if s <= 0 or s > M: continue
        h = (G / M) * s
        if h == 0: h = 1.0
        nby, nbx = M // s, M // s
        cimg = img[:nby*s, :nbx*s]; cmask = mask[:nby*s, :nbx*s]
        blocks = cimg.reshape(nby, s, nbx, s)
        mblocks = cmask.reshape(nby, s, nbx, s)
        valid = mblocks.any(axis=(1, 3))
        if not valid.any(): continue
        masked = np.where(mblocks, blocks, np.nan)
        with np.errstate(all="ignore"):
            bmin = np.nanmin(masked, axis=(1, 3)); bmax = np.nanmax(masked, axis=(1, 3))
        bmin, bmax = bmin[valid], bmax[valid]
        n_r = np.floor(bmax/h) - np.floor(bmin/h) + 1
        N_r.append(np.sum(n_r)); r_vals.append(M/s)
    if len(N_r) < 3: return np.nan
    return float(np.polyfit(np.log(r_vals), np.log(N_r), 1)[0])

def fd_buggy(gray_raw, mask):
    """BUG 1 + BUG 2 as in the original script: raw-intensity uint8 cast
       for the brain mask, and zero-filled background before DBC."""
    _, brain = cv2.threshold(gray_raw.astype(np.uint8), 10, 1, cv2.THRESH_BINARY)
    rest = cv2.bitwise_and(brain, cv2.bitwise_not(mask))
    I_t = np.where(mask == 1, gray_raw, 0)
    I_b = np.where(rest == 1, gray_raw, 0)
    Ft = differential_box_counting(I_t, mask=mask)
    Fb = differential_box_counting(I_b, mask=rest)
    if not (np.isfinite(Ft) and np.isfinite(Fb)) or Fb == 0: return None
    return Ft, Fb, Ft/Fb, int(brain.sum()), int(rest.sum())

def fd_fixed(gray_raw, mask):
    """Same DBC estimator, both defects corrected: the brain mask is
       thresholded on properly [0,255]-scaled intensity, and each region
       is rescaled using its OWN min/max rather than the zero-padded
       array-wide range."""
    g = gray_raw.astype(np.float64)
    g8 = (255.0*(g-g.min())/(g.max()-g.min())).astype(np.uint8) if g.max()>g.min() else g.astype(np.uint8)
    _, brain = cv2.threshold(g8, 10, 1, cv2.THRESH_BINARY)
    brain = ndimage.binary_fill_holes(brain).astype(np.uint8)
    rest = cv2.bitwise_and(brain, cv2.bitwise_not(mask))

    def region_scaled_fd(region_mask):
        vals = g[region_mask > 0]
        if vals.size < 16: return np.nan
        lo, hi = vals.min(), vals.max()
        if hi - lo < 1e-6: return np.nan
        scaled = np.zeros_like(g)
        scaled[region_mask > 0] = (g[region_mask > 0]-lo)/(hi-lo)*255.0
        return differential_box_counting(scaled, mask=region_mask)

    Ft = region_scaled_fd(mask); Fb = region_scaled_fd(rest)
    if not (np.isfinite(Ft) and np.isfinite(Fb)) or Fb == 0: return None
    return Ft, Fb, Ft/Fb, int(brain.sum()), int(rest.sum())

# =================================================================
# PART B — symmetry-based seeding (fixed version)
# =================================================================
def brain_mask_of(gray_0_255):
    m = gray_0_255 > max(gray_0_255.mean()*0.25, 10)
    m = ndimage.binary_fill_holes(m); m = ndimage.binary_opening(m, iterations=2)
    lbl, n = ndimage.label(m)
    if n >= 1:
        s = np.bincount(lbl.ravel()); s[0] = 0; m = lbl == s.argmax()
    return m.astype(np.uint8)

def symmetry_seed(gray_0_255, brain):
    """The connected region of highest bilateral asymmetry. Otsu
       thresholding adapts to the actual bimodal split instead of
       assuming a fixed percentile, and the winning component is chosen
       by mean asymmetry rather than by area, so a small strong lesion is
       not out-voted by a larger patch of weak noise."""
    H, W = gray_0_255.shape
    cols = np.where(brain.any(axis=0))[0]
    axis_col = int((cols.min()+cols.max())/2) if len(cols) else W//2
    mirrored = np.roll(gray_0_255[:, ::-1], 2*axis_col-(W-1), axis=1)
    diff = np.abs(gray_0_255.astype(float) - mirrored.astype(float))
    diff = ndimage.gaussian_filter(diff, 1.0) * brain
    vals = diff[brain > 0]
    if vals.size < 32: return None
    try: thr = threshold_otsu(vals)
    except Exception: thr = np.percentile(vals, 90)
    seed = ((diff > thr) & (brain > 0)).astype(np.uint8)
    seed = ndimage.binary_opening(seed, iterations=1).astype(np.uint8)
    lbl, n = ndimage.label(seed)
    if n < 1: return None
    best, best_score = None, -1.0
    for i in range(1, n+1):
        comp = lbl == i
        if comp.sum() < 8: continue
        score = diff[comp].mean()
        if score > best_score: best_score, best = score, comp
    return best.astype(np.uint8) if best is not None else None

def intensity_seed(gray_0_255, brain, pct=90):
    inside = gray_0_255 * brain; vals = inside[brain > 0]
    if vals.size < 32: return None
    thr = np.percentile(vals, pct)
    seed = ((inside > thr) & (brain > 0)).astype(np.uint8)
    seed = ndimage.binary_opening(seed, iterations=1).astype(np.uint8)
    if seed.sum() < 8: return None
    lbl, n = ndimage.label(seed)
    if n >= 1:
        s = np.bincount(lbl.ravel()); s[0] = 0; seed = (lbl == s.argmax()).astype(np.uint8)
    return seed

def evolve_contour(gray_0_255, brain, seed, iters=60):
    if seed is None: return None
    ls = morphological_chan_vese(gray_0_255/255.0, num_iter=iters,
                                 init_level_set=seed, smoothing=2,
                                 lambda1=1, lambda2=2)
    a = (np.asarray(ls) > 0).astype(np.uint8) * brain
    if a.sum() < 8: return None
    lbl, n = ndimage.label(a)
    if n >= 1:
        s = np.bincount(lbl.ravel()); s[0] = 0; a = (lbl == s.argmax()).astype(np.uint8)
    if a.sum() > 0.6*max(brain.sum(), 1): return None
    return a

def dice(a, b):
    a = a.astype(bool); b = b.astype(bool)
    return 2.0*(a & b).sum()/max(a.sum()+b.sum(), 1)

# =================================================================
# Data
# =================================================================
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

def build(max_n=None):
    if os.path.exists(CACHE):
        d = np.load(CACHE, allow_pickle=True)
        print(f"  loaded cache: {d['buggy'].shape}")
        return {k: d[k] for k in d.files}
    files = []
    for dp, _, fns in os.walk(DATA_ROOT):
        for fn in fns:
            if fn.lower().endswith(".mat"): files.append(os.path.join(dp, fn))
    files = sorted(files)
    if max_n: files = files[:max_n]
    print(f"  found {len(files)} .mat files (using {len(files)})")
    if not files:
        raise SystemExit("No .mat files under /kaggle/input. Attach the Figshare "
                         "collection as a DATASET, not as a notebook.")
    buggy, fixed, Y, PID = [], [], [], []
    dice_int, dice_sym = [], []
    n_bug_fail = n_fix_fail = 0
    for k, p in enumerate(files):
        try:
            with h5py.File(p, "r") as f:
                cj = f["cjdata"]
                lab = int(np.array(cj["label"]).ravel()[0])
                img_raw = np.array(cj["image"]).astype(np.float64)
                if img_raw.ndim == 2: img_raw = img_raw.T
                msk_raw = np.array(cj["tumorMask"]).astype(np.uint8)
                if msk_raw.ndim == 2: msk_raw = msk_raw.T
                pid = read_pid(cj)
        except Exception:
            continue
        if pid is None: continue

        # Part A: exactly as the colleague's script sees the data -- the
        # RAW image and mask, no resizing, no prior normalisation.
        rb = fd_buggy(img_raw, msk_raw)
        rf = fd_fixed(img_raw, msk_raw)
        if rb is None: n_bug_fail += 1
        if rf is None: n_fix_fail += 1
        if rb is None or rf is None: continue
        buggy.append(rb[:3]); fixed.append(rf[:3]); Y.append(lab-1); PID.append(pid)

        # Part B: on the same 128x128 pipeline used throughout this project
        g128, m128 = resize_pair(img_raw, msk_raw, 128)
        brain = brain_mask_of(g128)
        si = intensity_seed(g128, brain); ss = symmetry_seed(g128, brain)
        ai = evolve_contour(g128, brain, si); asym = evolve_contour(g128, brain, ss)
        dice_int.append(dice(ai, m128) if ai is not None else 0.0)
        dice_sym.append(dice(asym, m128) if asym is not None else 0.0)

        if (k+1) % 300 == 0:
            print(f"    processed {k+1}/{len(files)}   "
                  f"(A rejects: bug {n_bug_fail}, fixed {n_fix_fail} | "
                  f"B mean Dice: intensity {np.mean(dice_int):.3f}, "
                  f"symmetry {np.mean(dice_sym):.3f})")

    out = dict(buggy=np.array(buggy), fixed=np.array(fixed),
              Y=np.array(Y), PID=np.array(PID),
              dice_int=np.array(dice_int), dice_sym=np.array(dice_sym),
              n_bug_fail=n_bug_fail, n_fix_fail=n_fix_fail)
    np.savez_compressed(CACHE, **out)
    print(f"  cached: {out['buggy'].shape}")
    return out

# =================================================================
print("=" * 68); print("  PROCESSING (both parts share one pass)"); print("=" * 68)
d = build()
buggy, fixed, Y, PID = d["buggy"], d["fixed"], d["Y"], d["PID"]
dice_int, dice_sym = d["dice_int"], d["dice_sym"]
print(f"  Part A usable slices: {len(buggy)}  (rejected: buggy {int(d['n_bug_fail'])}, "
      f"fixed {int(d['n_fix_fail'])})")
print(f"  Part B usable slices: {len(dice_int)}")
results = {}

# ---------------------------------------------------------------
print("\n" + "=" * 68)
print("  PART A — DOES THE EFFECT SURVIVE THE FIX?")
print("=" * 68)
names3 = ["Ft", "Fb", "C"]
class_names = {0: "meningioma", 1: "glioma", 2: "pituitary"}
for tag, arr in [("BUGGY (as run by the colleague)", buggy),
                 ("FIXED (both defects corrected)", fixed)]:
    print(f"\n  {tag}")
    C = arr[:, 2]
    groups = [C[Y == c] for c in sorted(set(Y))]
    fstat, p_anova = stats.f_oneway(*groups)
    hstat, p_kw = stats.kruskal(*groups)
    for c in sorted(set(Y)):
        v = C[Y == c]
        print(f"    {class_names.get(c, c):12s} n={len(v):4d}  "
              f"C = {v.mean():.4f} +/- {v.std():.4f}")
    print(f"    ANOVA  : F={fstat:.2f}  p={p_anova:.3e}")
    print(f"    Kruskal: H={hstat:.2f}  p={p_kw:.3e}")
    results[tag] = {"anova_p": p_anova, "kruskal_p": p_kw,
                    "means": {class_names.get(c, c): float(C[Y==c].mean())
                             for c in sorted(set(Y))}}

corr = np.corrcoef(buggy[:, 2], fixed[:, 2])[0, 1]
print(f"\n  correlation between buggy-C and fixed-C across the same slices: {corr:.3f}")
results["buggy_vs_fixed_corr"] = float(corr)

print("\n  VERDICT:")
p_bug = results["BUGGY (as run by the colleague)"]["anova_p"]
p_fix = results["FIXED (both defects corrected)"]["anova_p"]
if p_fix > 1e-3 and p_bug < 1e-6:
    print(f"  The separation collapses once the defects are fixed "
          f"(p: {p_bug:.1e} -> {p_fix:.1e}). The original result was")
    print(f"  produced by the two code defects, not by a genuine property")
    print(f"  of the tumour tissue. It should not be used.")
elif p_fix < 1e-3:
    print(f"  The separation survives the fix (p={p_fix:.1e} after correction).")
    print(f"  This means C DOES separate the classes even with both defects")
    print(f"  removed, and the earlier finding, while computed incorrectly,")
    print(f"  pointed at something real. The correctly-computed effect size")
    print(f"  should now be used in its place.")
else:
    print(f"  Ambiguous: p_bug={p_bug:.1e}, p_fix={p_fix:.1e}. Inspect the")
    print(f"  per-class means above before drawing a conclusion.")

# ---------------------------------------------------------------
print("\n" + "=" * 68)
print("  PART B — SYMMETRY-BASED SEEDING VS BRIGHTEST-REGION SEEDING")
print("=" * 68)
print(f"  intensity seed -> contour : Dice {dice_int.mean():.3f} +/- {dice_int.std():.3f}")
print(f"  symmetry seed  -> contour : Dice {dice_sym.mean():.3f} +/- {dice_sym.std():.3f}")
try: p_seed = stats.wilcoxon(dice_sym, dice_int).pvalue
except Exception: p_seed = 1.0
print(f"  paired Wilcoxon p = {p_seed:.4g}   "
      f"(symmetry better on {int((dice_sym>dice_int).sum())}/{len(dice_int)} slices)")
results["seeding"] = {"intensity_mean": float(dice_int.mean()),
                      "symmetry_mean": float(dice_sym.mean()), "p": float(p_seed)}

both_fail = int(((dice_int < 0.1) & (dice_sym < 0.1)).sum())
either_ok = int(((dice_int > 0.5) | (dice_sym > 0.5)).sum())
print(f"\n  slices where BOTH seedings fail (Dice<0.1): {both_fail} "
      f"({100*both_fail/len(dice_int):.1f}%)")
print(f"  slices where AT LEAST ONE reaches Dice>0.5: {either_ok} "
      f"({100*either_ok/len(dice_int):.1f}%)")
print("""
  Recall from Stage 11: intensity seeding reached Dice 0.049 overall (the
  earlier full-dataset run). If symmetry seeding is materially better here,
  brightest-region was the wrong prior for this dataset, not a fundamental
  limit on automatic delineation. If both are weak, and few slices are
  rescued by taking whichever seed works, automatic delineation needs a
  learned localiser rather than a better heuristic, and that conclusion
  should be stated plainly rather than tested a third time with a new guess.
""")

with open(RESULTS, "w") as f:
    json.dump(results, f, indent=2, default=float)
print(f"  written to {RESULTS}")
print("\nDONE. Copy back both parts in full.\n")
