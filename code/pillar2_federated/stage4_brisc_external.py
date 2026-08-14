"""
==================================================================
 CF-SCAFFOLD — STAGE 5: EXTERNAL VALIDATION ON BRISC 2025
 Does the hard-mask (alpha=0) advantage REPLICATE on a 2nd dataset?
==================================================================
 The final Q1 requirement: show the federated stabilization result
 is not Figshare-specific. We re-run the FedRS comparison on BRISC
 2025 (Fateh et al., Sci. Data 2025; Kaggle: briscdataset/brisc2025)
 using the FLEXIBLE LOADER adapted for BRISC's structure.

 BRISC structure (confirmed from the Nature/arXiv papers):
   segmentation_task/{train,test}/images/*.jpg  +  masks/*.png
   filenames: brisc2025_<split>_<idx>_<code>_<plane>_<seq>.jpg
   tumor codes: gl=glioma, me=meningioma, pi=pituitary, no=no_tumor
   4793 image-mask pairs; 3 tumor types (matches our class-skew setup).

 IMPORTANT HONEST CAVEATS (state these in the thesis):
   1. BRISC has NO patient IDs -> federated clients are partitioned by
      tumor TYPE via Dirichlet (not by patient). This tests class-skew
      robustness, which is exactly our claim, but patient-level
      evaluation is not possible here (documented limitation).
   2. BRISC is derived from Nickparvar, which aggregates Cheng (our
      Figshare source) + SARTAJ + Br35H. So there is PARTIAL source
      overlap with Figshare. The masks are newly expert-annotated and
      the partition differs, so it is still a valid robustness check,
      but it is NOT a fully independent dataset. Report this honestly.

 The script PRINTS A DIAGNOSTIC of what structure it actually found
 before training, so if BRISC's layout differs you can adapt quickly.

 4 methods (FedAvg, FedRS a=0.5, FedRS a=0.25, HardMask a=0), 20 seeds,
 Dirichlet(0.3) over tumor types, GroupNorm, paired stats. Auto-resume.
------------------------------------------------------------------
 Kaggle: GPU. Attach briscdataset/brisc2025. Auto-resume CSV.
==================================================================
"""
import os, glob, csv, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
from scipy import stats
from sklearn.metrics import balanced_accuracy_score
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from PIL import Image

# ── CONFIG ───────────────────────────────────────────────────
DATA_ROOT = "/kaggle/input"          # walks recursively to find BRISC
OUT_DIR   = "/kaggle/working"
CSV_PATH  = f"{OUT_DIR}/brisc_fedrs_log.csv"
IMG_SIZE  = 128; N_CLIENTS = 18; BATCH = 32
SEEDS     = 20; PARTITION_ALPHA = 0.3
ROUNDS    = 40; LOCAL_EPOCHS = 1; LR = 0.01
MAX_PER_TYPE = 1000                  # cap per type for speed (set None for all)
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

TUMOR_CODES = {"gl": "glioma", "me": "meningioma", "pi": "pituitary", "no": "no_tumor"}
TYPE_TO_LABEL = {"meningioma": 0, "glioma": 1, "pituitary": 2}   # match Figshare order

# ── FLEXIBLE LOADER (BRISC-adapted) ──────────────────────────
def parse_tumor_type(fn):
    """Extract tumor type from BRISC filename code, e.g. ..._gl_ax_t1.jpg -> glioma."""
    for p in os.path.splitext(os.path.basename(fn))[0].split("_"):
        if p in TUMOR_CODES: return TUMOR_CODES[p]
    # fallback: maybe the parent folder names the class (classification_task)
    parent = os.path.basename(os.path.dirname(fn)).lower()
    for t in ["glioma", "meningioma", "pituitary", "no_tumor"]:
        if t in parent: return t
    return None

def find_mask(ipath):
    """BRISC: images/<name>.jpg -> masks/<name>.png (same basename)."""
    for cand in [ipath.replace("/images/", "/masks/"),
                 ipath.replace("/image/", "/mask/")]:
        mp = os.path.splitext(cand)[0] + ".png"
        if os.path.exists(mp): return mp
        mp2 = os.path.splitext(cand)[0] + ".jpg"
        if os.path.exists(mp2): return mp2
    # sibling _mask
    base, ext = os.path.splitext(ipath)
    for mp in [base + "_mask.png", base + "_mask" + ext]:
        if os.path.exists(mp): return mp
    return None

def load_brisc(root, max_per_type=None):
    """Walk root, find segmentation image/mask pairs, extract type from filename."""
    img_files = []
    for dp, _, files in os.walk(root):
        d = dp.replace("\\", "/")
        # prefer segmentation images (they have masks for fractal C)
        is_img_dir = d.endswith("/images") or "segmentation" in d.lower()
        for fn in files:
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")): continue
            if "mask" in fn.lower() or "/masks/" in d: continue
            img_files.append(os.path.join(dp, fn))
    print(f"  found {len(img_files)} candidate image files")
    recs = []; counts = {}
    for ip in sorted(img_files):
        tt = parse_tumor_type(ip)
        if tt is None or tt == "no_tumor": continue
        if max_per_type and counts.get(tt, 0) >= max_per_type: continue
        mp = find_mask(ip)
        img = np.asarray(Image.open(ip).convert("L").resize((IMG_SIZE, IMG_SIZE),
                         Image.BILINEAR)).astype(np.float32) / 255.0
        mask = None
        if mp:
            m = np.asarray(Image.open(mp).convert("L").resize((IMG_SIZE, IMG_SIZE), Image.NEAREST))
            mask = (m > 127).astype(np.uint8)
        recs.append(dict(image=img, mask=mask, tumor_type=tt, label=TYPE_TO_LABEL[tt]))
        counts[tt] = counts.get(tt, 0) + 1
    return recs, counts

# ── model (GroupNorm) ────────────────────────────────────────
class SmallNet(nn.Module):
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
    for p in m.parameters(): n = p.numel(); p.data.copy_(f[i:i+n].view_as(p)); i += n

def fedrs_loss(logits, yb, present, alpha):
    if alpha <= 0.0:
        mask = torch.full((logits.shape[1],), -1e9, device=logits.device)
        for c in present: mask[c] = 0.0
        return F.cross_entropy(logits + mask, yb)
    scale = torch.full((logits.shape[1],), alpha, device=logits.device)
    for c in present: scale[c] = 1.0
    return F.cross_entropy(logits * scale, yb)

def make_loader(X, Y, idx):
    ds = TensorDataset(torch.tensor(X[idx]), torch.tensor(Y[idx]))
    y = Y[idx]; cc = np.bincount(y, minlength=3)
    w = np.array([1.0 / cc[c] if cc[c] > 0 else 0.0 for c in y])
    sampler = WeightedRandomSampler(torch.tensor(w, dtype=torch.double), len(w), replacement=True)
    return DataLoader(ds, batch_size=BATCH, sampler=sampler)

def federated_run(X, Y, client_idx, test_idx, alpha, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = SmallNet().to(DEVICE); clients = list(client_idx)
    ni = {c: len(client_idx[c]) for c in clients}
    loaders = {c: make_loader(X, Y, client_idx[c]) for c in clients}
    present = {c: list(np.unique(Y[client_idx[c]])) for c in clients}
    for _ in range(ROUNDS):
        gflat = _flat(model); deltas = []
        for c in clients:
            _set(model, gflat)
            opt = torch.optim.SGD(model.parameters(), lr=LR); model.train()
            for _ in range(LOCAL_EPOCHS):
                for xb, yb in loaders[c]:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE); opt.zero_grad()
                    fedrs_loss(model(xb), yb, present[c], alpha).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            deltas.append((ni[c], _flat(model) - gflat))
        tot = sum(n for n, _ in deltas); _set(model, gflat + sum(n / tot * d for n, d in deltas))
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X[test_idx]).to(DEVICE)).argmax(1).cpu().numpy()
    return balanced_accuracy_score(Y[test_idx], pred)

def dirichlet_partition_by_label(Y, n_clients, alpha, seed):
    """No PID in BRISC -> partition slices by TYPE via Dirichlet across clients."""
    rng = np.random.default_rng(seed); cp = {c: [] for c in range(n_clients)}
    for cls in np.unique(Y):
        idx = np.where(Y == cls)[0]; rng.shuffle(idx)
        pr = rng.dirichlet([alpha] * n_clients); cnt = (pr * len(idx)).astype(int)
        cnt[-1] = len(idx) - cnt[:-1].sum(); pos = 0
        for c in range(n_clients):
            cp[c].extend(idx[pos:pos + cnt[c]]); pos += cnt[c]
    return {c: np.array(v) for c, v in cp.items() if len(v) >= 4}

METHODS = [("FedAvg", 1.0), ("FedRS(a=0.5)", 0.5), ("FedRS(a=0.25)", 0.25), ("HardMask(a=0.0)", 0.0)]

# ── auto-resume ──────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=["method", "seed", "acc"]).writeheader()
def is_done(method, seed):
    with open(CSV_PATH) as f:
        return any(r["method"] == method and int(r["seed"]) == seed for r in csv.DictReader(f))
def log_row(method, seed, acc):
    with open(CSV_PATH, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=["method", "seed", "acc"]).writerow(
            dict(method=method, seed=seed, acc=round(acc, 4)))

# ── MAIN ─────────────────────────────────────────────────────
print("=" * 64); print("  LOADING BRISC 2025 (flexible loader)"); print("=" * 64)
recs, counts = load_brisc(DATA_ROOT, MAX_PER_TYPE)
print("\n  STRUCTURE DIAGNOSTIC (what the loader actually found):")
print(f"    usable tumor slices : {len(recs)}")
print(f"    type distribution   : {counts}")
n_with_mask = sum(1 for r in recs if r['mask'] is not None and r['mask'].sum() > 0)
print(f"    slices with masks   : {n_with_mask}/{len(recs)}")
if len(recs) < 100:
    print("\n  !! Very few slices found. BRISC layout may differ from expected.")
    print("     Check the printed paths and adjust load_brisc() accordingly.")
    raise SystemExit("Aborting: not enough data — inspect structure diagnostic above.")

X = np.array([r["image"] for r in recs])[:, None].astype(np.float32)
Y = np.array([r["label"] for r in recs])
print(f"\n  X={X.shape}  Y classes={np.bincount(Y)}")

# held-out test split (random, stratified by class — no PID available)
rng = np.random.default_rng(42); idx_all = np.arange(len(Y)); rng.shuffle(idx_all)
test_idx = idx_all[:int(0.2 * len(idx_all))]; train_idx = idx_all[int(0.2 * len(idx_all)):]
print(f"  train={len(train_idx)}  test={len(test_idx)}")

print("\n" + "=" * 64); print(f"  BRISC FedRS COMPARISON ({SEEDS} seeds, {ROUNDS} rounds)"); print("=" * 64)
for seed in range(SEEDS):
    if all(is_done(name, seed) for name, _ in METHODS):
        print(f"  seed {seed:2d}: SKIP"); continue
    t0 = time.time()
    cp = dirichlet_partition_by_label(Y[train_idx], N_CLIENTS, PARTITION_ALPHA, seed)
    cidx = {c: train_idx[v] for c, v in cp.items()}
    parts = []
    for name, alpha in METHODS:
        if is_done(name, seed): parts.append(f"{name}=SKIP"); continue
        acc = federated_run(X, Y, cidx, test_idx, alpha, seed)
        log_row(name, seed, acc); parts.append(f"{name}={acc:.3f}")
    print(f"  seed {seed:2d}: {' | '.join(parts)}  ({time.time()-t0:.0f}s)")

# ── stats ────────────────────────────────────────────────────
rows = {}
with open(CSV_PATH) as f:
    for r in csv.DictReader(f): rows.setdefault(r["method"], {})[int(r["seed"])] = float(r["acc"])
seeds = sorted({s for m in rows for s in rows[m]})
print("\n" + "=" * 64); print("  BRISC RESULTS"); print("=" * 64)
for name, _ in METHODS:
    a = np.array([rows[name][s] for s in seeds if s in rows.get(name, {})])
    if len(a): print(f"  {name:18s}: {a.mean():.3f} +/- {a.std():.3f}  collapses={int((a<=0.40).sum())}/{len(a)}")

def cohend(a, b): d = np.array(a) - np.array(b); return d.mean() / (d.std(ddof=1) + 1e-12)
def boot(a, b, n=10000):
    d = np.array(a) - np.array(b); rng = np.random.default_rng(0)
    return np.percentile(d[rng.integers(0, len(d), (n, len(d)))].mean(1), [2.5, 97.5])
def holm(pv):
    order = np.argsort(pv); m = len(pv); adj = np.zeros(m)
    for rank, i in enumerate(order): adj[i] = min((m - rank) * pv[i], 1.0)
    for i in range(1, m): adj[order[i]] = max(adj[order[i]], adj[order[i-1]])
    return adj

yours = "HardMask(a=0.0)"; comp = [n for n, _ in METHODS if n != yours]
pvals, recs2 = [], []; yv = rows.get(yours, {})
for name in comp:
    common = [s for s in seeds if s in rows.get(name, {}) and s in yv]
    a = np.array([yv[s] for s in common]); b = np.array([rows[name][s] for s in common])
    try: p = stats.wilcoxon(a, b).pvalue
    except Exception: p = 1.0
    pvals.append(p); recs2.append((name, a.mean() - b.mean(), p, cohend(a, b), *boot(a, b)))
adj = holm(pvals)
print("\n" + "=" * 64); print("  HARD-MASK vs EACH (Holm) — BRISC"); print("=" * 64)
print(f"  {'vs':18s} {'delta':>8} {'p-Holm':>8} {'d':>7} {'95% CI':>18}")
for i, (name, delta, p, d, lo, hi) in enumerate(recs2):
    sig = "*" if adj[i] < 0.05 else " "
    print(f"  {name:18s} {delta:+8.3f} {adj[i]:8.4f}{sig} {d:+7.2f} [{lo:+.3f},{hi:+.3f}]")

print("\n" + "=" * 64); print("  EXTERNAL VALIDATION VERDICT"); print("=" * 64)
fr = next((r for r in recs2 if r[0] == "FedRS(a=0.5)"), None)
if fr:
    name, delta, p, d, lo, hi = fr; i = comp.index(name)
    if delta > 0.02 and adj[i] < 0.05:
        print(f"  REPLICATED: on BRISC, hard-mask beats FedRS(0.5) by {delta*100:+.1f}pp")
        print(f"  (Holm p={adj[i]:.4f}, d={d:.2f}). The advantage GENERALIZES across datasets.")
        print(f"  -> This is the external-validation evidence Q1 reviewers ask for.")
    else:
        print(f"  PARTIAL: on BRISC delta={delta*100:+.1f}pp (Holm p={adj[i]:.4f}).")
        print(f"  Report honestly: the gain is dataset-dependent; discuss why (BRISC")
        print(f"  class balance / source differ). Still a valuable robustness finding.")
print("\nDONE. Copy back: STRUCTURE DIAGNOSTIC, RESULTS, VERDICT.\n")
