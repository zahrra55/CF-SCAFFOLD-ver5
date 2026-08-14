"""
==================================================================
 CF-SCAFFOLD — STAGE 7: BIO-ANCHORED CONTRASTIVE LOSS (Bio-ConFL)
 Does anchoring the latent space with continuous fractal descriptors
 improve federated training under missing-class skew?
==================================================================
 IDEA (client-side only; server aggregation untouched):
   Add an auxiliary contrastive term whose targets come from the
   CONTINUOUS biophysical descriptors (C = Ft/Fb, lacunarity, border
   gradient) instead of the discrete labels. Because the term never
   references labels, it keeps shaping the latent space even for
   classes a client does not hold.

     L_total = L_restricted_CE  +  lambda_bio * L_biocontrastive

   L_biocontrastive is a soft cross-entropy between
     P_bio (row-normalised Gaussian kernel over descriptor distances)
   and the latent cosine-similarity distribution.

 PRIVACY: sigma is the median pairwise descriptor distance WITHIN the
 local batch. Descriptors are min-max normalised per client. Nothing
 is shared with the server beyond the usual model delta.

 PREREGISTERED SUCCESS CRITERION (fixed BEFORE seeing results):
   The method counts as successful only if, versus the hard-mask
   baseline, it achieves ALL of:
     (a) mean gain >= +0.02 balanced accuracy,
     (b) paired Wilcoxon p < 0.05,
     (c) wins on >= 13 of 20 seeds.
   Anything less is reported as a null result and the idea goes to
   future work. This is stated here so the outcome cannot be
   rationalised afterwards.

 ARMS (all share the same partitions and seeds):
   A. hard-mask only            (current published method, baseline)
   B. hard-mask + bio, lambda=0.1
   C. hard-mask + bio, lambda=0.5
   D. bio only, no restricted softmax   (isolates the bio term)

 COST: 4 arms x 20 seeds x ~35 s/run  ~ 47 min/arm, ~3 h total.
------------------------------------------------------------------
 Kaggle: GPU. Attach the Figshare .mat dataset. Auto-resume via CSV.
==================================================================
"""
import os, csv, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
try: import h5py
except Exception: os.system("pip install h5py --quiet"); import h5py
from scipy import ndimage, stats
from sklearn.metrics import balanced_accuracy_score
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

# ── CONFIG ───────────────────────────────────────────────────
DATA_ROOT = "/kaggle/input"
OUT_DIR   = "/kaggle/working"
CSV_PATH  = f"{OUT_DIR}/bio_contrastive_log.csv"
IMG_SIZE  = 128; N_CLIENTS = 18; BATCH = 32
SEEDS     = 20; ALPHA_DIR = 0.3
ROUNDS    = 40; LOCAL_EPOCHS = 1; LR = 0.01
TAU       = 0.07          # standard SimCLR/SupCon value, fixed
SIGMA_SCALE = 1.0         # multiplies the local median sigma
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

# ── data loading (os.walk: robust to any Kaggle mount layout) ──
def find_mat_files(root):
    out = []
    for dp, _, fnames in os.walk(root):
        for fn in fnames:
            if fn.lower().endswith(".mat"):
                out.append(os.path.join(dp, fn))
    return sorted(out)

def read_pid(cj):
    if "PID" not in cj: return None
    raw = np.array(cj["PID"]).ravel()
    try:
        s = "".join(chr(int(c)) for c in raw if int(c) > 0)
        return s.strip() or None
    except Exception:
        return None

def read_record(path):
    try:
        with h5py.File(path, "r") as f:
            cj = f["cjdata"]
            label = int(np.array(cj["label"]).ravel()[0])
            img = np.array(cj["image"]).astype(np.float32)
            if img.ndim == 2: img = img.T
            mask = None
            if "tumorMask" in cj:
                mask = np.array(cj["tumorMask"]).astype(np.uint8)
                if mask.ndim == 2: mask = mask.T
            return img, label, mask, read_pid(cj)
    except Exception:
        return None, None, None, None

def resize_pair(img, mask, size):
    from PIL import Image
    im = Image.fromarray(((img - img.min()) / (img.max() - img.min() + 1e-8) * 255
                          ).astype(np.uint8)).resize((size, size), Image.BILINEAR)
    g = np.asarray(im).astype(np.float32)
    if mask is None: return g, None
    mk = Image.fromarray((mask * 255).astype(np.uint8)).resize((size, size), Image.NEAREST)
    return g, (np.asarray(mk) > 127).astype(np.uint8)

# ── biophysical descriptors (Pillar I features reused) ───────
def brain_mask_of(gray):
    m = gray > max(gray.mean() * 0.25, 10)
    m = ndimage.binary_fill_holes(m); m = ndimage.binary_opening(m, iterations=2)
    lbl, n = ndimage.label(m)
    if n >= 1:
        s = np.bincount(lbl.ravel()); s[0] = 0; m = lbl == s.argmax()
    return m.astype(np.uint8)

def boxcount_fd(gray, mask, n_sizes=8):
    v = gray[mask > 0]
    if v.size < 64: return np.nan
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-6: return np.nan
    g = np.zeros_like(gray, dtype=np.float64)
    g[mask > 0] = (gray[mask > 0] - lo) / (hi - lo) * 255.0
    H, W = g.shape; p = min(H, W); mk = int(np.log2(p))
    sizes = [2 ** k for k in range(1, mk)][:n_sizes]; Ns = []
    for s in sizes:
        nb = 0
        for i in range(0, H - H % s, s):
            for j in range(0, W - W % s, s):
                bm = mask[i:i + s, j:j + s]
                if bm.sum() == 0: continue
                blk = g[i:i + s, j:j + s][bm > 0]
                if blk.size == 0: continue
                nb += int(np.floor(blk.max() / s) - np.floor(blk.min() / s) + 1)
        Ns.append(max(nb, 1))
    sizes = np.array(sizes, float); Ns = np.array(Ns, float)
    return float(np.polyfit(np.log(1.0 / sizes), np.log(Ns), 1)[0])

def lacunarity(mask, box=8):
    H, W = mask.shape; counts = []
    for i in range(0, H - box, box):
        for j in range(0, W - box, box):
            counts.append(mask[i:i + box, j:j + box].sum())
    counts = np.array(counts, float)
    if counts.mean() < 1e-6: return 0.0
    return float(1 + counts.var() / (counts.mean() ** 2))

def border_gradient(gray, tumor):
    gx, gy = np.gradient(gray.astype(float))
    grad = np.sqrt(gx ** 2 + gy ** 2)
    border = ndimage.binary_dilation(tumor) & ~ndimage.binary_erosion(tumor)
    return float(grad[border > 0].mean()) if border.sum() else 0.0

def descriptors_for(img, mask):
    """Return (image, [C, lacunarity, border_grad], ok)."""
    g, mk = resize_pair(img, mask, IMG_SIZE)
    if mk is None or mk.sum() < 16: return None, None
    brain = brain_mask_of(g); tumor = (mk > 0) & (brain > 0)
    brain_wo = (brain > 0) & (~tumor)
    if tumor.sum() < 16 or brain_wo.sum() < 64: return None, None
    Ft = boxcount_fd(g, tumor); Fb = boxcount_fd(g, brain_wo)
    if not (np.isfinite(Ft) and np.isfinite(Fb) and Fb > 1e-6): return None, None
    b = np.array([Ft / Fb, lacunarity(tumor), border_gradient(g, tumor)], dtype=np.float32)
    if not np.all(np.isfinite(b)): return None, None
    return (g / 255.0).astype(np.float32), b

def load_all():
    files = find_mat_files(DATA_ROOT)
    print(f"Found {len(files)} .mat files")
    X, Y, PID, B = [], [], [], []
    for k, path in enumerate(files):
        img, label, mask, pid = read_record(path)
        if img is None or pid is None or mask is None: continue
        g, b = descriptors_for(img, mask)
        if g is None: continue
        X.append(g); Y.append(label - 1); PID.append(pid); B.append(b)
        if (k + 1) % 800 == 0: print(f"  processed {k+1}/{len(files)}")
    return (np.array(X)[:, None], np.array(Y), np.array(PID), np.array(B))

# ── model with an exposed embedding ──────────────────────────
class SmallNet(nn.Module):
    def __init__(self, n=3, embed=64):
        super().__init__()
        self.c1 = nn.Conv2d(1, 16, 3, padding=1); self.n1 = nn.GroupNorm(4, 16)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1); self.n2 = nn.GroupNorm(4, 32)
        self.c3 = nn.Conv2d(32, 64, 3, padding=1); self.n3 = nn.GroupNorm(8, 64)
        self.proj = nn.Linear(64, embed)     # projection head for the contrastive term
        self.fc = nn.Linear(64, n)
    def forward(self, x, return_embed=False):
        x = F.max_pool2d(F.relu(self.n1(self.c1(x))), 2)
        x = F.max_pool2d(F.relu(self.n2(self.c2(x))), 2)
        x = F.max_pool2d(F.relu(self.n3(self.c3(x))), 2)
        h = F.adaptive_avg_pool2d(x, 1).flatten(1)
        if return_embed: return self.fc(h), self.proj(h)
        return self.fc(h)

def _flat(m): return torch.cat([p.data.flatten() for p in m.parameters()])
def _set(m, f):
    i = 0
    for p in m.parameters(): n = p.numel(); p.data.copy_(f[i:i+n].view_as(p)); i += n

# ── losses ───────────────────────────────────────────────────
def hard_mask_ce(logits, yb, present):
    mask = torch.full((logits.shape[1],), -1e9, device=logits.device)
    for c in present: mask[c] = 0.0
    return F.cross_entropy(logits + mask, yb)

def bio_contrastive(z, bio, tau=TAU, sigma_scale=SIGMA_SCALE):
    """Soft contrastive loss with targets from continuous descriptors.
       Labels are never used, so absent classes cause no degeneracy.
       sigma is the median pairwise distance inside this batch (local)."""
    B = z.shape[0]
    if B <= 1: return torch.tensor(0.0, device=z.device, requires_grad=True)
    z = F.normalize(z, p=2, dim=1)
    sim = torch.matmul(z, z.T) / tau
    d = torch.cdist(bio, bio, p=2)
    pos = d[d > 0]
    sigma = torch.median(pos) if pos.numel() > 0 else torch.tensor(1.0, device=z.device)
    if (not torch.isfinite(sigma)) or sigma <= 0:
        sigma = torch.tensor(1.0, device=z.device)
    sigma = sigma * sigma_scale
    S = torch.exp(-(d ** 2) / (2 * sigma ** 2 + 1e-6))
    eye = torch.eye(B, device=z.device)
    S = S * (1 - eye)
    P = S / (S.sum(dim=1, keepdim=True) + 1e-8)
    exp_sim = torch.exp(sim) * (1 - eye)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
    return -(P * log_prob).sum(dim=1).mean()

# ── federated training ───────────────────────────────────────
def make_loader(X, Y, Bd, idx):
    ds = TensorDataset(torch.tensor(X[idx]), torch.tensor(Y[idx]), torch.tensor(Bd[idx]))
    y = Y[idx]; cc = np.bincount(y, minlength=3)
    w = np.array([1.0 / cc[c] if cc[c] > 0 else 0.0 for c in y])
    sampler = WeightedRandomSampler(torch.tensor(w, dtype=torch.double), len(w), replacement=True)
    return DataLoader(ds, batch_size=BATCH, sampler=sampler, drop_last=True)

def normalize_bio_local(Bd, idx):
    """Per-client min-max normalisation. Strictly local: no global statistics."""
    sub = Bd[idx].copy()
    lo = sub.min(axis=0, keepdims=True); hi = sub.max(axis=0, keepdims=True)
    rng = np.where((hi - lo) < 1e-8, 1.0, hi - lo)
    out = Bd.copy()
    out[idx] = (sub - lo) / rng
    return out

def federated_run(X, Y, Bd, client_idx, test_idx, seed, use_mask, lambda_bio):
    torch.manual_seed(seed); np.random.seed(seed)
    model = SmallNet().to(DEVICE)
    clients = list(client_idx)
    ni = {c: len(client_idx[c]) for c in clients}
    # local descriptor normalisation, done once per client
    Bn = Bd.copy()
    for c in clients:
        Bn = normalize_bio_local(Bn, client_idx[c])
    loaders = {c: make_loader(X, Y, Bn, client_idx[c]) for c in clients}
    present = {c: list(np.unique(Y[client_idx[c]])) for c in clients}
    for _ in range(ROUNDS):
        g = _flat(model); deltas = []
        for c in clients:
            _set(model, g)
            opt = torch.optim.SGD(model.parameters(), lr=LR); model.train()
            for _ in range(LOCAL_EPOCHS):
                for xb, yb, bb in loaders[c]:
                    xb, yb, bb = xb.to(DEVICE), yb.to(DEVICE), bb.to(DEVICE)
                    opt.zero_grad()
                    logits, emb = model(xb, return_embed=True)
                    loss = (hard_mask_ce(logits, yb, present[c]) if use_mask
                            else F.cross_entropy(logits, yb))
                    if lambda_bio > 0:
                        loss = loss + lambda_bio * bio_contrastive(emb, bb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
            deltas.append((ni[c], _flat(model) - g))
        tot = sum(n for n, _ in deltas)
        _set(model, g + sum(n / tot * d for n, d in deltas))
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X[test_idx]).to(DEVICE)).argmax(1).cpu().numpy()
    return balanced_accuracy_score(Y[test_idx], pred)

def dirichlet_partition(Y, PID, n_clients, alpha_dir, seed):
    rng = np.random.default_rng(seed); uniq = np.unique(PID)
    pl = {p: Y[np.where(PID == p)[0][0]] for p in uniq}
    cp = {c: [] for c in range(n_clients)}
    for cls in np.unique(Y):
        cps = [p for p in uniq if pl[p] == cls]; rng.shuffle(cps)
        pr = rng.dirichlet([alpha_dir] * n_clients); cnt = (pr * len(cps)).astype(int)
        if len(cps): cnt[-1] = len(cps) - cnt[:-1].sum()
        i = 0
        for c in range(n_clients):
            for _ in range(cnt[c]):
                if i < len(cps): cp[c].append(cps[i]); i += 1
    ci = {c: np.where(np.isin(PID, cp[c]))[0] for c in range(n_clients)}
    return {c: v for c, v in ci.items() if len(v) >= BATCH}

# ── arms ─────────────────────────────────────────────────────
ARMS = [
    ("hardmask",          True,  0.0),
    ("hardmask+bio0.1",   True,  0.1),
    ("hardmask+bio0.5",   True,  0.5),
    ("bio_only",          False, 0.5),
]

# ── auto-resume ──────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=["arm", "seed", "acc"]).writeheader()
def is_done(a, s):
    with open(CSV_PATH) as f:
        return any(r["arm"] == a and int(r["seed"]) == s for r in csv.DictReader(f))
def log_row(a, s, acc):
    with open(CSV_PATH, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=["arm", "seed", "acc"]).writerow(
            dict(arm=a, seed=s, acc=round(acc, 4)))

# ── MAIN ─────────────────────────────────────────────────────
print("=" * 64); print("  LOADING"); print("=" * 64)
X, Y, PID, Bd = load_all()
print(f"  {len(X)} slices with descriptors, {len(np.unique(PID))} patients")
print(f"  class counts: {np.bincount(Y)}")
print(f"  descriptor ranges: C=[{Bd[:,0].min():.3f},{Bd[:,0].max():.3f}] "
      f"lac=[{Bd[:,1].min():.2f},{Bd[:,1].max():.2f}] "
      f"grad=[{Bd[:,2].min():.1f},{Bd[:,2].max():.1f}]")

rng = np.random.default_rng(42); uniq = np.unique(PID); rng.shuffle(uniq)
test_pids = set(uniq[:int(0.2 * len(uniq))])
test_idx = np.where(np.isin(PID, list(test_pids)))[0]
train_mask = ~np.isin(PID, list(test_pids)); train_pos = np.where(train_mask)[0]
print(f"  train slices {train_mask.sum()}, test slices {len(test_idx)}")

print("\n" + "=" * 64)
print(f"  BIO-CONTRASTIVE EXPERIMENT ({SEEDS} seeds x {len(ARMS)} arms)")
print("=" * 64)
for seed in range(SEEDS):
    if all(is_done(a, seed) for a, _, _ in ARMS):
        print(f"  seed {seed:2d}: SKIP"); continue
    cp = dirichlet_partition(Y[train_mask], PID[train_mask], N_CLIENTS, ALPHA_DIR, seed)
    cp = {c: train_pos[v] for c, v in cp.items()}
    parts = []
    t0 = time.time()
    for name, use_mask, lam in ARMS:
        if is_done(name, seed): parts.append(f"{name}=SKIP"); continue
        acc = federated_run(X, Y, Bd, cp, test_idx, seed, use_mask, lam)
        log_row(name, seed, acc); parts.append(f"{name}={acc:.3f}")
    print(f"  seed {seed:2d}: {' | '.join(parts)}  ({time.time()-t0:.0f}s)")

# ── analysis against the preregistered criterion ─────────────
rows = {}
with open(CSV_PATH) as f:
    for r in csv.DictReader(f):
        rows.setdefault(r["arm"], {})[int(r["seed"])] = float(r["acc"])
seeds = sorted({s for a in rows for s in rows[a]})

print("\n" + "=" * 64); print("  RESULTS"); print("=" * 64)
for name, _, _ in ARMS:
    v = np.array([rows[name][s] for s in seeds if s in rows.get(name, {})])
    if len(v):
        print(f"  {name:18s}: {v.mean():.4f} +/- {v.std():.4f}  (min {v.min():.3f})")

print("\n" + "=" * 64); print("  VERSUS HARD-MASK BASELINE (paired)"); print("=" * 64)
base = rows.get("hardmask", {})
print(f"  {'arm':18s} {'delta':>8} {'p':>8} {'wins':>7}   verdict")
for name, _, _ in ARMS:
    if name == "hardmask": continue
    common = [s for s in seeds if s in rows.get(name, {}) and s in base]
    if len(common) < 3: continue
    a = np.array([rows[name][s] for s in common]); b = np.array([base[s] for s in common])
    delta = a.mean() - b.mean()
    try: p = stats.wilcoxon(a, b).pvalue
    except Exception: p = 1.0
    wins = int((a > b).sum())
    ok = (delta >= 0.02) and (p < 0.05) and (wins >= 13)
    verdict = "MEETS CRITERION" if ok else "does not meet criterion"
    print(f"  {name:18s} {delta:+8.4f} {p:8.4f} {wins:>3}/{len(common)}   {verdict}")

print("\n" + "=" * 64); print("  DECISION"); print("=" * 64)
best = None
for name, _, _ in ARMS:
    if name == "hardmask": continue
    common = [s for s in seeds if s in rows.get(name, {}) and s in base]
    if len(common) < 3: continue
    a = np.array([rows[name][s] for s in common]); b = np.array([base[s] for s in common])
    delta = a.mean() - b.mean()
    try: p = stats.wilcoxon(a, b).pvalue
    except Exception: p = 1.0
    wins = int((a > b).sum())
    if (delta >= 0.02) and (p < 0.05) and (wins >= 13):
        best = (name, delta, p, wins, len(common))
if best:
    n, d, p, w, t = best
    print(f"  SUCCESS: {n} gives {d*100:+.1f}pp over hard mask (p={p:.4f}, {w}/{t} seeds).")
    print("  The bio-anchored term adds real value. Integrate it as a second")
    print("  contribution and report the ablation honestly.")
else:
    print("  NULL RESULT: no arm meets the preregistered criterion.")
    print("  The bio-anchored contrastive term does not improve on the hard mask")
    print("  in this setting. This is a legitimate finding: the restricted softmax")
    print("  already removes the pathology the contrastive term targets, so there")
    print("  is little left for it to correct. Report as an informative negative")
    print("  result or place in future work, and keep the current paper as is.")
print("\nDONE. Copy back RESULTS, the comparison table, and DECISION.\n")
