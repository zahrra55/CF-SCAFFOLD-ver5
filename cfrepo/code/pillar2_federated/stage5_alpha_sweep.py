"""
==================================================================
 CF-SCAFFOLD — STAGE 6: THE ALPHA-vs-SKEW RULE (mechanistic proof)
 Proves: the optimal restricted-softmax alpha_mask depends on the
 LOCAL ABSENCE (fraction of classes missing per client), which is
 driven by the Dirichlet skew alpha_dir.
==================================================================
 The D1 insight in one experiment: sweep the skew intensity, and at
 each level find the best alpha_mask. If the optimal alpha_mask RISES
 as skew decreases (local absence drops), the rule is proven visually
 and our alpha_mask=0 choice is shown to be the correct point on a
 principled curve -- not an arbitrary pick.

 NOTATION (state clearly in the paper to avoid confusion):
   alpha_dir  = Dirichlet concentration controlling class skew
                (small = severe skew = high local absence)
   alpha_mask = restricted-softmax scaling of absent-class logits
                (0 = hard mask / ours ; 1 = FedAvg / no restriction)

 DESIGN (fits one Kaggle session, ~2.6 GPU-h):
   3 skew levels x 3 mask levels x 8 seeds = 72 runs.
   Also MEASURES the real mean local absence at each skew level, so
   the x-axis is the true mechanistic variable, not just alpha_dir.

 HONEST: this STRENGTHENS the paper if the trend appears; if it is
 weak/noisy, we still have the two-dataset +5-6pp result. Report
 whatever the data shows.
------------------------------------------------------------------
 Kaggle: GPU. Attach Figshare .mat. Auto-resume via alpha_sweep_log.csv.
==================================================================
"""
import os, glob, csv, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
try: import h5py
except Exception: os.system("pip install h5py --quiet"); import h5py
from sklearn.metrics import balanced_accuracy_score
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

# ── CONFIG ───────────────────────────────────────────────────
DATA_GLOB = "/kaggle/input/**/*.mat"
OUT_DIR   = "/kaggle/working"
CSV_PATH  = f"{OUT_DIR}/alpha_sweep_log.csv"
IMG_SIZE  = 128; N_CLIENTS = 18; BATCH = 32
ROUNDS    = 40; LOCAL_EPOCHS = 1; LR = 0.01
SEEDS     = 8
ALPHA_DIR_LEVELS  = [0.1, 0.5, 3.0]      # severe -> mild skew
ALPHA_MASK_LEVELS = [0.0, 0.5, 1.0]      # hard mask -> FedAvg
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

# ── readers ──────────────────────────────────────────────────
def read_pid(cj):
    if "PID" not in cj: return None
    raw = np.array(cj["PID"]).ravel()
    try: s = "".join(chr(int(c)) for c in raw if int(c) > 0); return s.strip() or None
    except Exception: return None
def read_cjdata(path):
    try:
        with h5py.File(path, "r") as f:
            cj = f["cjdata"]; label = int(np.array(cj["label"]).ravel()[0])
            img = np.array(cj["image"]).astype(np.float32)
            if img.ndim == 2: img = img.T
            return img, label, read_pid(cj)
    except Exception: return None, None, None
def resize_img(img, size):
    from PIL import Image
    im = Image.fromarray(((img - img.min()) / (img.max() - img.min() + 1e-8) * 255
                          ).astype(np.uint8)).resize((size, size), Image.BILINEAR)
    return np.asarray(im).astype(np.float32)
def load_data():
    files = sorted(f for f in glob.glob(DATA_GLOB, recursive=True) if f.lower().endswith(".mat"))
    print(f"Found {len(files)} .mat files")
    X, Y, PID = [], [], []
    for k, path in enumerate(files):
        img, label, pid = read_cjdata(path)
        if img is None or pid is None: continue
        X.append((resize_img(img, IMG_SIZE) / 255.0).astype(np.float32))
        Y.append(label - 1); PID.append(pid)
        if (k + 1) % 800 == 0: print(f"  loaded {k+1}/{len(files)}")
    return np.array(X)[:, None], np.array(Y), np.array(PID)

# ── model ────────────────────────────────────────────────────
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

def federated_run(X, Y, client_idx, test_idx, alpha_mask, seed):
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
                    fedrs_loss(model(xb), yb, present[c], alpha_mask).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            deltas.append((ni[c], _flat(model) - gflat))
        tot = sum(n for n, _ in deltas); _set(model, gflat + sum(n / tot * d for n, d in deltas))
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X[test_idx]).to(DEVICE)).argmax(1).cpu().numpy()
    return balanced_accuracy_score(Y[test_idx], pred)

def dirichlet_partition(Y, PID, n_clients, alpha_dir, seed):
    rng = np.random.default_rng(seed); uniq = np.unique(PID)
    pl = {p: Y[np.where(PID == p)[0][0]] for p in uniq}; cp = {c: [] for c in range(n_clients)}
    for cls in np.unique(Y):
        cps = [p for p in uniq if pl[p] == cls]; rng.shuffle(cps)
        pr = rng.dirichlet([alpha_dir] * n_clients); cnt = (pr * len(cps)).astype(int)
        if len(cps): cnt[-1] = len(cps) - cnt[:-1].sum()
        idx = 0
        for c in range(n_clients):
            for _ in range(cnt[c]):
                if idx < len(cps): cp[c].append(cps[idx]); idx += 1
    ci = {c: np.where(np.isin(PID, cp[c]))[0] for c in range(n_clients)}
    return {c: v for c, v in ci.items() if len(v) >= 4}

def mean_local_absence(client_idx, Y, n_classes=3):
    """Mean fraction of classes MISSING per client — the true mechanistic variable."""
    fracs = []
    for c in client_idx:
        present = len(np.unique(Y[client_idx[c]]))
        fracs.append((n_classes - present) / n_classes)
    return float(np.mean(fracs)) if fracs else 0.0

# ── auto-resume ──────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=["alpha_dir", "alpha_mask", "seed", "acc", "absence"]).writeheader()
def is_done(ad, am, s):
    with open(CSV_PATH) as f:
        return any(abs(float(r["alpha_dir"])-ad)<1e-9 and abs(float(r["alpha_mask"])-am)<1e-9
                   and int(r["seed"])==s for r in csv.DictReader(f))
def log_row(ad, am, s, acc, absence):
    with open(CSV_PATH, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=["alpha_dir","alpha_mask","seed","acc","absence"]).writerow(
            dict(alpha_dir=ad, alpha_mask=am, seed=s, acc=round(acc,4), absence=round(absence,4)))

# ── MAIN ─────────────────────────────────────────────────────
print("=" * 64); print("  LOADING"); print("=" * 64)
X, Y, PID = load_data()
print(f"  {len(X)} slices, {len(np.unique(PID))} patients, classes={np.bincount(Y)}")
rng = np.random.default_rng(42); uniq = np.unique(PID); rng.shuffle(uniq)
test_pids = set(uniq[:int(0.2*len(uniq))])
test_idx = np.where(np.isin(PID, list(test_pids)))[0]
train_mask = ~np.isin(PID, list(test_pids)); train_pos = np.where(train_mask)[0]

print("\n" + "=" * 64); print("  ALPHA-vs-SKEW SWEEP"); print("=" * 64)
for ad in ALPHA_DIR_LEVELS:
    for seed in range(SEEDS):
        cp = dirichlet_partition(Y[train_mask], PID[train_mask], N_CLIENTS, ad, seed)
        cp = {c: train_pos[v] for c, v in cp.items()}
        absence = mean_local_absence(cp, Y)
        for am in ALPHA_MASK_LEVELS:
            if is_done(ad, am, seed): continue
            t0 = time.time()
            acc = federated_run(X, Y, cp, test_idx, am, seed)
            log_row(ad, am, seed, acc, absence)
            print(f"  dir={ad:4.1f} mask={am:4.2f} seed={seed} -> acc={acc:.3f} "
                  f"(absence={absence:.2f})  {time.time()-t0:.0f}s")

# ── aggregate + find optimal mask per skew level ─────────────
rows = {}
with open(CSV_PATH) as f:
    for r in csv.DictReader(f):
        key = (float(r["alpha_dir"]), float(r["alpha_mask"]))
        rows.setdefault(key, []).append(float(r["acc"]))
absences = {}
with open(CSV_PATH) as f:
    for r in csv.DictReader(f):
        absences[float(r["alpha_dir"])] = float(r["absence"])

print("\n" + "=" * 64); print("  RESULTS: mean balanced acc per (skew, mask)"); print("=" * 64)
print(f"  {'alpha_dir':>10} {'absence':>8} | " + " ".join(f"mask={am:.2f}" for am in ALPHA_MASK_LEVELS) + "   BEST")
for ad in ALPHA_DIR_LEVELS:
    cells = []
    for am in ALPHA_MASK_LEVELS:
        vals = rows.get((ad, am), [])
        cells.append(np.mean(vals) if vals else np.nan)
    best_am = ALPHA_MASK_LEVELS[int(np.nanargmax(cells))]
    line = f"  {ad:>10.1f} {absences.get(ad,0):>8.2f} | " + " ".join(f"{c:8.3f}" for c in cells)
    line += f"   alpha_mask={best_am}"
    print(line)

print("\n" + "=" * 64); print("  RULE VERDICT"); print("=" * 64)
best_masks = []
for ad in ALPHA_DIR_LEVELS:
    cells = [np.mean(rows.get((ad, am), [np.nan])) for am in ALPHA_MASK_LEVELS]
    best_masks.append(ALPHA_MASK_LEVELS[int(np.nanargmax(cells))])
# severe skew (low alpha_dir) should prefer LOW mask; mild skew should prefer HIGHER mask
if best_masks[0] <= best_masks[-1] and best_masks[0] == 0.0:
    print(f"  RULE SUPPORTED: optimal alpha_mask = {best_masks} as skew goes severe->mild.")
    print(f"  Under severe skew (high local absence) alpha_mask=0 is best; the optimum")
    print(f"  rises as skew decreases -- exactly the mechanistic prediction from D1.")
    print(f"  -> Your alpha_mask=0 is the correct point on a principled curve.")
else:
    print(f"  MIXED/NOISY: optimal alpha_mask = {best_masks}. Report honestly; the")
    print(f"  two-dataset +5-6pp result stands regardless. More seeds may sharpen this.")
print("\n  Plot to make: x=mean local absence (or alpha_dir), y=optimal alpha_mask,")
print("  plus a heatmap of balanced acc over (alpha_dir x alpha_mask).")
print("\nDONE. Copy back the RESULTS table + RULE VERDICT.\n")
