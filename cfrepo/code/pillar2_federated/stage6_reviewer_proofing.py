"""
==================================================================
 CF-SCAFFOLD — STAGE 8: REVIEWER-PROOFING EXPERIMENTS
 Three additions that close the gaps a J-BHI reviewer will target.
==================================================================
 PART A — FedLC baseline (the critical missing comparison)
   FedLC (Zhang et al., ICML 2022) generalises FedRS by subtracting a
   frequency-dependent margin from each logit. Because we cite it as
   the closest generalisation of our method, omitting it empirically
   is the single most likely cause of rejection. We add it here.

 PART B — Centralized upper bound
   The accuracy obtainable by training on the pooled data with the
   same backbone and budget. This tells the reader how much of the
   federated gap the correction closes, which no current table shows.

 PART C — Second architecture (ResNet-18 style)
   FedRS reported that output-layer corrections shrink with a larger
   feature extractor. A reviewer will ask whether our gain survives.
   We repeat the key comparison with a deeper backbone.

 All parts share the same partitions, seeds and evaluation protocol
 as the main experiments, so the numbers drop directly into the paper.

 COST: Part A ~2.5 h, Part B ~10 min, Part C ~3 h. Run A+B first;
 Part C is gated behind RUN_PART_C so it can go in a second session.
------------------------------------------------------------------
 Kaggle: GPU. Attach the Figshare .mat dataset. Auto-resume via CSV.
==================================================================
"""
import os, csv, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
try: import h5py
except Exception: os.system("pip install h5py --quiet"); import h5py
from scipy import stats
from sklearn.metrics import balanced_accuracy_score
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

# ── CONFIG ───────────────────────────────────────────────────
DATA_ROOT = "/kaggle/input"
OUT_DIR   = "/kaggle/working"
CACHE     = f"{OUT_DIR}/stage8_cache.npz"
CSV_PATH  = f"{OUT_DIR}/stage8_log.csv"
IMG_SIZE  = 128; N_CLIENTS = 18; BATCH = 32
SEEDS     = 20; ALPHA_DIR = 0.3
ROUNDS    = 40; LOCAL_EPOCHS = 1; LR = 0.01
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

RUN_PART_A = True      # FedLC baseline
RUN_PART_B = True      # centralized upper bound
RUN_PART_C = False      # ResNet backbone (set True for the second session)

# ── data (cached so repeat sessions start instantly) ─────────
def find_mat(root):
    out = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(".mat"): out.append(os.path.join(dp, fn))
    return sorted(out)

def read_pid(cj):
    if "PID" not in cj: return None
    raw = np.array(cj["PID"]).ravel()
    try:
        s = "".join(chr(int(c)) for c in raw if int(c) > 0); return s.strip() or None
    except Exception: return None

def resize_img(img, size):
    from PIL import Image
    im = Image.fromarray(((img - img.min()) / (img.max() - img.min() + 1e-8) * 255
                          ).astype(np.uint8)).resize((size, size), Image.BILINEAR)
    return np.asarray(im).astype(np.float32)

def load_data():
    if os.path.exists(CACHE):
        d = np.load(CACHE, allow_pickle=True)
        print(f"  loaded from cache: {CACHE}")
        return d["X"], d["Y"], d["PID"]
    files = find_mat(DATA_ROOT)
    print(f"Found {len(files)} .mat files")
    X, Y, PID = [], [], []
    for k, p in enumerate(files):
        try:
            with h5py.File(p, "r") as f:
                cj = f["cjdata"]
                lab = int(np.array(cj["label"]).ravel()[0])
                img = np.array(cj["image"]).astype(np.float32)
                if img.ndim == 2: img = img.T
                pid = read_pid(cj)
        except Exception:
            continue
        if pid is None: continue
        X.append((resize_img(img, IMG_SIZE) / 255.0).astype(np.float32))
        Y.append(lab - 1); PID.append(pid)
        if (k + 1) % 800 == 0: print(f"  loaded {k+1}/{len(files)}")
    X = np.array(X)[:, None]; Y = np.array(Y); PID = np.array(PID)
    np.savez_compressed(CACHE, X=X, Y=Y, PID=PID)
    print(f"  cached to {CACHE}")
    return X, Y, PID

# ── backbones ────────────────────────────────────────────────
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

class ResBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.n1 = nn.GroupNorm(8, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.n2 = nn.GroupNorm(8, cout)
        self.short = (nn.Sequential() if stride == 1 and cin == cout else
                      nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                    nn.GroupNorm(8, cout)))
    def forward(self, x):
        o = F.relu(self.n1(self.c1(x)))
        o = self.n2(self.c2(o))
        return F.relu(o + self.short(x))

class ResNetGN(nn.Module):
    """ResNet-18 style with GroupNorm, sized for 128x128 single-channel MRI."""
    def __init__(self, n=3):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, 32, 7, 2, 3, bias=False),
                                  nn.GroupNorm(8, 32), nn.ReLU(),
                                  nn.MaxPool2d(3, 2, 1))
        self.l1 = nn.Sequential(ResBlock(32, 32), ResBlock(32, 32))
        self.l2 = nn.Sequential(ResBlock(32, 64, 2), ResBlock(64, 64))
        self.l3 = nn.Sequential(ResBlock(64, 128, 2), ResBlock(128, 128))
        self.fc = nn.Linear(128, n)
    def forward(self, x):
        x = self.stem(x); x = self.l1(x); x = self.l2(x); x = self.l3(x)
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))

def make_model(kind):
    return (ResNetGN() if kind == "resnet" else SmallNet()).to(DEVICE)

def _flat(m): return torch.cat([p.data.flatten() for p in m.parameters()])
def _set(m, f):
    i = 0
    for p in m.parameters(): n = p.numel(); p.data.copy_(f[i:i+n].view_as(p)); i += n

# ── losses: the four methods compared ────────────────────────
def loss_hard_mask(logits, yb, present, counts):
    mask = torch.full((logits.shape[1],), -1e9, device=logits.device)
    for c in present: mask[c] = 0.0
    return F.cross_entropy(logits + mask, yb)

def loss_fedrs(logits, yb, present, counts, alpha=0.5):
    scale = torch.full((logits.shape[1],), alpha, device=logits.device)
    for c in present: scale[c] = 1.0
    return F.cross_entropy(logits * scale, yb)

def loss_fedlc(logits, yb, present, counts, tau=1.0):
    """FedLC (Zhang et al., ICML 2022): subtract a frequency-dependent margin.
       Rare classes get a larger margin; absent classes get a fixed large margin,
       which is the generalisation of the FedRS restriction."""
    n = counts.float()
    margin = tau * n.clamp(min=1.0).pow(-0.25)
    margin = torch.where(n > 0, margin, torch.full_like(margin, 10.0))
    adj = logits - margin.unsqueeze(0)
    adj = adj + torch.zeros_like(adj).scatter_(
        1, yb.unsqueeze(1), margin[yb].unsqueeze(1))
    return F.cross_entropy(adj, yb)

def loss_fedavg(logits, yb, present, counts):
    return F.cross_entropy(logits, yb)

METHODS = [
    ("FedAvg",        loss_fedavg),
    ("FedRS(0.5)",    loss_fedrs),
    ("FedLC",         loss_fedlc),
    ("HardMask",      loss_hard_mask),
]

# ── federated engine ─────────────────────────────────────────
def make_loader(X, Y, idx):
    ds = TensorDataset(torch.tensor(X[idx]), torch.tensor(Y[idx]))
    y = Y[idx]; cc = np.bincount(y, minlength=3)
    w = np.array([1.0 / cc[c] if cc[c] > 0 else 0.0 for c in y])
    sampler = WeightedRandomSampler(torch.tensor(w, dtype=torch.double), len(w), replacement=True)
    return DataLoader(ds, batch_size=BATCH, sampler=sampler)

def federated_run(X, Y, cidx, test_idx, loss_fn, seed, backbone="small"):
    torch.manual_seed(seed); np.random.seed(seed)
    model = make_model(backbone)
    clients = list(cidx); ni = {c: len(cidx[c]) for c in clients}
    loaders = {c: make_loader(X, Y, cidx[c]) for c in clients}
    present = {c: list(np.unique(Y[cidx[c]])) for c in clients}
    counts = {c: torch.tensor(np.bincount(Y[cidx[c]], minlength=3),
                              dtype=torch.float, device=DEVICE) for c in clients}
    for _ in range(ROUNDS):
        g = _flat(model); deltas = []
        for c in clients:
            _set(model, g)
            opt = torch.optim.SGD(model.parameters(), lr=LR); model.train()
            for _ in range(LOCAL_EPOCHS):
                for xb, yb in loaders[c]:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    opt.zero_grad()
                    loss_fn(model(xb), yb, present[c], counts[c]).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
            deltas.append((ni[c], _flat(model) - g))
        tot = sum(n for n, _ in deltas)
        _set(model, g + sum(n / tot * d for n, d in deltas))
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X[test_idx]).to(DEVICE)).argmax(1).cpu().numpy()
    return balanced_accuracy_score(Y[test_idx], pred)

def centralized_run(X, Y, train_idx, test_idx, seed, backbone="small"):
    """PART B: pooled training with the same backbone and total step budget."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = make_model(backbone)
    loader = make_loader(X, Y, train_idx)
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    model.train()
    for _ in range(ROUNDS):          # same number of passes as federated rounds
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); F.cross_entropy(model(xb), yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
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
    return {c: v for c, v in ci.items() if len(v) >= 4}

# ── logging ──────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=["part", "method", "seed", "acc"]).writeheader()
def is_done(part, m, s):
    with open(CSV_PATH) as f:
        return any(r["part"] == part and r["method"] == m and int(r["seed"]) == s
                   for r in csv.DictReader(f))
def log_row(part, m, s, acc):
    with open(CSV_PATH, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=["part", "method", "seed", "acc"]).writerow(
            dict(part=part, method=m, seed=s, acc=round(acc, 4)))

# ── MAIN ─────────────────────────────────────────────────────
print("=" * 64); print("  LOADING"); print("=" * 64)
X, Y, PID = load_data()
print(f"  {len(X)} slices, {len(np.unique(PID))} patients, classes={np.bincount(Y)}")
rng = np.random.default_rng(42); uniq = np.unique(PID); rng.shuffle(uniq)
test_pids = set(uniq[:int(0.2 * len(uniq))])
test_idx = np.where(np.isin(PID, list(test_pids)))[0]
train_mask = ~np.isin(PID, list(test_pids)); train_pos = np.where(train_mask)[0]

# ---------- PART A: FedLC baseline ----------
if RUN_PART_A:
    print("\n" + "=" * 64); print("  PART A — FedLC and baselines (small backbone)"); print("=" * 64)
    for seed in range(SEEDS):
        if all(is_done("A", n, seed) for n, _ in METHODS):
            print(f"  seed {seed:2d}: SKIP"); continue
        cp = dirichlet_partition(Y[train_mask], PID[train_mask], N_CLIENTS, ALPHA_DIR, seed)
        cp = {c: train_pos[v] for c, v in cp.items()}
        parts = []; t0 = time.time()
        for name, fn in METHODS:
            if is_done("A", name, seed): parts.append(f"{name}=SKIP"); continue
            acc = federated_run(X, Y, cp, test_idx, fn, seed, "small")
            log_row("A", name, seed, acc); parts.append(f"{name}={acc:.3f}")
        print(f"  seed {seed:2d}: {' | '.join(parts)}  ({time.time()-t0:.0f}s)")

# ---------- PART B: centralized upper bound ----------
if RUN_PART_B:
    print("\n" + "=" * 64); print("  PART B — centralized upper bound"); print("=" * 64)
    for seed in range(5):
        if is_done("B", "Centralized", seed): continue
        acc = centralized_run(X, Y, train_pos, test_idx, seed, "small")
        log_row("B", "Centralized", seed, acc)
        print(f"  seed {seed}: centralized={acc:.3f}")

# ---------- PART C: ResNet backbone ----------
if RUN_PART_C:
    print("\n" + "=" * 64); print("  PART C — ResNet-18 (GroupNorm) backbone"); print("=" * 64)
    for seed in range(10):
        if all(is_done("C", n, seed) for n, _ in METHODS):
            print(f"  seed {seed:2d}: SKIP"); continue
        cp = dirichlet_partition(Y[train_mask], PID[train_mask], N_CLIENTS, ALPHA_DIR, seed)
        cp = {c: train_pos[v] for c, v in cp.items()}
        parts = []; t0 = time.time()
        for name, fn in METHODS:
            if is_done("C", name, seed): parts.append(f"{name}=SKIP"); continue
            acc = federated_run(X, Y, cp, test_idx, fn, seed, "resnet")
            log_row("C", name, seed, acc); parts.append(f"{name}={acc:.3f}")
        print(f"  seed {seed:2d}: {' | '.join(parts)}  ({time.time()-t0:.0f}s)")

# ── analysis ─────────────────────────────────────────────────
rows = {}
with open(CSV_PATH) as f:
    for r in csv.DictReader(f):
        rows.setdefault((r["part"], r["method"]), {})[int(r["seed"])] = float(r["acc"])

def holm(pv):
    o = np.argsort(pv); m = len(pv); adj = np.zeros(m)
    for rank, i in enumerate(o): adj[i] = min((m - rank) * pv[i], 1.0)
    for i in range(1, m): adj[o[i]] = max(adj[o[i]], adj[o[i-1]])
    return adj

for part, title in [("A", "PART A — small backbone"), ("C", "PART C — ResNet backbone")]:
    present_methods = [n for n, _ in METHODS if (part, n) in rows]
    if not present_methods: continue
    print("\n" + "=" * 64); print(f"  {title}"); print("=" * 64)
    for n in present_methods:
        v = np.array(list(rows[(part, n)].values()))
        print(f"  {n:14s}: {v.mean():.4f} +/- {v.std():.4f}  (min {v.min():.3f})")
    if ("HardMask" in present_methods):
        base = rows[(part, "HardMask")]
        comp = [n for n in present_methods if n != "HardMask"]
        pv = []; rec = []
        for n in comp:
            common = sorted(set(rows[(part, n)]) & set(base))
            a = np.array([base[s] for s in common]); b = np.array([rows[(part, n)][s] for s in common])
            d = a.mean() - b.mean()
            try: p = stats.wilcoxon(a, b).pvalue
            except Exception: p = 1.0
            pv.append(p); rec.append((n, d, len(common), int((a > b).sum())))
        adj = holm(pv)
        print(f"\n  HardMask vs each (Holm-adjusted):")
        print(f"  {'baseline':14s} {'delta':>8} {'p':>8} {'wins':>8}")
        for i, (n, d, tot, w) in enumerate(rec):
            print(f"  {n:14s} {d:+8.4f} {adj[i]:8.4f} {w:>4}/{tot}")

if ("B", "Centralized") in rows:
    v = np.array(list(rows[("B", "Centralized")].values()))
    print("\n" + "=" * 64); print("  PART B — centralized upper bound"); print("=" * 64)
    print(f"  Centralized (pooled data): {v.mean():.4f} +/- {v.std():.4f}")
    if ("A", "HardMask") in rows:
        hm = np.array(list(rows[("A", "HardMask")].values()))
        fa = np.array(list(rows[("A", "FedAvg")].values())) if ("A","FedAvg") in rows else None
        print(f"  HardMask (federated)     : {hm.mean():.4f}")
        if fa is not None:
            gap = v.mean() - fa.mean()
            closed = (hm.mean() - fa.mean()) / gap * 100 if gap > 0 else float('nan')
            print(f"  FedAvg   (federated)     : {fa.mean():.4f}")
            print(f"  -> the correction closes {closed:.0f}% of the federated-to-centralized gap")

print("\nDONE. Copy back all sections.\n")
