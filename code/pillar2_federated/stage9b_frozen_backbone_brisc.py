"""
==================================================================
 CF-SCAFFOLD — STAGE 9B: FROZEN BACKBONE ON BRISC
 The frozen-extractor comparison, repeated on the second dataset.
==================================================================
 This is Stage 9 with one change: the data loader. BRISC ships JPEG
 images rather than MATLAB files, which is why running the Figshare
 version against it reports "Found 0 .mat files".

 EVERYTHING ELSE IS IDENTICAL so the two runs can be compared:
 the same frozen ImageNet ResNet-18, the same linear head, the same
 four local objectives, the same 20 seeds, the same Dirichlet(0.3)
 partition and the same statistics.

 ------------------------------------------------------------------
 A CAVEAT THAT MUST TRAVEL WITH THESE NUMBERS

 BRISC does not distribute patient identifiers. Clients and the test
 split are therefore formed from individual slices, and slices from
 one patient can fall on both sides of the split. Adjacent slices of
 the same patient look nearly identical, so this inflates absolute
 accuracy. Yagis et al. measured that inflation on brain MRI at 29 to
 55 points.

 What this does and does not permit:
   PERMITTED  — comparing the four methods against each other, since
                every method faces the identical partition.
   NOT PERMITTED — comparing these absolute numbers with the
                patient-level Figshare figures, or presenting them as
                clinical accuracy.

 The script prints this warning alongside the results so it cannot be
 detached from them.
 ------------------------------------------------------------------
 Kaggle: GPU. Attach briscdataset/brisc2025. Auto-resume via CSV.
==================================================================
"""

import os, csv, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
from scipy import stats
from sklearn.metrics import balanced_accuracy_score
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from PIL import Image

# ── CONFIG ───────────────────────────────────────────────────
DATA_ROOT    = "/kaggle/input"
OUT_DIR      = "/kaggle/working"
FEAT_CACHE   = f"{OUT_DIR}/brisc_frozen_features.npz"
CSV_PATH     = f"{OUT_DIR}/stage9b_brisc_log.csv"
IMG_SIZE     = 128
N_CLIENTS    = 18
BATCH        = 32
SEEDS        = 20
ALPHA_DIR    = 0.3
ROUNDS       = 40
LOCAL_EPOCHS = 1
LR           = 0.01
MAX_PER_TYPE = 1000          # same subset size as the earlier BRISC run
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

TUMOR_CODES   = {"gl": "glioma", "me": "meningioma",
                 "pi": "pituitary", "no": "no_tumor"}
# same order as Figshare, so the two experiments are directly comparable
TYPE_TO_LABEL = {"meningioma": 0, "glioma": 1, "pituitary": 2}

# ── BRISC loader ─────────────────────────────────────────────
def parse_tumor_type(path):
    """BRISC encodes the class in the filename for the segmentation subset
       and in the parent folder for the classification subset. Try both."""
    for part in os.path.splitext(os.path.basename(path))[0].split("_"):
        if part in TUMOR_CODES:
            return TUMOR_CODES[part]
    parent = os.path.basename(os.path.dirname(path)).lower()
    for t in ["glioma", "meningioma", "pituitary", "no_tumor"]:
        if t in parent:
            return t
    return None

def load_brisc(root, max_per_type=None):
    candidates = []
    for dp, _, files in os.walk(root):
        d = dp.replace("\\", "/")
        for fn in files:
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            if "mask" in fn.lower() or "/masks/" in d:
                continue                      # masks are not needed here
            candidates.append(os.path.join(dp, fn))
    print(f"  candidate image files found: {len(candidates)}")
    if not candidates:
        raise SystemExit(
            "No images found under /kaggle/input.\n"
            "Check that the BRISC dataset is attached as a DATASET (not a "
            "notebook). Expected: briscdataset/brisc2025.")

    X, Y, counts = [], [], {}
    for path in sorted(candidates):
        t = parse_tumor_type(path)
        if t is None or t == "no_tumor":
            continue                          # three tumor classes only
        if max_per_type and counts.get(t, 0) >= max_per_type:
            continue
        try:
            img = Image.open(path).convert("L").resize((IMG_SIZE, IMG_SIZE),
                                                       Image.BILINEAR)
        except Exception:
            continue
        X.append((np.asarray(img).astype(np.float32) / 255.0))
        Y.append(TYPE_TO_LABEL[t])
        counts[t] = counts.get(t, 0) + 1
        if len(X) % 500 == 0:
            print(f"  loaded {len(X)}")
    print(f"  class distribution: {counts}")
    return np.array(X)[:, None], np.array(Y, dtype=np.int64)

# ── frozen feature extraction ────────────────────────────────
def extract_frozen_features(X):
    from torchvision.models import resnet18, ResNet18_Weights
    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    extractor = nn.Sequential(*list(net.children())[:-1]).to(DEVICE).eval()
    for p in extractor.parameters():
        p.requires_grad = False
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE)
    feats = []
    with torch.no_grad():
        for i in range(0, len(X), 128):
            xb = torch.tensor(X[i:i + 128]).to(DEVICE).repeat(1, 3, 1, 1)
            xb = (xb - mean) / std
            feats.append(extractor(xb).flatten(1).cpu().numpy())
            if (i // 128) % 5 == 0:
                print(f"  extracted {min(i+128, len(X))}/{len(X)}")
    return np.concatenate(feats).astype(np.float32)

def get_features():
    if os.path.exists(FEAT_CACHE):
        d = np.load(FEAT_CACHE, allow_pickle=True)
        print(f"  loaded cached features: {d['Z'].shape}")
        return d["Z"], d["Y"]
    X, Y = load_brisc(DATA_ROOT, MAX_PER_TYPE)
    print(f"  {len(X)} slices, classes={np.bincount(Y)}")
    print("Extracting frozen ResNet-18 features (once)...")
    Z = extract_frozen_features(X)
    np.savez_compressed(FEAT_CACHE, Z=Z, Y=Y)
    print(f"  cached to {FEAT_CACHE}: {Z.shape}")
    return Z, Y

# ── the four local objectives (unchanged from Stage 9) ───────
def loss_fedavg(logits, yb, present, counts):
    return F.cross_entropy(logits, yb)

def loss_fedrs(logits, yb, present, counts, alpha=0.5):
    scale = torch.full((logits.shape[1],), alpha, device=logits.device)
    for c in present: scale[c] = 1.0
    return F.cross_entropy(logits * scale, yb)

def loss_fedlc(logits, yb, present, counts, tau=1.0):
    n = counts.float()
    margin = tau * n.clamp(min=1.0).pow(-0.25)
    margin = torch.where(n > 0, margin, torch.full_like(margin, 10.0))
    adj = logits - margin.unsqueeze(0)
    adj = adj + torch.zeros_like(adj).scatter_(
        1, yb.unsqueeze(1), margin[yb].unsqueeze(1))
    return F.cross_entropy(adj, yb)

def loss_hardmask(logits, yb, present, counts):
    mask = torch.full((logits.shape[1],), -1e9, device=logits.device)
    for c in present: mask[c] = 0.0
    return F.cross_entropy(logits + mask, yb)

METHODS = [("FedAvg", loss_fedavg), ("FedRS(0.5)", loss_fedrs),
           ("FedLC", loss_fedlc), ("HardMask", loss_hardmask)]

# ── federated training of the linear head ────────────────────
def make_loader(Z, Y, idx):
    ds = TensorDataset(torch.tensor(Z[idx]), torch.tensor(Y[idx]))
    y = Y[idx]; cc = np.bincount(y, minlength=3)
    w = np.array([1.0 / cc[c] if cc[c] > 0 else 0.0 for c in y])
    sampler = WeightedRandomSampler(torch.tensor(w, dtype=torch.double),
                                    len(w), replacement=True)
    return DataLoader(ds, batch_size=BATCH, sampler=sampler)

def _flat(m): return torch.cat([p.data.flatten() for p in m.parameters()])
def _set(m, f):
    i = 0
    for p in m.parameters():
        n = p.numel(); p.data.copy_(f[i:i+n].view_as(p)); i += n

def federated_run(Z, Y, cidx, test_idx, loss_fn, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    head = nn.Linear(Z.shape[1], 3).to(DEVICE)
    clients = list(cidx); ni = {c: len(cidx[c]) for c in clients}
    loaders = {c: make_loader(Z, Y, cidx[c]) for c in clients}
    present = {c: list(np.unique(Y[cidx[c]])) for c in clients}
    counts = {c: torch.tensor(np.bincount(Y[cidx[c]], minlength=3),
                              dtype=torch.float, device=DEVICE) for c in clients}
    for _ in range(ROUNDS):
        g = _flat(head); deltas = []
        for c in clients:
            _set(head, g)
            opt = torch.optim.SGD(head.parameters(), lr=LR); head.train()
            for _ in range(LOCAL_EPOCHS):
                for zb, yb in loaders[c]:
                    zb, yb = zb.to(DEVICE), yb.to(DEVICE)
                    opt.zero_grad()
                    loss_fn(head(zb), yb, present[c], counts[c]).backward()
                    torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                    opt.step()
            deltas.append((ni[c], _flat(head) - g))
        tot = sum(n for n, _ in deltas)
        _set(head, g + sum(n / tot * d for n, d in deltas))
    head.eval()
    with torch.no_grad():
        pred = head(torch.tensor(Z[test_idx]).to(DEVICE)).argmax(1).cpu().numpy()
    return balanced_accuracy_score(Y[test_idx], pred)

def centralized_run(Z, Y, train_idx, test_idx, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    head = nn.Linear(Z.shape[1], 3).to(DEVICE)
    loader = make_loader(Z, Y, train_idx)
    opt = torch.optim.SGD(head.parameters(), lr=LR); head.train()
    for _ in range(ROUNDS):
        for zb, yb in loader:
            zb, yb = zb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); F.cross_entropy(head(zb), yb).backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step()
    head.eval()
    with torch.no_grad():
        pred = head(torch.tensor(Z[test_idx]).to(DEVICE)).argmax(1).cpu().numpy()
    return balanced_accuracy_score(Y[test_idx], pred)

def dirichlet_partition_by_label(Y, n_clients, alpha_dir, seed):
    """No patient identifiers in BRISC, so slices are drawn directly."""
    rng = np.random.default_rng(seed); cp = {c: [] for c in range(n_clients)}
    for cls in np.unique(Y):
        idx = np.where(Y == cls)[0]; rng.shuffle(idx)
        pr = rng.dirichlet([alpha_dir] * n_clients)
        cnt = (pr * len(idx)).astype(int); cnt[-1] = len(idx) - cnt[:-1].sum()
        pos = 0
        for c in range(n_clients):
            cp[c].extend(idx[pos:pos + cnt[c]]); pos += cnt[c]
    return {c: np.array(v) for c, v in cp.items() if len(v) >= BATCH}

# ── logging ──────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=["method", "seed", "acc"]).writeheader()
def is_done(m, s):
    with open(CSV_PATH) as f:
        return any(r["method"] == m and int(r["seed"]) == s
                   for r in csv.DictReader(f))
def log_row(m, s, a):
    with open(CSV_PATH, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=["method", "seed", "acc"]).writerow(
            dict(method=m, seed=s, acc=round(a, 4)))

# ── MAIN ─────────────────────────────────────────────────────
print("=" * 64); print("  FROZEN BACKBONE ON BRISC"); print("=" * 64)
Z, Y = get_features()
print(f"  feature dimension: {Z.shape[1]}   trainable head parameters: "
      f"{Z.shape[1] * 3 + 3:,}")

rng = np.random.default_rng(42)
order = rng.permutation(len(Y))
n_test = int(0.2 * len(Y))
test_idx, train_pos = order[:n_test], order[n_test:]
print(f"  train slices {len(train_pos)}, test slices {len(test_idx)}")

print("\n" + "=" * 64); print("  SANITY CHECK ON THE FROZEN FEATURES"); print("=" * 64)
probe = centralized_run(Z, Y, train_pos, test_idx, 0)
print(f"  centralised linear probe on frozen features: {probe:.3f}")
if probe < 0.50:
    print("\n  !! The probe is close to chance. ImageNet features are not")
    print("     separating the three tumor types on this dataset, so the")
    print("     comparison below would measure feature weakness rather than")
    print("     the loss function. Do not report these numbers.")
    raise SystemExit(0)
print("  The features carry the signal; the comparison below is meaningful.")

print("\n" + "=" * 64)
print(f"  FEDERATED COMPARISON ({SEEDS} seeds)")
print("=" * 64)
for seed in range(SEEDS):
    if all(is_done(n, seed) for n, _ in METHODS):
        print(f"  seed {seed:2d}: SKIP"); continue
    cp = dirichlet_partition_by_label(Y[train_pos], N_CLIENTS, ALPHA_DIR, seed)
    cp = {c: train_pos[v] for c, v in cp.items()}
    parts = []; t0 = time.time()
    for name, fn in METHODS:
        if is_done(name, seed): parts.append(f"{name}=SKIP"); continue
        acc = federated_run(Z, Y, cp, test_idx, fn, seed)
        log_row(name, seed, acc); parts.append(f"{name}={acc:.3f}")
    print(f"  seed {seed:2d}: {' | '.join(parts)}  ({time.time()-t0:.0f}s)")

print("\n  centralised reference:")
for seed in range(5):
    if is_done("Centralized", seed): continue
    a = centralized_run(Z, Y, train_pos, test_idx, seed)
    log_row("Centralized", seed, a); print(f"    seed {seed}: {a:.3f}")

# ── analysis ─────────────────────────────────────────────────
rows = {}
with open(CSV_PATH) as f:
    for r in csv.DictReader(f):
        rows.setdefault(r["method"], {})[int(r["seed"])] = float(r["acc"])
seeds = sorted({s for m in rows for s in rows[m]})

def holm(pv):
    o = np.argsort(pv); m = len(pv); adj = np.zeros(m)
    for rank, i in enumerate(o): adj[i] = min((m - rank) * pv[i], 1.0)
    for i in range(1, m): adj[o[i]] = max(adj[o[i]], adj[o[i-1]])
    return adj

print("\n" + "=" * 64); print("  RESULTS — BRISC, frozen backbone"); print("=" * 64)
for n, _ in METHODS:
    v = np.array([rows[n][s] for s in seeds if s in rows.get(n, {})])
    if len(v):
        print(f"  {n:14s}: {v.mean():.4f} +/- {v.std():.4f}  (min {v.min():.3f})")
if "Centralized" in rows:
    v = np.array(list(rows["Centralized"].values()))
    print(f"  {'Centralized':14s}: {v.mean():.4f} +/- {v.std():.4f}")

base = rows.get("HardMask", {})
comp = [n for n, _ in METHODS if n != "HardMask"]
pv, rec = [], []
for n in comp:
    common = sorted(set(rows.get(n, {})) & set(base))
    if len(common) < 3: continue
    a = np.array([base[s] for s in common]); b = np.array([rows[n][s] for s in common])
    d = a - b
    try: p = stats.wilcoxon(a, b).pvalue
    except Exception: p = 1.0
    pv.append(p); rec.append((n, d.mean(), int((d > 0).sum()), len(common)))
if pv:
    adj = holm(pv)
    print(f"\n  HardMask vs each (Holm-adjusted):")
    print(f"  {'baseline':14s} {'delta':>9} {'p':>9} {'wins':>8}")
    for i, (n, dm, w, t) in enumerate(rec):
        print(f"  {n:14s} {dm:+9.4f} {adj[i]:9.4f} {w:>4}/{t}")

print("\n" + "=" * 64)
print("  HOW THESE NUMBERS MAY AND MAY NOT BE USED")
print("=" * 64)
print("""  BRISC carries no patient identifiers, so the split is by slice and
  slices from one patient may appear on both sides of it. Absolute
  accuracy is therefore inflated and is NOT comparable to the
  patient-level Figshare figures, nor is it a clinical accuracy.

  The comparison BETWEEN the four methods remains valid, because every
  method faces the identical partition on every seed. Report the
  ordering and the differences; do not report the absolute values
  without this caveat attached.""")
print("\nDONE. Copy back RESULTS and the comparison table.\n")
