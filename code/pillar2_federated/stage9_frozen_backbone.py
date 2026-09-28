"""
==================================================================
 CF-SCAFFOLD — STAGE 9: FROZEN PRETRAINED BACKBONE
 Does the correction matter more, or less, in the transfer-learning
 regime that clinical practice actually uses?
==================================================================
 MOTIVATION
   Stage 6 showed the benefit halving when the backbone was enlarged
   from a three-block CNN to a residual network trained from scratch,
   which is a genuine limitation: as the feature extractor takes on
   more of the classification, less is left for the final layer that
   our correction acts on.

   A reviewer will point out that real medical pipelines use large
   pretrained models, and will ask whether the result therefore
   matters in practice. This experiment answers that question, and
   the answer is not the one the Stage 6 trend suggests.

 THE ARGUMENT
   In transfer learning the extractor is FROZEN and only a
   classification head is trained. The head is then responsible for
   the entire decision, and the head is exactly where the
   absent-class gradient does its damage. So freezing should make the
   pathology WORSE and the correction MORE important, reversing the
   Stage 6 direction rather than continuing it.

   This is a falsifiable prediction. If the gain shrinks here too,
   the prediction is wrong and we report that.

 WHY IT IS CHEAP
   The extractor never updates, so features are computed ONCE for the
   whole dataset and cached. Federated training then operates on
   512-dimensional vectors with a linear head of 1,539 parameters.
   Each run takes seconds rather than minutes.

 ARMS (identical partitions, seeds and protocol to Stage 6)
   FedAvg, FedRS(0.5), FedLC, HardMask(0)
   plus a centralised reference on the same frozen features.
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
DATA_ROOT   = "/kaggle/input"
OUT_DIR     = "/kaggle/working"
FEAT_CACHE  = f"{OUT_DIR}/frozen_features.npz"
CSV_PATH    = f"{OUT_DIR}/stage9_frozen_log.csv"
IMG_SIZE    = 128
N_CLIENTS   = 18
BATCH       = 32
SEEDS       = 20
ALPHA_DIR   = 0.3
ROUNDS      = 40
LOCAL_EPOCHS= 1
LR          = 0.01
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# ── data reading ─────────────────────────────────────────────
def find_mat(root):
    out = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
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

def resize_img(img, size):
    from PIL import Image
    im = Image.fromarray(((img - img.min()) / (img.max() - img.min() + 1e-8) * 255
                          ).astype(np.uint8)).resize((size, size), Image.BILINEAR)
    return np.asarray(im).astype(np.float32)

def load_slices():
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
        if (k + 1) % 800 == 0: print(f"  read {k+1}/{len(files)}")
    return np.array(X)[:, None], np.array(Y), np.array(PID)

# ── frozen feature extraction (done once) ────────────────────
def extract_frozen_features(X):
    """Pass every slice through an ImageNet-pretrained ResNet-18 with the
       classifier removed. The extractor is never updated, so this is done
       once and cached. Grayscale is replicated to three channels and
       normalised with the statistics the network was trained on."""
    from torchvision.models import resnet18, ResNet18_Weights
    weights = ResNet18_Weights.IMAGENET1K_V1
    net = resnet18(weights=weights)
    extractor = nn.Sequential(*list(net.children())[:-1]).to(DEVICE).eval()
    for p in extractor.parameters():
        p.requires_grad = False

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE)

    feats = []
    with torch.no_grad():
        for i in range(0, len(X), 128):
            xb = torch.tensor(X[i:i + 128]).to(DEVICE)
            xb = xb.repeat(1, 3, 1, 1)
            xb = (xb - mean) / std
            feats.append(extractor(xb).flatten(1).cpu().numpy())
            if (i // 128) % 5 == 0:
                print(f"  extracted {min(i+128, len(X))}/{len(X)}")
    return np.concatenate(feats).astype(np.float32)

def get_features():
    if os.path.exists(FEAT_CACHE):
        d = np.load(FEAT_CACHE, allow_pickle=True)
        print(f"  loaded cached features: {d['Z'].shape}")
        return d["Z"], d["Y"], d["PID"]
    X, Y, PID = load_slices()
    print(f"  {len(X)} slices, {len(np.unique(PID))} patients, classes={np.bincount(Y)}")
    print("Extracting frozen ResNet-18 features (once)...")
    Z = extract_frozen_features(X)
    np.savez_compressed(FEAT_CACHE, Z=Z, Y=Y, PID=PID)
    print(f"  cached to {FEAT_CACHE}: {Z.shape}")
    return Z, Y, PID

# ── the four local objectives (identical to Stage 6) ─────────
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

METHODS = [
    ("FedAvg",        loss_fedavg),
    ("FedRS(0.5)",    loss_fedrs),
    ("FedLC",         loss_fedlc),
    ("HardMask",      loss_hardmask),
]

# ── federated training of the linear head only ───────────────
def make_loader(Z, Y, idx):
    ds = TensorDataset(torch.tensor(Z[idx]), torch.tensor(Y[idx]))
    y = Y[idx]; cc = np.bincount(y, minlength=3)
    w = np.array([1.0 / cc[c] if cc[c] > 0 else 0.0 for c in y])
    sampler = WeightedRandomSampler(torch.tensor(w, dtype=torch.double),
                                    len(w), replacement=True)
    return DataLoader(ds, batch_size=BATCH, sampler=sampler)

def federated_run(Z, Y, cidx, test_idx, loss_fn, seed):
    """Only a linear head is trained; the extractor is frozen and its output
       is already cached, so the whole federation operates on 512-d vectors."""
    torch.manual_seed(seed); np.random.seed(seed)
    head = nn.Linear(Z.shape[1], 3).to(DEVICE)
    clients = list(cidx)
    ni = {c: len(cidx[c]) for c in clients}
    loaders = {c: make_loader(Z, Y, cidx[c]) for c in clients}
    present = {c: list(np.unique(Y[cidx[c]])) for c in clients}
    counts = {c: torch.tensor(np.bincount(Y[cidx[c]], minlength=3),
                              dtype=torch.float, device=DEVICE) for c in clients}

    def flat(m): return torch.cat([p.data.flatten() for p in m.parameters()])
    def setflat(m, f):
        i = 0
        for p in m.parameters():
            n = p.numel(); p.data.copy_(f[i:i+n].view_as(p)); i += n

    for _ in range(ROUNDS):
        g = flat(head); deltas = []
        for c in clients:
            setflat(head, g)
            opt = torch.optim.SGD(head.parameters(), lr=LR); head.train()
            for _ in range(LOCAL_EPOCHS):
                for zb, yb in loaders[c]:
                    zb, yb = zb.to(DEVICE), yb.to(DEVICE)
                    opt.zero_grad()
                    loss_fn(head(zb), yb, present[c], counts[c]).backward()
                    torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                    opt.step()
            deltas.append((ni[c], flat(head) - g))
        tot = sum(n for n, _ in deltas)
        setflat(head, g + sum(n / tot * d for n, d in deltas))

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
print("=" * 64); print("  FROZEN PRETRAINED BACKBONE"); print("=" * 64)
Z, Y, PID = get_features()
print(f"  feature dimension: {Z.shape[1]}   trainable head parameters: "
      f"{Z.shape[1] * 3 + 3:,}")

# Sanity check before any federated run. If the frozen ImageNet features
# cannot separate the three tumor types even when trained centrally, then
# every federated number below is measuring the weakness of the features
# rather than the effect of the loss, and the experiment says nothing.
# This guard makes that failure visible instead of silent.
rng = np.random.default_rng(42); uniq = np.unique(PID); rng.shuffle(uniq)
test_pids = set(uniq[:int(0.2 * len(uniq))])
test_idx = np.where(np.isin(PID, list(test_pids)))[0]
train_mask = ~np.isin(PID, list(test_pids)); train_pos = np.where(train_mask)[0]
print(f"  train slices {train_mask.sum()}, test slices {len(test_idx)}")

# --- guard: do the frozen features carry the signal at all? -----------
print("\n" + "=" * 64); print("  SANITY CHECK ON THE FROZEN FEATURES"); print("=" * 64)
_probe = centralized_run(Z, Y, train_pos, test_idx, 0)
print(f"  centralised linear probe on frozen features: {_probe:.3f}")
if _probe < 0.50:
    print("\n  !! The probe is close to chance (0.333). ImageNet features are not")
    print("     separating the three tumor types, so any federated comparison")
    print("     below would be measuring feature weakness, not the loss function.")
    print("     Do not report these numbers. Options: fine-tune the last residual")
    print("     block instead of freezing everything, or use a backbone pretrained")
    print("     on medical images (for example RadImageNet).")
    print("     Stopping here.")
    raise SystemExit(0)
print("  The features carry the signal; the comparison below is meaningful.\n")

print("\n" + "=" * 64)
print(f"  FEDERATED COMPARISON ON FROZEN FEATURES ({SEEDS} seeds)")
print("=" * 64)
for seed in range(SEEDS):
    if all(is_done(n, seed) for n, _ in METHODS):
        print(f"  seed {seed:2d}: SKIP"); continue
    cp = dirichlet_partition(Y[train_mask], PID[train_mask],
                             N_CLIENTS, ALPHA_DIR, seed)
    cp = {c: train_pos[v] for c, v in cp.items()}
    parts = []; t0 = time.time()
    for name, fn in METHODS:
        if is_done(name, seed): parts.append(f"{name}=SKIP"); continue
        acc = federated_run(Z, Y, cp, test_idx, fn, seed)
        log_row(name, seed, acc); parts.append(f"{name}={acc:.3f}")
    print(f"  seed {seed:2d}: {' | '.join(parts)}  ({time.time()-t0:.0f}s)")

print("\n  centralised reference on the same frozen features:")
for seed in range(5):
    if is_done("Centralized", seed): continue
    a = centralized_run(Z, Y, train_pos, test_idx, seed)
    log_row("Centralized", seed, a)
    print(f"    seed {seed}: {a:.3f}")

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

print("\n" + "=" * 64); print("  RESULTS — frozen backbone"); print("=" * 64)
for n, _ in METHODS:
    v = np.array([rows[n][s] for s in seeds if s in rows.get(n, {})])
    if len(v):
        print(f"  {n:14s}: {v.mean():.4f} +/- {v.std():.4f}  "
              f"(min {v.min():.3f})  collapses={int((v<=0.40).sum())}/{len(v)}")
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
    pv.append(p); rec.append((n, d.mean(), d.mean()/(d.std(ddof=1)+1e-12),
                              int((d > 0).sum()), len(common)))
if pv:
    adj = holm(pv)
    print(f"\n  HardMask vs each (Holm-adjusted):")
    print(f"  {'baseline':14s} {'delta':>8} {'p':>8} {'d':>7} {'wins':>8}")
    for i, (n, dm, dd, w, t) in enumerate(rec):
        print(f"  {n:14s} {dm:+8.4f} {adj[i]:8.4f} {dd:+7.2f} {w:>4}/{t}")

print("\n" + "=" * 64); print("  WHAT THIS MEANS"); print("=" * 64)
if rec:
    hm_vs_avg = next((r for r in rec if r[0] == "FedAvg"), None)
    if hm_vs_avg:
        gain = hm_vs_avg[1] * 100
        print(f"  Gain of masking over FedAvg on frozen features: {gain:+.1f} points.")
        print(f"  For comparison, the same quantity measured in earlier stages:")
        print(f"    small CNN trained from scratch : +6.5 points")
        print(f"    residual network from scratch  : +2.9 points")
        print()
        if gain > 6.5:
            print("  The gain is LARGER than with the small network trained from")
            print("  scratch. Freezing the extractor moves the whole decision into")
            print("  the linear head, which is where the absent-class gradient acts,")
            print("  so the pathology is amplified rather than diluted. The Stage 6")
            print("  trend does not extend to the transfer-learning regime, and the")
            print("  regime that clinical practice actually uses is the one where")
            print("  the correction matters most.")
        elif gain > 2.9:
            print("  The gain sits between the two from-scratch settings. Freezing")
            print("  partially restores the effect but does not amplify it beyond")
            print("  the small-network case. Report the ordering as measured.")
        else:
            print("  The gain does NOT recover under freezing. The prediction that")
            print("  motivated this experiment is not supported, and the limitation")
            print("  identified in Stage 6 stands as stated. Report this as a")
            print("  negative result.")
print("\nDONE. Copy back RESULTS, the comparison table, and WHAT THIS MEANS.\n")
