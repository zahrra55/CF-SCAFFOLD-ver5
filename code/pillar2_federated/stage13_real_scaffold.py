"""
==================================================================
 CF-SCAFFOLD — STAGE 13: REAL SCAFFOLD (Option II, Karimireddy 2020)
==================================================================
 The project title references SCAFFOLD, but every previously validated
 result was produced by federated_train() in the main pipeline, which is
 plain weighted FedAvg with a restricted-softmax LOSS — it has no control
 variates. This script implements the real thing, in isolation, so the
 already-documented results are not put at risk while this is tested.

 WHAT SCAFFOLD ADDS (Karimireddy, Kale, Mohri, Reddi, Stich, Suresh,
 "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning",
 ICML 2020, PMLR 119:5132-5143):

   Each client k keeps a persistent control variate c_k (a vector shaped
   like the model). During local SGD, the correction (c - c_k) is added
   to every gradient, where c is the server's global control variate:

       g_local <- g_local + (c - c_k)

   After K local steps at learning rate eta_l, client k reports an
   updated c_k (Option II):

       c_k_new = c_k - c + (x - y_k) / (K * eta_l)

   and the server updates the global model AND the global control
   variate from the participating clients' deltas.

 WHY THIS IS A DIFFERENT MECHANISM FROM THE THESIS'S CORRECTION
   The restricted-softmax loss corrects the OUTPUT LAYER by removing
   the gradient on classes a client does not hold. SCAFFOLD corrects
   DRIFT IN THE SHARED FEATURE EXTRACTOR caused by client heterogeneity
   in general, regardless of which classes are present. They are not
   competitors; the thesis has argued from the start that they target
   different parts of the model. This script tests whether adding real
   SCAFFOLD on top of the existing correction changes anything, and by
   how much — not whether SCAFFOLD replaces the correction.

 PREREGISTERED SUCCESS CRITERION (fixed before running on real data):
   Real SCAFFOLD is considered SAFE TO ADOPT into the main pipeline only
   if, versus the current FedAvg-based training, it does BOTH:
     (a) does not reduce mean balanced accuracy by more than 1 point,
     (b) does not increase the collapse rate (accuracy <= 0.40) at all.
   If either fails, the honest conclusion is that the project's name
   should be corrected instead of the code, and the documented FedAvg
   results stand unchanged as the thesis's real results.

 IMPLEMENTATION SAFEGUARDS AGAINST THE FOUR SILENT-FAILURE MODES
 IDENTIFIED BEFORE WRITING THIS CODE:
   1. Sign checked explicitly: correction is ADDED to the gradient.
   2. c_k is stored in a dict keyed by a STABLE client id and persists
      across rounds; it is never reset inside the round loop.
   3. The global c is updated scaled by the participating fraction
      |S|/N, not a plain average over all N clients.
   4. Clients dropped by the Dirichlet partition (too few samples) never
      enter the c_k dict, so indices cannot desynchronise across rounds.
------------------------------------------------------------------
 Kaggle: GPU. Attach the Figshare .mat dataset. Auto-resume via CSV.
==================================================================
"""

import os, glob, csv, time, warnings
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
CSV_PATH    = f"{OUT_DIR}/stage13_scaffold_log.csv"
IMG_SIZE    = 128
N_CLIENTS   = 18
BATCH       = 32
SEEDS       = 5              # SMALL first: confirm safety before spending 20
ALPHA_DIR   = 0.3
ROUNDS      = 40
LOCAL_EPOCHS = 1
LR          = 0.01
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# ── PREREGISTERED THRESHOLDS (do not edit after seeing results) ─
MAX_ACC_DROP     = 0.01   # (a) at most 1 point worse than FedAvg-based training
MAX_EXTRA_COLLAPSE = 0    # (b) zero additional collapses allowed

# ── data (identical to the validated loader) ─────────────────
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
    files = find_mat(DATA_ROOT)
    print(f"Found {len(files)} .mat files")
    X, Y, PID = [], [], []
    for k, path in enumerate(files):
        try:
            with h5py.File(path, "r") as f:
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
    return np.array(X)[:, None], np.array(Y), np.array(PID)

# ── model (identical architecture to the validated SmallNet) ──
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
def _zeros_like_flat(m): return torch.zeros_like(_flat(m))

def restricted_ce(logits, yb, present):
    """The thesis's correction: unchanged, so the only variable introduced
       is the presence or absence of real SCAFFOLD control variates."""
    mask = torch.full((logits.shape[1],), -1e9, device=logits.device)
    for c in present: mask[c] = 0.0
    return F.cross_entropy(logits + mask, yb)

def make_loader(X, Y, idx):
    ds = TensorDataset(torch.tensor(X[idx]), torch.tensor(Y[idx]))
    y = Y[idx]; cc = np.bincount(y, minlength=3)
    w = np.array([1.0 / cc[c] if cc[c] > 0 else 0.0 for c in y])
    sampler = WeightedRandomSampler(torch.tensor(w, dtype=torch.double), len(w), replacement=True)
    return DataLoader(ds, batch_size=BATCH, sampler=sampler)

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

# ── run A: FedAvg + restricted-softmax (reproduces the documented result) ──
def run_fedavg_restricted(X, Y, client_idx, test_idx, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = SmallNet().to(DEVICE)
    clients = list(client_idx); ni = {c: len(client_idx[c]) for c in clients}
    loaders = {c: make_loader(X, Y, client_idx[c]) for c in clients}
    present = {c: list(np.unique(Y[client_idx[c]])) for c in clients}
    for _ in range(ROUNDS):
        g = _flat(model); deltas = []
        for c in clients:
            _set(model, g)
            opt = torch.optim.SGD(model.parameters(), lr=LR); model.train()
            for _ in range(LOCAL_EPOCHS):
                for xb, yb in loaders[c]:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE); opt.zero_grad()
                    restricted_ce(model(xb), yb, present[c]).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            deltas.append((ni[c], _flat(model) - g))
        tot = sum(n for n, _ in deltas); _set(model, g + sum(n/tot*d for n, d in deltas))
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X[test_idx]).to(DEVICE)).argmax(1).cpu().numpy()
    return balanced_accuracy_score(Y[test_idx], pred)

# ── run B: REAL SCAFFOLD + restricted-softmax ─────────────────
def run_scaffold_restricted(X, Y, client_idx, test_idx, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = SmallNet().to(DEVICE)
    clients = list(client_idx); ni = {c: len(client_idx[c]) for c in clients}
    loaders = {c: make_loader(X, Y, client_idx[c]) for c in clients}
    present = {c: list(np.unique(Y[client_idx[c]])) for c in clients}

    c_global = _zeros_like_flat(model)
    # SAFEGUARD 2: one persistent control-variate vector per stable client id,
    # created once, updated in place across every round.
    c_local = {c: _zeros_like_flat(model) for c in clients}

    for rd in range(ROUNDS):
        g = _flat(model); deltas = []; c_deltas = []
        for c in clients:
            _set(model, g)
            opt = torch.optim.SGD(model.parameters(), lr=LR); model.train()
            steps = 0
            for _ in range(LOCAL_EPOCHS):
                for xb, yb in loaders[c]:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE); opt.zero_grad()
                    restricted_ce(model(xb), yb, present[c]).backward()
                    # SAFEGUARD 1: correction is ADDED, sign matches the
                    # derivation verified before writing this code.
                    with torch.no_grad():
                        i = 0
                        for p in model.parameters():
                            n = p.numel()
                            corr = (c_global[i:i+n] - c_local[c][i:i+n]).view_as(p)
                            p.grad.add_(corr)
                            i += n
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step(); steps += 1
            y_k = _flat(model)
            eff_steps = max(steps, 1)
            # Option II update, exactly as verified analytically beforehand.
            new_c_k = c_local[c] - c_global + (g - y_k) / (eff_steps * LR)
            c_deltas.append((ni[c], new_c_k - c_local[c]))
            c_local[c] = new_c_k
            deltas.append((ni[c], y_k - g))
        tot = sum(n for n, _ in deltas)
        _set(model, g + sum(n/tot*d for n, d in deltas))
        # SAFEGUARD 3: scale by participating fraction |S|/N, not a plain mean.
        participation = len(clients) / N_CLIENTS
        c_global = c_global + participation * sum(n/tot*d for n, d in c_deltas)
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X[test_idx]).to(DEVICE)).argmax(1).cpu().numpy()
    return balanced_accuracy_score(Y[test_idx], pred)

# ── logging ──────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=["method", "seed", "acc"]).writeheader()
def is_done(m, s):
    with open(CSV_PATH) as f:
        return any(r["method"] == m and int(r["seed"]) == s for r in csv.DictReader(f))
def log_row(m, s, a):
    with open(CSV_PATH, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=["method", "seed", "acc"]).writerow(
            dict(method=m, seed=s, acc=round(a, 4)))

# ── MAIN ─────────────────────────────────────────────────────
print("=" * 66); print("  LOADING"); print("=" * 66)
X, Y, PID = load_data()
print(f"  {len(X)} slices, {len(np.unique(PID))} patients, classes={np.bincount(Y)}")
rng = np.random.default_rng(42); uniq = np.unique(PID); rng.shuffle(uniq)
test_pids = set(uniq[:int(0.2 * len(uniq))])
test_idx = np.where(np.isin(PID, list(test_pids)))[0]
train_mask = ~np.isin(PID, list(test_pids)); train_pos = np.where(train_mask)[0]

print("\n" + "=" * 66)
print(f"  FedAvg-restricted  vs  REAL SCAFFOLD-restricted  ({SEEDS} seeds)")
print("=" * 66)
for seed in range(SEEDS):
    if is_done("FedAvg_restricted", seed) and is_done("SCAFFOLD_restricted", seed):
        print(f"  seed {seed}: SKIP"); continue
    cp = dirichlet_partition(Y[train_mask], PID[train_mask], N_CLIENTS, ALPHA_DIR, seed)
    cp = {c: train_pos[v] for c, v in cp.items()}
    t0 = time.time()
    if not is_done("FedAvg_restricted", seed):
        a = run_fedavg_restricted(X, Y, cp, test_idx, seed)
        log_row("FedAvg_restricted", seed, a)
    else:
        a = None
    t1 = time.time()
    if not is_done("SCAFFOLD_restricted", seed):
        b = run_scaffold_restricted(X, Y, cp, test_idx, seed)
        log_row("SCAFFOLD_restricted", seed, b)
    else:
        b = None
    t2 = time.time()
    print(f"  seed {seed}: FedAvg={a if a is not None else '(cached)'}  "
         f"({t1-t0:.0f}s)   SCAFFOLD={b if b is not None else '(cached)'}  "
         f"({t2-t1:.0f}s)")

# ── analysis against the PREREGISTERED criterion ──────────────
rows = {}
with open(CSV_PATH) as f:
    for r in csv.DictReader(f):
        rows.setdefault(r["method"], {})[int(r["seed"])] = float(r["acc"])
fa = rows.get("FedAvg_restricted", {}); sc = rows.get("SCAFFOLD_restricted", {})
common = sorted(set(fa) & set(sc))

print("\n" + "=" * 66); print("  RESULTS"); print("=" * 66)
if common:
    a = np.array([fa[s] for s in common]); b = np.array([sc[s] for s in common])
    print(f"  FedAvg + restricted-softmax   : {a.mean():.4f} +/- {a.std():.4f}  "
         f"collapses={int((a<=0.40).sum())}/{len(a)}")
    print(f"  SCAFFOLD + restricted-softmax : {b.mean():.4f} +/- {b.std():.4f}  "
         f"collapses={int((b<=0.40).sum())}/{len(b)}")
    delta = b.mean() - a.mean()
    try: p = stats.wilcoxon(b, a).pvalue
    except Exception: p = 1.0
    print(f"  delta (SCAFFOLD - FedAvg): {delta*100:+.2f}pp, Wilcoxon p={p:.4f}")

    extra_collapse = int((b <= 0.40).sum()) - int((a <= 0.40).sum())
    cond_a = delta >= -MAX_ACC_DROP
    cond_b = extra_collapse <= MAX_EXTRA_COLLAPSE

    print("\n" + "=" * 66); print("  PREREGISTERED VERDICT"); print("=" * 66)
    print(f"  (a) accuracy drop <= {MAX_ACC_DROP*100:.0f}pp?  "
         f"{'PASS' if cond_a else 'FAIL'}  (observed: {-delta*100:+.2f}pp)")
    print(f"  (b) no extra collapses?  "
         f"{'PASS' if cond_b else 'FAIL'}  (observed: {extra_collapse:+d})")
    if cond_a and cond_b:
        print("\n  SAFE TO ADOPT: real SCAFFOLD control variates can replace the")
        print("  FedAvg aggregation in the main pipeline without materially")
        print("  changing the documented results. Proceed to re-run the full")
        print("  20-seed comparisons with SCAFFOLD before updating any paper.")
    else:
        print("\n  DO NOT ADOPT: at least one criterion failed. The honest,")
        print("  lower-risk path is to keep the validated FedAvg-based results")
        print("  exactly as documented and correct the project's name instead")
        print("  of the code -- for example to a name that does not claim")
        print("  SCAFFOLD, such as a class-presence-guided FedAvg framing.")
        print("  This was the fallback agreed before running this script, and")
        print("  it is a legitimate outcome, not a failure of the thesis.")
else:
    print("  not enough paired seeds yet.")

print("\nDONE. Copy back everything above.\n")
