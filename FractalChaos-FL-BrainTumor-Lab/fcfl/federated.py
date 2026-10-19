"""Hospital partitions, models and federated training."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .config import (
    DEVICE,
    FED_ROUNDS,
    OBJECTIVE_BATCH,
    OBJECTIVE_LOCAL_EPOCHS,
    OBJECTIVE_LR,
    OBJECTIVE_ROUNDS,
)


def dirichlet_partition(Y, PID, n_clients, alpha, seed):
    """Split patients into hospitals with Dirichlet-distributed class proportions."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(PID)
    pl = {p: Y[np.where(PID == p)[0][0]] for p in uniq}
    cp = {c: [] for c in range(n_clients)}
    for cls in np.unique(Y):
        cps = [p for p in uniq if pl[p] == cls]
        rng.shuffle(cps)
        cnt = (rng.dirichlet([alpha] * n_clients) * len(cps)).astype(int)
        if len(cps):
            cnt[-1] = len(cps) - cnt[:-1].sum()
        i = 0
        for c in range(n_clients):
            for _ in range(cnt[c]):
                if i < len(cps):
                    cp[c].append(cps[i])
                    i += 1
    ci = {c: np.where(np.isin(PID, cp[c]))[0] for c in range(n_clients)}
    return {c: v for c, v in ci.items() if len(v) >= 4}


def _flat(m):
    return torch.cat([p.data.flatten() for p in m.parameters()])


def _set(m, f):
    i = 0
    for p in m.parameters():
        n = p.numel()
        p.data.copy_(f[i : i + n].view_as(p))
        i += n


class SmallNet(nn.Module):

    def __init__(self, n=3):
        super().__init__()
        self.c1 = nn.Conv2d(1, 16, 3, padding=1)
        self.n1 = nn.GroupNorm(4, 16)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.n2 = nn.GroupNorm(4, 32)
        self.c3 = nn.Conv2d(32, 64, 3, padding=1)
        self.n3 = nn.GroupNorm(8, 64)
        self.fc = nn.Linear(64, n)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.n1(self.c1(x))), 2)
        x = F.max_pool2d(F.relu(self.n2(self.c2(x))), 2)
        x = F.max_pool2d(F.relu(self.n3(self.c3(x))), 2)
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))


class FeatNet(nn.Module):

    def __init__(self, d, n=3):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, n))

    def forward(self, x):
        return self.f(x)


def train_and_predict(kind, Xr, Y, train_sets, test_idx, seed):
    """Train with the same recipe centrally ({0: all training slices}) or federated; return test probabilities."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    lr = 0.01 if kind == "cnn" else 0.05
    Xt = torch.tensor(Xr, dtype=torch.float32)
    Yt = torch.tensor(Y)
    model = (SmallNet() if kind == "cnn" else FeatNet(Xr.shape[1])).to(DEVICE)
    n = sum((len(v) for v in train_sets.values()))
    for _ in range(FED_ROUNDS):
        g0 = _flat(model)
        acc = None
        for c, idx in train_sets.items():
            _set(model, g0)
            opt = torch.optim.SGD(model.parameters(), lr=lr)
            model.train()
            cc = np.bincount(Y[idx], minlength=3)
            w = np.array([1.0 / cc[c_] for c_ in Y[idx]])
            order = rng.choice(idx, len(idx), p=w / w.sum())
            for s in range(0, len(order), 32):
                b = order[s : s + 32]
                opt.zero_grad()
                F.cross_entropy(model(Xt[b].to(DEVICE)), Yt[b].to(DEVICE)).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            u = _flat(model) * (len(idx) / n)
            acc = u if acc is None else acc + u
        _set(model, acc)
    model.eval()
    with torch.no_grad():
        return torch.cat(
            [
                torch.softmax(model(Xt[test_idx[s : s + 512]].to(DEVICE)), 1).cpu()
                for s in range(0, len(test_idx), 512)
            ]
        ).numpy()


def make_model(kind="small"):
    """CNN used in the image-based federated experiments."""
    return SmallNet().to(DEVICE)


def make_loader(X, Y, idx):
    """Class-balanced data loader."""
    ds = TensorDataset(torch.tensor(X[idx]), torch.tensor(Y[idx]))
    y = Y[idx]
    cc = np.bincount(y, minlength=3)
    w = np.array([1.0 / cc[c] if cc[c] > 0 else 0.0 for c in y])
    sampler = WeightedRandomSampler(torch.tensor(w, dtype=torch.double), len(w), replacement=True)
    return DataLoader(ds, batch_size=OBJECTIVE_BATCH, sampler=sampler)


def loss_hard_mask(logits, yb, present, counts):
    """Cross-entropy restricted to the classes present at the client."""
    mask = torch.full((logits.shape[1],), -1000000000.0, device=logits.device)
    for c in present:
        mask[c] = 0.0
    return F.cross_entropy(logits + mask, yb)


def loss_fedrs(logits, yb, present, counts, alpha=0.5):
    """FedRS: logits of absent classes scaled by alpha."""
    scale = torch.full((logits.shape[1],), alpha, device=logits.device)
    for c in present:
        scale[c] = 1.0
    return F.cross_entropy(logits * scale, yb)


def loss_fedlc(logits, yb, present, counts, tau=1.0):
    """FedLC: frequency-dependent margin subtracted from the logits."""
    n = counts.float()
    margin = tau * n.clamp(min=1.0).pow(-0.25)
    margin = torch.where(n > 0, margin, torch.full_like(margin, 10.0))
    adj = logits - margin.unsqueeze(0)
    adj = adj + torch.zeros_like(adj).scatter_(1, yb.unsqueeze(1), margin[yb].unsqueeze(1))
    return F.cross_entropy(adj, yb)


def loss_fedavg(logits, yb, present, counts):
    """Plain cross-entropy."""
    return F.cross_entropy(logits, yb)


def federated_run(X, Y, cidx, test_idx, loss_fn, seed, backbone="small"):
    """Federated averaging with a given local objective; returns test balanced accuracy."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = make_model(backbone)
    clients = list(cidx)
    ni = {c: len(cidx[c]) for c in clients}
    loaders = {c: make_loader(X, Y, cidx[c]) for c in clients}
    present = {c: list(np.unique(Y[cidx[c]])) for c in clients}
    counts = {
        c: torch.tensor(np.bincount(Y[cidx[c]], minlength=3), dtype=torch.float, device=DEVICE)
        for c in clients
    }
    for _ in range(OBJECTIVE_ROUNDS):
        g = _flat(model)
        deltas = []
        for c in clients:
            _set(model, g)
            opt = torch.optim.SGD(model.parameters(), lr=OBJECTIVE_LR)
            model.train()
            for _ in range(OBJECTIVE_LOCAL_EPOCHS):
                for xb, yb in loaders[c]:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    opt.zero_grad()
                    loss_fn(model(xb), yb, present[c], counts[c]).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
            deltas.append((ni[c], _flat(model) - g))
        tot = sum((n for n, _ in deltas))
        _set(model, g + sum((n / tot * d for n, d in deltas)))
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X[test_idx]).to(DEVICE)).argmax(1).cpu().numpy()
    return balanced_accuracy_score(Y[test_idx], pred)


def centralized_run(X, Y, train_idx, test_idx, seed, backbone="small"):
    """Training on pooled data with the same backbone and number of passes."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = make_model(backbone)
    loader = make_loader(X, Y, train_idx)
    opt = torch.optim.SGD(model.parameters(), lr=OBJECTIVE_LR)
    model.train()
    for _ in range(OBJECTIVE_ROUNDS):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X[test_idx]).to(DEVICE)).argmax(1).cpu().numpy()
    return balanced_accuracy_score(Y[test_idx], pred)
