"""Experiment 5: local objectives for federated learning on images under label skew.

Eighteen hospitals with Dirichlet(0.3) class proportions, twenty seeds, the same test patients for every seed.
Compared: plain cross-entropy (FedAvg), FedRS (alpha 0.5), FedLC and the class-presence restriction, plus a
centralised model trained on the pooled data with the same backbone and number of passes. Results are logged as
they are produced, so the run resumes after an interruption. Tests over seeds are conditional on the test patients.
"""

import csv
import os

import numpy as np
from scipy import stats

from _common import heading
from fcfl.config import CENTRAL_SEEDS, N_CLIENTS, OBJECTIVE_SEEDS, OUTPUT_DIR, TEST_SPLIT_SEED
from fcfl.federated import (
    centralized_run,
    dirichlet_partition,
    federated_run,
    loss_fedavg,
    loss_fedlc,
    loss_fedrs,
    loss_hard_mask,
)
from fcfl.io import load_slices_cnn
from fcfl.stats import holm

METHODS = [
    ("FedAvg", loss_fedavg),
    ("FedRS (alpha 0.5)", loss_fedrs),
    ("FedLC", loss_fedlc),
    ("Class-presence restriction", loss_hard_mask),
]
LOG = os.path.join(OUTPUT_DIR, "fl_local_objectives.csv")
ALPHA_DIR = 0.3


def read_log():
    if not os.path.exists(LOG):
        return {}
    with open(LOG) as f:
        return {(r["method"], int(r["seed"])): float(r["acc"]) for r in csv.DictReader(f)}


def write_log(method, seed, acc):
    new = not os.path.exists(LOG)
    with open(LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "seed", "acc"])
        if new:
            w.writeheader()
        w.writerow(dict(method=method, seed=seed, acc=round(acc, 4)))


def main():
    X, Y, PID = load_slices_cnn(OUTPUT_DIR)
    heading("LOCAL OBJECTIVES FOR FEDERATED LEARNING ON IMAGES")
    print(f"  {len(X)} slices, {len(np.unique(PID))} patients, classes {np.bincount(Y)}")
    rng = np.random.default_rng(TEST_SPLIT_SEED)
    uniq = np.unique(PID)
    rng.shuffle(uniq)
    test_pids = set(uniq[: int(0.2 * len(uniq))])
    test_idx = np.where(np.isin(PID, list(test_pids)))[0]
    train_mask = ~np.isin(PID, list(test_pids))
    train_pos = np.where(train_mask)[0]
    done = read_log()
    for seed in range(OBJECTIVE_SEEDS):
        if all((n, seed) in done for n, _ in METHODS):
            continue
        cp = dirichlet_partition(Y[train_mask], PID[train_mask], N_CLIENTS, ALPHA_DIR, seed)
        cp = {c: train_pos[v] for c, v in cp.items()}
        for name, fn in METHODS:
            if (name, seed) not in done:
                write_log(name, seed, federated_run(X, Y, cp, test_idx, fn, seed, "small"))
        print(f"  seed {seed}: done")
    for seed in range(CENTRAL_SEEDS):
        if ("Centralised", seed) not in done:
            write_log("Centralised", seed, centralized_run(X, Y, train_pos, test_idx, seed, "small"))

    rows = read_log()
    res = {m: np.array([rows[(m, s)] for s in range(OBJECTIVE_SEEDS) if (m, s) in rows]) for m, _ in METHODS}
    print(f"\n  {'local objective':28s} balanced accuracy")
    for m, _ in METHODS:
        print(f"  {m:28s} {res[m].mean():.3f} +/- {res[m].std():.3f}  (minimum {res[m].min():.3f})")
    central = np.array([rows[("Centralised", s)] for s in range(CENTRAL_SEEDS) if ("Centralised", s) in rows])
    print(f"  {'Centralised (pooled data)':28s} {central.mean():.3f} +/- {central.std():.3f}")
    ref = "Class-presence restriction"
    others = [m for m, _ in METHODS if m != ref]
    pv = [stats.wilcoxon(res[ref], res[m]).pvalue if np.any(res[ref] != res[m]) else 1.0 for m in others]
    print(f"\n  {ref} against each objective (Wilcoxon over seeds, Holm-adjusted)")
    for m, p in zip(others, holm(pv)):
        print(
            f"    vs {m:22s} {100 * (res[ref].mean() - res[m].mean()):+6.2f} pp  p = {p:.4f}  "
            f"higher in {int((res[ref] > res[m]).sum())}/{len(res[ref])} seeds"
        )


if __name__ == "__main__":
    main()
