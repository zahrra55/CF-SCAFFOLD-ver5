"""Experiment 4: damage caused by heterogeneity to federated learning, CNN on images versus descriptors.

The same recipe is used for centralised and federated training (class-balanced sampling, gradient clipping,
plain cross-entropy, sample-weighted federated averaging), with the same hospitals and the same test patients
for both representations. Damage is centralised minus federated balanced accuracy. The interval of the damage
gap comes from a bootstrap over test patients. Predictions are saved, so the run resumes after an interruption.
"""

import os

import numpy as np
from scipy import stats
from sklearn.metrics import balanced_accuracy_score

from _common import heading, load_final_set
from fcfl.config import ALPHAS, FED_SEEDS, N_BOOT, N_CLIENTS, OUTPUT_DIR, PRIMARY_ALPHA, TEST_SPLIT_SEED
from fcfl.data import build_figshare
from fcfl.descriptors import BASE19, cols
from fcfl.federated import dirichlet_partition, train_and_predict
from fcfl.stats import boot_ci, patient_truth, patient_vote


def main():
    D, info = build_figshare(), load_final_set()
    X, Y, P = D["X"], D["Y"], D["P"]
    heading("HETEROGENEITY DAMAGE: CNN ON IMAGES VERSUS DESCRIPTORS")
    rng = np.random.default_rng(TEST_SPLIT_SEED)
    uniq = np.unique(P)
    rng.shuffle(uniq)
    test_p = set(uniq[: int(0.2 * len(uniq))])
    te = np.where(np.isin(P, list(test_p)))[0]
    trm = ~np.isin(P, list(test_p))
    tr = np.where(trm)[0]

    def standardised(names):
        Xs = X[:, cols(names)]
        fill = np.nansum(Xs[tr], 0) / np.maximum((~np.isnan(Xs[tr])).sum(0), 1)
        Xs = np.where(np.isnan(Xs), fill, Xs)
        mu, sd = Xs[tr].mean(0), Xs[tr].std(0) + 1e-12
        return (Xs - mu) / sd

    reps = {
        "cnn": ("cnn", (D["IMG"].astype(np.float32) / 255.0)[:, None]),
        "desc19": ("mlp", standardised(BASE19)),
    }
    if info["final"] != BASE19:
        reps["descselected"] = ("mlp", standardised(info["final"]))
    run_dir = os.path.join(OUTPUT_DIR, "heterogeneity_runs")
    os.makedirs(run_dir, exist_ok=True)
    parts, dropped = {}, []
    for a in ALPHAS:
        for s in range(FED_SEEDS):
            cl = dirichlet_partition(Y[trm], P[trm], N_CLIENTS, a, s)
            parts[(a, s)] = {c: tr[v] for c, v in cl.items()}
            dropped.append(len(tr) - sum(len(v) for v in cl.values()))
    print(
        f"  test patients {len(test_p)} ({len(te)} slices); training slices dropped by the partition "
        f"(hospitals with fewer than 4 slices): mean {np.mean(dropped):.1f}, at most {max(dropped)} of {len(tr)}"
    )
    settings = ["central"] + [f"fed{a}" for a in ALPHAS]
    for rep, (kind, Xr) in reps.items():
        for st in settings:
            for s in range(FED_SEEDS):
                f = os.path.join(run_dir, f"{rep}__{st}__{s}.npy")
                if not os.path.exists(f):
                    ts = {0: tr} if st == "central" else parts[(float(st[3:]), s)]
                    np.save(f, train_and_predict(kind, Xr, Y, ts, te, s))
        print(f"  {rep}: all runs done")

    yt, pids = Y[te], P[te]
    up = np.unique(pids)
    pidx = {q: np.where(pids == q)[0] for q in up}
    prob = {
        (r, st, s): np.load(os.path.join(run_dir, f"{r}__{st}__{s}.npy"))
        for r in reps
        for st in settings
        for s in range(FED_SEEDS)
    }
    acc = {k: balanced_accuracy_score(yt, v.argmax(1)) for k, v in prob.items()}
    pt = patient_truth(yt, pids)

    def patient_acc(k):
        pv = patient_vote(prob[k].argmax(1), prob[k], pids)
        return balanced_accuracy_score([pt[q] for q in sorted(pv)], [pv[q] for q in sorted(pv)])

    print(
        f"\n  {'representation':14s} {'centralised':>12s}"
        + "".join(f"{'fed a=' + str(a):>12s}{'damage':>9s}" for a in ALPHAS)
    )
    for rep in reps:
        c = np.mean([acc[(rep, "central", s)] for s in range(FED_SEEDS)])
        line = f"  {rep:14s} {c:12.3f}"
        for a in ALPHAS:
            fm = np.mean([acc[(rep, f"fed{a}", s)] for s in range(FED_SEEDS)])
            line += f"{fm:12.3f}{100 * (c - fm):+8.1f}"
        print(line)
    print(
        f"\n  patient-level federated accuracy at alpha {PRIMARY_ALPHA}: "
        + "; ".join(
            f"{rep} {np.mean([patient_acc((rep, f'fed{PRIMARY_ALPHA}', s)) for s in range(FED_SEEDS)]):.3f}"
            for rep in reps
        )
    )

    T = np.array([[np.sum(yt[pidx[q]] == c) for c in range(3)] for q in up], float)
    Cc = {
        k: np.array(
            [[np.sum((v.argmax(1)[pidx[q]] == c) & (yt[pidx[q]] == c)) for c in range(3)] for q in up], float
        )
        for k, v in prob.items()
    }
    rb = np.random.default_rng(1)
    W = np.stack(
        [np.bincount(rb.integers(0, len(up), len(up)), minlength=len(up)) for _ in range(N_BOOT)]
    ).astype(float)
    tot = W @ T
    valid = tot > 0

    def bacc_boot(k):
        rec = np.where(valid, (W @ Cc[k]) / np.where(valid, tot, 1), np.nan)
        return np.nanmean(rec, axis=1)

    def mean_over_seeds(rep, st):
        return np.mean([bacc_boot((rep, st, s)) for s in range(FED_SEEDS)], 0)

    print("\n  Damage gap (CNN damage minus descriptor damage), 95% bootstrap interval over test patients")
    for drep in [r for r in reps if r != "cnn"]:
        for a in ALPHAS:
            gap = (
                mean_over_seeds("cnn", "central")
                - mean_over_seeds("cnn", f"fed{a}")
                - mean_over_seeds(drep, "central")
                + mean_over_seeds(drep, f"fed{a}")
            )
            seeds = np.array(
                [
                    acc[("cnn", "central", s)]
                    - acc[("cnn", f"fed{a}", s)]
                    - acc[(drep, "central", s)]
                    + acc[(drep, f"fed{a}", s)]
                    for s in range(FED_SEEDS)
                ]
            )
            lo, hi = boot_ci(gap)
            pw = stats.wilcoxon(seeds).pvalue if np.any(seeds != 0) else 1.0
            print(
                f"    {drep:12s} alpha {a:<4}: {100 * seeds.mean():+6.1f} pp  [{100 * lo:+.1f}, {100 * hi:+.1f}]"
                f"  (Wilcoxon over seeds p = {pw:.4f}, conditional on the test patients)"
            )
    print("  Descriptors use the radiologist's tumour mask; the CNN sees the image only.")


if __name__ == "__main__":
    main()
