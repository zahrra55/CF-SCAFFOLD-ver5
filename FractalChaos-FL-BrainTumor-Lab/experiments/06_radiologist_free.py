"""Experiment 6: cost of replacing the radiologist's tumour mask by the U-Net mask, three-class task.

The same U-Net masks as experiment 2 (one network per patient-grouped fold, trained on that fold's training
patients only). Each classifier is trained on descriptors from radiologist masks and applied to the test patients
twice, with radiologist-mask and with U-Net-mask descriptors. The interval of the cost comes from a paired
bootstrap over patients.
"""

import numpy as np
from sklearn.model_selection import GroupKFold

from _common import heading, load_final_set
from fcfl.config import N_FOLDS
from fcfl.data import build_figshare, unet_features
from fcfl.descriptors import ALL, FEATURE_ORDER
from fcfl.evaluation import make_search
from fcfl.stats import bal_acc_from, paired_boot, patient_truth, patient_vote

TUMOUR = [
    "Ft",
    "GLCM_t",
    "correlation_t",
    "Dispersion_t",
    "lacun_t",
    "size",
    "solidity",
    "eccentricity",
    "extent",
    "circularity",
    "edge_density",
    "mean_grad",
    "border_grad",
]
SETS = {"full model (17)": FEATURE_ORDER, "tumour region only (13)": TUMOUR}


def main():
    D, info = build_figshare(), load_final_set()
    X, Y, P = D["X"], D["Y"], D["P"]
    heading("COST OF REMOVING THE RADIOLOGIST, THREE CLASSES")
    XA, okA, dice_all = unet_features(D, info["final"])
    print(
        f"  U-Net Dice over all {len(Y)} slices: mean {dice_all.mean():.3f}, median {np.median(dice_all):.3f}; "
        f"automatic descriptors on {int(okA.sum())} slices"
    )
    preds = {name: {"ref": {}, "auto": {}} for name in SETS}
    for tr, te in GroupKFold(N_FOLDS).split(X, Y, P):
        te_ok = te[okA[te]]
        for name, fs in SETS.items():
            idx = [ALL.index(n) for n in fs]
            clf = make_search(X[tr][:, idx], Y[tr], P[tr]).best_estimator_
            preds[name]["ref"].update(
                patient_vote(clf.predict(X[te_ok][:, idx]), clf.predict_proba(X[te_ok][:, idx]), P[te_ok])
            )
            preds[name]["auto"].update(
                patient_vote(clf.predict(XA[te_ok][:, idx]), clf.predict_proba(XA[te_ok][:, idx]), P[te_ok])
            )
    pt = patient_truth(Y, P)
    print(
        f"\n  {'descriptor set':26s} {'radiologist':>12s} {'U-Net':>8s} {'cost':>9s}   95% CI (paired, patients)"
    )
    for name in SETS:
        keys = sorted(set(preds[name]["ref"]) & set(preds[name]["auto"]))
        r, a = bal_acc_from(pt, preds[name]["ref"], keys), bal_acc_from(pt, preds[name]["auto"], keys)
        lo, hi = paired_boot(pt, preds[name]["ref"], preds[name]["auto"], keys)
        print(
            f"  {name:26s} {r:12.3f} {a:8.3f} {100 * (a - r):+8.1f} pp  [{100 * lo:+.1f}, {100 * hi:+.1f}]  ({len(keys)} patients)"
        )
    ok = np.where(okA)[0]
    print("\n  Agreement between radiologist-mask and U-Net-mask descriptors (Pearson r)")
    for n in TUMOUR + ["Fb"]:
        a, b = X[ok, ALL.index(n)], XA[ok, ALL.index(n)]
        r = np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else float("nan")
        print(f"    {n:14s} r = {r:.3f}")


if __name__ == "__main__":
    main()
