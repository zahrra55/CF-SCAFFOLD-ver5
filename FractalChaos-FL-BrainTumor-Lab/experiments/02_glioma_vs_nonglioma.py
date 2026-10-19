"""Experiment 2: glioma versus non-glioma (meningioma and pituitary) after automatic segmentation.

A proxy for malignant versus predominantly benign tumours. One U-Net per patient-grouped fold, trained on that
fold's training patients only, localises the tumour; the classifier is trained on descriptors from radiologist
masks and applied to descriptors from the automatic masks. A patient whose slices all fail localisation counts as
an error. Head-mask geometry alone is the shortcut control. All arms use the same split; intervals come from a
bootstrap over patients.
"""

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

from _common import heading, load_final_set
from fcfl.config import N_BOOT, N_FOLDS, N_REP
from fcfl.data import build_figshare, unet_features
from fcfl.descriptors import cols
from fcfl.evaluation import cv_per_fold, rf
from fcfl.stats import boot_ci, patient_truth, patient_vote


def main():
    D, info = build_figshare(), load_final_set()
    X, Y, P, G = D["X"], D["Y"], D["P"], D["G"]
    yb = (Y == 1).astype(int)
    final = info["final"]
    heading("GLIOMA VERSUS NON-GLIOMA")
    print(
        f"  {int(yb.sum())} glioma slices, {int((1 - yb).sum())} meningioma and pituitary slices; descriptor set {info['final_name']}"
    )
    sets = {
        "selected descriptors": cols(final),
        "fractal-chaos family": cols(info["fam"]),
        "strict tumour-only": cols(info["strict"]),
    }
    _, rep = cv_per_fold(X, yb, P, sets)
    _, rep_g = cv_per_fold(G, yb, P, {"geometry": list(range(G.shape[1]))})
    print(
        f"  radiologist masks, {N_REP} x {N_FOLDS}-fold: "
        + "; ".join(f"{n} {rep[n].mean():.3f}" for n in sets)
        + f"; geometry control {rep_g['geometry'].mean():.3f}"
    )

    XA, okA, dice_all = unet_features(D, final)
    print(
        f"  U-Net Dice over all {len(Y)} slices (failures counted as 0): mean {dice_all.mean():.3f}, "
        f"median {np.median(dice_all):.3f}, above 0.7 on {100 * (dice_all > 0.7).mean():.1f}%; "
        f"automatic descriptors on {int(okA.sum())} slices"
    )

    pt = patient_truth(yb, P)
    arms = {
        a: {}
        for a in (
            "radiologist",
            "U-Net (intention-to-treat)",
            "geometry control",
            "fractal-chaos family",
            "strict tumour-only",
        )
    }
    failed = 0
    for tr, te in GroupKFold(N_FOLDS).split(X, Y, P):
        for arm, Xm, c in (
            ("radiologist", X, cols(final)),
            ("geometry control", G, list(range(G.shape[1]))),
            ("fractal-chaos family", X, cols(info["fam"])),
            ("strict tumour-only", X, cols(info["strict"])),
        ):
            m = rf().fit(Xm[tr][:, c], yb[tr])
            arms[arm].update(patient_vote(m.predict(Xm[te][:, c]), m.predict_proba(Xm[te][:, c]), P[te]))
        m = rf().fit(X[tr][:, cols(final)], yb[tr])
        ok = te[okA[te]]
        pv = (
            patient_vote(m.predict(XA[ok][:, cols(final)]), m.predict_proba(XA[ok][:, cols(final)]), P[ok])
            if len(ok)
            else {}
        )
        for q in set(P[te]):
            if q not in pv:
                pv[q] = 1 - pt[q]
                failed += 1
        arms["U-Net (intention-to-treat)"].update(pv)

    keys = sorted(pt)
    yt = np.array([pt[q] for q in keys])
    pred = {a: np.array([arms[a][q] for q in keys]) for a in arms}
    point = {a: balanced_accuracy_score(yt, pred[a]) for a in arms}
    rng = np.random.default_rng(0)
    boot, diff = {a: [] for a in arms}, []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(keys), len(keys))
        if len(set(yt[i])) < 2:
            continue
        for a in arms:
            boot[a].append(balanced_accuracy_score(yt[i], pred[a][i]))
        diff.append(boot["U-Net (intention-to-treat)"][-1] - boot["geometry control"][-1])
    print(
        f"\n  Same split for every arm ({len(keys)} patients; {failed} with no usable automatic slice, counted as errors)"
    )
    for a in arms:
        lo, hi = boot_ci(boot[a])
        print(f"    {a:30s} {point[a]:.3f}  [{lo:.3f}, {hi:.3f}]")
    lo, hi = boot_ci(diff)
    print(
        f"    U-Net minus geometry control: {point['U-Net (intention-to-treat)'] - point['geometry control']:+.3f}  [{lo:+.3f}, {hi:+.3f}]"
    )


if __name__ == "__main__":
    main()
