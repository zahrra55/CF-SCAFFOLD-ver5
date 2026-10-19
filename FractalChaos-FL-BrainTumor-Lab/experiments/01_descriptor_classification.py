"""Experiment 1: three-class classification (meningioma, glioma, pituitary) from the descriptors.

Ten repeats of five patient-grouped folds on the Figshare dataset. Descriptor sets are compared with the
corrected repeated cross-validation t-test. FuzEn2D replaces SampEn2D only if it is defined on at least as many
slices and is non-inferior (lower 95% bound of the difference >= -0.5 points); each additional block is then
accepted only if its gain is significant after Holm correction. Nested selection gives an estimate free of
selection optimism. The selected set is saved for experiments 2, 4 and 6.
"""

import numpy as np
from scipy.stats import spearmanr

from _common import heading, report_test, save_final_set
from fcfl.config import N_FOLDS, N_REP, NESTED_REP
from fcfl.data import build_figshare
from fcfl.descriptors import ALL, BASE19, BLOCK_A, BLOCK_B, FUZ19, NEW, cols, derive_sets
from fcfl.evaluation import cv_per_fold, nested_selection
from fcfl.stats import corrected_t, holm


def main():
    D = build_figshare()
    X, Y, P = D["X"], D["Y"], D["P"]
    heading("THREE-CLASS CLASSIFICATION FROM THE DESCRIPTORS")
    print(
        f"  {len(Y)} slices, {len(set(P))} patients, classes {np.bincount(Y)}; {N_REP} x {N_FOLDS}-fold patient-grouped CV"
    )
    nan = {n: int(np.isnan(X[:, ALL.index(n)]).sum()) for n in NEW}
    print(
        "  undefined values (imputed inside training folds): " + ", ".join(f"{n} {v}" for n, v in nan.items())
    )
    size = X[:, ALL.index("size")]
    print(
        "  Spearman correlation with tumour size: "
        + ", ".join(
            f"{n} {spearmanr(X[:, ALL.index(n)], size, nan_policy='omit')[0]:+.2f}"
            for n in ["Ft", "lacun_t", "FuzEn_t", "GLac2_t", "GLacSlope_t", "Fbound", "mean_grad_in"]
        )
    )

    S = {
        f"{e}{b}": base + blk
        for e, base in (("BASE19", BASE19), ("FUZ19", FUZ19))
        for b, blk in (("", []), ("+A", BLOCK_A), ("+B", BLOCK_B), ("+A+B", BLOCK_A + BLOCK_B))
    }
    folds, pooled = cv_per_fold(X, Y, P, {n: cols(s) for n, s in S.items()})
    print(f"\n  {'set':16s} patient-level balanced accuracy (mean +/- sd over repeats)")
    for n in S:
        print(f"  {n:16s} {pooled[n].mean():.3f} +/- {pooled[n].std():.3f}")

    print("\n  Decisions")
    m, p, ci = corrected_t(folds["FUZ19"] - folds["BASE19"])
    coverage = nan["FuzEn_t"] + nan["FuzEn_b"] <= nan["SampEn_t"] + nan["SampEn_b"]
    use_fuzen = coverage and ci[0] >= -0.005
    report_test("FuzEn2D instead of SampEn2D", m, p, ci)
    print(
        f"    undefined {nan['FuzEn_t'] + nan['FuzEn_b']} vs {nan['SampEn_t'] + nan['SampEn_b']}; "
        f"non-inferiority bound -0.50 pp -> {'adopted' if use_fuzen else 'not adopted'}"
    )
    base = "FUZ19" if use_fuzen else "BASE19"
    tA, tB = corrected_t(folds[base + "+A"] - folds[base]), corrected_t(folds[base + "+B"] - folds[base])
    adj = holm([tA[1], tB[1]])
    accepted = []
    for label, t, p_adj, block in (
        ("boundary fractal dimension", tA, adj[0], BLOCK_A),
        ("grayscale lacunarity", tB, adj[1], BLOCK_B),
    ):
        ok = t[0] > 0 and p_adj < 0.05
        accepted += block if ok else []
        report_test(f"{label} added to {base}", *t)
        print(f"    Holm p = {p_adj:.4f} -> {'accepted' if ok else 'not accepted'}")

    final = (FUZ19 if use_fuzen else BASE19) + accepted
    name = base + ("+A" if BLOCK_A[0] in accepted else "") + ("+B" if BLOCK_B[0] in accepted else "")
    fam, tum, strict = derive_sets(final)
    rest = [f for f in final if f not in fam]
    folds2, pooled2 = cv_per_fold(
        X,
        Y,
        P,
        {
            "family": cols(fam),
            "tumour-only": cols(tum),
            "strict tumour-only": cols(strict),
            "without family": cols(rest),
        },
    )
    print(
        f"\n  Selected set: {name} ({len(final)} descriptors), {pooled[name].mean():.3f} +/- {pooled[name].std():.3f}"
    )
    for label, key in (
        ("fractal-chaos family", "family"),
        ("tumour-only", "tumour-only"),
        ("strict tumour-only", "strict tumour-only"),
        ("without the family", "without family"),
    ):
        print(f"  {label:24s} {pooled2[key].mean():.3f} +/- {pooled2[key].std():.3f}")
    report_test(
        "contribution of the fractal-chaos family", *corrected_t(folds[name] - folds2["without family"])
    )
    report_test(
        "strict tumour-only vs tumour-only",
        *corrected_t(folds2["strict tumour-only"] - folds2["tumour-only"]),
    )

    scores, chosen = nested_selection(X, Y, P, {n: cols(s) for n, s in S.items()})
    same_folds = folds[name][:NESTED_REP].ravel()
    print(
        f"\n  Nested selection ({NESTED_REP} x {N_FOLDS} outer folds): {scores.mean():.3f} +/- {scores.std():.3f} per fold; "
        f"selected set on the same folds {same_folds.mean():.3f}"
    )
    print("  sets chosen: " + ", ".join(f"{n} {chosen.count(n)}" for n in S if chosen.count(n)))
    save_final_set(dict(final=final, fam=fam, tum=tum, strict=strict, final_name=name))


if __name__ == "__main__":
    main()
