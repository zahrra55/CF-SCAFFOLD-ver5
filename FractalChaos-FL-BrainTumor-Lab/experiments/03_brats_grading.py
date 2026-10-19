"""Experiment 3: high-grade versus low-grade glioma on BraTS 2020.

A question of grade within one malignant tumour type, not of benign versus malignant. The tumour is the whole
tumour including oedema, and slice selection uses the expert segmentation. The decision threshold is chosen by
inner cross-validation on the training patients of each fold. Brain-mask geometry is the shortcut control.
"""

import numpy as np

from _common import heading, load_final_set, report_test
from fcfl.data import build_brats
from fcfl.descriptors import ALL, BASE19, FLAGS, SUB, cols
from fcfl.evaluation import cv_threshold_per_fold
from fcfl.stats import corrected_t, patient_truth


def main():
    final = load_final_set()["final"]
    B = build_brats()
    X, S, G, Y, P = B["X"], B["S"], B["G"], B["Y"], B["P"]
    heading("BraTS: HIGH-GRADE VERSUS LOW-GRADE GLIOMA")
    pt = patient_truth(Y, P)
    print(
        f"  {len(Y)} slices, {len(pt)} patients: {sum(v == 1 for v in pt.values())} HGG, {sum(v == 0 for v in pt.values())} LGG; "
        f"enhancing tumour on {int(S[:, 4].sum())} slices, NCR/NET on {int(S[:, 5].sum())}"
    )
    XS, pool = np.hstack([X, S]), ALL + SUB
    sets = {
        "19 descriptors": cols(BASE19, pool),
        "19 + sub-region block": cols(BASE19 + SUB, pool),
        "19 + presence flags": cols(BASE19 + FLAGS, pool),
        "presence flags alone": cols(FLAGS, pool),
        "selected set": cols(final, pool),
        "selected set + sub-regions": cols(final + SUB, pool),
    }
    fo = cv_threshold_per_fold(XS, Y, P, sets)
    fg = cv_threshold_per_fold(G, Y, P, {"geometry": list(range(G.shape[1]))})
    for n in sets:
        print(f"  {n:28s} balanced accuracy {fo[n].mean():.3f} +/- {fo[n].std():.3f}")
    print(
        f"  {'brain geometry (control)':28s} balanced accuracy {fg['geometry'].mean():.3f} +/- {fg['geometry'].std():.3f}"
    )
    report_test(
        "sub-region block added to the 19 descriptors",
        *corrected_t(fo["19 + sub-region block"] - fo["19 descriptors"]),
    )
    report_test(
        "sub-region measures vs presence flags",
        *corrected_t(fo["19 + sub-region block"] - fo["19 + presence flags"]),
    )


if __name__ == "__main__":
    main()
