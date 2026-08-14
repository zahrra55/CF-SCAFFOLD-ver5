"""
Leakage guard for per-patient adaptation protocols.

Background
----------
A support/query split of one patient's slices was used to evaluate per-patient
personalisation. It produced a gain of 32 balanced-accuracy points with Cohen's
d above 10 and no patient harmed, which is not a plausible outcome for a
personalisation method.

The cause is structural. Each patient carries a single tumor type across all
their slices, so the labels of the support set determine the labels of the query
set. The protocol leaks the answer.

This module demonstrates the leak directly: a predictor that never looks at the
images, and simply copies the majority label of the support set, is scored on the
query set. If that predictor scores near 1.0, the protocol cannot measure image
information and any accuracy obtained under it is uninterpretable.

Run this before adopting any patient-level adaptation protocol on a dataset where
the label is constant within a patient.
"""

import numpy as np
from sklearn.metrics import balanced_accuracy_score


def image_blind_support_query_score(labels, patient_ids, support_fraction=0.5, seed=0):
    """Score an image-blind predictor under a support/query split.

    For each patient, slices are split into a support set and a query set. The
    predictor ignores the images entirely and predicts the majority label of the
    support set for every query slice.

    Parameters
    ----------
    labels : array of int, one label per slice
    patient_ids : array, one patient identifier per slice
    support_fraction : float, fraction of each patient's slices used as support
    seed : int, controls the shuffling of slices within each patient

    Returns
    -------
    float
        Balanced accuracy of the image-blind predictor on the pooled query set.
        A value near 1.0 means the protocol leaks the label and cannot be used.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    patient_ids = np.asarray(patient_ids)

    query_true, query_pred = [], []

    for pid in np.unique(patient_ids):
        idx = np.where(patient_ids == pid)[0]
        rng.shuffle(idx)
        n_support = max(1, int(support_fraction * len(idx)))
        support, query = idx[:n_support], idx[n_support:]
        if len(query) == 0:
            continue
        majority = int(np.bincount(labels[support]).argmax())
        query_true.extend(labels[query])
        query_pred.extend([majority] * len(query))

    if not query_true:
        return float("nan")
    return balanced_accuracy_score(query_true, query_pred)


def assert_protocol_is_measurable(labels, patient_ids, threshold=0.60, seed=0):
    """Raise if an image-blind predictor scores above `threshold`.

    Intended as a guard to place before any per-patient adaptation experiment.
    """
    score = image_blind_support_query_score(labels, patient_ids, seed=seed)
    if score > threshold:
        raise ValueError(
            f"Support/query protocol leaks the label: an image-blind predictor "
            f"scores {score:.3f} balanced accuracy on the query set. Any result "
            f"obtained under this protocol measures leakage, not model skill."
        )
    return score


if __name__ == "__main__":
    # Demonstration on the structure of the Figshare dataset, where every slice
    # from a patient carries that patient's single tumor type.
    rng = np.random.default_rng(0)
    labels, pids = [], []
    for patient in range(46):
        tumor_type = int(rng.integers(0, 3))
        n_slices = int(rng.integers(8, 20))
        labels.extend([tumor_type] * n_slices)
        pids.extend([f"P{patient:03d}"] * n_slices)

    score = image_blind_support_query_score(labels, pids)
    print(f"Image-blind predictor, balanced accuracy on query set: {score:.3f}")
    print()
    if score > 0.95:
        print("The support set determines the query labels. Per-patient")
        print("support/query adaptation cannot be evaluated on this data, and")
        print("any accuracy reported under it would measure the leak.")
