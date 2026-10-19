"""Statistical tests, patient-level aggregation and bootstrap intervals."""

import numpy as np
from scipy import stats
from sklearn.metrics import balanced_accuracy_score

from .config import TEST_TRAIN_RATIO


def corrected_t(diff_folds, ratio=TEST_TRAIN_RATIO):
    """Corrected repeated cross-validation t-test (Nadeau and Bengio, 2003). Returns mean, p and 95% CI."""
    d = np.asarray(diff_folds, float).ravel()
    J = len(d)
    m = d.mean()
    s2 = d.var(ddof=1)
    if s2 == 0:
        return (float(m), 1.0 if m == 0 else 0.0, (float(m), float(m)))
    se = np.sqrt((1.0 / J + ratio) * s2)
    t = m / se
    p = 2 * stats.t.sf(abs(t), J - 1)
    h = stats.t.ppf(0.975, J - 1) * se
    return (float(m), float(p), (float(m - h), float(m + h)))


def holm(pvals):
    """Holm-adjusted p-values."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def patient_vote(pred, proba, pids):
    """Majority vote per patient; ties broken by mean class probability."""
    out = {}
    for pid in np.unique(pids):
        m = pids == pid
        cnt = np.bincount(pred[m], minlength=3)
        win = np.flatnonzero(cnt == cnt.max())
        out[pid] = int(win[0]) if len(win) == 1 else int(win[np.argmax(proba[m][:, win].mean(0))])
    return out


def patient_truth(Y, pids):
    """Label of each patient."""
    return {pid: int(np.bincount(Y[pids == pid]).argmax()) for pid in np.unique(pids)}


def boot_ci(v):
    """95% percentile interval."""
    return np.percentile(v, [2.5, 97.5])


def bal_acc_from(pt, pv, keys):
    return balanced_accuracy_score([pt[k] for k in keys], [pv[k] for k in keys])


def paired_boot(pt, pv_ref, pv_auto, keys, n=2000, seed=0):
    """95% CI of the difference in patient-level balanced accuracy, resampling patients."""
    rng = np.random.default_rng(seed)
    keys = np.array(keys)
    out = []
    for _ in range(n):
        k = rng.choice(keys, len(keys), replace=True)
        y = [pt[x] for x in k]
        if len(set(y)) < 3:
            continue
        out.append(
            balanced_accuracy_score(y, [pv_auto[x] for x in k])
            - balanced_accuracy_score(y, [pv_ref[x] for x in k])
        )
    return np.percentile(out, [2.5, 97.5])
