"""Patient-grouped cross-validation, threshold selection and nested selection."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import N_FOLDS, N_REP, NESTED_REP
from .stats import patient_truth, patient_vote


def rf(balanced=False):
    """Imputation, scaling and a 300-tree random forest."""
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            (
                "m",
                RandomForestClassifier(
                    n_estimators=300, random_state=0, n_jobs=-1, class_weight="balanced" if balanced else None
                ),
            ),
        ]
    )


def fold_bacc(Yte, Pte, pred, proba):
    """Patient-level balanced accuracy of one fold."""
    pv, pt = patient_vote(pred, proba, Pte), patient_truth(Yte, Pte)
    k = sorted(pv)
    return balanced_accuracy_score([pt[p] for p in k], [pv[p] for p in k])


def cv_per_fold(X, Y, P, sets, n_rep=N_REP, balanced=False):
    """Per-fold patient-level balanced accuracy for several descriptor sets on the same splits."""
    folds = {n: np.zeros((n_rep, N_FOLDS)) for n in sets}
    pooled = {n: [] for n in sets}
    for r in range(n_rep):
        allp = {n: (np.zeros_like(Y), np.zeros((len(Y), len(np.unique(Y))))) for n in sets}
        for k, (tr, te) in enumerate(
            StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=r).split(X, Y, P)
        ):
            for n, c in sets.items():
                m = rf(balanced).fit(X[tr][:, c], Y[tr])
                pr = m.predict(X[te][:, c])
                pb = m.predict_proba(X[te][:, c])
                folds[n][r, k] = fold_bacc(Y[te], P[te], pr, pb)
                allp[n][0][te] = pr
                allp[n][1][te] = pb
        for n in sets:
            pooled[n].append(fold_bacc(Y, P, allp[n][0], allp[n][1]))
    return (folds, {n: np.array(v) for n, v in pooled.items()})


def cv_threshold_per_fold(X, Y, P, sets, n_rep=N_REP):
    """Binary task; decision threshold chosen by inner cross-validation on training patients."""
    folds = {n: np.zeros((n_rep, N_FOLDS)) for n in sets}
    for r in range(n_rep):
        for k, (tr, te) in enumerate(
            StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=r).split(X, Y, P)
        ):
            pt = patient_truth(Y, P)
            for n, c in sets.items():
                oof = np.zeros(len(tr))
                for itr, ite in StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=100 + r).split(
                    X[tr], Y[tr], P[tr]
                ):
                    oof[ite] = (
                        rf(True).fit(X[tr][itr][:, c], Y[tr][itr]).predict_proba(X[tr][ite][:, c])[:, 1]
                    )
                ktr = sorted(set(P[tr]))
                ptr = np.array([oof[P[tr] == q].mean() for q in ktr])
                ytr = np.array([pt[q] for q in ktr])
                grid = np.linspace(0.05, 0.95, 91)
                thr = grid[np.argmax([balanced_accuracy_score(ytr, (ptr >= t).astype(int)) for t in grid])]
                m = rf(True).fit(X[tr][:, c], Y[tr])
                pr = m.predict_proba(X[te][:, c])[:, 1]
                kte = sorted(set(P[te]))
                folds[n][r, k] = balanced_accuracy_score(
                    [pt[q] for q in kte], [int(pr[P[te] == q].mean() >= thr) for q in kte]
                )
    return folds


def nested_selection(X, Y, P, candidates, n_rep=NESTED_REP):
    """Choose the best candidate set inside each outer training fold and score it on the outer test fold."""
    scores, chosen = [], []
    for r in range(n_rep):
        for tr, te in StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=r).split(X, Y, P):
            inner = {n: [] for n in candidates}
            for itr, ite in StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=1000 + r).split(
                X[tr], Y[tr], P[tr]
            ):
                for n, c in candidates.items():
                    m = rf().fit(X[tr][itr][:, c], Y[tr][itr])
                    inner[n].append(
                        fold_bacc(
                            Y[tr][ite],
                            P[tr][ite],
                            m.predict(X[tr][ite][:, c]),
                            m.predict_proba(X[tr][ite][:, c]),
                        )
                    )
            best = max(candidates, key=lambda n: np.mean(inner[n]))
            c = candidates[best]
            m = rf().fit(X[tr][:, c], Y[tr])
            scores.append(fold_bacc(Y[te], P[te], m.predict(X[te][:, c]), m.predict_proba(X[te][:, c])))
            chosen.append(best)
    return (np.array(scores), chosen)


def make_search(X_tr, Y_tr, P_tr):
    """Random forest with a small grid search over depth and leaf size, grouped by patient."""
    pipe = Pipeline(
        [
            ("sc", StandardScaler()),
            ("rf", RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)),
        ]
    )
    grid = {"rf__max_depth": [None, 12], "rf__min_samples_leaf": [1, 2]}
    gs = GridSearchCV(
        pipe, grid, cv=list(GroupKFold(3).split(X_tr, Y_tr, P_tr)), scoring="balanced_accuracy", n_jobs=-1
    )
    return gs.fit(X_tr, Y_tr)
