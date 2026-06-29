"""
Evaluation functions for the WDR91 DEL-ML pipeline.

Model performance is validated by cross-validation on the labelled training
set, reporting AUROC and AUPRC with a confidence interval across folds. Folds
are grouped on building block BB2 so that combinatorial DEL siblings never span
the train/validation boundary (preventing leakage).

AUPRC is the more informative metric here because the data are highly imbalanced
(~7.7% hits); AUROC is reported alongside it for reference.
"""

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score


def cv_metrics_with_ci(model_ctor, X, y, groups, n_splits=5, seed=42):
    """
    Grouped cross-validated AUROC and AUPRC with a 95% confidence interval.

    Parameters
    ----------
    model_ctor : callable
        Zero-argument constructor returning a fresh, unfitted classifier.
    X : array-like                feature matrix
    y : np.ndarray                binary labels
    groups : np.ndarray           grouping variable (building block BB2)
    n_splits : int                number of CV folds
    seed : int                    random seed for fold assignment

    Returns
    -------
    dict with per-fold scores and (mean, lo, hi) summaries for AUROC and AUPRC.
    The confidence interval is the 2.5/97.5 percentile across folds.
    """
    from tqdm import tqdm
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aurocs, auprcs = [], []
    for tr, va in tqdm(sgkf.split(X, y, groups), total=n_splits, desc="CV folds"):
        m = model_ctor()
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[va])[:, 1]
        aurocs.append(roc_auc_score(y[va], p))
        auprcs.append(average_precision_score(y[va], p))

    aurocs, auprcs = np.array(aurocs), np.array(auprcs)

    def summarize(scores):
        lo, hi = np.percentile(scores, [2.5, 97.5])
        return float(scores.mean()), float(lo), float(hi)

    return {
        "auroc_folds": aurocs.tolist(),
        "auprc_folds": auprcs.tolist(),
        "auroc": summarize(aurocs),   # (mean, lo, hi)
        "auprc": summarize(auprcs),
    }


def print_cv_report(metrics):
    """Print the cross-validated metrics with confidence intervals."""
    am, alo, ahi = metrics["auroc"]
    pm, plo, phi = metrics["auprc"]
    print(f"  AUROC: {am:.3f}  (95% CI {alo:.3f}-{ahi:.3f})  "
          f"folds={np.round(metrics['auroc_folds'], 3).tolist()}")
    print(f"  AUPRC: {pm:.3f}  (95% CI {plo:.3f}-{phi:.3f})  "
          f"folds={np.round(metrics['auprc_folds'], 3).tolist()}")
