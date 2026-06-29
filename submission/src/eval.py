"""
Evaluation functions for the WDR91 DEL-ML pipeline.

Model performance is validated by cross-validation on the labelled *training*
set, reporting AUROC and AUPRC with a confidence interval across folds. Folds
are grouped on building block BB2 so that combinatorial DEL siblings never span
the train/validation boundary (preventing optimistic leakage).

AUPRC is the more informative metric here because the data are highly imbalanced
(~7.7% hits); AUROC is reported alongside it for reference.
"""

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score


def ensemble_cv_metrics_with_ci(ctor, configs, fps, fp_matrix, train, sub_idx,
                                y, groups, n_splits=5, seed=42):
    """
    Grouped CV AUROC/AUPRC for the fingerprint-AVERAGED ensemble, with 95% CI.

    Validates the actual final model: it trains one classifier per fingerprint
    on each fold, averages their held-out predictions, and scores that average.
    The CI is the 2.5/97.5 percentile across folds.
    """
    from tqdm import tqdm
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    # precompute each fingerprint's matrix once
    Xs = {fp: fp_matrix(train.iloc[sub_idx], fp) for fp in fps}
    aurocs, auprcs = [], []
    for tr, va in tqdm(sgkf.split(Xs[fps[0]], y, groups), total=n_splits,
                       desc="ensemble CV folds"):
        fold_preds = []
        for fp in fps:
            m = ctor(configs[fp]); m.fit(Xs[fp][tr], y[tr])
            fold_preds.append(m.predict_proba(Xs[fp][va])[:, 1])
        p = np.mean(fold_preds, axis=0)
        aurocs.append(roc_auc_score(y[va], p))
        auprcs.append(average_precision_score(y[va], p))

    aurocs, auprcs = np.array(aurocs), np.array(auprcs)

    def summarize(s):
        lo, hi = np.percentile(s, [2.5, 97.5])
        return float(s.mean()), float(lo), float(hi)

    return {
        "auroc_folds": aurocs.tolist(),
        "auprc_folds": auprcs.tolist(),
        "auroc": summarize(aurocs),
        "auprc": summarize(auprcs),
    }


def print_cv_report(metrics):
    """Pretty-print the cross-validated metrics with confidence intervals."""
    am, alo, ahi = metrics["auroc"]
    pm, plo, phi = metrics["auprc"]
    print(f"  AUROC: {am:.3f}  (95% CI {alo:.3f}-{ahi:.3f})  "
          f"folds={np.round(metrics['auroc_folds'], 3).tolist()}")
    print(f"  AUPRC: {pm:.3f}  (95% CI {plo:.3f}-{phi:.3f})  "
          f"folds={np.round(metrics['auprc_folds'], 3).tolist()}")
