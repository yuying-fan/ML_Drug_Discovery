"""
train_model.py — reproduce the winning WDR91 DEL-ML submission.

The final model is a rank-weighted fusion of two complementary predictors:

    final_score = 0.8 * rank(LightGBM ensemble) + 0.2 * rank(TabPFN)

  * LightGBM ensemble: seven per-fingerprint LightGBM models (ECFP6, RDK,
    ATOMPAIR, ECFP4, AVALON, FCFP4, TOPTOR), each tuned by grouped
    cross-validation, their predicted probabilities averaged.
  * TabPFN: a tabular foundation model trained in-context on Set 1
    (ECFP6 + RDK + ATOMPAIR, PCA-compressed to 2,000 dims), using an
    upsampled 100k subsample.

Running this script:
  1. loads the data,
  2. reports local cross-validated AUROC/AUPRC with confidence intervals,
  3. trains the final models on all data,
  4. saves the LightGBM ensemble + fusion config to models/best_model.pkl,
  5. writes submission.csv.

NOTE ON HARDWARE: the TabPFN component runs far faster on a GPU. The script
auto-detects CUDA and uses it if available, otherwise falls back to CPU
(functional but very slow on the full test set).

NOTE ON REPRODUCIBILITY: TabPFN is not bit-for-bit deterministic across runs.
To reproduce the exact submitted result, pass the included original TabPFN 
predictions via --tabpfn-file new_tabpfn_set1.npy. The LightGBM ensemble 
is deterministic (same predictions every run).
"""

import argparse
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from lightgbm import LGBMClassifier

from src.dataset import load_data, test_ids, fp_matrix, build_set1_pca, SET123
from src.eval import ensemble_cv_metrics_with_ci, print_cv_report

warnings.filterwarnings("ignore", message="X does not have valid feature names")

# ---- paths ----
TRAIN_PATH = "data/crosstalk_train.parquet"
TEST_PATH = "data/crosstalk_test_inputs.parquet"
LGB_CONFIG_PATH = "lgb_configs_settings.json"            # tuned hyperparameters

# ---- fusion weight (chosen on the robust 0.70-0.80 plateau; 0.80 = winning submission) ----
FUSION_W = 0.80


def lgb_ctor(cfg):
    """Construct a LightGBM classifier from a tuned per-fingerprint config."""
    return LGBMClassifier(**cfg, is_unbalance=True, n_jobs=4,
                          random_state=42, verbose=-1)


def lightgbm_ensemble_predict(train, test, y_train, configs):
    """Train one tuned LightGBM per fingerprint and average their probabilities."""
    from tqdm import tqdm
    per_fp = []
    for fp in tqdm(SET123, desc="LightGBM ensemble (7 fingerprints)"):
        tqdm.write(f"  training {fp}...")
        model = lgb_ctor(configs[fp])
        model.fit(fp_matrix(train, fp), y_train)
        per_fp.append(model.predict_proba(fp_matrix(test, fp))[:, 1])
    return np.mean(per_fp, axis=0), per_fp


def tabpfn_predict(train, test, y_train):
    """Train TabPFN on the Set 1 / upsampled-100k features and predict the test set."""
    import os
    from tabpfn import TabPFNClassifier
    import torch

    # TabPFN requires an API token; read it from the environment
    if "TABPFN_TOKEN" not in os.environ:
        raise RuntimeError(
            "TabPFN requires an API token. Set it before running:\n"
            "    export TABPFN_TOKEN=your_token_here\n"
            "Obtain a token from the TabPFN provider, or run with --skip-tabpfn "
            "to produce the LightGBM-only baseline.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  TabPFN device: {device}"
          + ("" if device == "cuda" else "  (CPU — this will be slow)"))

    xs, ys, X_test = build_set1_pca(train, test, y_train)
    clf = TabPFNClassifier(device=device, ignore_pretraining_limits=True)
    clf.fit(xs, ys)

    from tqdm import tqdm
    preds = np.concatenate([
        clf.predict_proba(X_test[i:i + 5000])[:, 1]
        for i in tqdm(range(0, len(X_test), 5000), desc="TabPFN predicting")
    ])
    return preds


def fuse(lgb_scores, tabpfn_scores, w=FUSION_W):
    """Rank-weighted fusion: w * rank(LightGBM) + (1 - w) * rank(TabPFN)."""
    lr = rankdata(lgb_scores) / len(lgb_scores)
    tr = rankdata(tabpfn_scores) / len(tabpfn_scores)
    return w * lr + (1 - w) * tr


def main(skip_tabpfn=False, skip_validation=False, tabpfn_file=None):
    print("Loading data...")
    train, test, y_train = load_data(TRAIN_PATH, TEST_PATH)
    ids = test_ids(TEST_PATH)
    configs = json.load(open(LGB_CONFIG_PATH))
    print(f"  train: {train.shape} | test: {test.shape} | "
          f"hit rate: {y_train.mean():.4f}")

    # ---- 1. local validation: grouped CV AUROC/AUPRC with confidence intervals ----
    #    Validates the ACTUAL final model (set1+2+3 ensemble) via grouped CV on a
    #    60k subsample (grouped on BB2 so DEL siblings never span the train/val
    #    split). Uses only the labelled training set.
    if not skip_validation:
        print("\nValidating LightGBM ensemble (set1+2+3, grouped 5-fold CV)...")
        from sklearn.model_selection import train_test_split
        # stratified 60k subsample
        sub_idx, _ = train_test_split(np.arange(len(y_train)), train_size=60000,
                                      stratify=y_train, random_state=42)
        groups = train["DEL_ID"].str.split("-").str[2].to_numpy()[sub_idx]   # BB2
        y_sub = y_train[sub_idx]
        metrics = ensemble_cv_metrics_with_ci(
            lgb_ctor, configs, SET123, fp_matrix, train, sub_idx, y_sub, groups)
        print_cv_report(metrics)

    # ---- 2. LightGBM ensemble (train on all data) ----
    print("\nTraining LightGBM ensemble (7 fingerprints)...")
    lgb_scores, _ = lightgbm_ensemble_predict(train, test, y_train, configs)

    # ---- 3. TabPFN ----
    if skip_tabpfn:
        print("\nSkipping TabPFN (--skip-tabpfn); submitting LightGBM only.")
        final_scores = lgb_scores
    else:
        if tabpfn_file:
            print(f"\nLoading saved TabPFN predictions from {tabpfn_file}")
            tabpfn_scores = np.load(tabpfn_file)
        else:
            print("\nTraining TabPFN component (Set 1, upsampled 100k)...")
            tabpfn_scores = tabpfn_predict(train, test, y_train)
        print(f"\nFusing: {FUSION_W} * LightGBM + {1 - FUSION_W:.1f} * TabPFN")
        final_scores = fuse(lgb_scores, tabpfn_scores)

    # ---- 4. save model + fusion config ----
    model_bundle = {
        "lgb_configs": configs,
        "fingerprints": SET123,
        "fusion_weight": FUSION_W,
        "description": "0.8 * LightGBM(7-fingerprint ensemble) + 0.2 * TabPFN(Set1)",
    }
    joblib.dump(model_bundle, "models/best_model.pkl")
    print("\nSaved model bundle to models/best_model.pkl")

    # ---- 5. write submission ----
    pd.DataFrame({"RandomID": ids, "DELLabel": final_scores}) \
        .to_csv("submission.csv", index=False)
    print("Wrote submission.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-tabpfn", action="store_true",
                        help="train LightGBM only (skip the GPU TabPFN step)")
    parser.add_argument("--skip-validation", action="store_true",
                        help="skip the cross-validation step (faster)")
    parser.add_argument("--tabpfn-file", default=None,
                        help="path to saved TabPFN predictions (.npy); if given, "
                             "these are used instead of running TabPFN fresh ")
    args = parser.parse_args()
    main(skip_tabpfn=args.skip_tabpfn, skip_validation=args.skip_validation,
         tabpfn_file=args.tabpfn_file)