"""
Data loaders and feature builders for the WDR91 DEL-ML pipeline.

The competition provides molecules as count-based fingerprints stored as
comma-separated strings (one column per fingerprint type). This module parses
those into numeric matrices and provides the feature-construction helpers used
by both the LightGBM ensemble and the TabPFN component.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA


# The seven fingerprints used by the LightGBM ensemble (Set 1 + Set 2 + Set 3)
SET123 = ["ECFP6", "RDK", "ATOMPAIR", "ECFP4", "AVALON", "FCFP4", "TOPTOR"]

# The three fingerprints used for the TabPFN component (from Set 1)
SET1 = ["ECFP6", "RDK", "ATOMPAIR"]


def parse_fp(s):
    """Parse a single comma-separated count-fingerprint string into a float32 vector."""
    return np.array(s.split(","), dtype=np.float32)


def fp_matrix(df, fp):
    """Build a sparse (n_molecules x 2048) matrix for one fingerprint column."""
    return csr_matrix(np.stack(df[fp].map(parse_fp).to_numpy()))


def load_data(train_path, test_path):
    """
    Load the training and test parquet files.

    Returns
    -------
    train : DataFrame  (375,595 rows; includes DELLabel + fingerprint columns)
    test  : DataFrame  (339,258 rows; fingerprint columns, no labels)
    y_train : np.ndarray  binary hit labels for the training set
    """
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)
    y_train = train["DELLabel"].to_numpy().astype(int)
    return train, test, y_train


def test_ids(test_path):
    """Return the RandomID column from the test set, in file order."""
    return pd.read_parquet(test_path, columns=["RandomID"])["RandomID"].values


def build_set1_pca(train, test, y_train, n_components=2000, n_total=100_000,
                   seed=42):
    """
    Build the TabPFN feature matrices for Set 1 (ECFP6 + RDK + ATOMPAIR).

    Concatenates the three fingerprints (6,144 dims), fits PCA to 2,000
    components on the training set, and applies it to both train and test.
    Training rows are an upsampled 100k subsample: all hits plus a random
    sample of non-hits (~29% hit rate) to give TabPFN a balanced context.

    Returns
    -------
    xs : np.ndarray  (n_total x n_components)  upsampled training features
    ys : np.ndarray  (n_total,)                training labels
    X_test : np.ndarray  (n_test x n_components)  test features (file order)
    """
    Xtr = np.hstack([fp_matrix(train, fp).toarray() for fp in SET1])
    Xte = np.hstack([fp_matrix(test, fp).toarray() for fp in SET1])

    pca = PCA(n_components=n_components, random_state=seed).fit(Xtr)
    Xtr_pca = pca.transform(Xtr).astype(np.float32)
    Xte_pca = pca.transform(Xte).astype(np.float32)

    # upsampled subsample: all hits + non-hits to fill to n_total
    rng = np.random.default_rng(seed)
    hit_idx = np.where(y_train == 1)[0]
    non_idx = np.where(y_train == 0)[0]
    n_non = n_total - len(hit_idx)
    sub = np.concatenate([hit_idx, rng.choice(non_idx, n_non, replace=False)])
    rng.shuffle(sub)

    xs = Xtr_pca[sub]
    ys = y_train[sub]
    return xs, ys, Xte_pca
