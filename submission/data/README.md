# Dataset setup

Place the following files in this `data/` directory before running
`train_model.py`:

| File | Description |
|------|-------------|
| `crosstalk_train.parquet` | DEL training set (375,595 molecules: 28,778 hits + 346,817 non-hits) as count fingerprints, MW, and ALOGP. |
| `crosstalk_test_inputs.parquet` | ASMS test set (339,258 molecules) as fingerprints, MW, AlogP — no labels. Predictions are generated for these. |

These files were provided by the challenge organizers via the AIRCHECK
platform. Download them from the source you were given for the competition and
copy them here.

Validation is performed by cross-validation on the labelled training set (see
`src/eval.py`), so no labelled test file is required.

## Fingerprint columns

Each fingerprint is a comma-separated string of 2,048 integer counts. The
pipeline uses seven of them:
`ECFP6, RDK, ATOMPAIR, ECFP4, AVALON, FCFP4, TOPTOR`.
