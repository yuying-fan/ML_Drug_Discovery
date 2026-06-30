# WDR91 DEL-ML — Team "DELulu is the Solulu" final submission

Final model for the WDR91 DEL-ML Kaggle challenge. 
Best submission: `combo_tl_80_weighted` (Hits@200 = 11 on the public leaderboard)

## Model writeup

**Fingerprints:** The model uses count-based fingerprints provided by the
organizers.  
The LightGBM ensemble draws on seven fingerprints
(ECFP6, RDK, ATOMPAIR, ECFP4, AVALON, FCFP4, TOPTOR). A per-fingerprint
analysis showed no single representation dominated, so averaging
across all seven gave the most robust ranking.  
The TabPFN component uses Set 1 (ECFP6 + RDK + ATOMPAIR) — ECFP6 was the strongest individual fingerprint, and its larger radius captures richer substructural context.  
ECFP4 is used separately as the similarity metric for any diversity re-ranking, as it is the
field standard for Tanimoto-based diversity.

**Model, tuning, and validation:** The final predictor is a rank-weighted
fusion of two models: `0.8 * rank(LightGBM) + 0.2 * rank(TabPFN)`.
The LightGBM ensemble is seven per-fingerprint gradient-boosted models whose
probabilities are averaged. Each fingerprint's hyperparameters were tuned by
grouped cross-validation (grouped on building block BB2 to prevent leakage).  
The TabPFN component is a tabular foundation model trained in-context on the Set 1 fingerprints (concatenated and PCA-compressed to 2,000 dimensions, retaining ~95% of the variance) over an upsampled 100k  subsample.  
The two models are weakly correlated (Spearman ~0.04), which is why fusing them helped: TabPFN ranks some true hits moderately well that LightGBM ranks just outside its top compounds, and the fusion likelt surfaces them.

Model performance is validated by **grouped 5-fold cross-validation of the
full set1+2+3 ensemble**, computed on a 60k stratified subsample of the training
set and grouped on building block BB2. **AUPRC** and **AUROC** are reported,
each as a mean with a confidence interval across folds. Running `train_model.py` prints these CV
metrics before training the final models.

## Structure

```
submission_template/
├── README.md             <- This file
├── requirements.txt      <- Package requirements
├── train_model.py        <- Training + prediction script
├── lgb_configs_settings.json  <- Tuned LightGBM hyperparameters (per fingerprint)
├── data/
│   └── README.md         <- Dataset setup instructions
├── models/
│   └── best_model.pkl    <- Saved LightGBM configs + fusion recipe (written by the script)
└── src/
    ├── __init__.py
    ├── dataset.py        <- Data loaders + feature builders
    └── eval.py           <- Cross-validated AUROC/AUPRC with confidence intervals
```

## How to run

1. Install packages:
   ```
   pip install -r requirements.txt
   ```
2. Place datasets in `data/` (see `data/README.md`). The tuned LightGBM
   hyperparameters are in `lgb_configs_settings.json` (loaded automatically).
3. Run:
   ```
   python train_model.py
   ```
   This validates locally (cross-validated AUROC/AUPRC with confidence
   intervals), trains the final models on all data, saves
   `models/best_model.pkl`, and writes `submission.csv`.

### Note on TabPFN

The TabPFN component runs far faster on a **GPU**; the script auto-detects CUDA
and uses it when available, falling back to CPU otherwise (functional but very slow
over the full 339k-compound test set). The original predictions for submission were
generated with TabPFN on a Google Colab L4 GPU.

TabPFN requires an API token. Set it as an environment variable before running:
```
export TABPFN_TOKEN=your_token_here
```
To reproduce the LightGBM-only baseline predictions without TabPFN:
```
python train_model.py --skip-tabpfn
```

### Reproducibility note

The LightGBM ensemble is deterministic — it produces the same predictions every run.
TabPFN is not; although it uses a fixed `random_state` (default 0), GPU floating-point
operations are not bit-for-bit reproducible between runs. Also, regenerating the PCA
features in a different environment can introduce small numerical differences. As a
result, a freshly regenerated TabPFN can shift compounds, changing the fusion result 
slightly.

To reproduce the exact submitted result, pass the included original TabPFN predictions:
```
python train_model.py --tabpfn-file new_tabpfn_set1.npy
```
Run without this flag to regenerate TabPFN from scratch

### A note on `best_model.pkl`

Because the final model is a *fusion*, `best_model.pkl` stores the artifacts
needed to rebuild it: the tuned LightGBM hyperparameters, the fingerprint list,
and the fusion weight. The TabPFN component is a foundation model loaded from its
pretrained checkpoint at run time rather than a fitted object, so it is
reconstructed by the script rather than pickled. Running `train_model.py`
rebuilds the LightGBM ensemble and regenerates TabPFN; see the reproducibility 
note above regarding the exact submitted result.
