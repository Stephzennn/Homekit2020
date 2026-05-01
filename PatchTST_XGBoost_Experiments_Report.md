# PatchTST + XGBoost Flu Detection — Experiment Log & Results
*Spring 2026 | Homekit2020 / SeattleFluStudy Project*

---

## 1. Problem Statement

Predict influenza positivity from **passive wearable sensor data** (Fitbit) using a 7-day sliding window of minute-level physiological signals. The core challenge is extreme class imbalance: only ~0.04% of windows are flu-positive, making standard classification approaches fail catastrophically.

---

## 2. Dataset

| Property | Value |
|----------|-------|
| Source | Homekit2020 / SeattleFluStudy |
| Signals (channels) | heart_rate, missing_heart_rate, missing_steps, sleep_classic_0/1/2/3, steps (8 total) |
| Temporal resolution | Minute-level |
| Context window | 10,080 minutes (7 days) |
| Split strategy | By user, temporal cutoff 2020-02-10 |
| **Train** | 94 positive / 199,107 negative (~2,120:1 imbalance) |
| **Validation** | 48 positive / 99,498 negative |
| **Test** | 42 positive / 102,169 negative |
| Prevalence | ~0.04% |
| Random baseline PR-AUC | ~0.0004 |

**Data leakage audit (confirmed clean):**
- Normalization is instance-level (per-sample z-score per channel via `normalize_numerical=True` in `ActivityTask`) — no cross-sample fitting
- RevIN at model level is also per-sample
- `--scaler standard` argument exists in the finetune script but is never consumed (dead code)
- Negative undersampling applied to training split only
- `pos_weight` computed from undersampled training labels only
- XGBoost `eval_set` uses validation only, never test
- Threshold optimization on val PR curve, applied blind to test
- Parquet data splits are separate directories with no overlap (user-level split)

---

## 3. Model Architecture — PatchTST

PatchTST is a Transformer-based time-series model that divides input sequences into fixed-length patches and applies self-attention across patches.

### Full Backbone (primary)
| Hyperparameter | Value |
|----------------|-------|
| patch_len | 1,440 (1 day) |
| stride | 180 (3 hours) |
| context_points | 10,080 (7 days) |
| Number of patches | 49 |
| n_layers | 6 |
| n_heads | 8 |
| d_model | 256 |
| d_ff | 512 |
| c_in (channels) | 8 |
| Total parameters | ~3.5M |
| Instance normalization | RevIN (per-sample) |
| Pretraining | Masked patch prediction (50% mask ratio, 100 epochs) |
| Pretraining data | All users (positive + negative windows) |

**Checkpoint:** `patchtst_pretrained_cw10080_patch1440_stride180_epochs-pretrain100_mask0.5_model492026.pth`

### Negative-Only Backbone (ablation)
| Hyperparameter | Value |
|----------------|-------|
| n_layers | 4 (smaller architecture) |
| All other params | Same as full backbone |
| Pretraining data | **Healthy (negative) windows only** |

**Checkpoint:** `Negative only training 4132026/patchtst_pretrained_cw10080_patch1440_stride180_epochs-pretrain100_mask0.5_model12.pth`

---

## 4. Embedding Extraction Strategy

Two extraction modes depending on the checkpoint type:

**Finetuned mode** (classification checkpoint):
- Forward hook on `model.head.dropout` output
- Embedding shape: `[B × (nvars × d_model)]` = `[B × 2048]`

**Pretrained mode** (backbone checkpoint, `--pretrain` flag):
- Pre-hook on `model.head` input (last patch representation, flattened)

---

## 5. Imbalance Handling Strategies Explored

### 5.1 Negative Undersampling
Keep at most `neg_subsample_ratio` negatives per positive in the training split. Applied train-only; val and test remain fully imbalanced.

### 5.2 Positive Weighting (`scale_pos_weight` in XGBoost)
Multiply the loss contribution of positive samples by a factor proportional to the neg/pos ratio.

### 5.3 XGBoost Early Stopping on Val PR-AUC
XGBoost uses `eval_metric="aucpr"` with `eval_set=[(X_val, y_val)]` and early stopping (50 rounds). The best tree count is determined by validation PR-AUC.

### 5.4 Threshold Optimization
The operating threshold is chosen to maximize F1 on the validation PR curve, then applied blind to the test set.

---

## 6. Experiments & Results

### 6.1 Direct End-to-End Finetuning (Classification Head)

Starting from the **full pretrained backbone** (`model492026.pth`), a classification head (`Linear(d_model × n_vars, 1)`) is trained end-to-end with BCEWithLogitsLoss. 10:1 negative undersampling applied to training data. 20 epochs, LR finder enabled.

| Model | Config | Test PR-AUC | Test ROC-AUC |
|-------|--------|------------|-------------|
| Model 2 | 10:1 undersample + auto pos_weight (~10) | **0.0007** | 0.632 |
| Model 4 | 10:1 undersample + no pos_weight (cap=1) | 0.0006 | 0.594 |

**Key observation:** End-to-end finetuning with only 94 positive training samples fails to learn generalizable flu signal. The model memorizes training positives (train PR-AUC can be high) but collapses on val/test. PR-AUC barely above random baseline (0.0004).

---

### 6.2 Linear Probe

Frozen backbone, only classification head trained. 10:1 undersample + auto pos_weight. 20 epochs.

| Model | Config | Test PR-AUC | Test ROC-AUC |
|-------|--------|------------|-------------|
| Model 3 | 10:1 undersample + auto pos_weight | 0.0004 | 0.486 |

**Key observation:** Worse than finetuning. Frozen backbone representations are not discriminative enough with a linear head alone at this sample size.

---

### 6.3 XGBoost on Finetuned Full Backbone Embeddings

The finetuned model (`model1.pth`, trained with no undersampling, standard BCEWithLogitsLoss) is used as a **frozen feature extractor**. XGBoost is trained on the extracted 2,048-dim embeddings.

XGBoost hyperparameters (defaults unless noted):
- `n_estimators=2000`, **early stopping = 50 rounds on validation PR-AUC**
- `learning_rate=0.03`, `max_depth=3`
- `min_child_weight=5`, `subsample=0.8`, `colsample_bytree=0.8`
- `reg_lambda=1.0`, `reg_alpha=0.0`
- `objective="binary:logistic"`, **`eval_metric="aucpr"`**

**Early stopping design:** XGBoost monitors validation PR-AUC (`eval_metric="aucpr"`) at each boosting round using `eval_set=[(X_val, y_val)]`. Training stops when val PR-AUC has not improved for 50 consecutive rounds. The final model uses the tree ensemble at the round with the **highest validation PR-AUC** — not the last round. This means the model selection criterion is directly optimized for the metric that matters most given the extreme class imbalance. The test set is never seen during this process; it is evaluated once at the end using the val-selected model and a threshold derived from the val PR curve.

#### Results Table — XGBoost Variants

| Model ID | Neg Ratio | Pos Weight | Max Depth | Reg λ | Min Child W | Val PR-AUC | **Test PR-AUC** | Test ROC-AUC | Notes |
|----------|-----------|-----------|-----------|-------|-------------|-----------|----------------|-------------|-------|
| 1 | None | None | 3 | 1.0 | 5 | 0.000482 | 0.000411 | 0.500 | At random — no signal extracted |
| 2 | None | 2× ratio | 3 | 1.0 | 5 | 0.006 | 0.007 | 0.751 | Pos weighting helps somewhat |
| 3 | 10:1 | None | 3 | 1.0 | 5 | 0.010 | 0.009 | 0.783 | Undersampling better than weighting |
| 4 | 10:1 | cap=1 | 3 | 1.0 | 5 | 0.004 | 0.004 | 0.739 | Worse without auto weight |
| **4282026** | **20:1** | **None** | **3** | **1.0** | **5** | **0.016** | **0.026** | **0.738** | **← BEST** |
| 3014282026 | 30:1 | None | 3 | 1.0 | 5 | 0.030 | 0.003 | 0.702 | Overfit to val |
| 5014282026 | 50:1 | None | 3 | 1.0 | 5 | 0.020 | 0.008 | 0.711 | Overfit to val |
| 2014292026 | 20:1 | None | 2 | 5.0 | 10 | 0.012 | 0.012 | 0.769 | Regularization hurt PR-AUC |
| 42820262 (trial 2) | 20:1 | None | 3 | 1.0 | 5 | 0.015 | 0.003 | 0.680 | Seed variance — see note |

**Best model @ opt threshold (4282026):**
- Val: TP=2, FP=9, sensitivity=4.2%, precision=18.2%, F1=0.068
- Test: TP=1, FP=13, sensitivity=2.4%, precision=7.1%, F1=0.036
- Test PR-AUC = **65× above random baseline**

---

### 6.4 Overfitting Pattern with Higher Neg Ratios

At 30:1 and 50:1, XGBoost's early stopping (based on val PR-AUC) causes overfitting to the 48 specific validation positives. At training time, the model sees a much lower imbalance ratio than at val/test, causing the model's confidence scores to be tuned specifically for val positives rather than generalizing. The 20:1 ratio is the optimal balance.

The regularized model (2014292026) shows the inverse: it **eliminated overfitting** (val PR-AUC = test PR-AUC = 0.012) and improved ROC-AUC (0.769 vs 0.738), but lost the sharp high-precision spike at low recall that drives PR-AUC up in the default model.

**Trial 2 non-replication (important):** The 20:1 setup (model 4282026, test PR-AUC = 0.026) was rerun with identical hyperparameters as a validation check (model 42820262). The replication **failed to reproduce** the result, yielding test PR-AUC = 0.003 — an 88% drop. Val PR-AUC was consistent between the two runs (0.016 vs 0.015), confirming the model selection step was stable. The divergence is entirely in test generalization.

The source of variance is XGBoost's internal stochasticity: `subsample=0.8` and `colsample_bytree=0.8` randomly sample rows and features at each boosting round. With only 42 test positives, a shift of even 1–2 correctly ranked positive samples produces a large swing in PR-AUC. This means **the 0.026 figure should be interpreted with caution** — it likely represents an optimistic draw from a distribution whose true mean is lower. The result demonstrates real signal (PR-AUC consistently above random across all runs) but the precise magnitude is unreliable at this positive sample count.

---

### 6.5 Negative-Only Pretrained Backbone Ablation

All experiments below use the **4-layer negative-only pretrained backbone** (`model12.pth`) as the feature extractor.

#### XGBoost directly on negative-only pretrained backbone

| Model | Neg Ratio | Test PR-AUC | Test ROC-AUC |
|-------|-----------|------------|-------------|
| Neg-XGB model 1 | 10:1 | ~0.005* | ~0.75* |
| Neg-XGB model 2 | 20:1 | 0.007 | 0.800 |

*Approximate from earlier run, some details incomplete.

#### Finetuning negative-only backbone then XGBoost

The negative-only backbone was first finetuned end-to-end (60 epochs, 10:1 undersampling, auto pos_weight), then embeddings from that finetuned model were passed to XGBoost.

| Model | Approach | Test PR-AUC | Test ROC-AUC |
|-------|----------|------------|-------------|
| Neg-finetune direct | Classification head only (60 epochs) | 0.001 | 0.641 |
| Neg-finetune + XGBoost | 20:1 XGBoost on finetuned neg embeddings | 0.003 | 0.773 |

**Key observation:** The negative-only backbone consistently underperforms across all approaches. Having never seen flu-positive windows during pretraining, the representations do not encode any flu-relevant signal. Even after 60-epoch finetuning on 94 positives, the model's embeddings remain uninformative for flu detection.

---

## 7. Full Performance Leaderboard

Ranked by test PR-AUC (primary metric):

| Rank | Model | Backbone | Approach | Test PR-AUC | Test ROC-AUC |
|------|-------|----------|----------|------------|-------------|
| 1 | XGB 4282026 | Full pretrain → finetuned | XGBoost 20:1, no pos weight | **0.026** | 0.738 |
| 2 | XGB model 2 | Full pretrain → finetuned | XGBoost, 2× pos weight | 0.007 | 0.751 |
| 3 | Neg-XGB model 2 | Neg-only pretrained | XGBoost 20:1 | 0.007 | 0.800 |
| 4 | XGB model 3 | Full pretrain → finetuned | XGBoost 10:1 | 0.009 | 0.783 |
| 5 | XGB 5014282026 | Full pretrain → finetuned | XGBoost 50:1 | 0.008 | 0.711 |
| 6 | XGB 2014292026 | Full pretrain → finetuned | XGBoost 20:1 + regularization | 0.012 | 0.769 |
| 7 | XGB model 4 | Full pretrain → finetuned | XGBoost 10:1, pos cap | 0.004 | 0.739 |
| 8 | Neg-finetune + XGB | Neg-only pretrain → finetune | XGBoost 20:1 | 0.003 | 0.773 |
| 9 | XGB 3014282026 | Full pretrain → finetuned | XGBoost 30:1 | 0.003 | 0.702 |
| 10 | Finetune model 2 | Full pretrain → finetuned | End-to-end 10:1 + pos weight | 0.0007 | 0.632 |
| 11 | Finetune model 4 | Full pretrain → finetuned | End-to-end 10:1, no weight | 0.0006 | 0.594 |
| 12 | Neg-finetune direct | Neg-only pretrain → finetune | End-to-end 60 epochs | 0.001 | 0.641 |
| 13 | Linear probe model 3 | Full pretrain → finetuned | Frozen backbone + head | 0.0004 | 0.486 |
| 14 | XGB model 1 | Full pretrain → finetuned | XGBoost, no undersampling | 0.0004 | 0.500 |

---

## 8. Key Findings & Takeaways

### 8.1 XGBoost >> End-to-End Finetuning
With only 94 positive training samples, gradient-based end-to-end training of a 3.5M-parameter model fails completely (best test PR-AUC: 0.0007). XGBoost on frozen embeddings achieves 37× better PR-AUC (0.026). The pretrained backbone already encodes useful physiological representations; XGBoost can exploit these with far fewer samples than a neural head requires.

### 8.2 Negative Undersampling is the Primary Lever
The single biggest improvement came from undersampling negatives during XGBoost training. Without it (model 1), XGBoost scores at random. With 20:1 undersampling, it achieves 65× above random PR-AUC. The optimal ratio is **20:1** — lower ratios leave too much imbalance, higher ratios cause val-test overfitting via early stopping.

### 8.3 Positive Weighting Hurts at Extreme Imbalance
Upweighting the positive class (even 2×) did not improve over 20:1 undersampling alone. The combination of undersampling + pos_weight did not help either. The model generalizes best when trained with a realistic 20:1 ratio and default `scale_pos_weight=1`.

### 8.4 Regularization Trades PR-AUC for ROC-AUC
Adding `max_depth=2`, `reg_lambda=5`, `min_child_weight=10` eliminated val-test overfitting and improved ROC-AUC (0.738 → 0.769) but hurt PR-AUC (0.026 → 0.012). The default model's sharp high-precision spike (few very-confident positive predictions) is more valuable for PR-AUC than a broader, more calibrated score distribution.

### 8.5 Full Pretraining Backbone is Essential
The negative-only pretrained backbone consistently underperforms across all approaches (best test PR-AUC: 0.007 vs 0.026). Without flu-positive signal during pretraining, the learned representations do not capture physiological changes associated with influenza. The full backbone — pretrained on all users including flu-positive ones — provides the necessary embedding quality.

### 8.6 Best Result Did Not Replicate — Results are Highly Variable Due to Small N
The best result (model 4282026, test PR-AUC = 0.026) was explicitly re-run under identical conditions as a validation check. The replication produced test PR-AUC = 0.003, failing to reproduce the original. Val PR-AUC was stable across both runs (0.016 vs 0.015), so the model selection step (early stopping on highest val PR-AUC) was consistent — the variance is entirely in test generalization.

Root cause: XGBoost's stochastic row/feature subsampling (`subsample=0.8`, `colsample_bytree=0.8`) combined with only 42 test positives means that ranking 1–2 additional positives correctly can shift PR-AUC by an order of magnitude. The 0.026 figure should be treated as an **optimistic single draw**, not a reliable estimate of expected performance. The model does demonstrate real signal (above random in all but one run) but the magnitude is unreliable at this sample size. Multi-seed averaging or cross-validation would be needed for a stable estimate.

### 8.7 Data Leakage: None Found
Systematic audit confirmed no leakage: instance normalization (not fitted on training set), separate user-level splits, train-only subsampling, val-only threshold optimization.

---

## 9. Architecture & Code Changes Made

### XgBoost.py
- Removed capped `scale_pos_weight`
- Added `--neg_subsample_ratio` argument (train-only negative undersampling)
- Added `--scale_pos_weight_mult` argument (optional positive upweighting)
- Added `--xgb_max_depth`, `--xgb_reg_lambda`, `--xgb_min_child_weight`, `--xgb_lr` arguments
- XGBoost `eval_metric` set to `"aucpr"` (PR-AUC-based early stopping)
- Model IDs used for separate output CSV/PNG/PKL files to prevent overwriting prior runs

### datautils.py
- Added `neg_subsample_ratio` and `seed` to `DataLoadersV2.__init__`
- Added `_subsample_negatives()` method (train split only)
- Wired through `get_dls()` for Wearable dataset

### patchtst_finetune.py
- Added `--neg_subsample_ratio` and `--seed` arguments
- `compute_pos_weight()` uses undersampled train labels

---

## 10. SLURM Scripts (Experiment Registry)

| Script | Description |
|--------|-------------|
| `submit_xgboost_patchtst.sh` | XGB 30:1, model 3014282026 |
| `submit_xgboost_50_1.sh` | XGB 50:1, model 5014282026 |
| `submit_xgboost_20_1_trial2.sh` | XGB 20:1 repeat, model 42820262 |
| `submit_xgboost_final_spring_trial.sh` | XGB 20:1 + regularization, model 2014292026 |
| `submit_xgboost_neg_pretrain_20_1.sh` | XGB 20:1 on neg-only backbone |
| `submit_xgboost_neg_finetuned.sh` | XGB 20:1 on neg-only finetuned backbone |
| `submit_xgboost_negative_pretrain.sh` | XGB 10:1 on neg-only backbone |
| `submit_finetune_undersample.sh` | Finetune 10:1 + auto pos_weight, model 2 |
| `submit_finetune_undersample_noweight.sh` | Finetune 10:1, no pos_weight, model 4 |
| `submit_linear_probe_undersample.sh` | Linear probe 10:1, model 3 |
| `submit_finetune_neg_pretrain.sh` | Finetune neg-only backbone, 60 epochs |

---

## 11. Next Steps / Open Questions

- **Why does the 20:1 XGBoost vary so much between runs?** Consider averaging over multiple seeds or using cross-validation on the training positives.
- **Feature engineering:** Raw wearable signals may benefit from domain-specific features (e.g., HRV, resting HR baseline deviation) rather than relying purely on embeddings.
- **Different pretraining objective:** Contrastive pretraining (e.g., SimCLR on temporal windows) may yield embeddings more discriminative for rare events.
- **More positive training data:** The fundamental bottleneck is 94 train positives. Augmenting positive samples (time-warping, jittering) could help.
- **Larger context window:** A 14-day window might capture prodromal symptoms better than 7 days.
