# =============================================================================
# finetune.py — LearnedPatchTST Full Training Script
# =============================================================================
#
# RESEARCH HYPOTHESIS:
#
#   Fixed stride patching imposes an arbitrary inductive bias on the temporal
#   structure of physiological signals. We propose learning the patching scheme
#   directly from the classification task using a lightweight supervised
#   clustering module, bypassing the combinatorial search over patch/stride
#   hyperparameters. The learned scheme — which may assign non-contiguous
#   timesteps to the same patch and adapt the number of patches to input
#   complexity — is then used as the tokenization function for masked
#   self-supervised pretraining, decoupling patch discovery from backbone
#   training. We hypothesize this produces a better-informed pretraining
#   tokenization than manual stride selection, and test this on flu positivity
#   prediction from wearable sensor data.
#
#   Critically, this method tailors the patching scheme to the specific
#   downstream task. The optimal patch structure for flu positivity prediction
#   may differ substantially from that of symptom severity, headache
#   positivity, or fatigue — each task may exploit different temporal
#   groupings of the physiological signal. Rather than using a single fixed
#   patching scheme across all tasks, LearnedPatchTST discovers a
#   task-specific tokenization, making the representation maximally
#   informative for the prediction objective at hand.
#
# -----------------------------------------------------------------------------
# PURPOSE:
#   End-to-end training script for LearnedPatchTST.
#   Mirrors the structure of PatchTST_self_supervised/patchtst_finetune.py
#   but replaces the fixed-stride patching step with the learned patch
#   assignments produced by the frozen ANN from patch_learner.py.
#
# -----------------------------------------------------------------------------
# OVERALL TRAINING PIPELINE (two phases):
#
#   PHASE 1 — ANN patch assignment training (see patch_learner.py):
#       1. Initialise the ANN (small MLP) that maps [timestep features +
#          positional embedding] → patch probability distribution [K].
#       2. For each run (e.g. 3 runs):
#           a. Re-initialise the upper transformer with fresh random weights.
#           b. Freeze the upper transformer.
#           c. Train the ANN for 3–4 epochs using:
#               - Soft weighted mean-pooling for differentiable patch embeddings
#               - classification_loss + λ * balancing_loss
#           d. ANN weights are retained across runs (not reset between runs).
#       3. After all runs: freeze the ANN permanently.
#
#   PHASE 2 — Full model training (standard finetuning):
#       1. Load the frozen ANN.
#       2. For each lookback window: run every timestep through the frozen ANN,
#          take argmax → hard patch ID per timestep.
#       3. Group timesteps by patch ID → mean-pool → patch embeddings [B, K, D].
#       4. Feed patch embeddings into the transformer encoder (unfrozen).
#       5. Train the full model (transformer + classifier head) as normal,
#          using the same training loop as patchtst_finetune.py:
#               - BCEWithLogitsLoss with pos_weight (capped)
#               - ValidationROCAUCCB for ROC-AUC tracking
#               - SaveModelCB monitoring valid_roc_auc
#               - EarlyStoppingCB
#               - LR finder (rank 0 only, broadcast to all ranks if DDP)
#
# -----------------------------------------------------------------------------
# KEY HYPERPARAMETERS (to be added to argparse):
#   --num_patches       K: number of patches per lookback window
#   --ann_hidden_dim    hidden layer size of the patch assignment ANN
#   --ann_runs          number of re-initialisation runs in Phase 1 (e.g. 3)
#   --ann_epochs        epochs per run in Phase 1 (e.g. 3–4)
#   --balancing_lambda  λ: weight of the balancing loss in Phase 1
#   --pos_weight_cap    cap on BCE pos_weight (same as patchtst_finetune.py)
#
# -----------------------------------------------------------------------------
# REUSED FROM patchtst_finetune.py:
#   - DataLoaders / DictDataset / DataLoadersV2
#   - BCEWithLogitsLoss squeeze wrapper
#   - compute_pos_weight()
#   - ValidationROCAUCCB
#   - SaveModelCB, EarlyStoppingCB, TrackTimerCB, PrintResultsCB
#   - LR finder with rank-0 broadcast pattern (DDP safe)
#   - Focal Loss and LibAUC options (commented out, for future experiments)
#
# =============================================================================
