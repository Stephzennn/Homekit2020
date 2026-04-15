# =============================================================================
# patching.py — Learned Patch Assignment Bridge
# =============================================================================
#
# PURPOSE:
#   This module is the bridge between Step 0 (supervised clustering) and
#   Step 1 (masked pretraining). It takes the trained SoftKMeansPatchLearner
#   from patch_learner.py, runs every window in the dataset through it to
#   compute hard patch assignments, and saves the assignment map to disk.
#   The pretraining script then reads this map instead of computing fixed
#   stride/patch_len patches.
#
# -----------------------------------------------------------------------------
# WHERE THIS FITS IN THE PIPELINE:
#
#   Step 0: Train PatchLearnerClassifier on labeled data (patch_learner.py)
#               ↓
#   Step 0.5: THIS FILE — extract and save hard patch assignments
#               ↓
#   Step 1: Masked pretraining using learned patch assignments
#               ↓
#   Step 2: Linear probe → full finetune (patchtst_finetune.py)
#
# -----------------------------------------------------------------------------
# WHAT THIS SCRIPT DOES:
#
#   1. LOAD TRAINED CLUSTERING MODULE
#      Load the saved PatchLearnerClassifier checkpoint from Step 0.
#      Extract only the SoftKMeansPatchLearner (discard the classification head
#      — it is no longer needed after Step 0).
#      Freeze all weights.
#
#   2. RUN HARD ASSIGNMENT ON FULL DATASET
#      Iterate over every window in the dataset (train + val + test,
#      labeled and unlabeled). For each window [T x C]:
#          patch_ids = patch_learner.hard_assignments(x)  → [T] int tensor
#      Each timestep gets a discrete patch ID in {0, 1, ..., K-1}.
#      This is deterministic — same window always produces same assignment.
#
#   3. SAVE ASSIGNMENT MAP TO DISK
#      Save as a dictionary keyed by window index (or participant_id + date):
#          {
#              window_id: patch_ids tensor [T],
#              ...
#          }
#      Saved as a .pt file (torch.save) alongside the dataset.
#      The pretraining script loads this file at startup.
#
#   4. STATISTICS / SANITY CHECK
#      After generating assignments, print:
#          - Distribution of patch sizes (how many timesteps per patch on avg)
#          - Number of empty patches (patches that received zero timesteps)
#          - Entropy of marginal patch distribution (should be high — near log K)
#      These diagnostics confirm the clustering module did not collapse and
#      produced a meaningful spread of assignments.
#
# -----------------------------------------------------------------------------
# HOW PRETRAINING USES THE ASSIGNMENT MAP:
#
#   In patchtst_pretrain.py, the current patching step is:
#       patches = x[:, t:t+patch_len]  for t in range(0, T, stride)
#
#   With learned patches, this is replaced by:
#       assignment_map = torch.load(assignment_map_path)
#       for k in range(K):
#           mask = (assignment_map[window_id] == k)   # timesteps in patch k
#           patch_k = x[:, mask, :].mean(dim=1)       # mean pool → [B, C]
#
#   Masking for the reconstruction task happens at the patch level:
#       randomly select a fraction of the K patches to mask
#       reconstruction target = mean-pooled embedding of masked patch timesteps
#       (same objective as PatchTST but with learned non-contiguous patches)
#
# -----------------------------------------------------------------------------
# INPUTS:
#   --clustering_model_path   path to saved PatchLearnerClassifier checkpoint
#   --assignment_save_path    where to save the output assignment map .pt file
#   --dataset                 which dataset to run (Wearable, etc.)
#   --batch_size              batch size for inference (no gradients needed)
#
# OUTPUTS:
#   assignment_map.pt         dict mapping window_id → patch_ids tensor [T]
#
# =============================================================================
