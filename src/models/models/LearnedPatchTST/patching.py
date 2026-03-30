# =============================================================================
# patching.py — Learned Patch Embedding Layer
# =============================================================================
#
# PURPOSE:
#   This module replaces PatchTST's fixed-stride patch embedding with a
#   dynamic one driven by the frozen ANN from patch_learner.py.
#   It takes a lookback window and produces K patch embeddings — one per patch —
#   regardless of how many timesteps ended up in each patch.
#
# -----------------------------------------------------------------------------
# INPUTS:
#   - x: raw lookback window, shape [B, T, C]
#         B = batch size, T = number of timesteps, C = number of features
#   - patch_assignments: output of frozen ANN
#       * During Phase 2 training: hard integer IDs, shape [B, T]
#         (each timestep has one patch ID in {0, 1, ..., K-1})
#       * During Phase 1 ANN training: soft probabilities, shape [B, T, K]
#         (used for differentiable weighted pooling — see patch_learner.py)
#
# -----------------------------------------------------------------------------
# OUTPUT:
#   - patch_embeddings: shape [B, K, D]
#       K = number of patches (fixed hyperparameter, e.g. lookback / patch_size)
#       D = patch embedding dimension
#   These are then passed into the transformer encoder exactly as PatchTST does.
#
# -----------------------------------------------------------------------------
# HOW PATCH EMBEDDINGS ARE FORMED:
#
#   Phase 1 (soft, differentiable):
#       patch_k_embedding = sum_t(p(t→k) * x_t) / sum_t(p(t→k))
#       i.e. a weighted mean of all timestep feature vectors, weighted by
#       how likely the ANN thinks they belong to patch k.
#       Every timestep contributes a little to every patch — but most weight
#       flows to the patch with the highest probability.
#
#   Phase 2 (hard, after ANN is frozen):
#       Timesteps are grouped by their argmax patch ID.
#       patch_k_embedding = mean of x_t for all t where argmax(ANN(t)) == k
#       i.e. simple mean pooling over the timesteps assigned to patch k.
#       Variable group sizes are fine — mean pooling produces fixed-size output.
#
# -----------------------------------------------------------------------------
# NOTES:
#   - The number of timesteps per patch will vary (unlike PatchTST where every
#     patch has exactly patch_len timesteps). This is expected and handled by
#     mean pooling.
#   - The resulting [B, K, D] tensor has the same shape as PatchTST's patch
#     embeddings, so the transformer encoder above can be reused unchanged.
#   - Positional embeddings for the K patches may need rethinking: since patches
#     are no longer ordered by time, standard sequential positional encoding
#     is not directly meaningful. Options: learnable patch-level positional
#     embeddings, or no positional encoding at the patch level (TBD).
#
# =============================================================================
