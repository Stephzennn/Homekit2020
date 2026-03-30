# =============================================================================
# patch_learner.py — Learned Patch Assignment ANN
# =============================================================================
#
# PURPOSE:
#   Replace PatchTST's fixed-stride patching with a learned assignment.
#   Instead of assuming that semantic meaning only comes from sequential
#   neighbours (e.g. timesteps 1–16 always form patch 1), a small ANN
#   decides which patch each timestep belongs to. This removes the
#   temporal-locality inductive bias.
#
# -----------------------------------------------------------------------------
# INPUT PER TIMESTEP:
#   - The full feature row at time t: shape [C] where C = number of features
#     (e.g. 8 wearable channels).
#   - A positional embedding added on top, so the ANN also knows *where*
#     in the lookback window this timestep sits.
#   Together: [C + pos_embedding_dim] or [C] with positional encoding added.
#
# OUTPUT PER TIMESTEP:
#   - A probability distribution over K patch IDs: shape [K]
#   - During training: soft probabilities (kept differentiable via softmax)
#   - During inference / Phase 2: hard assignment via argmax → scalar patch ID
#
# -----------------------------------------------------------------------------
# TRAINING LOOP (Phase 1 — ANN training):
#
#   The ANN is trained while the upper transformer architecture is frozen.
#   The upper model is re-initialized (new random weights) between runs,
#   so the ANN sees different "pressure" each time and must find assignments
#   that generalise regardless of the upper model's initialisation.
#
#   Structure:
#       for run in range(num_runs):           # e.g. 3 runs
#           re-initialise upper model weights (frozen)
#           for epoch in range(epochs_per_run):  # e.g. 3–4 epochs
#               for each batch of lookback windows:
#                   1. Feed each timestep [features + pos_emb] through ANN
#                      → soft patch probabilities [T, K]
#                   2. Compute patch embeddings via soft weighted mean-pooling:
#                      patch_k = sum_t(p(t→k) * x_t) / sum_t(p(t→k))
#                      This is fully differentiable — no Gumbel-Softmax needed.
#                   3. Feed patch embeddings into frozen upper model → logits
#                   4. Compute total loss:
#                      loss = classification_loss + λ * balancing_loss
#                   5. Backprop through upper model (no grad) → patch embeddings
#                      → soft probabilities → ANN weights
#                   6. Update ANN weights only
#
#   After all runs: retain the final ANN weights.
#
# -----------------------------------------------------------------------------
# BALANCING LOSS (anti-collapse):
#
#   Without a regulariser, the ANN tends to assign most timesteps to a single
#   patch (cluster collapse). The balancing loss prevents this by penalising
#   unequal patch usage across the batch.
#
#   Concretely: compute the average patch assignment distribution across all
#   timesteps in the batch, then penalise its deviation from a uniform
#   distribution (e.g. via KL divergence or entropy maximisation).
#
#       marginal_k = mean over all timesteps of p(t → k)
#       balancing_loss = KL(marginal, Uniform(K))
#                      = -entropy(marginal)   [maximise entropy = force balance]
#
#   λ is a hyperparameter controlling how strongly balance is enforced.
#   Solves both collapse (one dominant patch) and empty patches simultaneously.
#
# -----------------------------------------------------------------------------
# AFTER PHASE 1:
#   - Freeze ANN weights.
#   - For every lookback window: run each timestep through the frozen ANN,
#     take argmax → hard patch ID per timestep.
#   - These hard assignments define the patches used in Phase 2 (full training).
#
# -----------------------------------------------------------------------------
# KEY DIFFERENCE FROM PatchTST:
#   - PatchTST: patch boundary = fixed stride (e.g. every 16 steps)
#   - LearnedPatchTST: patch boundary = ANN decision, can be non-contiguous
#     (timestep 5 and timestep 800 can share a patch if the ANN assigns them
#     to the same group based on their feature values + position)
#
# =============================================================================
