# =============================================================================
# patch_learner.py — Supervised Soft K-Means Patch Assignment Module
# =============================================================================
#
# RESEARCH HYPOTHESIS (see finetune.py for full writeup):
#   Fixed stride patching imposes an arbitrary inductive bias. This module
#   learns a task-driven patching scheme directly from labeled data via
#   differentiable soft k-means clustering attached to a classification head.
#
# -----------------------------------------------------------------------------
# TRAINING STAGE (Step 0 — before any pretraining):
#
#   Architecture used in this stage:
#       Raw input [B x T x C]
#           ↓
#       SoftKMeansPatchLearner  ← this file
#           ↓  soft assignments [B x T x K]
#           ↓  weighted mean pool → patch embeddings [B x K x C]
#           ↓  global average pool → [B x C]
#           ↓
#       Linear classification head
#           ↓
#       BCEWithLogitsLoss + pos_weight (same as patchtst_finetune.py)
#
#   No transformer backbone in this stage — the clustering module and
#   classification head are trained alone. This eliminates the joint
#   instability problem (co-adapting backbone + assignment module).
#
#   After convergence:
#       - Freeze the clustering module
#       - Run each training window through it, take argmax → hard patch IDs
#       - Use these hard assignments as the patching scheme for pretraining
#         (replaces fixed stride/patch_len in patchtst_pretrain.py)
#
# -----------------------------------------------------------------------------
# SOFT K-MEANS ASSIGNMENT (differentiable):
#
#   Each timestep x_t ∈ R^C is assigned to K cluster centroids via softmax
#   over negative squared distances:
#
#       d(t, k) = ||x_t - centroid_k||^2
#       p(t → k) = softmax(-d / temperature)[k]
#
#   This is fully differentiable. Gradients flow from the classification loss
#   back through the pooling → assignments → centroids.
#
#   At inference (hard assignment):
#       patch_id(t) = argmin_k d(t, k)  =  argmax_k p(t → k)
#
# -----------------------------------------------------------------------------
# PATCH EMBEDDINGS (weighted mean pooling):
#
#   patch_k = Σ_t [ p(t→k) · x_t ] / Σ_t [ p(t→k) ]
#
#   Fully differentiable during training. At inference, replaced by simple
#   mean pooling over hard-assigned timesteps.
#
# -----------------------------------------------------------------------------
# BALANCING LOSS (anti-collapse):
#
#   Without regularisation, all timesteps collapse to one centroid.
#   We maximise entropy of the marginal patch distribution:
#
#       marginal_k = mean over all timesteps of p(t → k)   shape [K]
#       balancing_loss = -entropy(marginal)
#                      = Σ_k marginal_k * log(marginal_k)
#
#   This forces roughly equal usage of all K patches.
#
# -----------------------------------------------------------------------------
# KEY HYPERPARAMETERS:
#   K              number of patches/clusters (default 150)
#   temperature    softmax temperature — lower = sharper assignments (default 1.0)
#   balancing_lambda  weight of balancing loss (default 0.1)
#
# =============================================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_pos_enc(T: int, C: int, device) -> torch.Tensor:
    """
    Standard sinusoidal positional encoding, shape [T, C].

    Fully deterministic — no learned parameters, no randomness.
    The same T and C always produce the exact same tensor, making it
    safe to apply identically in Phase 1 and Phase 2.

        PE(t, 2i)   = sin(t / 10000^(2i / C))
        PE(t, 2i+1) = cos(t / 10000^(2i / C))

    Handles both even and odd C correctly.

    Parameters
    ----------
    T : int    sequence length (e.g. 10080)
    C : int    feature dimension (e.g. 8 wearable channels)
    device     torch device

    Returns
    -------
    pe : torch.Tensor  shape [T, C], dtype float32
    """
    pe = torch.zeros(T, C, device=device)
    position = torch.arange(T, dtype=torch.float, device=device).unsqueeze(1)  # [T, 1]
    i_even   = torch.arange(0, C, 2, dtype=torch.float, device=device)         # [C//2]
    div_term = torch.exp(i_even * (-math.log(10000.0) / C))                    # [C//2]

    pe[:, 0::2] = torch.sin(position * div_term)                  # even indices
    pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])  # odd indices
    return pe


class SoftKMeansPatchLearner(nn.Module):
    """
    Differentiable soft k-means clustering module.

    Maps a lookback window [B x T x C] to K patch embeddings [B x K x C]
    via soft distance-based assignment to K learned centroids.

    Used in Step 0 (supervised training without transformer) to discover
    a task-driven patching scheme for subsequent pretraining.

    Parameters
    ----------
    n_features : int
        Number of input channels C (e.g. 8 wearable channels).
    n_patches : int
        Number of clusters K. Default 150.
    temperature : float
        Softmax temperature for assignment sharpness.
        Lower = harder assignments. Default 1.0.
    balancing_lambda : float
        Weight of the entropy balancing loss. Default 0.1.
    """

    def __init__(self, n_features: int, n_patches: int = 150,
                 temperature: float = 1.0, balancing_lambda: float = 0.1):
        super().__init__()

        self.n_patches = n_patches
        self.temperature = temperature
        self.balancing_lambda = balancing_lambda

        # K cluster centroids in R^C — learned via backprop from task loss.
        # Initialised with small random values; will be updated by gradient
        # descent alongside the classification head.
        self.centroids = nn.Parameter(
            torch.randn(n_patches, n_features) * 0.01
        )

    def forward(self, x: torch.Tensor):
        """
        Compute soft patch embeddings from input window.

        Parameters
        ----------
        x : torch.Tensor  shape [B, T, C]
            Raw lookback window — B samples, T timesteps, C features.

        Returns
        -------
        patch_embeddings : torch.Tensor  shape [B, K, C]
            One embedding per patch, computed as weighted mean of timesteps.
        assignments : torch.Tensor  shape [B, T, K]
            Soft assignment probabilities — p(t → k) for each timestep t
            and patch k. Useful for computing the balancing loss.
        """
        B, T, C = x.shape

        # --- Compute pairwise squared distances [B, T, K] ---
        # x_flat: [B*T, C], centroids: [K, C]
        x_flat = x.reshape(B * T, C)

        # ||x_t - c_k||^2 = ||x_t||^2 + ||c_k||^2 - 2 x_t · c_k
        x_sq = (x_flat ** 2).sum(dim=1, keepdim=True)          # [B*T, 1]
        c_sq = (self.centroids ** 2).sum(dim=1, keepdim=True).T # [1, K]
        cross = x_flat @ self.centroids.T                        # [B*T, K]
        distances = x_sq + c_sq - 2 * cross                     # [B*T, K]
        distances = distances.reshape(B, T, self.n_patches)      # [B, T, K]

        # --- Soft assignments via softmax over negative distances ---
        # Higher probability = closer to centroid = more likely assigned.
        assignments = F.softmax(-distances / self.temperature, dim=-1)  # [B, T, K]

        # --- Weighted mean pooling → patch embeddings ---
        # patch_k = Σ_t [ p(t→k) · x_t ] / Σ_t [ p(t→k) ]
        # assignments: [B, T, K] → transpose to [B, K, T] for bmm
        # x: [B, T, C]
        a_t = assignments.permute(0, 2, 1)                   # [B, K, T]
        numerator = torch.bmm(a_t, x)                        # [B, K, C]
        denominator = a_t.sum(dim=2, keepdim=True) + 1e-8    # [B, K, 1]
        patch_embeddings = numerator / denominator            # [B, K, C]

        return patch_embeddings, assignments

    def balancing_loss(self, assignments: torch.Tensor) -> torch.Tensor:
        """
        Entropy maximisation loss to prevent cluster collapse.

        Computes the marginal patch distribution (average assignment over all
        timesteps and batch items), then returns negative entropy — minimising
        this loss maximises entropy, forcing balanced patch usage.

        Parameters
        ----------
        assignments : torch.Tensor  shape [B, T, K]
            Soft assignment probabilities from forward().

        Returns
        -------
        loss : torch.Tensor  scalar
        """
        # Average assignment across all timesteps and batch items → [K]
        marginal = assignments.mean(dim=(0, 1))  # [K]

        # Negative entropy: Σ_k p_k * log(p_k)  (minimise = maximise entropy)
        loss = (marginal * torch.log(marginal + 1e-8)).sum()

        return self.balancing_lambda * loss

    def hard_assignments(self, x: torch.Tensor,
                         active_centroid_ids: list = None) -> torch.Tensor:
        """
        Compute discrete patch IDs at inference.

        DESIGN DECISION — why we compute the full K distribution first:
        ---------------------------------------------------------------
        Each timestep is sent through the full softmax over all K centroids
        (exactly as in training), producing a distribution [B, T, K].  We
        then restrict the argmax to the active subset (K_eff columns).

        An alternative would be to compute distances only to the K_eff active
        centroids and skip the dead ones entirely.  Both approaches return the
        same winner because softmax is monotonically increasing and preserves
        distance ordering — argmax(softmax(-d)[active]) == argmin(d[active]).
        We use the full-distribution route because:
          1. It is consistent with the Phase 1 training framing (every
             timestep always sees all K centroids).
          2. The slice [:, :, active_ids] is a single index op — no extra
             distance computation is needed.
          3. It makes the active_centroid_ids filter a post-processing step
             rather than a change in the model's computation graph, keeping
             the two phases symmetric.

        GUARANTEE — every timestep gets exactly one assignment:
        ---------------------------------------------------------------
        Because argmax is computed over [B, T, K_eff] (the active columns
        only), the result is always in [0, K_eff-1].  A timestep whose
        highest-probability centroid in the full K distribution happens to
        be a dead one will still be assigned to its highest-probability
        *active* centroid.  No timestep can be left unassigned.

        If active_centroid_ids is None, argmax is over all K centroids
        (original behaviour — only safe when no centroids have been filtered,
        e.g. during diagnostic inspection of Phase 1 outputs).

        Parameters
        ----------
        x : torch.Tensor  shape [B, T, C]
        active_centroid_ids : list[int] or None
            Indices of live centroids from *_meta.json.
            When provided, the returned IDs are in [0, K_eff-1] — position
            within the active list, not the original centroid index.

        Returns
        -------
        patch_ids : torch.Tensor  shape [B, T]  dtype=torch.long
            Every timestep is guaranteed exactly one assignment.
        """
        with torch.no_grad():
            _, assignments = self.forward(x)        # [B, T, K]  full distribution

            if active_centroid_ids is None:
                patch_ids = assignments.argmax(dim=-1)              # [B, T]
            else:
                active_ids = torch.tensor(active_centroid_ids,
                                          device=x.device, dtype=torch.long)
                active_probs = assignments[:, :, active_ids]        # [B, T, K_eff]
                patch_ids = active_probs.argmax(dim=-1)             # [B, T]

        return patch_ids

    def topk_patch_embeddings(self,
                              x: torch.Tensor,
                              active_centroid_ids: list,
                              top_k: int = 2) -> tuple:
        """
        Compute sparse soft patch embeddings using top-k membership per timestep.

        Instead of hard assignment (each timestep → exactly one patch) or full
        soft assignment (each timestep → all K patches), this method keeps only
        the top-k highest softmax weights per timestep and zeros out the rest.

        DESIGN DECISION — raw weights, NO renormalization:
        --------------------------------------------------
        The raw softmax probabilities are kept as-is for the top-k clusters.
        We deliberately do NOT renormalize the surviving weights to sum to 1.

        Why: the raw weights already encode confidence.
            p(t→k1)=0.6,  p(t→k2)=0.39,  rest dropped (total ~0.99)
        If we renormalize: k1 → 0.606, k2 → 0.394 — barely changes anything
        here because the dropped mass was near zero.

        But consider a borderline timestep:
            p(t→k1)=0.35,  p(t→k2)=0.33,  p(t→k3)=0.30,  rest=0.02
        After dropping k3 (top_k=2): k1=0.35, k2=0.33  (sum=0.68, not 1.0)
        Renormalized:                k1=0.515, k2=0.485  (sum=1.0)

        Renormalization INFLATES the weights of the two survivors beyond
        their true confidence — it pretends the timestep has zero probability
        of belonging anywhere else, which is false for a borderline timestep.
        The low sum (0.68) is itself informative: it tells the patch that this
        timestep was genuinely uncertain, and its contribution should be
        down-weighted accordingly in the patch embedding denominator.

        ALTERNATIVE TO TRY IF ISSUES ARISE:
        ------------------------------------
        If training is unstable or patch embeddings have inconsistent scale
        (because denominators vary with the surviving weight sum), try
        renormalizing:
            top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-8)
        This fixes the denominator issue at the cost of losing confidence
        information. A middle ground: renormalize only when sum < threshold
        (e.g., 0.5), keeping raw weights when the top-k already dominate.

        HOW COMPUTATION SCALES:
        -----------------------
        full soft (top_k=K): O(T*K) weights — expensive
        top_k=1            : O(T*1)   → same as hard assignment but weighted mean
        top_k=2            : O(T*2)   → boundary timesteps split between 2 patches
        top_k=3            : O(T*3)   → smoother boundaries, slight extra cost
        Recommended: top_k=2 for balance between overlap and efficiency.

        Parameters
        ----------
        x                  : torch.Tensor  shape [B, T, C]
        active_centroid_ids: list[int]  from *_meta.json
        top_k              : int  number of patches each timestep contributes to

        Returns
        -------
        patch_embeddings   : torch.Tensor  shape [B, K_eff, C]
            Weighted mean of timesteps assigned to each active centroid,
            using raw (non-renormalized) top-k softmax weights.
        sparse_assignments : torch.Tensor  shape [B, T, K_eff]
            Sparse weight matrix — top-k weights kept, rest zeroed out.
            Useful for diagnostics and computing the balancing loss.
        """
        B, T, C = x.shape

        # Step 1 — compute full softmax assignments over all K centroids
        _, assignments = self.forward(x)                              # [B, T, K]

        # Step 2 — restrict to active centroids
        active_ids = torch.tensor(active_centroid_ids,
                                  device=x.device, dtype=torch.long)
        K_eff = len(active_ids)
        active_probs = assignments[:, :, active_ids]                  # [B, T, K_eff]

        # Step 3 — keep only top-k weights per timestep, zero out the rest
        # topk returns (values, indices) along dim=-1 (the K_eff dimension)
        top_vals, top_idx = active_probs.topk(
            min(top_k, K_eff), dim=-1, largest=True, sorted=False
        )                                                              # [B, T, top_k]

        # Scatter top-k values back into a full [B, T, K_eff] sparse matrix
        sparse_assignments = torch.zeros_like(active_probs)           # [B, T, K_eff]
        sparse_assignments.scatter_(dim=-1, index=top_idx, src=top_vals)
        # Result: each timestep has exactly top_k non-zero entries,
        # with their original raw softmax weights (NOT renormalized).

        # Step 4 — weighted mean pooling → patch embeddings [B, K_eff, C]
        # patch_k = Σ_t [ w(t→k) · x_t ] / Σ_t [ w(t→k) ]
        # where w(t→k) is the raw top-k weight (0 if timestep not in top-k for k)
        a_t = sparse_assignments.permute(0, 2, 1)                     # [B, K_eff, T]
        numerator   = torch.bmm(a_t, x)                               # [B, K_eff, C]
        denominator = a_t.sum(dim=2, keepdim=True) + 1e-8             # [B, K_eff, 1]
        patch_embeddings = numerator / denominator                     # [B, K_eff, C]
        # NOTE: denominator < 1.0 for clusters whose timesteps had low raw weights
        # (borderline members). This is intentional — see design note above.

        return patch_embeddings, sparse_assignments


    def cluster_patch_content(self,
                              x: torch.Tensor,
                              active_centroid_ids: list,
                              P: int,
                              weight_threshold: float = None) -> tuple:
        """
        For each active cluster k, gather the top-P timesteps by soft weight
        from the CLUSTER's perspective (cluster picks its top-P members).

        This is the raw-content counterpart to topk_patch_embeddings.
        Instead of returning a C-dim weighted mean per cluster, this returns
        the actual P timestep vectors that most belong to each cluster —
        analogous to patch_len×C in fixed-stride PatchTST.

        Natural population of cluster k in a sample:
            nat_pop(b, k) = number of timesteps with soft weight > weight_threshold
            Default threshold = 1/K_eff (uniform baseline — above-chance membership).

        Padding rule:
            nat_pop(b, k) >= P  → take top-P by weight, pad_mask all True
            nat_pop(b, k) <  P  → take all real members (ranked by weight),
                                   pad remaining P - nat_pop positions with 0.0,
                                   pad_mask False at padded positions

        Parameters
        ----------
        x                   : torch.Tensor  [B, T, C]
        active_centroid_ids : list[int]     from *_meta.json
        P                   : int           fixed patch size (members per cluster)
                                            recommended: derived from 75th percentile
                                            of natural population distribution
                                            (see compute_cluster_P in phase2_pretrain.py)
        weight_threshold    : float or None uniform-baseline default = 1/K_eff

        Returns
        -------
        content  : torch.Tensor  [B, K_eff, P, C]
            Raw timestep values for each cluster's top-P members.
            Zero-filled at padded positions.
        pad_mask : torch.Tensor  [B, K_eff, P]  dtype=bool
            True  = real member position
            False = padded position (loss should ignore these)
        """
        B, T, C = x.shape

        # Full softmax over all K centroids (consistent with training)
        _, assignments = self.forward(x)                          # [B, T, K]

        # Restrict to active centroids
        active_ids_t = torch.tensor(active_centroid_ids,
                                    device=x.device, dtype=torch.long)
        K_eff = len(active_ids_t)
        active_probs = assignments[:, :, active_ids_t]            # [B, T, K_eff]

        # Default threshold: uniform baseline — a timestep "belongs" to cluster k
        # if its weight exceeds what a random assignment would give
        if weight_threshold is None:
            weight_threshold = 1.0 / K_eff

        # Natural population per (b, k): timesteps with weight above baseline
        # shape [B, K_eff]
        natural_pop = (active_probs > weight_threshold).sum(dim=1).long()

        # Rank timesteps per cluster by descending weight → top-P indices
        # active_probs: [B, T, K_eff] → permute to [B, K_eff, T]
        probs_bkt = active_probs.permute(0, 2, 1)                 # [B, K_eff, T]
        _, top_idx = probs_bkt.topk(
            min(P, T), dim=-1, largest=True, sorted=True
        )                                                          # [B, K_eff, P]

        # Gather raw timestep values — memory-efficient: no T-axis expansion
        # Flatten [B, K_eff, P] → [B, K_eff*P], gather from x [B, T, C]
        flat_idx     = top_idx.reshape(B, K_eff * P)              # [B, K_eff*P]
        flat_idx_exp = flat_idx.unsqueeze(-1).expand(-1, -1, C)   # [B, K_eff*P, C]
        content_flat = x.gather(dim=1, index=flat_idx_exp)        # [B, K_eff*P, C]
        content      = content_flat.reshape(B, K_eff, P, C)       # [B, K_eff, P, C]

        # Build padding mask: position p is real iff p < natural_pop[b, k]
        ranks    = torch.arange(P, device=x.device)               # [P]
        pad_mask = ranks.unsqueeze(0).unsqueeze(0) < \
                   natural_pop.unsqueeze(-1)                       # [B, K_eff, P]

        # Zero out padded positions so they don't pollute the input signal
        content = content * pad_mask.unsqueeze(-1).float()        # [B, K_eff, P, C]

        return content, pad_mask


class PatchLearnerClassifier(nn.Module):
    """
    Step 0 full model: SoftKMeansPatchLearner + classification head.

    Trained end-to-end on labeled data. No transformer backbone.
    After training, only the SoftKMeansPatchLearner is retained as the
    patching scheme for pretraining.

    Parameters
    ----------
    n_features : int
        Number of input channels C.
    n_patches : int
        Number of clusters K. Default 150.
    temperature : float
        Assignment softmax temperature. Default 1.0.
    balancing_lambda : float
        Weight of balancing loss. Default 0.1.
    n_classes : int
        Number of output classes. Use 1 for binary classification
        with BCEWithLogitsLoss. Default 1.
    """

    def __init__(self, n_features: int, n_patches: int = 150,
                 temperature: float = 1.0, balancing_lambda: float = 0.1,
                 n_classes: int = 1):
        super().__init__()

        self.patch_learner = SoftKMeansPatchLearner(
            n_features=n_features,
            n_patches=n_patches,
            temperature=temperature,
            balancing_lambda=balancing_lambda,
        )

        # Global average pool collapses [B, K, C] → [B, C] so the classifier
        # receives a fixed-size input regardless of K or variable patch sizes.
        # Linear classifier on top — outputs raw logits for BCEWithLogitsLoss.
        self.classifier = nn.Linear(n_features, n_classes)

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : torch.Tensor  shape [B, T, C]

        Returns
        -------
        logits : torch.Tensor  shape [B, n_classes]
        balancing_loss : torch.Tensor  scalar
            Add to classification loss during training:
            total_loss = classification_loss + balancing_loss
        """
        # Step 1: soft k-means → patch embeddings [B, K, C]
        patch_embeddings, assignments = self.patch_learner(x)

        # Step 2: global average pool over patches → [B, C]
        pooled = patch_embeddings.mean(dim=1)  # [B, C]

        # Step 3: linear classifier → logits [B, n_classes]
        logits = self.classifier(pooled)

        # Step 4: balancing loss (add to task loss during training)
        bal_loss = self.patch_learner.balancing_loss(assignments)

        return logits, bal_loss
