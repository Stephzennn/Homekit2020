# =============================================================================
# phase2_pretrain.py — Phase 2: Masked Pretraining on Learned Cluster Patches
# =============================================================================
#
# PIPELINE POSITION:
#   Phase 1  → discover task-driven patches via clustering
#   Phase 2  → THIS FILE: masked self-supervised pretraining on those patches
#   Phase 3  → fine-tune the pretrained transformer for flu classification
#
# HOW IT DIFFERS FROM patchtst_pretrain.py:
#   Fixed-stride PatchTST:
#       [B, T, C] → unfold(patch_len, stride) → [B, K, C, patch_len]
#                → W_P: Linear(patch_len → d_model) per channel
#                → transformer over K tokens
#                → reconstruct raw timestep values
#
#   This file (cluster-based):
#       [B, T, C] → frozen patch learner (top-k soft weights) → [B, K_eff, C]
#                → W_P: Linear(C → d_model)  joint projection
#                → transformer over K_eff tokens
#                → reconstruct soft patch embeddings [B, K_eff, C]
#
#   Key difference: patches are data-driven, variable-size, non-overlapping
#   in hard mode but soft in top-k mode. The transformer sees K_eff tokens
#   instead of a fixed num_patch.
#
# COMPONENTS:
#   ClusterPatchTST       — transformer model accepting [B, K_eff, C] input
#   ClusterPatchMaskCB    — callback replacing PatchMaskCB: builds soft patch
#                           embeddings via topk_patch_embeddings, then masks
#   phase2_pretrain.py    — training script (this file)
#
# USAGE:
#   torchrun --nproc_per_node=2 \
#     .../LearnedPatchTST/phase2_pretrain.py \
#     --dset_pretrain Wearable \
#     --context_points 10080 \
#     --patch_learner_path saved_models/.../model3_patch_learner.pth \
#     --meta_path         saved_models/.../model3_meta.json \
#     --top_k 2 \
#     --mask_ratio 0.5 \
#     --n_epochs_pretrain 40 \
#     --n_layers 6 --n_heads 8 --d_model 256 --d_ff 512 \
#     --pretrained_model_id 1 \
#     --model_type LearnedPatch_phase2
# =============================================================================

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn

_PATCHTST_DIR = os.path.join(os.path.dirname(__file__), '..', 'PatchTST_self_supervised')
sys.path.insert(0, os.path.abspath(_PATCHTST_DIR))

from datautils import get_dls
from src.learner import Learner, transfer_weights
from src.callback.core import Callback
from src.callback.tracking import *
from src.callback.transforms import RevInCB
from src.basics import set_device

from patch_learner import SoftKMeansPatchLearner, sinusoidal_pos_enc
from src.callback.patch_mask import random_masking_3D


# =============================================================================
# compute_cluster_P — diagnostic pass to derive P from natural population dist
# =============================================================================
def compute_cluster_P(patch_learner, train_loader, active_ids, revin, pos_enc,
                      device, percentile: int = 75) -> int:
    """
    Run one pass over the training set to measure the natural population of
    each active cluster, then return P as the chosen percentile of that
    distribution.

    Natural population of cluster k in sample b:
        nat_pop(b, k) = number of timesteps whose soft weight to cluster k
                        exceeds the uniform baseline (1 / K_eff).

    P = percentile(mean_nat_pop over clusters, percentile)
        e.g. percentile=75 → 75% of clusters have natural pop ≤ P,
             so only 25% need truncation. The rest are padded at most lightly.

    Parameters
    ----------
    patch_learner : SoftKMeansPatchLearner   frozen Phase 1 model
    train_loader  : DataLoader
    active_ids    : list[int]                from meta.json
    revin         : RevIN or None
    pos_enc       : torch.Tensor or None     [T, C] sinusoidal
    device        : torch.device
    percentile    : int                      default 75

    Returns
    -------
    P : int   fixed patch content size (timesteps per cluster)
    """
    # TO FIX — P SELECTION SHOULD USE RANK-ORDERING, NOT A THRESHOLD:
    # The current approach uses 1/K_eff as a membership threshold to count natural
    # population per cluster, then takes the 75th percentile as P. This is flawed:
    # at tau=0.3 the softmax never truly zeros out, so 1/K_eff is too permissive —
    # almost every timestep exceeds it for many clusters, inflating natural populations
    # (observed: mean=589, expected uniform=71, P ended up at 872 instead of ~100).
    #
    # Correct approach (no threshold needed):
    #   1. For each cluster k, rank ALL T timesteps by hard-assignment population
    #      (how many timesteps have cluster k as their argmax across the training set).
    #      No threshold — just sort the K_eff clusters by their mean population.
    #   2. Look at the distribution of those K_eff population values.
    #      If the distribution is flat (low Gini) → P = T // K_eff, no sifting needed.
    #      If the distribution has a natural elbow (some clusters much smaller) →
    #      P = population at the elbow. Clusters below the elbow either get padded
    #      or dropped (same Gini logic as filter_centroids.py, but applied to patch size
    #      rather than cluster survival).
    #   3. This is the same rank-ordering logic the user described: the population
    #      distribution itself is the sifting factor, not an arbitrary threshold or
    #      percentile. The bottom clusters having roughly equal populations means the
    #      distribution is healthy and P = min population. A steep drop at the bottom
    #      signals near-dead clusters that should be excluded from the P calculation.

    K_eff     = len(active_ids)
    threshold = 1.0 / K_eff
    active_t  = torch.tensor(active_ids, device=device, dtype=torch.long)

    pop_accum = torch.zeros(K_eff)
    n_samples = 0

    patch_learner.eval()
    with torch.no_grad():
        for raw_batch in train_loader:
            xb = raw_batch['inputs_embeds'] if isinstance(raw_batch, dict) \
                 else raw_batch[0]
            xb = xb.to(device)

            if revin is not None:
                xb = revin(xb, mode='norm')
            if pos_enc is not None:
                xb = xb + pos_enc.unsqueeze(0).to(device)

            _, assignments = patch_learner.forward(xb)           # [B, T, K]
            active_probs   = assignments[:, :, active_t]         # [B, T, K_eff]

            # Natural pop per (b, k) → mean over batch → accumulate
            nat_pop = (active_probs > threshold).float().sum(dim=1)  # [B, K_eff]
            pop_accum += nat_pop.cpu().sum(dim=0)
            n_samples += xb.shape[0]

    mean_pops = (pop_accum / n_samples).numpy()   # [K_eff] mean nat pop per cluster

    P = int(np.percentile(mean_pops, percentile))
    P = max(P, 1)

    print(f'[compute_cluster_P] Natural population stats over {n_samples} samples:')
    print(f'  min={mean_pops.min():.1f}  median={np.median(mean_pops):.1f}'
          f'  mean={mean_pops.mean():.1f}  max={mean_pops.max():.1f}')
    print(f'  P = {percentile}th percentile = {P} timesteps per cluster')
    print(f'  Clusters needing truncation (nat_pop > P): '
          f'{(mean_pops > P).sum()} / {K_eff}')
    print(f'  Clusters needing padding    (nat_pop < P): '
          f'{(mean_pops < P).sum()} / {K_eff}')

    return P


# =============================================================================
# ClusterPatchTST — transformer operating on [B, K_eff, P*C] patch content
# =============================================================================
class ClusterPatchTST(nn.Module):
    """
    Transformer backbone for cluster-based masked pretraining and fine-tuning.

    Unlike the original PatchTST which operates channel-independently on
    [B*C, num_patch, patch_len] tensors, this model operates jointly over
    all channels because each cluster token already encodes all C channels.

    Architecture (pretrain):
        Input  [B, K_eff, patch_content_size]   where patch_content_size = P*C
            → W_P: Linear(P*C → d_model)        joint projection
            → + W_pos (sinusoidal, fixed)        positional encoding over K_eff
            → TSTEncoder (n_layers transformer)
            → pretrain head: Linear(d_model → P*C)   reconstruct patch content

    Architecture (classification):
        Same encoder → GAP over K_eff → Linear(d_model → n_classes)

    Parameters
    ----------
    patch_content_size : int    P*C — flattened content vector per cluster token.
                                For the current cluster-mean path (design flaw,
                                see DESIGN FLAW comment in ClusterPatchMaskCB):
                                patch_content_size = C.
                                For Approach C (correct): patch_content_size = P*C.
    K_eff              : int    number of active cluster patches (sequence length)
    d_model            : int    transformer hidden dim
    n_layers           : int    number of transformer encoder layers
    n_heads            : int    number of attention heads
    d_ff               : int    feedforward dim inside transformer
    dropout            : float
    head_dropout       : float
    head_type          : str    'pretrain' or 'classification'
    n_classes          : int    output classes for classification (default 1)
    """

    def __init__(self,
                 patch_content_size: int,
                 K_eff: int,
                 d_model: int = 256,
                 n_layers: int = 6,
                 n_heads: int = 8,
                 d_ff: int = 512,
                 dropout: float = 0.1,
                 head_dropout: float = 0.1,
                 head_type: str = 'pretrain',
                 n_classes: int = 1):
        super().__init__()

        assert head_type in ('pretrain', 'classification'), \
            "head_type must be 'pretrain' or 'classification'"

        self.K_eff             = K_eff
        self.d_model           = d_model
        self.head_type         = head_type
        self.patch_content_size = patch_content_size

        # Input projection: P*C → d_model
        # For cluster-mean path: P*C = C (trivial, 8-dim)
        # For Approach C:        P*C = 71*8 = 568
        self.W_P = nn.Linear(patch_content_size, d_model)

        # Fixed sinusoidal positional encoding over K_eff patch positions
        pe = self._make_pos_enc(K_eff, d_model)
        self.register_buffer('W_pos', pe)

        self.dropout = nn.Dropout(dropout)

        from src.models.patchTST import TSTEncoder
        self.encoder = TSTEncoder(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            norm='BatchNorm',
            attn_dropout=0.0,
            dropout=dropout,
            activation='gelu',
            res_attention=False,
            n_layers=n_layers,
        )

        if head_type == 'pretrain':
            # Reconstruct full patch content [B, K_eff, P*C]
            self.head = nn.Sequential(
                nn.Dropout(head_dropout),
                nn.Linear(d_model, patch_content_size),
            )
        else:
            # Binary classification: global average pool → linear
            self.head = nn.Sequential(
                nn.Dropout(head_dropout),
                nn.Linear(d_model, n_classes),
            )

    @staticmethod
    def _make_pos_enc(length: int, d_model: int) -> torch.Tensor:
        pe       = torch.zeros(length, d_model)
        position = torch.arange(length, dtype=torch.float).unsqueeze(1)
        i_even   = torch.arange(0, d_model, 2, dtype=torch.float)
        div_term = torch.exp(i_even * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, K_eff, patch_content_size]

        pretrain output  : [B, K_eff, patch_content_size]
        classify output  : [B, n_classes]
        """
        x = self.W_P(x)                                    # [B, K_eff, d_model]
        x = self.dropout(x + self.W_pos.unsqueeze(0))      # [B, K_eff, d_model]
        z = self.encoder(x)                                # [B, K_eff, d_model]

        if self.head_type == 'pretrain':
            return self.head(z)                            # [B, K_eff, patch_content_size]
        else:
            z = z.mean(dim=1)
            return self.head(z)                            # [B, n_classes]


# =============================================================================
# ClusterPatchMaskCB — replaces PatchMaskCB for the cluster pipeline
# =============================================================================
class ClusterPatchMaskCB(Callback):
    """
    Callback that replaces the fixed-stride PatchMaskCB for cluster-based
    masked pretraining.

    Before each forward pass:
      1. Apply RevIN-normalised xb through the frozen patch learner using
         top-k soft weights → patch_embeddings [B, K_eff, C]
      2. Randomly mask mask_ratio fraction of the K_eff patches using
         random_masking_3D (reused from patch_mask.py)
      3. Set learner.xb = masked embeddings  [B, K_eff, C]
         Set learner.yb = original embeddings [B, K_eff, C]  (reconstruction target)

    Loss (overrides learner.loss_func):
      MSE on masked positions only — same logic as PatchMaskCB._loss.

    Parameters
    ----------
    patch_learner_path : str   path to *_patch_learner.pth from Phase 1
    meta_path          : str   path to *_meta.json from Phase 1
    mask_ratio         : float fraction of K_eff patches to mask
    top_k              : int   soft membership per timestep (default 2)
    use_pos_enc        : bool  apply sinusoidal PE before patch learner
                               (must match Phase 1 training — read from meta)
    context_points     : int   T — needed to build pos_enc
    """

    def __init__(self,
                 patch_learner_path: str,
                 meta_path: str,
                 mask_ratio: float = 0.5,
                 P: int = None,
                 pop_percentile: int = 75):
        self.patch_learner_path = patch_learner_path
        self.meta_path          = meta_path
        self.mask_ratio         = mask_ratio
        self.pop_percentile     = pop_percentile

        # P can be pre-computed and passed in (recommended — avoids a second
        # diagnostic pass in before_fit).  If None, before_fit will compute it.
        self.P                  = P

        # Set in before_fit
        self.patch_learner      = None
        self.active_ids         = None
        self.pos_enc            = None
        self.n_features         = None
        self.mask               = None    # [B, K_eff]       True = masked patch token
        self.pad_mask           = None    # [B, K_eff, P]    True = real member

    def before_fit(self):
        """
        Load frozen patch learner, run diagnostic pass to compute P via
        Approach C (distribution-informed), then override the learner's
        loss function and rebuild the model's W_P / head to match P*C.
        """
        device = self.learner.device

        with open(self.meta_path, 'r') as f:
            meta = json.load(f)

        self.active_ids  = meta['active_centroid_ids']
        n_patches_total  = meta['n_patches_total']
        use_pos_enc      = meta.get('use_pos_enc', True)
        context_points   = meta.get('context_points', 10080)
        self.n_features  = meta.get('n_features', 8)
        # Use Phase 1 end_tau — centroids were shaped at this temperature,
        # so soft assignments at this tau best reflect the final trained state.
        # Falls back to 0.3 for old checkpoints that pre-date end_tau in meta.
        end_tau          = meta.get('end_tau', 0.3)

        # Build and freeze patch learner
        state = torch.load(self.patch_learner_path, map_location=device)
        self.patch_learner = SoftKMeansPatchLearner(
            n_features=self.n_features,
            n_patches=n_patches_total,
            temperature=end_tau,
        ).to(device)
        self.patch_learner.load_state_dict(state)
        self.patch_learner.eval()
        for p in self.patch_learner.parameters():
            p.requires_grad_(False)

        # Positional encoding — must match Phase 1
        if use_pos_enc:
            self.pos_enc = sinusoidal_pos_enc(context_points, self.n_features, device)
        else:
            self.pos_enc = None

        # Build a temporary RevIN for the diagnostic pass to match training normalisation
        from src.callback.transforms import RevInCB
        revin_tmp = getattr(self.learner, '_revin_tmp', None)
        # Simpler: just use the learner's own revin if RevInCB is present
        revin_cb = next((cb for cb in self.learner.cbs
                         if cb.__class__.__name__ == 'RevInCB'), None)
        revin_diag = revin_cb.revin if revin_cb is not None else None

        # Approach C — P is pre-computed in main() before model construction
        # (because W_P input size = P*C must be known at ClusterPatchTST.__init__).
        # If P was not passed at construction time, run the diagnostic here as fallback.
        if self.P is None:
            self.P = compute_cluster_P(
                patch_learner = self.patch_learner,
                train_loader  = self.learner.dls.train,
                active_ids    = self.active_ids,
                revin         = revin_diag,
                pos_enc       = self.pos_enc,
                device        = device,
                percentile    = self.pop_percentile,
            )

        # Override learner loss with pad-mask-aware masked MSE
        self.learner.loss_func = self._loss

        print(f'[ClusterPatchMaskCB] patch_learner  : {self.patch_learner_path}')
        print(f'[ClusterPatchMaskCB] active_ids     : {len(self.active_ids)} centroids')
        print(f'[ClusterPatchMaskCB] P              : {self.P} timesteps/cluster '
              f'({self.pop_percentile}th percentile)')
        print(f'[ClusterPatchMaskCB] patch_content  : P*C = {self.P}*{self.n_features}'
              f' = {self.P * self.n_features}')
        print(f'[ClusterPatchMaskCB] mask_ratio     : {self.mask_ratio}')
        print(f'[ClusterPatchMaskCB] use_pos_enc    : {use_pos_enc}')

    def before_forward(self):
        """
        Build raw patch content via Approach C, mask patches, set learner xb/yb.

        Steps:
          1. Apply RevIN normalisation (handled by RevInCB before this callback)
          2. Apply sinusoidal PE if Phase 1 used it
          3. Call cluster_patch_content → [B, K_eff, P, C] + pad_mask [B, K_eff, P]
          4. Flatten to [B, K_eff, P*C]
          5. Random mask over K_eff patch tokens
          6. Zero out masked tokens in xb, keep originals as yb
        """
        # Lazy-load patch learner if before_fit was never called
        # (e.g. learn.test() path skips before_fit)
        if self.patch_learner is None:
            self.before_fit()

        xb     = self.xb      # [B, T, C]  RevIN already applied by RevInCB
        device = xb.device

        if next(self.patch_learner.parameters()).device != device:
            self.patch_learner = self.patch_learner.to(device)

        if self.pos_enc is not None:
            xb = xb + self.pos_enc.to(device).unsqueeze(0)

        with torch.no_grad():
            # Approach C: raw content [B, K_eff, P, C] + pad_mask [B, K_eff, P]
            content, self.pad_mask = self.patch_learner.cluster_patch_content(
                xb,
                active_centroid_ids=self.active_ids,
                P=self.P,
            )  # content: [B, K_eff, P, C]

        B, K_eff, P, C = content.shape
        # Flatten patch content for transformer input: [B, K_eff, P*C]
        content_flat = content.reshape(B, K_eff, P * C)

        # Random masking over the K_eff patch dimension
        xb_masked, _, self.mask, _ = random_masking_3D(
            content_flat, self.mask_ratio
        )  # xb_masked: [B, K_eff, P*C], mask: [B, K_eff]
        self.mask = self.mask.bool()

        self.learner.xb = xb_masked      # [B, K_eff, P*C]  masked input to transformer
        self.learner.yb = content_flat   # [B, K_eff, P*C]  reconstruction target

    def _loss(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Masked MSE loss with two levels of masking:

          mask     [B, K_eff]      True = this patch token was masked (reconstruct it)
          pad_mask [B, K_eff, P]   True = real timestep member (not padding)

        Only compute loss where BOTH conditions hold: the patch was masked AND
        the individual timestep position is a real cluster member (not padding).

        preds  : [B, K_eff, P*C]
        target : [B, K_eff, P*C]
        """
        B = preds.shape[0]
        P = self.P
        C = self.n_features

        target   = target.to(preds.device,         non_blocking=True)
        mask     = self.mask.to(preds.device,       non_blocking=True)   # [B, K_eff]
        pad_mask = self.pad_mask.to(preds.device,   non_blocking=True)   # [B, K_eff, P]

        # Reshape to [B, K_eff, P, C] for per-timestep loss
        preds_4d  = preds.reshape(B, -1, P, C)   # [B, K_eff, P, C]
        target_4d = target.reshape(B, -1, P, C)  # [B, K_eff, P, C]

        # MSE per (b, k, p): mean over C channels → [B, K_eff, P]
        loss_4d = ((preds_4d - target_4d) ** 2).mean(dim=-1)   # [B, K_eff, P]

        # Combined mask: patch was masked AND position is a real member
        # mask: [B, K_eff] → expand to [B, K_eff, P]
        combined = mask.unsqueeze(-1) & pad_mask                # [B, K_eff, P]

        loss = (loss_4d * combined.float()).sum() / (combined.sum() + 1e-8)
        return loss


# =============================================================================
# ClusterPatchCB — no-masking patch callback for Phase 3 fine-tuning
# =============================================================================
class ClusterPatchCB(Callback):
    """
    Patch callback for Phase 3 fine-tuning (classification).

    Identical setup to ClusterPatchMaskCB but WITHOUT masking:
      - Loads the frozen patch learner from Phase 1
      - Builds cluster patch content [B, K_eff, P*C] via Approach C
      - Sets learner.xb to the patch content for ClusterPatchTST input
      - Leaves learner.yb (classification labels [B]) UNTOUCHED

    Parameters
    ----------
    patch_learner_path : str   path to *_patch_learner.pth from Phase 1
    meta_path          : str   path to *_meta.json from Phase 1
    P                  : int   timesteps per cluster (pre-computed in main)
    pop_percentile     : int   percentile used if P is None (fallback)
    """

    def __init__(self,
                 patch_learner_path: str,
                 meta_path: str,
                 P: int = None,
                 pop_percentile: int = 75):
        self.patch_learner_path = patch_learner_path
        self.meta_path          = meta_path
        self.pop_percentile     = pop_percentile
        self.P                  = P

        # Set in before_fit
        self.patch_learner  = None
        self.active_ids     = None
        self.pos_enc        = None
        self.n_features     = None

    def before_fit(self):
        device = self.learner.device

        with open(self.meta_path, 'r') as f:
            meta = json.load(f)

        self.active_ids  = meta['active_centroid_ids']
        n_patches_total  = meta['n_patches_total']
        use_pos_enc      = meta.get('use_pos_enc', True)
        context_points   = meta.get('context_points', 10080)
        self.n_features  = meta.get('n_features', 8)
        end_tau          = meta.get('end_tau', 0.3)

        state = torch.load(self.patch_learner_path, map_location=device)
        self.patch_learner = SoftKMeansPatchLearner(
            n_features=self.n_features,
            n_patches=n_patches_total,
            temperature=end_tau,
        ).to(device)
        self.patch_learner.load_state_dict(state)
        self.patch_learner.eval()
        for p in self.patch_learner.parameters():
            p.requires_grad_(False)

        if use_pos_enc:
            self.pos_enc = sinusoidal_pos_enc(context_points, self.n_features, device)
        else:
            self.pos_enc = None

        if self.P is None:
            revin_cb = next((cb for cb in self.learner.cbs
                             if cb.__class__.__name__ == 'RevInCB'), None)
            revin_diag = revin_cb.revin if revin_cb is not None else None
            self.P = compute_cluster_P(
                patch_learner = self.patch_learner,
                train_loader  = self.learner.dls.train,
                active_ids    = self.active_ids,
                revin         = revin_diag,
                pos_enc       = self.pos_enc,
                device        = device,
                percentile    = self.pop_percentile,
            )

        print(f'[ClusterPatchCB] patch_learner  : {self.patch_learner_path}')
        print(f'[ClusterPatchCB] active_ids     : {len(self.active_ids)} centroids')
        print(f'[ClusterPatchCB] P              : {self.P} timesteps/cluster '
              f'({self.pop_percentile}th percentile)')
        print(f'[ClusterPatchCB] patch_content  : P*C = {self.P}*{self.n_features}'
              f' = {self.P * self.n_features}')
        print(f'[ClusterPatchCB] use_pos_enc    : {use_pos_enc}')

    def before_forward(self):
        """
        Build raw patch content via Approach C, set learner.xb.
        learner.yb (classification label [B]) is left untouched.

        Steps:
          1. RevIN normalisation already applied by RevInCB (before this CB)
          2. Apply sinusoidal PE if Phase 1 used it
          3. cluster_patch_content → [B, K_eff, P, C]
          4. Flatten to [B, K_eff, P*C] → set as learner.xb
        """
        # Lazy-load patch learner if before_fit was never called
        # (e.g. learn.test() path skips before_fit)
        if self.patch_learner is None:
            self.before_fit()

        xb     = self.xb      # [B, T, C]  RevIN already applied
        device = xb.device

        if next(self.patch_learner.parameters()).device != device:
            self.patch_learner = self.patch_learner.to(device)

        if self.pos_enc is not None:
            xb = xb + self.pos_enc.to(device).unsqueeze(0)

        with torch.no_grad():
            content, _ = self.patch_learner.cluster_patch_content(
                xb,
                active_centroid_ids=self.active_ids,
                P=self.P,
            )  # content: [B, K_eff, P, C]

        B, K_eff, P, C = content.shape
        self.learner.xb = content.reshape(B, K_eff, P * C)   # [B, K_eff, P*C]
        # learner.yb (labels) stays as-is


# =============================================================================
# Argument parser
# =============================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Phase 2: Cluster-based masked pretraining')

    # Data
    parser.add_argument('--dset_pretrain', type=str, default='Wearable')
    parser.add_argument('--context_points', type=int, default=10080)
    parser.add_argument('--target_points', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--scaler', type=str, default='standard')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--revin', type=int, default=1)

    # Phase 1 outputs
    parser.add_argument('--patch_learner_path', type=str, required=True,
                        help='path to *_patch_learner.pth from Phase 1')
    parser.add_argument('--meta_path', type=str, required=True,
                        help='path to *_meta.json from Phase 1')

    # Soft patching (used by Phase 1 frozen model — kept for CB compat)
    parser.add_argument('--top_k', type=int, default=2,
                        help='soft membership per timestep (1=hard, 2=recommended)')

    # Approach C: distribution-informed patch content size
    parser.add_argument('--pop_percentile', type=int, default=75,
                        help='percentile of natural cluster population distribution '
                             'used to set P (timesteps per cluster). '
                             '75 → only 25%% of clusters need truncation. '
                             'Higher = larger patches, fewer truncations, more padding.')

    # Masking
    parser.add_argument('--mask_ratio', type=float, default=0.5)

    # Transformer architecture
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--head_dropout', type=float, default=0.1)

    # Training
    parser.add_argument('--n_epochs_pretrain', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-4)

    # Saving
    parser.add_argument('--pretrained_model_id', type=int, default=1)
    parser.add_argument('--model_type', type=str, default='LearnedPatch_phase2')

    # Resume
    parser.add_argument('--resume_from', type=str, default=None,
                        help='path to a Phase 2 checkpoint to resume from')

    # Wandb
    parser.add_argument('--use_wandb', type=int, default=0)
    parser.add_argument('--wandb_project', type=str, default='PatchTST-Wearable')
    parser.add_argument('--wandb_run_name', type=str, default=None)

    return parser


# =============================================================================
# DDP setup — identical to patchtst_pretrain.py
# =============================================================================
def setup_ddp():
    using_torchrun = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ

    if using_torchrun:
        rank       = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f'cuda:{local_rank}')
        else:
            device = torch.device('cpu')

        if world_size > 1:
            os.environ.setdefault('NCCL_IB_DISABLE', '1')
            if not dist.is_initialized():
                dist.init_process_group(backend='nccl', init_method='env://')
            is_distributed = True
        else:
            is_distributed = False
    else:
        rank = world_size = 0
        local_rank = 0
        world_size = 1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        is_distributed = False

    if torch.cuda.is_available():
        import time as _t
        _t0 = _t.time()
        _r = int(os.environ.get('RANK', 0))
        print(f'[rank {_r}] CUDA warmup start...', flush=True)
        torch.zeros(1, device=device)
        torch.cuda.synchronize(device)
        print(f'[rank {_r}] CUDA warmup done ({_t.time()-_t0:.1f}s)', flush=True)

    return is_distributed, rank, world_size, local_rank, device


# =============================================================================
# Save path helpers
# =============================================================================
def finalize_args(args, K_eff):
    args.dset = args.dset_pretrain

    args.save_pretrained_model = (
        f'clusterpatchtst_pretrained'
        f'_cw{args.context_points}'
        f'_K{K_eff}'
        f'_topk{args.top_k}'
        f'_epochs{args.n_epochs_pretrain}'
        f'_mask{args.mask_ratio}'
        f'_model{args.pretrained_model_id}'
    )

    args.save_path = (
        f'saved_models/{args.dset_pretrain}/learned_patchtst/{args.model_type}/'
    )
    return args


# =============================================================================
# Pre-compute P — must run before model is built (P*C is W_P's input size)
# =============================================================================
def load_patch_learner_and_compute_P(args, K_eff, device, rank) -> tuple:
    """
    Load the frozen patch learner and run one diagnostic pass over the
    training set to compute P (Approach C). Returns (patch_learner, P).

    Must be called before get_model so that patch_content_size = P*C
    is known at model construction time.
    """
    with open(args.meta_path, 'r') as f:
        meta = json.load(f)

    n_patches_total = meta['n_patches_total']
    use_pos_enc     = meta.get('use_pos_enc', True)
    context_points  = meta.get('context_points', 10080)
    n_features      = meta.get('n_features', 8)
    active_ids      = meta['active_centroid_ids']
    # Phase 1 end_tau — must match the temperature the centroids converged at
    end_tau         = meta.get('end_tau', 0.3)

    state = torch.load(args.patch_learner_path, map_location=device)
    patch_learner = SoftKMeansPatchLearner(
        n_features=n_features,
        n_patches=n_patches_total,
        temperature=end_tau,
    ).to(device)
    patch_learner.load_state_dict(state)
    patch_learner.eval()
    for p in patch_learner.parameters():
        p.requires_grad_(False)

    pos_enc = sinusoidal_pos_enc(context_points, n_features, device) \
              if use_pos_enc else None

    # Small RevIN for diagnostic normalisation — no learned parameters needed
    # here since this is just a population count (weights don't need to be
    # trained): use None and accept slight mismatch vs training normalisation.
    # The population counts are robust to this — they are ordinal (rank-based),
    # not sensitive to the exact scale of the weights.
    revin_diag = None

    dls = get_dls(args)

    if rank == 0:
        P = compute_cluster_P(
            patch_learner = patch_learner,
            train_loader  = dls.train,
            active_ids    = active_ids,
            revin         = revin_diag,
            pos_enc       = pos_enc,
            device        = device,
            percentile    = args.pop_percentile,
        )
    else:
        P = 0   # will be broadcast from rank 0

    return patch_learner, P, dls


# =============================================================================
# Model factory
# =============================================================================
def get_model(patch_content_size, K_eff, args, rank):
    model = ClusterPatchTST(
        patch_content_size=patch_content_size,
        K_eff=K_eff,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        head_type='pretrain',
    )

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'ClusterPatchTST | K_eff={K_eff} | patch_content_size={patch_content_size}'
              f' | params={n_params:,}')

    return model


# =============================================================================
# Pretrain function
# =============================================================================
def pretrain_func(args, lr, K_eff, P, rank, is_distributed, dls=None):
    if dls is None:
        dls = get_dls(args)

    n_features         = dls.vars
    patch_content_size = P * n_features
    model = get_model(patch_content_size, K_eff, args, rank)

    if args.resume_from is not None:
        model = transfer_weights(args.resume_from, model)
        if rank == 0:
            print(f'Resumed from: {args.resume_from}')

    # Callbacks — RevIN must come before ClusterPatchMaskCB so the patch
    # learner sees normalised inputs, matching Phase 1 training.
    cbs = [RevInCB(n_features, denorm=False)] if args.revin else []
    cbs += [
        ClusterPatchMaskCB(
            patch_learner_path=args.patch_learner_path,
            meta_path=args.meta_path,
            mask_ratio=args.mask_ratio,
            P=P,
            pop_percentile=args.pop_percentile,
        ),
        SaveModelCB(
            monitor='valid_loss',
            fname=args.save_pretrained_model,
            path=args.save_path,
        ),
    ]

    if args.use_wandb and rank == 0:
        cbs.append(WandbCB(
            project=args.wandb_project,
            run_name=args.wandb_run_name or args.save_pretrained_model,
            config=vars(args),
        ))

    learn = Learner(
        dls,
        model,
        loss_func=nn.MSELoss(reduction='mean'),  # overridden by ClusterPatchMaskCB
        lr=lr,
        cbs=cbs,
        metrics=[],
    )

    if is_distributed:
        learn.to_distributed()

    learn.fit_one_cycle(args.n_epochs_pretrain, lr_max=lr)

    if rank == 0:
        train_loss = learn.recorder['train_loss']
        valid_loss = learn.recorder['valid_loss']
        pd.DataFrame({'train_loss': train_loss, 'valid_loss': valid_loss}).to_csv(
            args.save_path + args.save_pretrained_model + '_losses.csv',
            float_format='%.6f', index=False,
        )


# =============================================================================
# LR finder
# =============================================================================
def find_lr(args, K_eff, P, rank, dls=None):
    if dls is None:
        dls = get_dls(args)

    patch_content_size = P * dls.vars
    model = get_model(patch_content_size, K_eff, args, rank)

    # Load resumed weights before running the LR finder so the loss landscape
    # the finder sees matches the checkpoint, not a random initialisation.
    # Without this, the suggested LR is calibrated to a cold-start model and
    # will typically be too high for a partially-trained one.
    if args.resume_from is not None:
        model = transfer_weights(args.resume_from, model)
        if rank == 0:
            print(f'[find_lr] Loaded resume checkpoint: {args.resume_from}')

    cbs = [RevInCB(dls.vars, denorm=False)] if args.revin else []
    cbs += [ClusterPatchMaskCB(
        patch_learner_path=args.patch_learner_path,
        meta_path=args.meta_path,
        mask_ratio=args.mask_ratio,
        P=P,
        pop_percentile=args.pop_percentile,
    )]

    learn = Learner(dls, model, loss_func=nn.MSELoss(), lr=args.lr, cbs=cbs)
    suggested_lr = learn.lr_finder(end_lr=args.lr)

    if rank == 0:
        print(f'Suggested LR: {suggested_lr:.6f}')

    return suggested_lr, dls


# =============================================================================
# Resume checkpoint P-consistency check
# =============================================================================
def _verify_P_vs_checkpoint(args, P: int, dls=None) -> None:
    """
    When --resume_from is set, load the checkpoint's W_P.weight and verify
    that its input dimension matches P * n_features.

    If there is a mismatch, transfer_weights will silently skip W_P (shape
    mismatch) and the model resumes with a randomly-initialised projection
    layer — effectively training from scratch with wrong patch content size.
    Better to catch this early and tell the user what P the checkpoint expects.
    """
    if args.resume_from is None:
        return

    ckpt = torch.load(args.resume_from, map_location='cpu')
    if 'W_P.weight' not in ckpt:
        print(f'[P-check] WARNING: W_P.weight not found in checkpoint — cannot verify P.')
        return

    ckpt_patch_content_size = ckpt['W_P.weight'].shape[1]
    # n_features: use dls.vars if available, else fall back to meta
    if dls is not None:
        n_features = dls.vars
    else:
        with open(args.meta_path) as f:
            n_features = json.load(f).get('n_features', 8)

    ckpt_P = ckpt_patch_content_size // n_features

    print(f'[P-check] Computed P          : {P}')
    print(f'[P-check] Checkpoint P        : {ckpt_P}  '
          f'(W_P input={ckpt_patch_content_size}, n_features={n_features})')

    if P == ckpt_P:
        print(f'[P-check] MATCH — safe to resume.')
    else:
        raise ValueError(
            f'[P-check] MISMATCH: diagnostic pass produced P={P} but checkpoint '
            f'was built with P={ckpt_P}. Resume would silently corrupt W_P.\n'
            f'Fix: re-run Phase 1 diagnostic with --pop_percentile adjusted until '
            f'P={ckpt_P}, or pass --pop_percentile that reproduces the original P.'
        )


# =============================================================================
# Main
# =============================================================================
def main():
    parser = build_parser()
    args   = parser.parse_args()

    is_distributed, rank, world_size, local_rank, device = setup_ddp()

    # Read K_eff from meta before finalizing args (needed for save name)
    with open(args.meta_path, 'r') as f:
        meta = json.load(f)
    K_eff = meta['effective_k']

    if rank == 0:
        print(f'K_eff from meta: {K_eff}')

    args = finalize_args(args, K_eff)

    if rank == 0:
        os.makedirs(args.save_path, exist_ok=True)
        print('args:', args)
        print(f'Save path: {args.save_path}')
        print(f'Save name: {args.save_pretrained_model}')

    if not is_distributed and torch.cuda.is_available():
        set_device()

    if is_distributed:
        device_t  = torch.device(f'cuda:{local_rank}')
        # Broadcast both LR and P from rank 0 so all ranks use identical values
        lr_p_tensor = torch.zeros(2, dtype=torch.float32, device=device_t)

        if rank == 0:
            # Step 1 — compute P from natural population distribution
            # Must happen before model construction (W_P input size = P*C)
            _, P, cached_dls = load_patch_learner_and_compute_P(
                args, K_eff, device_t, rank
            )
            print(f'[main] Computed P={P}  (patch_content_size={P * K_eff})')

            # If resuming, verify P matches the checkpoint's W_P input size.
            # A mismatch means the diagnostic pass produced a different P than
            # what the checkpoint was built with — transfer_weights would then
            # silently fail to load W_P and the model would train from scratch.
            _verify_P_vs_checkpoint(args, P, dls=cached_dls)

            print(f'Broadcasting P={P} to all ranks...')

            # Step 2 — LR finder using the correct patch_content_size
            suggested_lr, cached_dls = find_lr(args, K_eff, P, rank, dls=cached_dls)
            suggested_lr = float(suggested_lr)
            print(f'Broadcasting LR={suggested_lr:.6f} to all ranks...')

            lr_p_tensor[0] = suggested_lr
            lr_p_tensor[1] = float(P)

        dist.broadcast(lr_p_tensor, src=0)
        suggested_lr = lr_p_tensor[0].item()
        P            = int(lr_p_tensor[1].item())

        pretrain_func(
            args=args, lr=suggested_lr, K_eff=K_eff, P=P,
            rank=rank, is_distributed=is_distributed,
            dls=cached_dls if rank == 0 else None,
        )
    else:
        # Single GPU / CPU path
        _, P, cached_dls = load_patch_learner_and_compute_P(
            args, K_eff, device, rank
        )
        print(f'[main] Computed P={P}  (patch_content_size={P * K_eff})')
        _verify_P_vs_checkpoint(args, P, dls=cached_dls)
        suggested_lr, cached_dls = find_lr(args, K_eff, P, rank, dls=cached_dls)
        pretrain_func(
            args=args, lr=float(suggested_lr), K_eff=K_eff, P=P,
            rank=rank, is_distributed=is_distributed,
            dls=cached_dls,
        )

    if rank == 0:
        print('Phase 2 pretraining complete.')


if __name__ == '__main__':
    main()
