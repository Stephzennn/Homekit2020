# =============================================================================
# phase1_cluster_classification.py — Phase 1: Task-Driven Patch Learning
# =============================================================================
#
# PURPOSE:
#   Train a SoftKMeansPatchLearner + linear classifier end-to-end on the
#   Wearable flu-positivity task. No transformer backbone — just the clustering
#   module and a single linear head. The goal is to discover a task-driven
#   patch assignment scheme that captures temporal patterns relevant to flu
#   onset. Classification performance here is a proxy signal, not the end goal.
#
# OUTPUT:
#   A saved .pth checkpoint of the full model (clustering + head).
#   Phase 2 loads the SoftKMeansPatchLearner from this checkpoint and
#   discretises the soft assignments into a fixed patch map for pretraining.
#
# PIPELINE POSITION:
#   Phase 1 (this file)  → discover patches from labeled data
#   Phase 2              → discretise assignments, extract patch schema
#   Phase 3              → masked pretraining on learned patches
#   Phase 4              → fine-tuning for classification
#
# KEY DESIGN DECISIONS:
#   - Gumbel-Softmax with temperature annealing:
#       Training:   soft assignments with Gumbel noise (stochastic, differentiable)
#       Temperature decreases linearly each epoch: --start_tau → --end_tau
#       Inference:  temperature → 0 → pure argmax (discrete)
#   - Over-specified K (--n_patches, default 150):
#       Many centroids intentionally; dead ones accumulate zero assignment.
#       Phase 2 filters dead centroids → effective K is data-driven, not fixed.
#   - Loss: BCE(pos_weight) + balancing_lambda * entropy_regulariser
#   - Saves on best validation ROC-AUC (primary metric for imbalanced data)
#   - DDP supported via torchrun --nproc_per_node=2
#   - Same data pipeline as patchtst_finetune.py (datautils.get_dls)
#
# EXAMPLE COMMAND:
#   torchrun --nproc_per_node=2 \
#     .../LearnedPatchTST/phase1_cluster_classification.py \
#     --dset_finetune Wearable \
#     --context_points 10080 \
#     --n_patches 150 \
#     --n_epochs 30 \
#     --lr 1e-3 \
#     --model_id 1 \
#     --model_type LearnedPatch_phase1
# =============================================================================

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Add PatchTST_self_supervised to sys.path so we can reuse datautils, RevIN,
# and the same data pipeline without duplicating any code.
# ---------------------------------------------------------------------------
_PATCHTST_DIR = os.path.join(os.path.dirname(__file__), '..', 'PatchTST_self_supervised')
sys.path.insert(0, os.path.abspath(_PATCHTST_DIR))

from datautils import get_dls                                  # same loader as finetune
from src.models.layers.revin import RevIN                      # per-instance normalisation

# ---------------------------------------------------------------------------
# Import the clustering model defined in patch_learner.py (same directory).
# PatchLearnerClassifier = SoftKMeansPatchLearner + linear head.
# ---------------------------------------------------------------------------
from patch_learner import PatchLearnerClassifier, sinusoidal_pos_enc


# =============================================================================
# Argument parser
# =============================================================================
parser = argparse.ArgumentParser(description='Phase 1: Task-driven patch learning')

# --- Data ---
parser.add_argument('--dset_finetune', type=str, default='Wearable')
parser.add_argument('--context_points', type=int, default=10080,
                    help='lookback window length in timesteps')
parser.add_argument('--target_points', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--scaler', type=str, default='standard')
parser.add_argument('--features', type=str, default='M')

# --- Clustering model ---
parser.add_argument('--n_patches', type=int, default=150,
                    help='number of cluster centroids K (over-specify; dead ones are '
                         'filtered out in Phase 2)')
parser.add_argument('--balancing_lambda', type=float, default=0.5,
                    help='weight of entropy balancing loss (prevents cluster collapse)')

# --- Gumbel-Softmax temperature annealing ---
parser.add_argument('--start_tau', type=float, default=1.0,
                    help='initial softmax temperature (high = soft/uniform assignments)')
parser.add_argument('--end_tau', type=float, default=0.1,
                    help='final softmax temperature (low = sharp/discrete assignments)')

# --- Training ---
parser.add_argument('--n_epochs', type=int, default=30)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--pos_weight_cap', type=float, default=-1.0,
                    help='cap on BCEWithLogitsLoss pos_weight; -1 = no cap')
parser.add_argument('--revin', type=int, default=1,
                    help='apply per-instance RevIN normalisation (1=yes)')
parser.add_argument('--use_pos_enc', type=int, default=1,
                    help='add sinusoidal positional encoding to xb before '
                         'clustering (1=yes, default). Deterministic — same '
                         'encoding must be applied in Phase 2.')

# --- Saving ---
parser.add_argument('--model_id', type=int, default=1,
                    help='identifier appended to saved model filename')
parser.add_argument('--model_type', type=str, default='LearnedPatch_phase1',
                    help='subdirectory name under saved_models/')

# --- Wandb ---
parser.add_argument('--use_wandb', type=int, default=0)
parser.add_argument('--wandb_project', type=str, default='PatchTST-Wearable')
parser.add_argument('--wandb_run_name', type=str, default=None)

args = parser.parse_args()


# =============================================================================
# Distributed setup — mirrors patchtst_finetune.py exactly
# =============================================================================
is_distributed = 'LOCAL_RANK' in os.environ
local_rank = int(os.environ.get('LOCAL_RANK', 0))
rank = int(os.environ.get('RANK', 0))
world_size = int(os.environ.get('WORLD_SIZE', 1))

if is_distributed:
    torch.cuda.set_device(local_rank)
    os.environ.setdefault('NCCL_IB_DISABLE', '1')
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl', init_method='env://')

device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# Save path
# =============================================================================
args.dset = args.dset_finetune
save_dir = os.path.join(
    'saved_models', args.dset_finetune, 'learned_patchtst', args.model_type
)
if rank == 0:
    os.makedirs(save_dir, exist_ok=True)

save_name = (
    f'{args.dset_finetune}_phase1_cluster'
    f'_cw{args.context_points}'
    f'_K{args.n_patches}'
    f'_epochs{args.n_epochs}'
    f'_model{args.model_id}'
)
save_path = os.path.join(save_dir, save_name + '.pth')

if rank == 0:
    print('args:', args)
    print(f'Model will be saved to: {save_path}')


# =============================================================================
# Helper: compute pos_weight from label tensor
# Same logic as patchtst_finetune.py — caps at pos_weight_cap if set.
# =============================================================================
def compute_pos_weight(labels, device):
    n_pos = int((labels == 1).sum().item())
    n_neg = int((labels == 0).sum().item())
    raw = n_neg / n_pos
    capped = raw if args.pos_weight_cap < 0 else min(raw, args.pos_weight_cap)
    if rank == 0:
        print(f'pos_weight: raw={raw:.1f}, cap={args.pos_weight_cap}, '
              f'using={capped:.1f}  (pos={n_pos}, neg={n_neg})')
    return torch.tensor([capped], dtype=torch.float32, device=device)


# =============================================================================
# Helper: linearly anneal temperature tau from start_tau to end_tau.
# Called at the start of each epoch to update self.temperature on the
# patch learner before the forward pass.
# High tau → soft/uniform assignments (exploration).
# Low tau → sharp/discrete assignments (exploitation).
# =============================================================================
def anneal_tau(epoch: int, n_epochs: int) -> float:
    # Linear schedule: epoch 0 → start_tau, epoch n_epochs-1 → end_tau
    frac = epoch / max(n_epochs - 1, 1)
    return args.start_tau + frac * (args.end_tau - args.start_tau)


# =============================================================================
# Helper: compute ROC-AUC across all DDP ranks.
# Each rank collects its own val predictions, then all_gather combines them
# so every rank computes the same global ROC-AUC. Mirrors ValidationROCAUCCB.
# =============================================================================
def compute_roc_auc_distributed(all_logits, all_labels):
    # all_logits, all_labels: Python lists of tensors accumulated over val batches
    logits = torch.cat(all_logits).to(device)   # [N_local]
    labels = torch.cat(all_labels).to(device)   # [N_local]

    if is_distributed and world_size > 1:
        # Gather from all ranks onto all ranks
        gathered_logits = [None] * world_size
        gathered_labels = [None] * world_size
        dist.all_gather_object(gathered_logits, logits.cpu())
        dist.all_gather_object(gathered_labels, labels.cpu())
        logits = torch.cat(gathered_logits)
        labels = torch.cat(gathered_labels)

    probs = torch.sigmoid(logits).numpy().reshape(-1)
    y_true = labels.numpy().reshape(-1).astype(int)

    if len(np.unique(y_true)) < 2:
        return 0.0  # undefined if only one class present in split
    return roc_auc_score(y_true, probs)


# =============================================================================
# Main training function
# =============================================================================
def main():
    # -------------------------------------------------------------------------
    # Step 1: Load data using the same get_dls pipeline as patchtst_finetune.py.
    # This gives us the same train/val/test splits, scaler, and class labels.
    # -------------------------------------------------------------------------
    dls = get_dls(args)
    n_features = dls.vars   # number of wearable channels (C)

    if rank == 0:
        print(f'n_features (channels): {n_features}')
        print(f'n_patches (K): {args.n_patches}')

    # -------------------------------------------------------------------------
    # Step 2: Compute pos_weight for weighted BCE loss.
    # With ~94 positives / 199k samples, pos_weight ≈ 2118 without cap.
    # -------------------------------------------------------------------------
    pos_weight = compute_pos_weight(dls.train.dataset.data['label'], device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # -------------------------------------------------------------------------
    # Step 3: Build the model.
    # PatchLearnerClassifier = SoftKMeansPatchLearner (K centroids) + Linear head.
    # No transformer backbone — this is intentionally lightweight.
    # -------------------------------------------------------------------------
    model = PatchLearnerClassifier(
        n_features=n_features,
        n_patches=args.n_patches,
        temperature=args.start_tau,         # will be annealed each epoch
        balancing_lambda=args.balancing_lambda,
        n_classes=1,
    ).to(device)

    # Wrap with DDP if running multi-GPU
    if is_distributed:
        model = DDP(model, device_ids=[local_rank])

    # -------------------------------------------------------------------------
    # Step 4: RevIN — per-instance normalisation.
    # Mirrors RevInCB(denorm=False) from patchtst_finetune.py.
    # Normalises each sample independently so the clustering module sees
    # z-scored channels rather than raw scale differences.
    # -------------------------------------------------------------------------
    revin = RevIN(n_features).to(device) if args.revin else None

    # -------------------------------------------------------------------------
    # Step 4b: Sinusoidal positional encoding [T, C] — fixed, no parameters.
    # Created once here and reused every forward pass. Deterministic: the same
    # T and C always produce the same tensor, so Phase 2 can reproduce it
    # exactly by calling sinusoidal_pos_enc with the same arguments.
    # -------------------------------------------------------------------------
    pos_enc = sinusoidal_pos_enc(args.context_points, n_features, device) \
              if args.use_pos_enc else None

    if rank == 0 and pos_enc is not None:
        print(f'Positional encoding: ON  (T={args.context_points}, C={n_features})')
    elif rank == 0:
        print('Positional encoding: OFF')

    # -------------------------------------------------------------------------
    # Step 5: Optimiser — Adam, same LR interface as patchtst_finetune.py.
    # -------------------------------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # -------------------------------------------------------------------------
    # Step 6: Wandb initialisation (rank 0 only).
    # -------------------------------------------------------------------------
    if args.use_wandb and rank == 0:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or save_name,
            config=vars(args)
        )

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    best_val_loss = float('inf')
    best_epoch = -1

    for epoch in range(args.n_epochs):

        # --- Anneal temperature: high at start (soft), low at end (sharp) ---
        tau = anneal_tau(epoch, args.n_epochs)
        # Access the patch learner whether model is wrapped in DDP or not
        _patch_learner = model.module.patch_learner if is_distributed else model.patch_learner
        _patch_learner.temperature = tau

        # =====================================================================
        # TRAINING PASS
        # =====================================================================
        model.train()
        train_losses = []

        for raw_batch in dls.train:
            # DataLoader yields dicts: {"inputs_embeds": [B,T,C], "label": [B]}
            if isinstance(raw_batch, dict):
                xb, yb = raw_batch["inputs_embeds"], raw_batch["label"]
            else:
                xb, yb = raw_batch  # fallback for tuple-style datasets

            # xb: [B, T, C]  (timesteps first — confirmed by RevInCB)
            # yb: [B]         binary labels
            xb = xb.to(device)
            yb = yb.to(device).float()

            # --- RevIN: normalise each instance independently ---
            # RevIN expects [B, T, C] and outputs [B, T, C], reducing over T.
            if revin is not None:
                xb = revin(xb, mode='norm')

            # --- Positional encoding: add fixed [T, C] to every sample ---
            if pos_enc is not None:
                xb = xb + pos_enc.unsqueeze(0)   # broadcast over batch

            # No transpose needed: SoftKMeansPatchLearner expects [B, T, C]
            x = xb   # [B, T, C]

            # --- Forward pass ---
            # model returns (logits [B,1], balancing_loss scalar)
            logits, bal_loss = model(x)

            # --- Reconcile shapes: logits [B,1] → [B] to match yb [B] ---
            logits = logits.squeeze(-1)

            # --- Total loss: classification BCE + entropy balancing ---
            cls_loss = loss_fn(logits, yb)
            total_loss = cls_loss + bal_loss

            # --- Backward + optimiser step ---
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            train_losses.append(cls_loss.item())  # track classification loss only (balancing loss is internal)

        avg_train_loss = np.mean(train_losses)

        # =====================================================================
        # VALIDATION PASS — no gradients, collect logits for ROC-AUC
        # =====================================================================
        model.eval()
        val_logits, val_labels = [], []
        val_losses = []

        with torch.no_grad():
            for raw_batch in dls.valid:
                if isinstance(raw_batch, dict):
                    xb, yb = raw_batch["inputs_embeds"], raw_batch["label"]
                else:
                    xb, yb = raw_batch

                xb = xb.to(device)
                yb = yb.to(device).float()

                if revin is not None:
                    xb = revin(xb, mode='norm')
                if pos_enc is not None:
                    xb = xb + pos_enc.unsqueeze(0)

                x = xb  # [B, T, C]
                logits, bal_loss = model(x)
                logits = logits.squeeze(-1)

                cls_loss = loss_fn(logits, yb)
                val_losses.append(cls_loss.item())  # track classification loss only (balancing loss is internal)

                val_logits.append(logits.detach().cpu())
                val_labels.append(yb.detach().cpu())

        avg_val_loss = np.mean(val_losses)

        # --- Compute ROC-AUC across all ranks (all_gather) ---
        val_roc = compute_roc_auc_distributed(val_logits, val_labels)

        # --- Count active clusters (centroids with mean assignment > threshold) ---
        # This tells us the effective K — how many centroids are actually being used.
        # Computed on a single batch for efficiency; logged for monitoring.
        with torch.no_grad():
            sample_raw = next(iter(dls.valid))
            if isinstance(sample_raw, dict):
                sample_xb = sample_raw["inputs_embeds"]
            else:
                sample_xb = sample_raw[0]
            sample_xb = sample_xb.to(device)
            if revin is not None:
                sample_xb = revin(sample_xb, mode='norm')
            if pos_enc is not None:
                sample_xb = sample_xb + pos_enc.unsqueeze(0)
            sample_x = sample_xb  # [B, T, C]
            _, sample_assignments = (_patch_learner)(sample_x)
            # marginal[k] = mean assignment probability for centroid k
            marginal = sample_assignments.mean(dim=(0, 1)).cpu()
            # Active centroid = any centroid whose marginal > 1/(K*10) threshold
            active_k = int((marginal > (1.0 / (args.n_patches * 10))).sum().item())

        if rank == 0:
            print(
                f'Epoch {epoch:3d} | tau={tau:.3f} | '
                f'train_loss={avg_train_loss:.6f} | val_loss={avg_val_loss:.6f} | '
                f'val_ROC-AUC={val_roc:.4f} | active_K={active_k}/{args.n_patches}'
            )

            # --- Save on best validation ROC-AUC ---
            # ROC-AUC is the right criterion here (not val_loss) because:
            #   1. Class imbalance makes absolute loss values misleading
            #   2. We care about ranking ability (does the model distinguish
            #      flu-positive from flu-negative windows?)
            #   3. val_loss includes the balancing term which is a regulariser,
            #      not a direct performance signal
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                torch.save(model.state_dict() if not is_distributed
                           else model.module.state_dict(), save_path)
                print(f'  → New best! Saved to {save_path}')

            # --- Wandb logging ---
            if args.use_wandb:
                import wandb
                wandb.log({
                    'epoch': epoch,
                    'tau': tau,
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'val_roc_auc': val_roc,
                    'active_K': active_k,
                }, step=epoch)

    # -------------------------------------------------------------------------
    # End of training summary
    # -------------------------------------------------------------------------
    if rank == 0:
        print(f'\nTraining complete. Best val loss={best_val_loss:.6f} at epoch {best_epoch}.')
        print(f'Saved model: {save_path}')

        # ---------------------------------------------------------------------
        # Post-training: reload best checkpoint, compute effective K over the
        # full training set, and save the patch learner separately for Phase 2.
        # ---------------------------------------------------------------------

        # Reload best weights into a plain (non-DDP) model at sharpest tau
        best_model = PatchLearnerClassifier(
            n_features=n_features,
            n_patches=args.n_patches,
            temperature=args.end_tau,
            balancing_lambda=args.balancing_lambda,
            n_classes=1,
        ).to(device)
        best_model.load_state_dict(torch.load(save_path, map_location=device))
        best_model.eval()

        # Accumulate marginal centroid usage over the full training set.
        # A single batch (used during the loop) is not reliable enough for
        # deciding which centroids are dead before Phase 2 discretization.
        marginal_accum = torch.zeros(args.n_patches)
        n_batches = 0
        with torch.no_grad():
            for raw_batch in dls.train:
                if isinstance(raw_batch, dict):
                    xb = raw_batch["inputs_embeds"]
                else:
                    xb = raw_batch[0]
                xb = xb.to(device)
                if revin is not None:
                    xb = revin(xb, mode='norm')
                if pos_enc is not None:
                    xb = xb + pos_enc.unsqueeze(0)
                _, assignments = best_model.patch_learner(xb)
                marginal_accum += assignments.mean(dim=(0, 1)).cpu()
                n_batches += 1

        marginal = marginal_accum / n_batches          # [K] mean assignment prob
        threshold = 1.0 / (args.n_patches * 10)
        active_mask = marginal > threshold
        effective_k = int(active_mask.sum().item())
        active_ids = active_mask.nonzero(as_tuple=True)[0].tolist()

        print(f'\nEffective K (active centroids): {effective_k} / {args.n_patches}')
        print(f'Active centroid IDs (first 20): {active_ids[:20]}{"..." if len(active_ids) > 20 else ""}')

        # Save just the patch learner state dict — this is what Phase 2 loads.
        # The classifier head is discarded; only the centroids are needed.
        patch_learner_path = save_path.replace('.pth', '_patch_learner.pth')
        torch.save(best_model.patch_learner.state_dict(), patch_learner_path)
        print(f'Patch learner saved to: {patch_learner_path}')

        # Save Phase 2 metadata: effective K, active centroid IDs, threshold used
        import json
        meta = {
            'effective_k': effective_k,
            'n_patches_total': args.n_patches,
            'active_centroid_ids': active_ids,
            'marginal_threshold': threshold,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'end_tau': args.end_tau,   # final temperature — Phase 2 reads this
            'use_pos_enc': bool(args.use_pos_enc),
            'context_points': args.context_points,
            'n_features': n_features,
        }
        meta_path = save_path.replace('.pth', '_meta.json')
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f'Phase 2 metadata saved to: {meta_path}')

        # ---------------------------------------------------------------------
        # Diagnostic: patch size distribution over the full training set.
        #
        # Run one more pass using hard_assignments (with active_centroid_ids)
        # to count how many timesteps each active centroid captures per sample.
        # This tells us the layout before Phase 2 — if the distribution is
        # wildly skewed (one centroid owns 80% of timesteps), the clustering
        # did not spread well and Phase 1 may need retuning (higher lambda,
        # more epochs, or a different K).
        # ---------------------------------------------------------------------
        size_accum = torch.zeros(effective_k)       # [K_eff] total timestep counts
        size_sq_accum = torch.zeros(effective_k)    # for per-centroid std
        per_sample_min = torch.full((effective_k,), float('inf'))
        per_sample_max = torch.zeros(effective_k)
        n_samples_diag = 0
        coverage_ok = True

        with torch.no_grad():
            for raw_batch in dls.train:
                if isinstance(raw_batch, dict):
                    xb = raw_batch["inputs_embeds"]
                else:
                    xb = raw_batch[0]
                xb = xb.to(device)
                if revin is not None:
                    xb = revin(xb, mode='norm')
                if pos_enc is not None:
                    xb = xb + pos_enc.unsqueeze(0)

                # patch_ids: [B, T] values in [0, K_eff-1] — every timestep
                # is guaranteed one assignment (see hard_assignments docstring)
                patch_ids = best_model.patch_learner.hard_assignments(
                    xb, active_centroid_ids=active_ids
                )
                B, T = patch_ids.shape

                # Count timesteps per centroid per sample: [B, K_eff]
                counts = torch.zeros(B, effective_k, device=device)
                counts.scatter_add_(
                    1, patch_ids,
                    torch.ones(B, T, device=device)
                )

                # Coverage check: every timestep should be accounted for
                if not (counts.sum(dim=1) == T).all():
                    coverage_ok = False

                counts_cpu = counts.cpu()
                size_accum    += counts_cpu.sum(dim=0)          # [K_eff]
                size_sq_accum += (counts_cpu ** 2).sum(dim=0)
                per_sample_min = torch.minimum(per_sample_min,
                                               counts_cpu.min(dim=0).values)
                per_sample_max = torch.maximum(per_sample_max,
                                               counts_cpu.max(dim=0).values)
                n_samples_diag += B

        # Mean and std of patch size per centroid across all training samples
        mean_sizes = size_accum / n_samples_diag               # [K_eff]
        std_sizes  = (size_sq_accum / n_samples_diag
                      - mean_sizes ** 2).clamp(min=0).sqrt()   # [K_eff]

        # Distribution of mean patch sizes across centroids
        sizes_sorted, _ = mean_sizes.sort(descending=True)
        top5_vals,  top5_idx  = mean_sizes.topk(min(5, effective_k))
        bot5_vals,  bot5_idx  = mean_sizes.topk(min(5, effective_k), largest=False)

        print('\n' + '=' * 60)
        print('  PHASE 1 — CLUSTER LAYOUT DIAGNOSTICS')
        print('=' * 60)
        print(f'  Total centroids K            : {args.n_patches}')
        print(f'  Active centroids (effective K): {effective_k}')
        print(f'  Dead centroids filtered       : {args.n_patches - effective_k}')
        print(f'  Marginal threshold used       : {threshold:.5f}  (1 / K*10)')
        print(f'  Coverage (all timesteps hit)  : {"PASS" if coverage_ok else "FAIL — check hard_assignments"}')
        print()
        print(f'  Context window length T       : {args.context_points}')
        print(f'  Expected uniform patch size   : {args.context_points / effective_k:.1f}  (T / K_eff)')
        print()
        print('  Patch size distribution (mean timesteps per centroid):')
        print(f'    Mean across centroids : {mean_sizes.mean():.1f}')
        print(f'    Std  across centroids : {mean_sizes.std():.1f}')
        print(f'    Median                : {mean_sizes.median():.1f}')
        print(f'    Min  centroid (idx {bot5_idx[0].item():3d}): {mean_sizes.min():.1f} timesteps'
              f'  (per-sample range: {per_sample_min[bot5_idx[0]].item():.0f}–{per_sample_max[bot5_idx[0]].item():.0f})')
        print(f'    Max  centroid (idx {top5_idx[0].item():3d}): {mean_sizes.max():.1f} timesteps'
              f'  (per-sample range: {per_sample_min[top5_idx[0]].item():.0f}–{per_sample_max[top5_idx[0]].item():.0f})')
        print()
        print(f'  Top 5 largest patches (active_k index → mean timesteps ± std):')
        for i, (val, idx) in enumerate(zip(top5_vals, top5_idx)):
            orig_id = active_ids[idx.item()]
            print(f'    #{i+1}: active[{idx.item():3d}] orig_centroid[{orig_id:3d}]'
                  f' → {val:.1f} ± {std_sizes[idx].item():.1f} timesteps')
        print(f'  Bottom 5 smallest patches:')
        for i, (val, idx) in enumerate(zip(bot5_vals, bot5_idx)):
            orig_id = active_ids[idx.item()]
            print(f'    #{i+1}: active[{idx.item():3d}] orig_centroid[{orig_id:3d}]'
                  f' → {val:.1f} ± {std_sizes[idx].item():.1f} timesteps')
        print()

        # Rough imbalance ratio: largest / smallest patch
        imbalance = mean_sizes.max() / (mean_sizes.min() + 1e-8)
        print(f'  Imbalance ratio (max/min)     : {imbalance:.1f}x')
        if imbalance > 20:
            print('  WARNING: high imbalance — consider increasing --balancing_lambda'
                  ' or --n_epochs to encourage more even centroid usage.')
        elif imbalance < 3:
            print('  NOTE: very even distribution — patches are well-balanced.')
        print('=' * 60)

        if args.use_wandb:
            import wandb
            wandb.finish()

    if is_distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
