"""
dual_encoder_probe.py — Dual-encoder MLP linear probe on frozen PatchTST backbones.

Architecture
------------
Two pretrained PatchTST backbones (positive-only trained and negative-only trained)
are loaded and frozen.  For each input window both backbones produce an embedding:

    backbone → [B x nvars x d_model x num_patch]
             → mean over patches          → [B x nvars x d_model]
             → reshape                    → [B x nvars * d_model]

The two embeddings are concatenated → [B x 2*nvars*d_model] and fed into a small
MLP classifier whose depth is controlled by --n_hidden_layers:

    n_hidden_layers = 0  →  Linear(input_dim, 1)               (true linear probe)
    n_hidden_layers = 2  →  Linear → ReLU → Dropout  ×2  → Linear(1)

Embeddings are extracted ONCE from both frozen backbones and cached in RAM,
so each training epoch only runs the lightweight MLP (no repeated backbone passes).

Supports DDP (torchrun --nproc_per_node=2):
  - Each rank extracts embeddings independently from its own GPU copy of the backbones
  - LR finder runs on rank 0 only; result is broadcast to all ranks
  - MLP training uses DistributedSampler + DDP wrapper
  - Validation all-gathers predictions across ranks so metrics are computed on full val set
  - Model saving and test evaluation run on rank 0 only

Mirrors patchtst_finetune.py exactly on:
  - Argument parser style and naming
  - DDP detection and init pattern
  - BCEWithLogitsLoss + pos_weight (capped)
  - LR finder (LR range test) with broadcast pattern
  - Cosine annealing scheduler
  - Generalization-aware model saving (skip save when val_loss > train_loss)
  - PR-AUC as primary metric
  - PR-optimal threshold from val applied blind to test
  - Full confusion matrix, sensitivity, specificity, F1, F2, MCC, balanced accuracy
  - PR + ROC curve plots saved as PNG
  - Metrics CSV
  - W&B logging (optional)

Usage (single GPU)
------------------
python dual_encoder_probe.py \
    --positive_model /path/to/positive.pth \
    --negative_model /path/to/negative.pth \
    --n_hidden_layers 2 --hidden_dim 256 --n_epochs_finetune 50

Usage (2 GPUs via torchrun)
---------------------------
torchrun --nproc_per_node=2 dual_encoder_probe.py \
    --positive_model /path/to/positive.pth \
    --negative_model /path/to/negative.pth \
    --n_hidden_layers 2 --hidden_dim 256 --n_epochs_finetune 50
"""

import numpy as np
import pandas as pd
import os
import sys
import argparse

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score,
    recall_score, f1_score, fbeta_score, matthews_corrcoef,
    balanced_accuracy_score, confusion_matrix, roc_curve,
    precision_recall_curve,
)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
_PROJECT_ROOT = "/home/hice1/ezg6/projects/Homekit2020/src"
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.models.patchTST import PatchTST
from src.models.layers.revin import RevIN
from src.callback.patch_mask import create_patch
from datautils import get_dls


# ------------------------------------------------------------------
# Argument parser — mirrors patchtst_finetune.py style
# ------------------------------------------------------------------
parser = argparse.ArgumentParser()

# --- Backbone checkpoints ---
parser.add_argument('--positive_model', type=str, required=True,
                    help='Path to the positive-class pretrained .pth checkpoint')
parser.add_argument('--negative_model', type=str, required=True,
                    help='Path to the negative-class pretrained .pth checkpoint')

# --- MLP architecture ---
parser.add_argument('--n_hidden_layers', type=int, default=2,
                    help='Number of hidden layers in the MLP (0 = true linear probe)')
parser.add_argument('--hidden_dim', type=int, default=256,
                    help='Width of each hidden layer')
parser.add_argument('--mlp_dropout', type=float, default=0.2,
                    help='Dropout after each hidden layer')

# --- Backbone architecture (must match checkpoints) ---
parser.add_argument('--context_points', type=int, default=10080)
parser.add_argument('--patch_len',      type=int, default=1440)
parser.add_argument('--stride',         type=int, default=180)
parser.add_argument('--c_in',           type=int, default=8)
parser.add_argument('--d_model',        type=int, default=256)
parser.add_argument('--n_layers',       type=int, default=4)
parser.add_argument('--n_heads',        type=int, default=8)
parser.add_argument('--d_ff',           type=int, default=512)
parser.add_argument('--dropout',        type=float, default=0.1)
parser.add_argument('--head_dropout',   type=float, default=0.1)
parser.add_argument('--revin',          type=int, default=1)

# --- Data ---
parser.add_argument('--batch_size',           type=int, default=128,
                    help='Batch size per GPU for MLP training')
parser.add_argument('--backbone_batch_size',  type=int, default=64,
                    help='Batch size for backbone embedding extraction')
parser.add_argument('--num_workers',          type=int, default=6)
parser.add_argument('--dset',                 type=str, default='Wearable')

# --- Training ---
parser.add_argument('--n_epochs_finetune', type=int, default=50,
                    help='Number of MLP training epochs')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='Learning rate (ignored when --use_lr_finder=1)')
parser.add_argument('--use_lr_finder', type=int, default=1,
                    help='Run LR range test before training (1=yes, 0=use --lr directly)')
parser.add_argument('--lr_find_steps', type=int, default=200,
                    help='Mini-batch steps for the LR range test')
parser.add_argument('--use_scheduler', type=int, default=1,
                    help='Cosine annealing LR scheduler (1=yes)')
parser.add_argument('--pos_weight_cap', type=float, default=-1.0,
                    help='Cap for BCEWithLogitsLoss pos_weight; -1 = no cap')

# --- Save / identify ---
parser.add_argument('--model_type',        type=str, default='dual_encoder',
                    help='Sub-directory name under saved_models/.../masked_patchtst/')
parser.add_argument('--finetuned_model_id', type=int, default=1)

# --- W&B (mirrors patchtst_finetune.py) ---
parser.add_argument('--use_wandb',      type=int, default=0,
                    help='Enable Weights & Biases logging (1=yes, 0=no)')
parser.add_argument('--wandb_project',  type=str, default='PatchTST-Wearable')
parser.add_argument('--wandb_run_name', type=str, default=None)

args = parser.parse_args()


# ------------------------------------------------------------------
# Distributed detection — mirrors patchtst_finetune.py exactly
# ------------------------------------------------------------------
is_distributed = "LOCAL_RANK" in os.environ
local_rank = int(os.environ.get("LOCAL_RANK", 0))
rank       = int(os.environ.get("RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))

if is_distributed:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Distributed run requested but torch.cuda.is_available() is False. "
            "Run on an NVIDIA CUDA node."
        )
    torch.cuda.set_device(local_rank)
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")

    import time as _t
    _t0 = _t.time()
    print(f"[rank {rank}] CUDA warmup start...", flush=True)
    torch.zeros(1, device=torch.device(f"cuda:{local_rank}"))
    torch.cuda.synchronize()
    print(f"[rank {rank}] CUDA warmup done ({_t.time() - _t0:.1f}s)", flush=True)

device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

# ------------------------------------------------------------------
# Save path — mirrors patchtst_finetune.py exactly
# ------------------------------------------------------------------
args.save_path = 'saved_models/' + args.dset + '/masked_patchtst/' + args.model_type + '/'
if rank == 0:
    os.makedirs(args.save_path, exist_ok=True)

suffix_name = (
    '_cw' + str(args.context_points)
    + '_patch' + str(args.patch_len)
    + '_stride' + str(args.stride)
    + '_hl' + str(args.n_hidden_layers)
    + '_hd' + str(args.hidden_dim)
    + '_epochs' + str(args.n_epochs_finetune)
    + '_model' + str(args.finetuned_model_id)
)
args.save_finetuned_model = args.dset + '_dual_encoder_probe' + suffix_name

if rank == 0:
    print('args:', args)
    print(f'Device: {device}  |  world_size: {world_size}')


# ------------------------------------------------------------------
# pos_weight — mirrors patchtst_finetune.py exactly
# ------------------------------------------------------------------
def compute_pos_weight(labels, device):
    """Compute BCEWithLogitsLoss pos_weight from label tensor, capped by args.pos_weight_cap."""
    n_pos = int((labels == 1).sum().item())
    n_neg = int((labels == 0).sum().item())
    raw_ratio = n_neg / n_pos
    capped = raw_ratio if args.pos_weight_cap < 0 else min(raw_ratio, args.pos_weight_cap)
    if rank == 0:
        print(f'pos_weight: raw={raw_ratio:.1f}, cap={args.pos_weight_cap}, using={capped:.1f}  '
              f'(pos={n_pos}, neg={n_neg})')
    return torch.tensor([capped], dtype=torch.float32, device=device)


# ------------------------------------------------------------------
# Backbone loading — frozen
# ------------------------------------------------------------------
def load_backbone(ckpt_path):
    num_patch = (max(args.context_points, args.patch_len) - args.patch_len) // args.stride + 1
    model = PatchTST(
        c_in=args.c_in,
        target_dim=args.patch_len,
        patch_len=args.patch_len,
        stride=args.stride,
        num_patch=num_patch,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_model=args.d_model,
        shared_embedding=True,
        d_ff=args.d_ff,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        act='relu',
        head_type='pretrain',
        res_attention=False,
    )
    state = torch.load(ckpt_path, map_location='cpu')
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    model.load_state_dict(state, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(device)
    if rank == 0:
        print(f'  Loaded (frozen): {ckpt_path}')
        print(f'  Backbone params: {sum(p.numel() for p in model.parameters()):,}  (all frozen)')
    return model


def _register_embedding_hook(model):
    """Mean-pool patches → flatten → [B x nvars*d_model]."""
    captured = []

    def _pre_hook(module, inp):
        x = inp[0].detach().cpu()       # [B x nvars x d_model x num_patch]
        x = x.mean(dim=-1)              # [B x nvars x d_model]
        x = x.reshape(x.shape[0], -1)  # [B x nvars * d_model]
        captured.append(x)

    handle = model.head.register_forward_pre_hook(_pre_hook)
    return captured, handle


# ------------------------------------------------------------------
# Embedding extraction — cached once, reused every epoch
# Each rank extracts independently (backbones are frozen/deterministic)
# ------------------------------------------------------------------
@torch.no_grad()
def _extract_split(model, dataloader, revin):
    captured, hook = _register_embedding_hook(model)
    all_emb, all_lbl = [], []

    for batch in dataloader:
        if isinstance(batch, dict):
            xb = batch['inputs_embeds'].float().to(device)
            yb = batch['label'].float()
        else:
            xb = batch[0].float().to(device)
            yb = batch[1].float()

        if args.revin:
            xb = revin(xb, 'norm')

        xb_patch, _ = create_patch(xb, args.patch_len, args.stride)
        _ = model(xb_patch)

        all_emb.append(captured.pop())
        all_lbl.append(yb.reshape(-1).cpu())

    hook.remove()
    return torch.cat(all_emb, dim=0), torch.cat(all_lbl, dim=0)


def extract_all_embeddings(pos_model, neg_model, dls, revin):
    """
    Extract and cache embeddings from both frozen backbones for all splits.
    Returns {'train': (emb, lbl), 'val': (emb, lbl), 'test': (emb, lbl)}
    where emb shape is [N x 2*nvars*d_model].
    """
    split_map = {'train': dls.train, 'val': dls.valid, 'test': dls.test}
    result = {}

    for split_name, dl in split_map.items():
        if dl is None:
            continue
        if rank == 0:
            print(f'  [{split_name}] extracting positive model …')
        emb_pos, lbl = _extract_split(pos_model, dl, revin)
        if rank == 0:
            print(f'  [{split_name}] extracting negative model …')
        emb_neg, _   = _extract_split(neg_model, dl, revin)
        emb = torch.cat([emb_pos, emb_neg], dim=1)   # [N x 2*nvars*d_model]
        result[split_name] = (emb, lbl)
        if rank == 0:
            n_pos = int((lbl == 1).sum())
            print(f'    {split_name}: {len(lbl)} samples  '
                  f'(pos={n_pos}, neg={len(lbl)-n_pos})  '
                  f'emb_dim={emb.shape[1]}')

    return result


# ------------------------------------------------------------------
# MLP classifier
# ------------------------------------------------------------------
class DualEncoderMLP(nn.Module):
    """
    MLP on top of concatenated dual-backbone embeddings.

    n_hidden_layers=0  →  single Linear (true linear probe)
    n_hidden_layers>0  →  n × (Linear → ReLU → Dropout) → Linear(1)
    """

    def __init__(self, input_dim, hidden_dim, n_hidden_layers, dropout):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(n_hidden_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)   # [B]


# ------------------------------------------------------------------
# LR range test — rank 0 only; caller broadcasts result
# ------------------------------------------------------------------
def find_lr(emb_train, lbl_train, input_dim):
    """
    Sweep LR exponentially from 1e-7 to 1.0 over lr_find_steps mini-batches.
    Returns the LR at the point of steepest loss descent.
    Runs on rank 0 only — caller is responsible for broadcasting.
    """
    start_lr = 1e-7
    end_lr   = 1.0
    steps    = args.lr_find_steps

    model = DualEncoderMLP(input_dim, args.hidden_dim, args.n_hidden_layers,
                           args.mlp_dropout).to(device)
    pos_weight = compute_pos_weight(lbl_train, device)
    loss_fn    = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt        = torch.optim.Adam(model.parameters(), lr=start_lr)
    mult       = (end_lr / start_lr) ** (1.0 / steps)
    sched      = torch.optim.lr_scheduler.MultiplicativeLR(opt, lr_lambda=lambda _: mult)

    dl      = DataLoader(TensorDataset(emb_train, lbl_train),
                         batch_size=args.batch_size, shuffle=True)
    dl_iter = iter(dl)

    lrs, losses = [], []
    best_loss = float('inf')
    avg_loss  = 0.0
    beta      = 0.98

    model.train()
    for step in range(steps):
        try:
            xb, yb = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dl)
            xb, yb = next(dl_iter)

        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()

        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smooth   = avg_loss / (1 - beta ** (step + 1))
        current_lr = opt.param_groups[0]['lr']
        lrs.append(current_lr)
        losses.append(smooth)

        if smooth < best_loss:
            best_loss = smooth
        if smooth > 4 * best_loss:
            break

        sched.step()

    skip     = max(1, len(losses) // 10)
    grads    = np.gradient(losses[skip:])
    best_idx = int(np.argmin(grads)) + skip
    suggested_lr = lrs[best_idx]

    print(f'LR range test done ({len(lrs)} steps) — suggested_lr {suggested_lr:.2e}')
    return suggested_lr


# ------------------------------------------------------------------
# save_recorders — mirrors patchtst_finetune.py
# ------------------------------------------------------------------
def save_recorders(train_losses, val_losses, val_pr_aucs, val_roc_aucs, lrs):
    pd.DataFrame({
        'train_loss':  train_losses,
        'valid_loss':  val_losses,
        'valid_pr_auc':  val_pr_aucs,
        'valid_roc_auc': val_roc_aucs,
        'lr':          lrs,
    }).to_csv(
        args.save_path + args.save_finetuned_model + '_losses.csv',
        float_format='%.6f', index=False,
    )


# ------------------------------------------------------------------
# Linear probe training — mirrors linear_probe_func in patchtst_finetune.py
# ------------------------------------------------------------------
def linear_probe_func(lr=args.lr):
    if rank == 0:
        print('dual-encoder linear probe')

    emb_train, lbl_train = split_data['train']
    emb_val,   lbl_val   = split_data['val']

    input_dim = emb_train.shape[1]
    model = DualEncoderMLP(input_dim, args.hidden_dim, args.n_hidden_layers,
                           args.mlp_dropout).to(device)

    if is_distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    if rank == 0:
        base = model.module if is_distributed else model
        n_params = sum(p.numel() for p in base.parameters() if p.requires_grad)
        print(f'number of MLP params: {n_params:,}')
        print(f'MLP: input_dim={input_dim}  hidden_dim={args.hidden_dim}  '
              f'n_hidden_layers={args.n_hidden_layers}  mlp_dropout={args.mlp_dropout}')

    pos_weight = compute_pos_weight(lbl_train, device)
    loss_fn    = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimiser  = torch.optim.Adam(model.parameters(), lr=lr)

    if args.use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=args.n_epochs_finetune, eta_min=lr / 100
        )
        if rank == 0:
            print(f'Cosine annealing: lr {lr:.2e} → {lr/100:.2e} over {args.n_epochs_finetune} epochs')
    else:
        scheduler = None

    # DDP: each rank gets its own shard of the training set
    if is_distributed:
        train_sampler = DistributedSampler(TensorDataset(emb_train, lbl_train),
                                           num_replicas=world_size, rank=rank, shuffle=True)
        train_dl = DataLoader(TensorDataset(emb_train, lbl_train),
                              batch_size=args.batch_size, sampler=train_sampler,
                              num_workers=0, pin_memory=True)
    else:
        train_dl = DataLoader(TensorDataset(emb_train, lbl_train),
                              batch_size=args.batch_size, shuffle=True)

    val_dl = DataLoader(TensorDataset(emb_val, lbl_val),
                        batch_size=args.batch_size, shuffle=False)

    # W&B init (rank 0 only)
    if args.use_wandb and rank == 0:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or args.save_finetuned_model,
            config=vars(args),
        )

    best_val_loss = float('inf')
    train_losses, val_losses, val_pr_aucs, val_roc_aucs, lrs_tracked = [], [], [], [], []

    if rank == 0:
        print(f'\n  {"Epoch":>6}  {"train_loss":>12}  {"valid_loss":>12}  '
              f'{"valid_PR-AUC":>14}  {"valid_ROC-AUC":>15}  {"lr":>10}')

    for epoch in range(1, args.n_epochs_finetune + 1):

        if is_distributed:
            train_sampler.set_epoch(epoch)

        # ── Train ──
        model.train()
        ep_loss = 0.0
        ep_n    = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimiser.step()
            ep_loss += loss.item() * len(yb)
            ep_n    += len(yb)

        # Aggregate train loss across ranks
        if is_distributed:
            ep_loss_t = torch.tensor([ep_loss, float(ep_n)], device=device)
            dist.all_reduce(ep_loss_t, op=dist.ReduceOp.SUM)
            train_loss = ep_loss_t[0].item() / ep_loss_t[1].item()
        else:
            train_loss = ep_loss / len(lbl_train)

        # ── Validate — each rank runs its full val set, then all-gather ──
        model.eval()
        vl_loss   = 0.0
        all_preds, all_tgts = [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred    = model(xb)
                vl_loss += loss_fn(pred, yb).item() * len(yb)
                all_preds.append(torch.sigmoid(pred).cpu())
                all_tgts.append(yb.cpu())

        local_probs   = torch.cat(all_preds).numpy()
        local_targets = torch.cat(all_tgts).numpy().astype(int)
        local_val_loss = vl_loss / len(local_targets)

        # In DDP: gather from all ranks so metrics are on the full val set
        if is_distributed:
            gathered_probs   = [None] * world_size
            gathered_targets = [None] * world_size
            gathered_losses  = [None] * world_size
            dist.all_gather_object(gathered_probs,   local_probs)
            dist.all_gather_object(gathered_targets, local_targets)
            dist.all_gather_object(gathered_losses,  local_val_loss)
            probs   = np.concatenate(gathered_probs)
            targets = np.concatenate(gathered_targets).astype(int)
            val_loss = float(np.mean(gathered_losses))
        else:
            probs   = local_probs
            targets = local_targets
            val_loss = local_val_loss

        if len(np.unique(targets)) >= 2:
            pr_auc  = average_precision_score(targets, probs)
            roc_auc = roc_auc_score(targets, probs)
        else:
            pr_auc = roc_auc = float('nan')

        current_lr = optimiser.param_groups[0]['lr']
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_pr_aucs.append(pr_auc)
        val_roc_aucs.append(roc_auc)
        lrs_tracked.append(current_lr)

        if rank == 0:
            print(f'  {epoch:>6}  {train_loss:>12.6f}  {val_loss:>12.6f}  '
                  f'{pr_auc:>14.4f}  {roc_auc:>15.4f}  {current_lr:>10.2e}')
            print(f'  [val ROC-AUC: {roc_auc:.4f}  |  val PR-AUC: {pr_auc:.4f}]')

        if args.use_wandb and rank == 0:
            import wandb
            wandb.log({
                'train_loss': train_loss, 'valid_loss': val_loss,
                'valid_pr_auc': pr_auc, 'valid_roc_auc': roc_auc,
                'lr': current_lr, 'epoch': epoch,
            })

        # ── Generalization-aware save (rank 0 only) ──
        if rank == 0 and val_loss < best_val_loss:
            base = model.module if is_distributed else model
            if val_loss <= train_loss:
                best_val_loss = val_loss
                torch.save(base.state_dict(),
                           args.save_path + args.save_finetuned_model + '.pth')
                print(f'  Better model found at epoch {epoch} with valid_loss: {val_loss:.6f}')
            else:
                print(f'  Epoch {epoch}: valid_loss ({val_loss:.6f}) > '
                      f'train_loss ({train_loss:.6f}) — skipping save (overfitting guard)')

        if scheduler is not None:
            scheduler.step()

    if rank == 0:
        save_recorders(train_losses, val_losses, val_pr_aucs, val_roc_aucs, lrs_tracked)

    if args.use_wandb and rank == 0:
        import wandb
        wandb.finish()


# ------------------------------------------------------------------
# Metrics helpers — verbatim from patchtst_finetune.py
# ------------------------------------------------------------------
def _metrics_at_threshold(y_true, probs, threshold):
    """
    Compute the full suite of imbalance-aware classification metrics at a
    given decision threshold.  Returns a flat dict of name → value.

    WHY THESE METRICS (for highly imbalanced flu detection, ~0.04% prevalence):

    Sensitivity (Recall/TPR) = TP / (TP + FN)
        Fraction of true flu cases the model actually caught.
        MOST IMPORTANT here — missing a flu onset is dangerous.
        A model predicting all-negative gets 0.0.

    Specificity (TNR) = TN / (TN + FP)
        Fraction of healthy windows correctly identified as healthy.
        Almost always high with extreme imbalance — do not rely on this alone.

    Precision (PPV) = TP / (TP + FP)
        Of all windows flagged as flu, how many truly are?
        Low precision = many false alarms. Trades off with sensitivity.

    NPV (Negative Predictive Value) = TN / (TN + FN)
        Of all windows flagged as healthy, how many truly are?
        Always near 1.0 when prevalence is low — less informative.

    F1 = 2 * precision * recall / (precision + recall)
        Harmonic mean of precision and recall. Balanced view.

    F2 = (5 * precision * recall) / (4 * precision + recall)
        Like F1 but weights recall 2x more than precision.
        RIGHT choice for flu detection: missing a case is worse than a false alarm.

    MCC (Matthews Correlation Coefficient) — range [-1, 1]
        BEST SINGLE METRIC for imbalanced data. Accounts for all four
        confusion matrix cells. 0 = no better than random, 1 = perfect.
        Unlike accuracy/F1, not inflated by the dominant negative class.

    Balanced Accuracy = (sensitivity + specificity) / 2
        Accuracy corrected for imbalance. 0.5 = random, 1.0 = perfect.
        Not inflated by the large negative class.
    """
    y_pred = (probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv         = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    f1      = f1_score(y_true, y_pred, zero_division=0)
    f2      = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    mcc     = matthews_corrcoef(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    return dict(
        TP=int(tp), FP=int(fp), TN=int(tn), FN=int(fn),
        sensitivity=sensitivity,
        specificity=specificity,
        precision=precision,
        npv=npv,
        f1=f1,
        f2=f2,
        mcc=mcc,
        balanced_accuracy=bal_acc,
    )


def _interpret_sensitivity(s, n_pos):
    """Plain-English interpretation of sensitivity for flu detection."""
    caught = round(s * n_pos)
    missed = n_pos - caught
    if s == 0.0:
        return 'catching NO flu cases — model predicts all negative at this threshold'
    elif s < 0.3:
        return f'catching only {caught}/{n_pos} flu cases — missing {missed} ({(1-s)*100:.0f}%) — very poor for clinical use'
    elif s < 0.6:
        return f'catching {caught}/{n_pos} flu cases — missing {missed} ({(1-s)*100:.0f}%) — moderate'
    elif s < 0.8:
        return f'catching {caught}/{n_pos} flu cases — missing {missed} ({(1-s)*100:.0f}%) — reasonable'
    else:
        return f'catching {caught}/{n_pos} flu cases — missing only {missed} ({(1-s)*100:.0f}%) — strong'


def _interpret_mcc(mcc):
    """Plain-English interpretation of MCC."""
    if mcc <= 0.0:
        return 'no useful signal (≤ random)'
    elif mcc < 0.2:
        return 'weak signal'
    elif mcc < 0.4:
        return 'moderate signal'
    elif mcc < 0.6:
        return 'good signal'
    else:
        return 'strong signal'


def _interpret_pr_auc(pr_auc, prevalence):
    """Compare PR-AUC to random baseline (= prevalence)."""
    if prevalence <= 0:
        return ''
    lift = pr_auc / prevalence
    if lift < 2:
        return f'{lift:.1f}x above random — poor'
    elif lift < 10:
        return f'{lift:.1f}x above random — moderate'
    elif lift < 50:
        return f'{lift:.1f}x above random — good'
    else:
        return f'{lift:.1f}x above random — excellent'


def _find_pr_optimal_threshold(y_true, probs):
    """Find the threshold that maximises F1 on the Precision-Recall curve."""
    prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_true, probs)
    denom  = prec_arr[:-1] + rec_arr[:-1]
    f1_arr = np.where(denom > 0, 2 * prec_arr[:-1] * rec_arr[:-1] / denom, 0.0)
    best_idx = int(np.argmax(f1_arr))
    return float(thresh_arr[best_idx]), float(f1_arr[best_idx]), prec_arr, rec_arr, thresh_arr


def _save_curves(save_path, model_name, train_data, val_data, test_data,
                 pr_opt_thresh_train, pr_opt_thresh_val, youden_thresh_val):
    """Plot and save PR + ROC curves — verbatim from patchtst_finetune.py."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = {'train': '#1f77b4', 'val': '#ff7f0e', 'test': '#2ca02c'}

    ax_pr = axes[0]
    for y_true, probs, label in [train_data, val_data, test_data]:
        if len(np.unique(y_true)) < 2:
            continue
        prec, rec, _ = precision_recall_curve(y_true, probs)
        auc_val = average_precision_score(y_true, probs)
        ax_pr.plot(rec, prec, color=colors[label], lw=1.8,
                   label=f'{label}  (PR-AUC={auc_val:.3f})')

    val_true, val_probs, _ = val_data
    if len(np.unique(val_true)) >= 2:
        vp = np.array(val_probs)
        vp_pred = (vp >= pr_opt_thresh_val).astype(int)
        tp = ((vp_pred == 1) & (val_true == 1)).sum()
        fp = ((vp_pred == 1) & (val_true == 0)).sum()
        fn = ((vp_pred == 0) & (val_true == 1)).sum()
        p_pt = tp / (tp + fp) if (tp + fp) > 0 else 0
        r_pt = tp / (tp + fn) if (tp + fn) > 0 else 0
        ax_pr.scatter([r_pt], [p_pt], color=colors['val'], s=120, zorder=5,
                      marker='*', label=f'val PR-opt thresh={pr_opt_thresh_val:.3f}')

    test_true, _, _ = test_data
    prevalence = test_true.mean()
    ax_pr.axhline(prevalence, color='gray', lw=1, ls='--',
                  label=f'random baseline ({prevalence:.4f})')
    ax_pr.set_xlabel('Recall', fontsize=12)
    ax_pr.set_ylabel('Precision', fontsize=12)
    ax_pr.set_title('Precision-Recall Curves', fontsize=13)
    ax_pr.legend(fontsize=9)
    ax_pr.set_xlim([0, 1]); ax_pr.set_ylim([0, 1.02]); ax_pr.grid(alpha=0.3)

    ax_roc = axes[1]
    for y_true, probs, label in [train_data, val_data, test_data]:
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, probs)
        auc_val = roc_auc_score(y_true, probs)
        ax_roc.plot(fpr, tpr, color=colors[label], lw=1.8,
                    label=f'{label}  (ROC-AUC={auc_val:.3f})')

    if len(np.unique(val_true)) >= 2:
        fpr_v, tpr_v, _ = roc_curve(val_true, np.array(val_probs))
        j_idx = int(np.argmax(tpr_v - fpr_v))
        ax_roc.scatter([fpr_v[j_idx]], [tpr_v[j_idx]],
                       color=colors['val'], s=120, zorder=5, marker='*',
                       label=f"val Youden's J thresh={youden_thresh_val:.3f}")

    ax_roc.plot([0, 1], [0, 1], 'k--', lw=1, label='random baseline')
    ax_roc.set_xlabel('False Positive Rate', fontsize=12)
    ax_roc.set_ylabel('True Positive Rate', fontsize=12)
    ax_roc.set_title('ROC Curves', fontsize=13)
    ax_roc.legend(fontsize=9)
    ax_roc.set_xlim([0, 1]); ax_roc.set_ylim([0, 1.02]); ax_roc.grid(alpha=0.3)

    plt.tight_layout()
    out_path = save_path + model_name + '_pr_roc_curves.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Curves saved → {out_path}')


def _print_metrics_block(label, m, threshold, n_pos):
    """Print a formatted block of metrics — verbatim from patchtst_finetune.py."""
    print(f'  {"─"*60}')
    print(f'  Threshold : {threshold:.4f}  ({label})')
    print(f'  {"─"*60}')
    print(f'  Confusion matrix:')
    print(f'    TP (flu caught)      = {m["TP"]:6d}')
    print(f'    FN (flu missed)      = {m["FN"]:6d}   ← minimise this')
    print(f'    FP (false alarms)    = {m["FP"]:6d}')
    print(f'    TN (healthy correct) = {m["TN"]:6d}')
    print(f'  {"─"*60}')
    sens_note = _interpret_sensitivity(m['sensitivity'], n_pos)
    print(f'  Sensitivity (Recall)  : {m["sensitivity"]:.4f}  — {sens_note}')
    print(f'  Specificity           : {m["specificity"]:.4f}  — fraction of healthy windows correctly ID\'d')
    print(f'  Precision (PPV)       : {m["precision"]:.4f}  — of flagged windows, fraction truly flu')
    print(f'  NPV                   : {m["npv"]:.4f}  — of cleared windows, fraction truly healthy')
    print(f'  {"─"*60}')
    print(f'  F1                    : {m["f1"]:.4f}  — balanced precision/recall')
    print(f'  F2 (recall-weighted)  : {m["f2"]:.4f}  — weights missing a case 2x worse than false alarm')
    print(f'  MCC                   : {m["mcc"]:.4f}  — {_interpret_mcc(m["mcc"])}')
    print(f'  Balanced Accuracy     : {m["balanced_accuracy"]:.4f}  — accuracy corrected for imbalance (0.5=random)')


# ------------------------------------------------------------------
# Test / evaluation — rank 0 only; mirrors test_func in patchtst_finetune.py
# ------------------------------------------------------------------
def test_func(weight_path):
    input_dim = split_data['train'][0].shape[1]
    model = DualEncoderMLP(input_dim, args.hidden_dim, args.n_hidden_layers,
                           args.mlp_dropout).to(device)
    state = torch.load(weight_path + '.pth', map_location=device)
    model.load_state_dict(state)
    model.eval()

    def _infer(emb):
        dl = DataLoader(TensorDataset(emb), batch_size=args.batch_size, shuffle=False)
        logits = []
        with torch.no_grad():
            for (xb,) in dl:
                logits.append(model(xb.to(device)).cpu())
        return torch.cat(logits).numpy()

    logits_train = _infer(split_data['train'][0])
    logits_val   = _infer(split_data['val'][0])
    logits_test  = _infer(split_data['test'][0])

    train_probs = 1 / (1 + np.exp(-logits_train))
    val_probs   = 1 / (1 + np.exp(-logits_val))
    test_probs  = 1 / (1 + np.exp(-logits_test))

    train_true = split_data['train'][1].numpy().astype(int)
    val_true   = split_data['val'][1].numpy().astype(int)
    test_true  = split_data['test'][1].numpy().astype(int)

    # PR-optimal threshold from train and val
    if len(np.unique(train_true)) >= 2:
        pr_opt_thresh_train, pr_opt_f1_train, _, _, _ = _find_pr_optimal_threshold(train_true, train_probs)
    else:
        pr_opt_thresh_train, pr_opt_f1_train = 0.5, float('nan')

    if len(np.unique(val_true)) >= 2:
        pr_opt_thresh_val, pr_opt_f1_val, _, _, _ = _find_pr_optimal_threshold(val_true, val_probs)
        opt_threshold = pr_opt_thresh_val
    else:
        pr_opt_thresh_val, pr_opt_f1_val = 0.5, float('nan')
        opt_threshold = 0.5

    prevalence = test_true.mean()
    n_pos      = int(test_true.sum())

    if len(np.unique(test_true)) >= 2:
        roc_auc = roc_auc_score(test_true, test_probs)
        pr_auc  = average_precision_score(test_true, test_probs)
    else:
        roc_auc = pr_auc = float('nan')

    m_05  = _metrics_at_threshold(test_true, test_probs, threshold=0.5)
    m_opt = _metrics_at_threshold(test_true, test_probs, threshold=opt_threshold) \
            if not np.isnan(opt_threshold) else None

    print()
    print(f'  {"="*60}')
    print(f'  TEST RESULTS')
    print(f'  {"="*60}')
    print(f'  NOTE: with {prevalence*100:.3f}% prevalence, accuracy and ROC-AUC are')
    print(f'  misleading — a model predicting all-negative scores ~100% accuracy')
    print(f'  and ROC-AUC can look decent. Focus on PR-AUC, MCC, Sensitivity,')
    print(f'  and F2. The @optimal block is more informative than @0.5 because')
    print(f'  @0.5 almost always predicts all-negative at this prevalence.')
    print(f'  {"="*60}')
    print(f'  Dataset (test set):')
    print(f'    Total samples  : {len(test_true)}')
    print(f'    Positives      : {n_pos}  (flu-onset windows)')
    print(f'    Negatives      : {len(test_true) - n_pos}')
    print(f'    Prevalence     : {prevalence:.5f}  ({prevalence*100:.3f}%)')
    print(f'    Random PR-AUC  : {prevalence:.5f}  (baseline — a random model scores this)')
    print()
    print(f'  Optimal threshold selection (PR curve, max F1):')
    print(f'    Train PR-optimal threshold : {pr_opt_thresh_train:.4f}  (best F1={pr_opt_f1_train:.4f} on train)')
    print(f'    Val   PR-optimal threshold : {pr_opt_thresh_val:.4f}  (best F1={pr_opt_f1_val:.4f} on val)  ← used on test')
    print(f'    Val positives              : {int(val_true.sum())}  /  {len(val_true)}')
    print(f'    Train positives            : {int(train_true.sum())}  /  {len(train_true)}')
    print()
    print(f'  Threshold-independent metrics (test set):')
    print(f'    PR-AUC  (primary) : {pr_auc:.4f}  — {_interpret_pr_auc(pr_auc, prevalence)}')
    print(f'    ROC-AUC           : {roc_auc:.4f}  — ranking ability (0.5=random, 1.0=perfect)')
    print()
    _print_metrics_block('@0.5  (fixed reference)', m_05, threshold=0.5, n_pos=n_pos)
    if m_opt is not None:
        print()
        _print_metrics_block(
            'PR-optimal threshold from VAL (max F1 on val PR curve), applied to TEST',
            m_opt, threshold=opt_threshold, n_pos=n_pos,
        )
    print(f'  {"="*60}')

    _save_curves(
        save_path=args.save_path,
        model_name=weight_path.replace(args.save_path, ''),
        train_data=(train_true, train_probs, 'train'),
        val_data=(val_true,   val_probs,   'val'),
        test_data=(test_true,  test_probs,  'test'),
        pr_opt_thresh_train=pr_opt_thresh_train,
        pr_opt_thresh_val=pr_opt_thresh_val,
        youden_thresh_val=opt_threshold,
    )

    row = dict(
        roc_auc=roc_auc, pr_auc=pr_auc,
        pr_opt_threshold_train=pr_opt_thresh_train, pr_opt_f1_train=pr_opt_f1_train,
        pr_opt_threshold_val=pr_opt_thresh_val,     pr_opt_f1_val=pr_opt_f1_val,
        n_hidden_layers=args.n_hidden_layers,        hidden_dim=args.hidden_dim,
    )
    for k, v in m_05.items():
        row[f'{k}@0.5'] = v
    if m_opt is not None:
        for k, v in m_opt.items():
            row[f'{k}@opt'] = v

    pd.DataFrame([row]).to_csv(
        args.save_path + args.save_finetuned_model + '_acc.csv',
        float_format='%.6f', index=False,
    )

    scores = [roc_auc, pr_auc, m_05['precision']]
    return scores


# ------------------------------------------------------------------
# Main — mirrors patchtst_finetune.py __main__ block
# ------------------------------------------------------------------
if __name__ == '__main__':

    # Load frozen backbones — each rank loads its own copy onto its GPU
    if rank == 0:
        print('\nLoading backbones …')
    pos_model = load_backbone(args.positive_model)
    neg_model = load_backbone(args.negative_model)

    revin = RevIN(num_features=args.c_in, eps=1e-5, affine=False).to(device)

    if rank == 0:
        print('\nLoading data …')

    class _DataArgs:
        dset           = args.dset
        context_points = args.context_points
        target_points  = 1
        batch_size     = args.backbone_batch_size
        num_workers    = min(args.num_workers, 4)
        scaler         = 'standard'
        features       = 'M'
        patch_len      = args.patch_len
        stride         = args.stride
        revin          = args.revin
        label_filter   = 'all'

    dls = get_dls(_DataArgs())

    # Each rank extracts independently (frozen/deterministic — result is identical)
    if rank == 0:
        print('\nExtracting embeddings from both frozen backbones …')
    split_data = extract_all_embeddings(pos_model, neg_model, dls, revin)

    # LR finder: rank 0 only, then broadcast to all ranks
    if is_distributed:
        if rank == 0:
            if args.use_lr_finder:
                print('\nRunning lr_finder on rank 0 …')
                emb_train, lbl_train = split_data['train']
                suggested_lr = find_lr(emb_train, lbl_train, emb_train.shape[1])
                print(f'[rank 0] Suggested LR: {suggested_lr:.6f}. Broadcasting to all ranks …')
            else:
                suggested_lr = args.lr
                print(f'[rank 0] Skipping LR finder, using fixed LR: {suggested_lr:.6f}. Broadcasting …')
            lr_tensor = torch.tensor([suggested_lr], dtype=torch.float64).to(device)
        else:
            lr_tensor = torch.zeros(1, dtype=torch.float64).to(device)

        dist.broadcast(lr_tensor, src=0)
        suggested_lr = float(lr_tensor.item())
        if rank == 0:
            print(f'suggested_lr {suggested_lr}')
    else:
        if args.use_lr_finder:
            print('\nRunning lr_finder …')
            emb_train, lbl_train = split_data['train']
            suggested_lr = find_lr(emb_train, lbl_train, emb_train.shape[1])
        else:
            suggested_lr = args.lr
            print(f'Skipping LR finder, using fixed LR: {suggested_lr:.6f}')
        print(f'suggested_lr {suggested_lr}')

    # Train
    linear_probe_func(suggested_lr)
    if rank == 0:
        print('finetune completed')

    # Barrier: make sure all ranks finish training before rank 0 runs test_func.
    # Without this, rank 1 could hit destroy_process_group while rank 0 is
    # still in test_func, which may cause torchrun to terminate the job early.
    if is_distributed:
        dist.barrier()

    # Evaluate on rank 0 only
    if rank == 0:
        out = test_func(args.save_path + args.save_finetuned_model)
        print('----------- Complete! -----------')

    # Second barrier: rank 0 signals it has finished test_func so rank 1 can
    # exit cleanly at the same time rather than timing out.
    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()
