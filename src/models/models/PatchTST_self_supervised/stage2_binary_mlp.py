"""
Stage 2 — Binary MLP classifier on 3-class PatchTST softmax outputs.

Pipeline:
  1. Load frozen stage-1 checkpoint (3-class PatchTST finetuned)
  2. Run inference on all splits → softmax [p0, p1, p2] per window
  3. Binary labels: 1 if class-2 (Tested Positive), else 0
  4. Undersample train negatives to neg_subsample_ratio:1
  5. Train BinaryMLP(3 → hidden → 1) with BCEWithLogitsLoss + pos_weight
  6. Evaluate on test set with full metrics suite

Run from /home/hice1/ezg6/projects:
  python Homekit2020/src/models/models/PatchTST_self_supervised/stage2_binary_mlp.py \
    --checkpoint <path_to_3class.pth> ...
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, fbeta_score, matthews_corrcoef,
    balanced_accuracy_score, confusion_matrix,
    roc_curve, precision_recall_curve,
)
import argparse

# ── Path setup: same layout as patchtst_finetune.py ───────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_HOMEKIT_SRC = os.path.join(_HERE, '..', '..', '..', '..', '..')
for p in [_HERE, os.path.normpath(_HOMEKIT_SRC)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.models.patchTST import PatchTST
from src.learner import Learner, transfer_weights
from src.callback.core import *
from src.callback.patch_mask import *
from src.callback.transforms import *
from src.basics import set_device
from datautils import get_dls


# ══════════════════════════════════════════════════════════════════════════════
# Argument parser
# ══════════════════════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser()

# ── Stage-1 model (3-class PatchTST) ─────────────────────────────────────────
parser.add_argument('--checkpoint', type=str, required=True,
                    help='Path to the 3-class finetuned .pth checkpoint')
parser.add_argument('--context_points', type=int, default=10080)
parser.add_argument('--target_points',  type=int, default=3,
                    help='Must match stage-1 (3 classes)')
parser.add_argument('--patch_len',      type=int, default=1440)
parser.add_argument('--stride',         type=int, default=180)
parser.add_argument('--n_layers',       type=int, default=6)
parser.add_argument('--n_heads',        type=int, default=8)
parser.add_argument('--d_model',        type=int, default=256)
parser.add_argument('--d_ff',           type=int, default=512)
parser.add_argument('--dropout',        type=float, default=0.2)
parser.add_argument('--head_dropout',   type=float, default=0.2)
parser.add_argument('--revin',          type=int, default=1)
parser.add_argument('--head_type',      type=str, default='classification')
parser.add_argument('--batch_size',     type=int, default=32,
                    help='Batch size for PatchTST inference')
parser.add_argument('--num_workers',    type=int, default=2)
parser.add_argument('--scaler',         type=str, default='standard')
parser.add_argument('--features',       type=str, default='M')

# ── Stage-2 imbalance handling ────────────────────────────────────────────────
parser.add_argument('--neg_subsample_ratio', type=int, default=20,
                    help='Keep this many train negatives per positive; 0=all')
parser.add_argument('--pos_weight_cap', type=float, default=1.0,
                    help='Cap on BCEWithLogitsLoss pos_weight; 1=no upweight')
parser.add_argument('--seed', type=int, default=42)

# ── MLP architecture ──────────────────────────────────────────────────────────
parser.add_argument('--mlp_hidden', type=int, nargs='+', default=[64, 32],
                    help='Hidden layer sizes e.g. --mlp_hidden 64 32')
parser.add_argument('--mlp_dropout', type=float, default=0.3)

# ── MLP training ──────────────────────────────────────────────────────────────
parser.add_argument('--mlp_lr',         type=float, default=1e-3)
parser.add_argument('--mlp_wd',         type=float, default=1e-4,
                    help='Weight decay (L2 regularisation)')
parser.add_argument('--mlp_epochs',     type=int,   default=300)
parser.add_argument('--mlp_batch_size', type=int,   default=256)
parser.add_argument('--mlp_patience',   type=int,   default=30,
                    help='Early stop if val PR-AUC does not improve')

# ── Output ────────────────────────────────────────────────────────────────────
parser.add_argument('--save_path', type=str,
                    default='saved_models/Wearable_3class/masked_patchtst/Stage2_BinaryMLP/',
                    help='Directory for all outputs')
parser.add_argument('--model_id', type=str, default='stage2_mlp',
                    help='Short name used in all output filenames')

args = parser.parse_args()

os.makedirs(args.save_path, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
print('args:', args)


# ══════════════════════════════════════════════════════════════════════════════
# Binary MLP
# ══════════════════════════════════════════════════════════════════════════════
class BinaryMLP(nn.Module):
    def __init__(self, input_dim=3, hidden_dims=(64, 32), dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
# Metrics helpers  (mirrored from patchtst_finetune.py for consistency)
# ══════════════════════════════════════════════════════════════════════════════
def _metrics_at_threshold(y_true, probs, threshold):
    y_pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv         = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1  = f1_score(y_true, y_pred, zero_division=0)
    f2  = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    bal = balanced_accuracy_score(y_true, y_pred)
    return dict(
        TP=int(tp), FP=int(fp), TN=int(tn), FN=int(fn),
        sensitivity=sensitivity, specificity=specificity,
        precision=precision, npv=npv,
        f1=f1, f2=f2, mcc=mcc, balanced_accuracy=bal,
    )


def _interpret_sensitivity(s, n_pos):
    caught = round(s * n_pos)
    missed = n_pos - caught
    if s == 0.0:
        return 'catching NO flu cases'
    elif s < 0.3:
        return f'catching {caught}/{n_pos} — very poor'
    elif s < 0.6:
        return f'catching {caught}/{n_pos} — moderate'
    elif s < 0.8:
        return f'catching {caught}/{n_pos} — reasonable'
    else:
        return f'catching {caught}/{n_pos} — strong'


def _interpret_mcc(mcc):
    if mcc <= 0.0:   return 'no useful signal (≤ random)'
    elif mcc < 0.2:  return 'weak signal'
    elif mcc < 0.4:  return 'moderate signal'
    elif mcc < 0.6:  return 'good signal'
    else:            return 'strong signal'


def _interpret_pr_auc(pr_auc, prevalence):
    if prevalence <= 0: return ''
    lift = pr_auc / prevalence
    if lift < 2:    return f'{lift:.1f}x above random — poor'
    elif lift < 10: return f'{lift:.1f}x above random — moderate'
    elif lift < 50: return f'{lift:.1f}x above random — good'
    else:           return f'{lift:.1f}x above random — excellent'


def _find_pr_optimal_threshold(y_true, probs):
    prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_true, probs)
    denom  = prec_arr[:-1] + rec_arr[:-1]
    f1_arr = np.where(denom > 0, 2 * prec_arr[:-1] * rec_arr[:-1] / denom, 0.0)
    best   = int(np.argmax(f1_arr))
    return float(thresh_arr[best]), float(f1_arr[best]), prec_arr, rec_arr, thresh_arr


def _print_metrics_block(label, m, threshold, n_pos):
    print(f'  {"─"*60}')
    print(f'  Threshold : {threshold:.4f}  ({label})')
    print(f'  {"─"*60}')
    print(f'  Confusion matrix:')
    print(f'    TP (flu caught)      = {m["TP"]:6d}')
    print(f'    FN (flu missed)      = {m["FN"]:6d}   ← minimise this')
    print(f'    FP (false alarms)    = {m["FP"]:6d}')
    print(f'    TN (healthy correct) = {m["TN"]:6d}')
    print(f'  {"─"*60}')
    print(f'  Sensitivity (Recall)  : {m["sensitivity"]:.4f}  — {_interpret_sensitivity(m["sensitivity"], n_pos)}')
    print(f'  Specificity           : {m["specificity"]:.4f}')
    print(f'  Precision (PPV)       : {m["precision"]:.4f}')
    print(f'  NPV                   : {m["npv"]:.4f}')
    print(f'  {"─"*60}')
    print(f'  F1                    : {m["f1"]:.4f}')
    print(f'  F2 (recall-weighted)  : {m["f2"]:.4f}')
    print(f'  MCC                   : {m["mcc"]:.4f}  — {_interpret_mcc(m["mcc"])}')
    print(f'  Balanced Accuracy     : {m["balanced_accuracy"]:.4f}')


def _save_pr_roc_curves(save_path, model_name, splits_data,
                        pr_opt_thresh_val, youden_thresh_val):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = {'train': '#1f77b4', 'val': '#ff7f0e', 'test': '#2ca02c'}

    ax_pr, ax_roc = axes

    for y_true, probs, label in splits_data:
        if len(np.unique(y_true)) < 2:
            continue
        prec, rec, _ = precision_recall_curve(y_true, probs)
        auc_val = average_precision_score(y_true, probs)
        ax_pr.plot(rec, prec, color=colors[label], lw=1.8,
                   label=f'{label}  (PR-AUC={auc_val:.4f})')
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_val = roc_auc_score(y_true, probs)
        ax_roc.plot(fpr, tpr, color=colors[label], lw=1.8,
                    label=f'{label}  (ROC-AUC={roc_val:.4f})')

    # Mark val PR-optimal threshold
    val_true, val_probs, _ = splits_data[1]
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
        fpr_v, tpr_v, thresh_v = roc_curve(val_true, vp)
        j_idx = int(np.argmax(tpr_v - fpr_v))
        ax_roc.scatter([fpr_v[j_idx]], [tpr_v[j_idx]],
                       color=colors['val'], s=120, zorder=5, marker='*',
                       label=f"Youden's J={youden_thresh_val:.3f}")

    test_true, _, _ = splits_data[2]
    prevalence = test_true.mean()
    ax_pr.axhline(prevalence, color='gray', lw=1, ls='--',
                  label=f'random ({prevalence:.4f})')
    ax_pr.set_xlabel('Recall'); ax_pr.set_ylabel('Precision')
    ax_pr.set_title('Precision-Recall Curves'); ax_pr.legend(fontsize=9)
    ax_pr.set_xlim([0, 1]); ax_pr.set_ylim([0, 1.02]); ax_pr.grid(alpha=0.3)

    ax_roc.plot([0, 1], [0, 1], 'k--', lw=1, label='random')
    ax_roc.set_xlabel('FPR'); ax_roc.set_ylabel('TPR')
    ax_roc.set_title('ROC Curves'); ax_roc.legend(fontsize=9)
    ax_roc.set_xlim([0, 1]); ax_roc.set_ylim([0, 1.02]); ax_roc.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(save_path, model_name + '_pr_roc_curves.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Curves saved → {out}')


def _save_loss_curve(save_path, model_name, train_losses, val_losses, best_epoch):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, label='train BCE loss', color='#1f77b4')
    ax.plot(val_losses,   label='val BCE loss',   color='#ff7f0e')
    ax.axvline(best_epoch, color='green', ls='--', lw=1.2,
               label=f'best epoch={best_epoch}')
    ax.set_xlabel('Epoch'); ax.set_ylabel('BCE Loss')
    ax.set_title('Stage-2 MLP Training'); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(save_path, model_name + '_loss_curve.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Loss curve saved → {out}')


# ══════════════════════════════════════════════════════════════════════════════
# Stage-1 inference: extract softmax probs from frozen 3-class PatchTST
# ══════════════════════════════════════════════════════════════════════════════
def build_patchtst(n_vars):
    num_patch = (max(args.context_points, args.patch_len) - args.patch_len) // args.stride + 1
    model = PatchTST(
        c_in=n_vars,
        target_dim=args.target_points,
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
        head_type=args.head_type,
        res_attention=False,
    )
    print(f'PatchTST params: {sum(p.numel() for p in model.parameters()):,}')
    return model


def extract_probs_all_splits():
    """
    Run frozen 3-class PatchTST on all splits.
    Returns dict with keys 'train', 'val', 'test', each a tuple:
        (softmax_probs [N,3],  binary_labels [N],  raw_3class_labels [N])
    """
    print('\n── Stage 1: extracting softmax probs from frozen 3-class PatchTST ──')

    # Build dataloader using Wearable_3class task
    dls_args = argparse.Namespace(**vars(args))
    dls_args.dset = 'Wearable_3class'
    dls_args.dset_finetune = 'Wearable_3class'
    dls_args.neg_subsample_ratio = 0   # extract full dataset — we subsample in stage 2
    dls_args.seed = args.seed

    dls = get_dls(dls_args)

    model = build_patchtst(dls.vars).to(device)
    cbs  = [RevInCB(dls.vars, denorm=False)] if args.revin else []
    cbs += [PatchCB(patch_len=args.patch_len, stride=args.stride)]
    learn = Learner(dls, model, cbs=cbs)

    weight_path = args.checkpoint
    if weight_path.endswith('.pth'):
        weight_path = weight_path[:-4]

    results = {}
    for split, dl in [('train', dls.train), ('val', dls.valid), ('test', dls.test)]:
        print(f'  Inferring {split} ...')
        preds, targets = learn.test(dl, weight_path=weight_path + '.pth')
        logits = np.array(preds)          # [N, 3]
        targs  = np.array(targets).reshape(-1).astype(int)

        # Numerically stable softmax
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_p   = np.exp(shifted)
        probs   = exp_p / exp_p.sum(axis=1, keepdims=True)   # [N, 3]

        binary_labels = (targs == 2).astype(int)   # flu-positive vs rest
        print(f'    {split}: N={len(targs)}, pos(class-2)={binary_labels.sum()}, '
              f'neg={( binary_labels == 0).sum()}'
              f'  class counts: {dict(zip(*np.unique(targs, return_counts=True)))}')
        results[split] = (probs, binary_labels, targs)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Undersampling
# ══════════════════════════════════════════════════════════════════════════════
def undersample(probs, binary_labels, ratio, seed):
    """Randomly drop negatives so that neg:pos = ratio:1."""
    if ratio <= 0:
        return probs, binary_labels

    rng     = np.random.default_rng(seed)
    pos_idx = np.where(binary_labels == 1)[0]
    neg_idx = np.where(binary_labels == 0)[0]
    n_keep  = min(len(pos_idx) * ratio, len(neg_idx))

    chosen_neg = rng.choice(neg_idx, size=int(n_keep), replace=False)
    keep = np.concatenate([pos_idx, chosen_neg])
    rng.shuffle(keep)

    print(f'  Undersampled train: {len(pos_idx)} pos + {int(n_keep)} neg '
          f'({ratio}:1)  total={len(keep)}')
    return probs[keep], binary_labels[keep]


# ══════════════════════════════════════════════════════════════════════════════
# MLP training
# ══════════════════════════════════════════════════════════════════════════════
def train_mlp(X_train, y_train, X_val, y_val):
    print(f'\n── Stage 2: training BinaryMLP({[3]+args.mlp_hidden+[1]}) ──')

    # Tensors
    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32)
    Xv = torch.tensor(X_val,   dtype=torch.float32).to(device)
    yv = torch.tensor(y_val,   dtype=torch.float32).to(device)

    train_ds = TensorDataset(Xt, yt)
    train_dl = DataLoader(train_ds, batch_size=args.mlp_batch_size,
                          shuffle=True, drop_last=False)

    # Model
    mlp = BinaryMLP(input_dim=3,
                    hidden_dims=args.mlp_hidden,
                    dropout=args.mlp_dropout).to(device)

    # Loss — pos_weight from raw ratio, capped
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    raw_pw = n_neg / n_pos if n_pos > 0 else 1.0
    pw     = raw_pw if args.pos_weight_cap < 0 else min(raw_pw, args.pos_weight_cap)
    print(f'  pos_weight: raw={raw_pw:.1f}, cap={args.pos_weight_cap}, using={pw:.1f}')
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=device))

    opt = torch.optim.Adam(mlp.parameters(), lr=args.mlp_lr,
                           weight_decay=args.mlp_wd)

    best_val_pr_auc  = -1.0
    best_epoch       = 0
    patience_counter = 0
    best_state       = None
    train_losses, val_losses = [], []

    for epoch in range(args.mlp_epochs):
        # ── train ──
        mlp.train()
        epoch_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logit = mlp(xb)
            loss  = criterion(logit, yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(xb)
        train_loss = epoch_loss / len(y_train)
        train_losses.append(train_loss)

        # ── validate ──
        mlp.eval()
        with torch.no_grad():
            val_logit = mlp(Xv)
            val_loss  = criterion(val_logit, yv).item()
            val_probs = torch.sigmoid(val_logit).cpu().numpy()
        val_losses.append(val_loss)

        val_pr_auc = (average_precision_score(y_val, val_probs)
                      if len(np.unique(y_val)) >= 2 else 0.0)

        if epoch % 20 == 0 or epoch < 5:
            print(f'  Epoch {epoch:4d} | train_loss={train_loss:.5f} | '
                  f'val_loss={val_loss:.5f} | val_PR-AUC={val_pr_auc:.5f}')

        # ── early stopping on val PR-AUC ──
        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc  = val_pr_auc
            best_epoch       = epoch
            patience_counter = 0
            best_state       = {k: v.clone() for k, v in mlp.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= args.mlp_patience:
                print(f'  Early stop at epoch {epoch} '
                      f'(best val PR-AUC={best_val_pr_auc:.5f} at epoch {best_epoch})')
                break

    print(f'  Best epoch: {best_epoch}  |  best val PR-AUC: {best_val_pr_auc:.5f}')

    # Reload best weights
    mlp.load_state_dict(best_state)

    # Save checkpoint
    ckpt_path = os.path.join(args.save_path, f'{args.model_id}.pth')
    torch.save(best_state, ckpt_path)
    print(f'  MLP checkpoint saved → {ckpt_path}')

    return mlp, train_losses, val_losses, best_epoch


# ══════════════════════════════════════════════════════════════════════════════
# Full evaluation
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_and_report(mlp, splits_data):
    """
    Comprehensive test-set evaluation.
    splits_data: dict with 'train'/'val'/'test' → (softmax_probs[N,3], binary_labels[N], ...)
    """
    print('\n── Stage 2: full evaluation ──')

    def get_mlp_probs(probs_3class):
        X = torch.tensor(probs_3class, dtype=torch.float32).to(device)
        mlp.eval()
        with torch.no_grad():
            logits = mlp(X).cpu().numpy()
        return torch.sigmoid(torch.tensor(logits)).numpy()

    # ── Get probs from MLP for all splits ────────────────────────────────────
    train_probs3, train_bin, _ = splits_data['train']
    val_probs3,   val_bin,   _ = splits_data['val']
    test_probs3,  test_bin,  _ = splits_data['test']

    train_out = get_mlp_probs(train_probs3)
    val_out   = get_mlp_probs(val_probs3)
    test_out  = get_mlp_probs(test_probs3)

    # ── Threshold selection: PR-optimal on val (applied blind to test) ────────
    if len(np.unique(val_bin)) >= 2:
        pr_thresh_train, pr_f1_train, _, _, _ = _find_pr_optimal_threshold(train_bin, train_out)
        pr_thresh_val,   pr_f1_val,   _, _, _ = _find_pr_optimal_threshold(val_bin, val_out)
        opt_threshold = pr_thresh_val
    else:
        pr_thresh_train = pr_thresh_val = opt_threshold = 0.5
        pr_f1_train = pr_f1_val = float('nan')

    # Youden's J on val ROC (for ROC plot marker)
    if len(np.unique(val_bin)) >= 2:
        fpr_v, tpr_v, thresh_v = roc_curve(val_bin, val_out)
        youden_thresh = float(thresh_v[int(np.argmax(tpr_v - fpr_v))])
    else:
        youden_thresh = 0.5

    # ── Threshold-independent metrics ────────────────────────────────────────
    def safe_auc(y, p, fn):
        return fn(y, p) if len(np.unique(y)) >= 2 else float('nan')

    train_pr_auc  = safe_auc(train_bin, train_out, average_precision_score)
    train_roc_auc = safe_auc(train_bin, train_out, roc_auc_score)
    val_pr_auc    = safe_auc(val_bin,   val_out,   average_precision_score)
    val_roc_auc   = safe_auc(val_bin,   val_out,   roc_auc_score)
    test_pr_auc   = safe_auc(test_bin,  test_out,  average_precision_score)
    test_roc_auc  = safe_auc(test_bin,  test_out,  roc_auc_score)

    # ── Threshold-dependent metrics on test ───────────────────────────────────
    m_05  = _metrics_at_threshold(test_bin, test_out, 0.5)
    m_opt = _metrics_at_threshold(test_bin, test_out, opt_threshold)

    prevalence = test_bin.mean()
    n_pos      = int(test_bin.sum())

    # ── Print ─────────────────────────────────────────────────────────────────
    print()
    print(f'  {"="*60}')
    print(f'  STAGE-2 MLP TEST RESULTS')
    print(f'  {"="*60}')
    print(f'  Architecture : BinaryMLP(3 → {args.mlp_hidden} → 1)')
    print(f'  Checkpoint   : {args.checkpoint}')
    print()
    print(f'  Dataset (test set):')
    print(f'    Total     : {len(test_bin)}')
    print(f'    Positives : {n_pos}  (flu-positive windows)')
    print(f'    Negatives : {len(test_bin) - n_pos}')
    print(f'    Prevalence: {prevalence:.5f}  ({prevalence*100:.3f}%)')
    print(f'    Random PR-AUC baseline: {prevalence:.5f}')
    print()
    print(f'  ── Across-split summary ──────────────────────────────')
    print(f'    Train  PR-AUC : {train_pr_auc:.4f}   ROC-AUC : {train_roc_auc:.4f}  '
          f'  (pos={int(train_bin.sum())}, total={len(train_bin)})')
    print(f'    Val    PR-AUC : {val_pr_auc:.4f}   ROC-AUC : {val_roc_auc:.4f}  '
          f'  (pos={int(val_bin.sum())}, total={len(val_bin)})')
    print(f'    Test   PR-AUC : {test_pr_auc:.4f}   ROC-AUC : {test_roc_auc:.4f}  '
          f'  (pos={n_pos}, total={len(test_bin)})')
    print()
    print(f'  Stage-1 softmax inputs (test set mean per class):')
    print(f'    p0 (nonsymptomatic) : {test_probs3[:, 0].mean():.4f}')
    print(f'    p1 (tested-neg)     : {test_probs3[:, 1].mean():.4f}')
    print(f'    p2 (tested-pos)     : {test_probs3[:, 2].mean():.4f}')
    print()
    print(f'  Threshold-independent metrics (test):')
    print(f'    PR-AUC  (primary) : {test_pr_auc:.4f}  — {_interpret_pr_auc(test_pr_auc, prevalence)}')
    print(f'    ROC-AUC           : {test_roc_auc:.4f}')
    print()
    print(f'  Optimal threshold selection:')
    print(f'    Train PR-optimal  : {pr_thresh_train:.4f}  (F1={pr_f1_train:.4f})')
    print(f'    Val   PR-optimal  : {pr_thresh_val:.4f}  (F1={pr_f1_val:.4f}) ← applied to test')
    print()
    _print_metrics_block('@0.5 (fixed reference)', m_05, 0.5, n_pos)
    print()
    _print_metrics_block(
        'PR-optimal from VAL (applied blind to TEST)', m_opt, opt_threshold, n_pos,
    )
    print(f'  {"="*60}')

    # ── Save curves ───────────────────────────────────────────────────────────
    _save_pr_roc_curves(
        args.save_path, args.model_id,
        splits_data=[(train_bin, train_out, 'train'),
                     (val_bin,   val_out,   'val'),
                     (test_bin,  test_out,  'test')],
        pr_opt_thresh_val=pr_thresh_val,
        youden_thresh_val=youden_thresh,
    )

    # ── Save CSV ──────────────────────────────────────────────────────────────
    row = dict(
        model_id=args.model_id,
        checkpoint=args.checkpoint,
        neg_subsample_ratio=args.neg_subsample_ratio,
        mlp_hidden=str(args.mlp_hidden),
        mlp_dropout=args.mlp_dropout,
        pos_weight_cap=args.pos_weight_cap,
        train_pr_auc=train_pr_auc,  train_roc_auc=train_roc_auc,
        val_pr_auc=val_pr_auc,      val_roc_auc=val_roc_auc,
        test_pr_auc=test_pr_auc,    test_roc_auc=test_roc_auc,
        pr_opt_threshold_train=pr_thresh_train, pr_opt_f1_train=pr_f1_train,
        pr_opt_threshold_val=pr_thresh_val,     pr_opt_f1_val=pr_f1_val,
        prevalence=prevalence, n_pos_test=n_pos,
    )
    for k, v in m_05.items():
        row[f'{k}@0.5'] = v
    for k, v in m_opt.items():
        row[f'{k}@opt'] = v

    csv_path = os.path.join(args.save_path, f'{args.model_id}_metrics.csv')
    pd.DataFrame([row]).to_csv(csv_path, float_format='%.6f', index=False)
    print(f'  Metrics CSV saved → {csv_path}')

    return test_pr_auc, test_roc_auc


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Phase 1: extract [p0,p1,p2] from frozen 3-class PatchTST ────────────
    splits_data = extract_probs_all_splits()

    train_probs3, train_bin, _ = splits_data['train']
    val_probs3,   val_bin,   _ = splits_data['val']
    test_probs3,  test_bin,  _ = splits_data['test']

    # ── Phase 2: undersample train negatives ─────────────────────────────────
    print(f'\n── Undersampling train set ({args.neg_subsample_ratio}:1) ──')
    X_train_us, y_train_us = undersample(
        train_probs3, train_bin, args.neg_subsample_ratio, args.seed,
    )
    # Val and test keep full distribution (never undersample held-out sets)
    X_val  = val_probs3
    y_val  = val_bin

    # ── Phase 3: train binary MLP ─────────────────────────────────────────────
    mlp, train_losses, val_losses, best_epoch = train_mlp(
        X_train_us, y_train_us, X_val, y_val,
    )

    # ── Phase 4: full evaluation on test set ──────────────────────────────────
    pr_auc, roc_auc = evaluate_and_report(mlp, splits_data)

    # ── Save loss curve ───────────────────────────────────────────────────────
    _save_loss_curve(args.save_path, args.model_id,
                     train_losses, val_losses, best_epoch)

    print(f'\nDone.  Test PR-AUC={pr_auc:.4f}  ROC-AUC={roc_auc:.4f}')
