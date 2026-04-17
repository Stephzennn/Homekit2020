"""
dual_encoder_probe.py — Dual-encoder MLP classifier on frozen PatchTST backbones.

Architecture
------------
Two pretrained PatchTST backbones (one trained on positive/flu windows, one on
negative/healthy windows) are loaded and frozen.  For each input window both
backbones produce an embedding via the same hook used in plot_embeddings.py:

    backbone → [B x nvars x d_model x num_patch]
             → mean over patches          → [B x nvars x d_model]
             → reshape                    → [B x nvars * d_model]

The two embeddings are concatenated → [B x 2 * nvars * d_model] and fed into a
small MLP whose depth is controlled by --n_hidden_layers:

    n_hidden_layers = 0  →  Linear(input_dim, 1)              (true linear probe)
    n_hidden_layers = 2  →  Linear → ReLU → Dropout  ×2  → Linear(1)
    n_hidden_layers = 4  →  Linear → ReLU → Dropout  ×4  → Linear(1)

Embeddings are extracted ONCE from both frozen backbones and cached in RAM,
so each training epoch only runs the lightweight MLP — no repeated backbone
forward passes.

Evaluation mirrors patchtst_finetune.py exactly:
  - PR-AUC as primary metric (correct for severely imbalanced data)
  - PR-optimal threshold (max F1 on val PR curve) applied blind to test
  - Full confusion matrix, sensitivity, specificity, F1, F2, MCC, balanced accuracy
  - PR + ROC curve plots saved as PNG

Usage (run from the PatchTST_self_supervised directory)
-------------------------------------------------------
python dual_encoder_probe.py \
    --positive_model /path/to/positive.pth \
    --negative_model /path/to/negative.pth \
    --n_hidden_layers 2 \
    --hidden_dim 256 \
    --n_epochs 50 \
    --lr 1e-3 \
    --model_type "dual_encoder_v1" \
    --c_in 8 --d_model 256 --n_layers 4 --n_heads 8 --d_ff 512 \
    --patch_len 1440 --stride 180 --context_points 10080
"""

import os
import sys
import argparse

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
_PROJECT_ROOT = "/home/hice1/ezg6/projects/Homekit2020/src"
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, fbeta_score, matthews_corrcoef,
    balanced_accuracy_score, confusion_matrix, roc_curve,
    precision_recall_curve,
)

from src.models.patchTST import PatchTST
from src.models.layers.revin import RevIN
from src.callback.patch_mask import create_patch
from datautils import get_dls


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        description="Dual-encoder MLP probe on frozen PatchTST backbones",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Backbone checkpoints ---
    p.add_argument('--positive_model', type=str, required=True,
                   help='Path to the positive-class pretrained .pth checkpoint')
    p.add_argument('--negative_model', type=str, required=True,
                   help='Path to the negative-class pretrained .pth checkpoint')

    # --- MLP architecture ---
    p.add_argument('--n_hidden_layers', type=int, default=2,
                   help='Number of hidden layers in the MLP. 0 = true linear probe.')
    p.add_argument('--hidden_dim', type=int, default=256,
                   help='Width of each hidden layer')
    p.add_argument('--mlp_dropout', type=float, default=0.1,
                   help='Dropout applied after each hidden layer')

    # --- Backbone architecture (must match checkpoints) ---
    p.add_argument('--c_in',         type=int,   default=8)
    p.add_argument('--d_model',      type=int,   default=256)
    p.add_argument('--n_layers',     type=int,   default=4)
    p.add_argument('--n_heads',      type=int,   default=8)
    p.add_argument('--d_ff',         type=int,   default=512)
    p.add_argument('--dropout',      type=float, default=0.1)
    p.add_argument('--head_dropout', type=float, default=0.1)
    p.add_argument('--patch_len',      type=int, default=1440)
    p.add_argument('--stride',         type=int, default=180)
    p.add_argument('--context_points', type=int, default=10080)

    # --- Training ---
    p.add_argument('--n_epochs',    type=int,   default=50,   help='MLP training epochs')
    p.add_argument('--lr',          type=float, default=1e-3, help='Learning rate')
    p.add_argument('--batch_size',  type=int,   default=256,
                   help='Batch size for MLP training (can be larger since no backbone pass)')
    p.add_argument('--pos_weight_cap', type=float, default=-1.0,
                   help='Cap for BCEWithLogitsLoss pos_weight. -1 = no cap.')

    # --- Data ---
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--revin',       type=int, default=1)
    p.add_argument('--backbone_batch_size', type=int, default=32,
                   help='Batch size used when extracting backbone embeddings')

    # --- Save / identify ---
    p.add_argument('--model_type',  type=str, default='dual_encoder',
                   help='Sub-directory name under saved_models/.../masked_patchtst/')
    p.add_argument('--model_id',    type=int, default=1,
                   help='Integer ID appended to the saved model filename')
    p.add_argument('--dset',        type=str, default='Wearable')

    return p


# ---------------------------------------------------------------------------
# Backbone loading — identical to ood_detector.py
# ---------------------------------------------------------------------------
def load_backbone(ckpt_path, args, device):
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
    print(f'  Loaded (frozen): {ckpt_path}')
    return model


def _register_embedding_hook(model):
    """Mean-pool over patches → reshape → [B x nvars*d_model]. Same as ood_detector.py."""
    captured = []

    def _pre_hook(module, inp):
        x = inp[0].detach().cpu()       # [B x nvars x d_model x num_patch]
        x = x.mean(dim=-1)              # [B x nvars x d_model]
        x = x.reshape(x.shape[0], -1)  # [B x nvars * d_model]
        captured.append(x)

    handle = model.head.register_forward_pre_hook(_pre_hook)
    return captured, handle


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------
@torch.no_grad()
def _extract_split(model, dataloader, revin, args, device):
    """Run one dataloader through a frozen backbone; return (embeddings, labels)."""
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


def extract_all_embeddings(pos_model, neg_model, dls, revin, args, device):
    """
    Extract embeddings from both frozen backbones for all three splits.
    Returns a dict: {'train': (emb, lbl), 'val': (emb, lbl), 'test': (emb, lbl)}
    where emb is [N x 2*nvars*d_model] (positive and negative concatenated).
    """
    split_map = {'train': dls.train, 'val': dls.valid, 'test': dls.test}
    result = {}

    for split_name, dl in split_map.items():
        if dl is None:
            continue
        print(f'  Extracting {split_name} embeddings …')
        emb_pos, lbl = _extract_split(pos_model, dl, revin, args, device)
        emb_neg, _   = _extract_split(neg_model, dl, revin, args, device)
        emb = torch.cat([emb_pos, emb_neg], dim=1)  # [N x 2*nvars*d_model]
        result[split_name] = (emb, lbl)
        n_pos = int((lbl == 1).sum())
        print(f'    {split_name}: {len(lbl)} samples  '
              f'(pos={n_pos}, neg={len(lbl)-n_pos})  '
              f'emb_dim={emb.shape[1]}')

    return result


# ---------------------------------------------------------------------------
# MLP model
# ---------------------------------------------------------------------------
class DualEncoderMLP(nn.Module):
    """
    MLP classifier on top of concatenated dual-backbone embeddings.

    n_hidden_layers=0  →  single linear layer (true linear probe)
    n_hidden_layers>0  →  n hidden Linear→ReLU→Dropout layers, then output linear
    """

    def __init__(self, input_dim, hidden_dim, n_hidden_layers, dropout):
        super().__init__()
        layers = []
        in_dim = input_dim

        for _ in range(n_hidden_layers):
            layers += [
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)   # [B]


# ---------------------------------------------------------------------------
# pos_weight computation — mirrors patchtst_finetune.py
# ---------------------------------------------------------------------------
def compute_pos_weight(labels_tensor, device, pos_weight_cap):
    n_pos = int((labels_tensor == 1).sum().item())
    n_neg = int((labels_tensor == 0).sum().item())
    raw   = n_neg / n_pos
    capped = raw if pos_weight_cap < 0 else min(raw, pos_weight_cap)
    print(f'  pos_weight: raw={raw:.1f}, cap={pos_weight_cap}, using={capped:.1f}  '
          f'(pos={n_pos}, neg={n_neg})')
    return torch.tensor([capped], dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(split_data, args, save_path, save_name, device):
    emb_train, lbl_train = split_data['train']
    emb_val,   lbl_val   = split_data['val']

    input_dim = emb_train.shape[1]
    model = DualEncoderMLP(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        n_hidden_layers=args.n_hidden_layers,
        dropout=args.mlp_dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\n  MLP: input_dim={input_dim}  hidden_dim={args.hidden_dim}  '
          f'n_hidden_layers={args.n_hidden_layers}  params={n_params:,}')

    pos_weight = compute_pos_weight(lbl_train, device, args.pos_weight_cap)
    loss_fn    = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimiser  = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_dl = DataLoader(
        TensorDataset(emb_train, lbl_train),
        batch_size=args.batch_size, shuffle=True,
    )
    val_dl = DataLoader(
        TensorDataset(emb_val, lbl_val),
        batch_size=args.batch_size, shuffle=False,
    )

    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    val_pr_aucs,  val_roc_aucs = [], []

    print(f'\n  Training for {args.n_epochs} epochs …')
    print(f'  {"Epoch":>6}  {"train_loss":>12}  {"val_loss":>10}  '
          f'{"val_PR-AUC":>12}  {"val_ROC-AUC":>13}')

    for epoch in range(1, args.n_epochs + 1):

        # ── Train ──
        model.train()
        ep_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimiser.step()
            ep_loss += loss.item() * len(yb)
        train_loss = ep_loss / len(lbl_train)

        # ── Validate ──
        model.eval()
        vl_loss = 0.0
        all_preds, all_tgts = [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                vl_loss += loss_fn(pred, yb).item() * len(yb)
                all_preds.append(torch.sigmoid(pred).cpu())
                all_tgts.append(yb.cpu())
        val_loss = vl_loss / len(lbl_val)

        probs   = torch.cat(all_preds).numpy()
        targets = torch.cat(all_tgts).numpy().astype(int)

        if len(np.unique(targets)) >= 2:
            pr_auc  = average_precision_score(targets, probs)
            roc_auc = roc_auc_score(targets, probs)
        else:
            pr_auc = roc_auc = float('nan')

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_pr_aucs.append(pr_auc)
        val_roc_aucs.append(roc_auc)

        print(f'  {epoch:>6}  {train_loss:>12.6f}  {val_loss:>10.6f}  '
              f'{pr_auc:>12.4f}  {roc_auc:>13.4f}')

        # ── Save best ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_path, save_name + '.pth'))
            print(f'  → Saved (val_loss={val_loss:.6f})')

    # ── Save losses CSV ──
    pd.DataFrame({
        'train_loss': train_losses,
        'val_loss':   val_losses,
        'val_pr_auc': val_pr_aucs,
        'val_roc_auc': val_roc_aucs,
    }).to_csv(os.path.join(save_path, save_name + '_losses.csv'),
              float_format='%.6f', index=False)

    # Reload best checkpoint
    model.load_state_dict(torch.load(os.path.join(save_path, save_name + '.pth'),
                                     map_location=device))
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Metrics helpers — identical to patchtst_finetune.py
# ---------------------------------------------------------------------------
def _metrics_at_threshold(y_true, probs, threshold):
    y_pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv         = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    return dict(
        TP=int(tp), FP=int(fp), TN=int(tn), FN=int(fn),
        sensitivity=sensitivity, specificity=specificity,
        precision=precision, npv=npv,
        f1=f1_score(y_true, y_pred, zero_division=0),
        f2=fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        mcc=matthews_corrcoef(y_true, y_pred),
        balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
    )


def _find_pr_optimal_threshold(y_true, probs):
    prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_true, probs)
    denom  = prec_arr[:-1] + rec_arr[:-1]
    f1_arr = np.where(denom > 0, 2 * prec_arr[:-1] * rec_arr[:-1] / denom, 0.0)
    best   = int(np.argmax(f1_arr))
    return float(thresh_arr[best]), float(f1_arr[best]), prec_arr, rec_arr, thresh_arr


def _interpret_sensitivity(s, n_pos):
    caught = round(s * n_pos)
    missed = n_pos - caught
    if s == 0.0:   return 'catching NO flu cases — model predicts all negative'
    elif s < 0.3:  return f'catching {caught}/{n_pos} — very poor'
    elif s < 0.6:  return f'catching {caught}/{n_pos} — moderate'
    elif s < 0.8:  return f'catching {caught}/{n_pos} — reasonable'
    else:          return f'catching {caught}/{n_pos} — strong'


def _interpret_mcc(mcc):
    if mcc <= 0.0:  return 'no useful signal (≤ random)'
    elif mcc < 0.2: return 'weak signal'
    elif mcc < 0.4: return 'moderate signal'
    elif mcc < 0.6: return 'good signal'
    else:           return 'strong signal'


def _interpret_pr_auc(pr_auc, prevalence):
    if prevalence <= 0: return ''
    lift = pr_auc / prevalence
    if lift < 2:   return f'{lift:.1f}x above random — poor'
    elif lift < 10: return f'{lift:.1f}x above random — moderate'
    elif lift < 50: return f'{lift:.1f}x above random — good'
    else:           return f'{lift:.1f}x above random — excellent'


def _print_metrics_block(label, m, threshold, n_pos):
    print(f'  {"─"*60}')
    print(f'  Threshold : {threshold:.4f}  ({label})')
    print(f'  {"─"*60}')
    print(f'  Confusion matrix:')
    print(f'    TP={m["TP"]:6d}  FN={m["FN"]:6d}   ← minimise FN')
    print(f'    FP={m["FP"]:6d}  TN={m["TN"]:6d}')
    print(f'  {"─"*60}')
    print(f'  Sensitivity : {m["sensitivity"]:.4f}  — {_interpret_sensitivity(m["sensitivity"], n_pos)}')
    print(f'  Specificity : {m["specificity"]:.4f}')
    print(f'  Precision   : {m["precision"]:.4f}')
    print(f'  NPV         : {m["npv"]:.4f}')
    print(f'  {"─"*60}')
    print(f'  F1          : {m["f1"]:.4f}')
    print(f'  F2          : {m["f2"]:.4f}')
    print(f'  MCC         : {m["mcc"]:.4f}  — {_interpret_mcc(m["mcc"])}')
    print(f'  Bal. Acc.   : {m["balanced_accuracy"]:.4f}')


def _save_curves(save_path, model_name, train_data, val_data, test_data,
                 pr_opt_thresh_train, pr_opt_thresh_val):
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

    # Mark val PR-optimal threshold
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
                      marker='*', label=f'val PR-opt={pr_opt_thresh_val:.3f}')

    prevalence = test_data[0].mean()
    ax_pr.axhline(prevalence, color='gray', lw=1, ls='--',
                  label=f'random baseline ({prevalence:.4f})')
    ax_pr.set(xlabel='Recall', ylabel='Precision',
              title='Precision-Recall Curves', xlim=[0, 1], ylim=[0, 1.02])
    ax_pr.legend(fontsize=9)
    ax_pr.grid(alpha=0.3)

    ax_roc = axes[1]
    for y_true, probs, label in [train_data, val_data, test_data]:
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, probs)
        auc_val = roc_auc_score(y_true, probs)
        ax_roc.plot(fpr, tpr, color=colors[label], lw=1.8,
                    label=f'{label}  (ROC-AUC={auc_val:.3f})')
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=1)
    ax_roc.set(xlabel='FPR', ylabel='TPR', title='ROC Curves',
               xlim=[0, 1], ylim=[0, 1.02])
    ax_roc.legend(fontsize=9)
    ax_roc.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(save_path, model_name + '_pr_roc_curves.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Curves saved → {out}')


# ---------------------------------------------------------------------------
# Test / evaluation — mirrors patchtst_finetune.py test_func
# ---------------------------------------------------------------------------
@torch.no_grad()
def _infer(model, emb, batch_size, device):
    """Run MLP on precomputed embeddings, return raw logits as numpy."""
    dl = DataLoader(TensorDataset(emb), batch_size=batch_size, shuffle=False)
    logits = []
    model.eval()
    for (xb,) in dl:
        logits.append(model(xb.to(device)).cpu())
    return torch.cat(logits).numpy()


def test_func(model, split_data, args, save_path, save_name, device):
    # ── Inference on all three splits ──
    logits_train = _infer(model, split_data['train'][0], args.batch_size, device)
    logits_val   = _infer(model, split_data['val'][0],   args.batch_size, device)
    logits_test  = _infer(model, split_data['test'][0],  args.batch_size, device)

    train_probs = 1 / (1 + np.exp(-logits_train))
    val_probs   = 1 / (1 + np.exp(-logits_val))
    test_probs  = 1 / (1 + np.exp(-logits_test))

    train_true = split_data['train'][1].numpy().astype(int)
    val_true   = split_data['val'][1].numpy().astype(int)
    test_true  = split_data['test'][1].numpy().astype(int)

    # ── PR-optimal threshold from train and val ──
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

    # ── Threshold-independent metrics (test) ──
    prevalence = test_true.mean()
    n_pos      = int(test_true.sum())

    if len(np.unique(test_true)) >= 2:
        roc_auc = roc_auc_score(test_true, test_probs)
        pr_auc  = average_precision_score(test_true, test_probs)
    else:
        roc_auc = pr_auc = float('nan')

    # ── Threshold-dependent metrics ──
    m_05  = _metrics_at_threshold(test_true, test_probs, 0.5)
    m_opt = _metrics_at_threshold(test_true, test_probs, opt_threshold) \
            if not np.isnan(opt_threshold) else None

    # ── Print ──
    print()
    print(f'  {"="*60}')
    print(f'  TEST RESULTS  —  Dual-Encoder MLP  '
          f'(n_hidden={args.n_hidden_layers}, hidden_dim={args.hidden_dim})')
    print(f'  {"="*60}')
    print(f'  Dataset (test):')
    print(f'    Total    : {len(test_true)}')
    print(f'    Positives: {n_pos}')
    print(f'    Negatives: {len(test_true) - n_pos}')
    print(f'    Prevalence: {prevalence:.5f}  ({prevalence*100:.3f}%)')
    print(f'    Random PR-AUC: {prevalence:.5f}')
    print()
    print(f'  Optimal threshold selection (PR curve, max F1):')
    print(f'    Train PR-optimal: {pr_opt_thresh_train:.4f}  (F1={pr_opt_f1_train:.4f})')
    print(f'    Val   PR-optimal: {pr_opt_thresh_val:.4f}  (F1={pr_opt_f1_val:.4f})  ← used on test')
    print()
    print(f'  Threshold-independent (test):')
    print(f'    PR-AUC  (primary) : {pr_auc:.4f}  — {_interpret_pr_auc(pr_auc, prevalence)}')
    print(f'    ROC-AUC           : {roc_auc:.4f}')
    print()
    _print_metrics_block('@0.5  (fixed reference)', m_05, 0.5, n_pos)
    if m_opt is not None:
        print()
        _print_metrics_block(
            'PR-optimal from VAL (max F1), applied to TEST',
            m_opt, opt_threshold, n_pos,
        )
    print(f'  {"="*60}')

    # ── Curves ──
    _save_curves(
        save_path, save_name,
        train_data=(train_true, train_probs, 'train'),
        val_data=(val_true,   val_probs,   'val'),
        test_data=(test_true,  test_probs,  'test'),
        pr_opt_thresh_train=pr_opt_thresh_train,
        pr_opt_thresh_val=pr_opt_thresh_val,
    )

    # ── CSV ──
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
        os.path.join(save_path, save_name + '_acc.csv'),
        float_format='%.6f', index=False,
    )


# ---------------------------------------------------------------------------
# Data loading shim
# ---------------------------------------------------------------------------
def make_data_args(args):
    class _Args:
        dset           = args.dset
        context_points = args.context_points
        target_points  = 1
        batch_size     = args.backbone_batch_size
        num_workers    = args.num_workers
        scaler         = 'standard'
        features       = 'M'
        patch_len      = args.patch_len
        stride         = args.stride
        revin          = args.revin
        label_filter   = 'all'
    return _Args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = build_parser().parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')
    print(f'Positive model : {args.positive_model}')
    print(f'Negative model : {args.negative_model}')
    print(f'MLP depth      : {args.n_hidden_layers} hidden layers  '
          f'(0 = linear probe)')

    # ── Save path ──
    save_path = os.path.join(
        'saved_models', args.dset, 'masked_patchtst', args.model_type
    )
    os.makedirs(save_path, exist_ok=True)

    save_name = (
        f'{args.dset}_dual_encoder'
        f'_hl{args.n_hidden_layers}'
        f'_hd{args.hidden_dim}'
        f'_epochs{args.n_epochs}'
        f'_model{args.model_id}'
    )
    print(f'Save path : {save_path}/{save_name}')

    # ── Load frozen backbones ──
    print('\nLoading backbones …')
    pos_model = load_backbone(args.positive_model, args, device)
    neg_model = load_backbone(args.negative_model, args, device)

    revin = RevIN(num_features=args.c_in, eps=1e-5, affine=False).to(device)

    # ── Data ──
    print('\nLoading data …')
    dls = get_dls(make_data_args(args))

    # ── Extract embeddings once ──
    print('\nExtracting embeddings from both frozen backbones …')
    split_data = extract_all_embeddings(pos_model, neg_model, dls, revin, args, device)

    # ── Train MLP ──
    print('\n' + '='*60)
    print('Training MLP …')
    print('='*60)
    model = train(split_data, args, save_path, save_name, device)

    # ── Evaluate ──
    print('\n' + '='*60)
    print('Evaluating on test set …')
    print('='*60)
    test_func(model, split_data, args, save_path, save_name, device)

    print('\n----------- Complete! -----------')


if __name__ == '__main__':
    main()
