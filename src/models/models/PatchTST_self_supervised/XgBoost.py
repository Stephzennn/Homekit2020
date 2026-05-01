"""
XgBoost.py — PatchTST backbone + XGBoost classifier for flu detection.

Two modes:

  Train (--is_xgboost 1):
    Extracts embeddings from all splits using the frozen PatchTST backbone,
    trains XGBoost, saves the model + metrics CSV + PR/ROC curves.

  Test-only (default, --is_xgboost 0):
    Extracts val + test embeddings, loads a saved XGBoost model, evaluates.

Embedding source:

  Finetuned checkpoint (default):
    Pre-classification dropout output [B x nvars*d_model].

  Pretrained checkpoint (--pretrain):
    Backbone output: last patch → flatten → [B x nvars*d_model].

Usage (run from the PatchTST_self_supervised directory):

  Train:
    python XgBoost.py --is_xgboost 1 --pretrain \
        --checkpoint /path/to/pretrained.pth \
        --out_dir    /path/to/output \
        --c_in 8 --d_model 256 --n_layers 4 --n_heads 8 --d_ff 512 \
        --patch_len 1440 --stride 180 --context_points 10080

  Test-only:
    python XgBoost.py \
        --checkpoint /path/to/pretrained.pth \
        --out_dir    /path/to/output \
        --c_in 8 --d_model 256 --n_layers 4 --n_heads 8 --d_ff 512 \
        --patch_len 1440 --stride 180 --context_points 10080
"""

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# Path setup — mirrors what patchtst_finetune.py does
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

_PROJECT_ROOT = "/home/hice1/ezg6/projects/Homekit2020/src"
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
import joblib
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_recall_curve, roc_curve,
    f1_score, fbeta_score, matthews_corrcoef,
    balanced_accuracy_score, confusion_matrix,
)

from src.models.patchTST import PatchTST
from src.models.layers.revin import RevIN
from src.callback.patch_mask import create_patch
from datautils import get_dls
from xgboost import XGBClassifier


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        description="PatchTST backbone + XGBoost classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Mode ---
    p.add_argument("--is_xgboost", type=int, default=0,
                   help="1 = train XGBoost; 0 = test-only (load saved model)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    # --- Paths ---
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to the PatchTST .pth checkpoint")
    p.add_argument("--out_dir", type=str, default="saved_models/xgboost",
                   help="Directory where outputs (model, CSV, plots) are saved")
    p.add_argument("--xgb_model_id", type=int, default=1,
                   help="Integer suffix for the saved XGBoost model filename")

    # --- Architecture (must match the checkpoint) ---
    p.add_argument("--c_in",         type=int,   default=8,   help="Number of input channels")
    p.add_argument("--d_model",      type=int,   default=256, help="Transformer d_model")
    p.add_argument("--n_layers",     type=int,   default=6,   help="Number of Transformer layers")
    p.add_argument("--n_heads",      type=int,   default=8,   help="Number of attention heads")
    p.add_argument("--d_ff",         type=int,   default=512, help="Transformer FFN hidden dim")
    p.add_argument("--dropout",      type=float, default=0.2, help="Transformer dropout")
    p.add_argument("--head_dropout", type=float, default=0.2, help="Classification head dropout")
    p.add_argument("--head_type", type=str, default="classification",
                   choices=["classification", "resnet_classification"],
                   help="Head architecture (ignored when --pretrain is set)")

    # --- Pretrain mode ---
    p.add_argument("--pretrain", action="store_true", default=False,
                   help="Load a pretrained checkpoint; discard pretrain head and use backbone output")

    # --- Patching ---
    p.add_argument("--patch_len",      type=int, default=1440, help="Patch length")
    p.add_argument("--stride",         type=int, default=180,  help="Stride between patches")
    p.add_argument("--context_points", type=int, default=10080, help="Sequence length")

    # --- Data ---
    p.add_argument("--batch_size",  type=int, default=32, help="Batch size for inference")
    p.add_argument("--num_workers", type=int, default=0,  help="DataLoader worker processes")
    p.add_argument("--revin",       type=int, default=1,  help="Use RevIN normalization (1=yes)")
    p.add_argument("--use_raw_features", action="store_true", default=False,
                   help="Concatenate per-channel raw signal statistics (mean/std/min/max) "
                        "to the model embedding before XGBoost training")

    # --- t-SNE ---
    p.add_argument("--skip_tsne",       action="store_true", default=False,
                   help="Skip t-SNE visualization")
    p.add_argument("--tsne_perplexity", type=float, default=None,
                   help="t-SNE perplexity (defaults to min(30, N//10))")
    p.add_argument("--tsne_n_iter",     type=int,   default=1000, help="t-SNE iterations")

    # --- Splits ---
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                   choices=["train", "val", "test"],
                   help="Which splits to extract (train mode). Test-only always uses val+test.")

    # --- XGBoost hyperparameters ---
    p.add_argument("--xgb_n_estimators",          type=int,   default=2000,
                   help="Upper bound on trees; early stopping decides the actual number")
    p.add_argument("--xgb_early_stopping_rounds", type=int,   default=50,
                   help="Stop if val aucpr doesn't improve for this many rounds")
    p.add_argument("--xgb_lr",               type=float, default=0.03)
    p.add_argument("--xgb_max_depth",        type=int,   default=3)
    p.add_argument("--xgb_min_child_weight", type=float, default=5)
    p.add_argument("--xgb_subsample",        type=float, default=0.8)
    p.add_argument("--xgb_colsample_bytree", type=float, default=0.8)
    p.add_argument("--xgb_reg_lambda",       type=float, default=1.0)
    p.add_argument("--xgb_reg_alpha",        type=float, default=0.0)
    p.add_argument("--xgb_tree_method",      type=str,   default="hist")

    # --- Class imbalance ---
    p.add_argument("--scale_pos_weight_mult", type=float, default=0.0,
                   help="Multiply (n_neg/n_pos) by this to get scale_pos_weight. "
                        "0 = disabled (no positive weighting). E.g. 2.0 = 2x ratio.")
    p.add_argument("--neg_subsample_ratio",   type=int,   default=0,
                   help="Keep this many negatives per positive in the training set. "
                        "0 = disabled (use all negatives).")

    return p


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------
def build_model(args, device):
    num_patch = (max(args.context_points, args.patch_len) - args.patch_len) // args.stride + 1
    print(f"num_patch = {num_patch}   embedding_dim = {args.c_in * args.d_model}")

    if args.pretrain:
        # Pretrain head expects (d_model, patch_len, dropout) and reconstructs patches.
        # target_dim is patch_len for the pretrain head's linear projection.
        head_type  = "pretrain"
        target_dim = args.patch_len
    else:
        head_type  = args.head_type
        target_dim = 1

    model = PatchTST(
        c_in=args.c_in,
        target_dim=target_dim,
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
        act="relu",
        head_type=head_type,
        res_attention=False,
    )

    state = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]

    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing:
        print(f"[WARN] Missing keys: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected}")

    model.eval()
    model.to(device)
    mode_label = "pretrained (backbone only)" if args.pretrain else f"finetuned ({head_type} head)"
    print(f"Loaded [{mode_label}]: {args.checkpoint}")
    return model


# ---------------------------------------------------------------------------
# Forward hooks — two modes
#
# Finetuned (classification head):
#   ClassificationHead.forward:
#     x = x[:,:,:,-1]      [B x nvars x d_model]   (last patch)
#     x = flatten(x)       [B x nvars*d_model]
#     x = dropout(x)       [B x nvars*d_model]  ← hook fires HERE (post-dropout)
#     y = linear(x)        [B x 1]              ← skipped
#
# Pretrained (backbone only):
#   PatchTST.forward calls backbone → head.
#   We intercept the backbone output at the INPUT of model.head using a
#   forward_pre_hook, then aggregate it the same way the classification head
#   would: last patch → flatten → [B x nvars*d_model].
# ---------------------------------------------------------------------------
def register_finetune_embedding_hook(model):
    """Hook on head.dropout output — works for classification & resnet_classification heads."""
    captured = []

    def _hook(module, inp, out):
        captured.append(out.detach().cpu())

    handle = model.head.dropout.register_forward_hook(_hook)
    return captured, handle


def register_pretrain_embedding_hook(model):
    """
    Hook on the INPUT of model.head (= backbone output) using a pre-hook.
    Aggregates [B x nvars x d_model x num_patch] → [B x nvars*d_model]
    by taking the last patch and flattening — identical to what the
    classification head does before its linear layer.
    """
    captured = []

    def _pre_hook(module, inp):
        # inp is a tuple; inp[0] is the backbone output [B x nvars x d_model x num_patch]
        x = inp[0].detach().cpu()           # [B x nvars x d_model x num_patch]
        x = x[:, :, :, -1]                 # [B x nvars x d_model]  (last patch)
        x = x.flatten(start_dim=1)         # [B x nvars * d_model]
        captured.append(x)

    handle = model.head.register_forward_pre_hook(_pre_hook)
    return captured, handle


# ---------------------------------------------------------------------------
# Data loading shim
# ---------------------------------------------------------------------------
def make_data_args(args):
    class _Args:
        dset           = "Wearable"
        context_points = args.context_points
        target_points  = 1
        batch_size     = args.batch_size
        num_workers    = args.num_workers
        scaler         = "standard"
        features       = "M"
        patch_len      = args.patch_len
        stride         = args.stride
        revin          = args.revin
    return _Args()


# ---------------------------------------------------------------------------
# Embedding extraction for one DataLoader
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_embeddings(model, dataloader, revin, captured, args, device):
    all_emb = []
    all_lbl = []
    all_raw = []

    for batch in dataloader:
        if isinstance(batch, dict):
            xb = batch["inputs_embeds"].float().to(device)
            yb = batch["label"].float()
        elif isinstance(batch, (list, tuple)):
            xb = batch[0].float().to(device)
            yb = batch[1].float()
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

        # RevIN normalization
        if args.revin:
            xb = revin(xb, "norm")

        # Raw per-channel statistics: xb is [B x T x C]
        # Produces [B x C*4] — mean, std, min, max per channel over time
        raw_mean = xb.mean(dim=1)
        raw_std  = xb.std(dim=1)
        raw_min  = xb.min(dim=1).values
        raw_max  = xb.max(dim=1).values
        all_raw.append(
            torch.cat([raw_mean, raw_std, raw_min, raw_max], dim=1).detach().cpu()
        )

        # Patching  [B x T x C] → [B x num_patch x C x patch_len]
        xb_patch, _ = create_patch(xb, args.patch_len, args.stride)

        # Forward — hook captures the embedding
        _ = model(xb_patch)

        all_emb.append(captured.pop())
        all_lbl.append(yb.reshape(-1).cpu())

    if not all_emb:
        return None, None, None

    return (
        torch.cat(all_emb, dim=0).numpy(),
        torch.cat(all_lbl, dim=0).numpy(),
        torch.cat(all_raw, dim=0).numpy(),
    )


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def run_tsne(embeddings, args, n_samples):
    perplexity = args.tsne_perplexity or min(30, max(5, n_samples // 10))
    print(f"  t-SNE: perplexity={perplexity}, n_iter={args.tsne_n_iter}")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=args.tsne_n_iter,
        random_state=42,
        verbose=0,
    )
    return tsne.fit_transform(embeddings)


def save_individual_plot(z, labels, split_name, args, out_path):
    n_total = len(labels)
    n_pos   = int((labels == 1).sum())
    n_neg   = int((labels == 0).sum())

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(z[labels == 0, 0], z[labels == 0, 1],
               c="#4878d0", alpha=0.55, s=18, linewidths=0,
               label=f"Flu-Negative  (n={n_neg})")
    ax.scatter(z[labels == 1, 0], z[labels == 1, 1],
               c="#d65f5f", alpha=0.75, s=28, linewidths=0,
               label=f"Flu-Positive  (n={n_pos})")

    mode_tag = "pretrained backbone" if args.pretrain else "pre-classification"
    ax.set_title(
        f"{mode_tag} embedding (t-SNE) — {split_name} set\n"
        f"patch={args.patch_len}  stride={args.stride}  "
        f"d_model={args.d_model}  |  N={n_total}",
        fontsize=11,
    )
    ax.set_xlabel("t-SNE dim 1", fontsize=10)
    ax.set_ylabel("t-SNE dim 2", fontsize=10)
    ax.legend(fontsize=9, markerscale=1.8, framealpha=0.8)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def save_combined_plot(split_data, args, out_path):
    splits   = list(split_data.keys())
    n_splits = len(splits)
    fig, axes = plt.subplots(1, n_splits, figsize=(6 * n_splits, 5.5))
    if n_splits == 1:
        axes = [axes]

    for ax, split_name in zip(axes, splits):
        labels, z = split_data[split_name]
        n_total = len(labels)
        n_pos   = int((labels == 1).sum())
        n_neg   = int((labels == 0).sum())

        ax.scatter(z[labels == 0, 0], z[labels == 0, 1],
                   c="#4878d0", alpha=0.55, s=14, linewidths=0,
                   label=f"Negative (n={n_neg})")
        ax.scatter(z[labels == 1, 0], z[labels == 1, 1],
                   c="#d65f5f", alpha=0.75, s=22, linewidths=0,
                   label=f"Positive (n={n_pos})")

        ax.set_title(f"{split_name}  (N={n_total})", fontsize=11)
        ax.set_xlabel("t-SNE dim 1", fontsize=9)
        ax.set_ylabel("t-SNE dim 2", fontsize=9)
        ax.legend(fontsize=8, markerscale=1.5, framealpha=0.8)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    mode_tag = "pretrained backbone" if args.pretrain else "pre-classification"
    fig.suptitle(
        f"PatchTST {mode_tag} embeddings (t-SNE)\n"
        f"patch={args.patch_len}  stride={args.stride}  "
        f"d_model={args.d_model}  embedding_dim={args.c_in * args.d_model}",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Combined figure saved → {out_path}")

def train_xgboost_from_split_embeddings(split_embeddings, args):
    """
    Trains XGBoost using embeddings already extracted into RAM.

    split_embeddings format:
        {
            "train": (X_train, y_train),
            "val":   (X_val, y_val),
            "test":  (X_test, y_test)
        }
    """

    X_train, y_train = split_embeddings["train"]
    X_val, y_val = split_embeddings["val"]
    X_test, y_test = split_embeddings["test"]

    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    X_test = X_test.astype(np.float32)

    y_train = y_train.astype(np.int64)
    y_val = y_val.astype(np.int64)
    y_test = y_test.astype(np.int64)

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())

    if n_pos == 0:
        raise ValueError("No positive examples in the training split.")

    # --- Negative undersampling ---
    neg_subsample_ratio = getattr(args, "neg_subsample_ratio", 0)
    if neg_subsample_ratio > 0:
        rng = np.random.default_rng(getattr(args, "seed", 42))
        pos_idx = np.where(y_train == 1)[0]
        neg_idx = np.where(y_train == 0)[0]
        keep_n  = min(len(neg_idx), len(pos_idx) * neg_subsample_ratio)
        kept_neg = rng.choice(neg_idx, size=keep_n, replace=False)
        idx = np.concatenate([pos_idx, kept_neg])
        rng.shuffle(idx)
        X_train, y_train = X_train[idx], y_train[idx]
        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())

    # --- Positive class weighting ---
    scale_pos_weight_mult = getattr(args, "scale_pos_weight_mult", 0.0)
    if scale_pos_weight_mult > 0:
        scale_pos_weight = scale_pos_weight_mult * (n_neg / n_pos)
    else:
        scale_pos_weight = 1.0

    print("\n--- XGBoost training ---")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_val shape:   {X_val.shape}")
    print(f"  X_test shape:  {X_test.shape}")
    print(f"  Train positives: {n_pos}")
    print(f"  Train negatives: {n_neg}")
    if neg_subsample_ratio > 0:
        print(f"  Negative undersampling: {neg_subsample_ratio}:1 ratio")
    if scale_pos_weight_mult > 0:
        print(f"  scale_pos_weight: {scale_pos_weight:.2f}  ({scale_pos_weight_mult}x ratio={n_neg/n_pos:.1f})")
    else:
        print(f"  scale_pos_weight: disabled (no positive weighting)")

    clf = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",

        n_estimators=getattr(args, "xgb_n_estimators", 2000),
        learning_rate=getattr(args, "xgb_lr", 0.03),
        max_depth=getattr(args, "xgb_max_depth", 3),
        min_child_weight=getattr(args, "xgb_min_child_weight", 5),

        subsample=getattr(args, "xgb_subsample", 0.8),
        colsample_bytree=getattr(args, "xgb_colsample_bytree", 0.8),

        reg_lambda=getattr(args, "xgb_reg_lambda", 1.0),
        reg_alpha=getattr(args, "xgb_reg_alpha", 0.0),

        scale_pos_weight=scale_pos_weight,

        random_state=getattr(args, "seed", 42),
        n_jobs=-1,
        tree_method=getattr(args, "xgb_tree_method", "hist"),
    )

    early_stopping_rounds = getattr(args, "xgb_early_stopping_rounds", 50)
    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=early_stopping_rounds,
        verbose=50,
    )

    print(f"\n  Early stopping fired at round {clf.best_iteration} "
          f"(best val aucpr = {clf.best_score:.4f})")
    print(f"  Model will predict using {clf.best_iteration + 1} trees.")

    return clf

# ---------------------------------------------------------------------------
# Metrics helpers (ported from patchtst_finetune.py)
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
    best_idx = int(np.argmax(f1_arr))
    return float(thresh_arr[best_idx]), float(f1_arr[best_idx])


def _interpret_sensitivity(s, n_pos):
    caught, missed = round(s * n_pos), n_pos - round(s * n_pos)
    if s == 0.0:
        return 'catching NO flu cases — model predicts all negative at this threshold'
    elif s < 0.3:
        return f'catching only {caught}/{n_pos} flu cases — missing {missed} ({(1-s)*100:.0f}%) — very poor'
    elif s < 0.6:
        return f'catching {caught}/{n_pos} flu cases — missing {missed} ({(1-s)*100:.0f}%) — moderate'
    elif s < 0.8:
        return f'catching {caught}/{n_pos} flu cases — missing {missed} ({(1-s)*100:.0f}%) — reasonable'
    else:
        return f'catching {caught}/{n_pos} flu cases — missing only {missed} ({(1-s)*100:.0f}%) — strong'


def _interpret_mcc(mcc):
    if mcc <= 0.0:   return 'no useful signal (≤ random)'
    elif mcc < 0.2:  return 'weak signal'
    elif mcc < 0.4:  return 'moderate signal'
    elif mcc < 0.6:  return 'good signal'
    else:            return 'strong signal'


def _interpret_pr_auc(pr_auc, prevalence):
    if prevalence <= 0:  return ''
    lift = pr_auc / prevalence
    if lift < 2:    return f'{lift:.1f}x above random — poor'
    elif lift < 10: return f'{lift:.1f}x above random — moderate'
    elif lift < 50: return f'{lift:.1f}x above random — good'
    else:           return f'{lift:.1f}x above random — excellent'


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


# ---------------------------------------------------------------------------
# XGBoost results: CSV + PR/ROC curves
# ---------------------------------------------------------------------------
def _save_xgb_results(clf, split_embeddings, args):
    """Evaluate clf on val and test, print full metrics, save CSV + curves."""
    xgb_model_name = f"xgboost_model_{args.xgb_model_id}"

    print("\n--- XGBoost evaluation ---")

    # Collect probs/labels for val and test
    split_results = {}
    for split_name in ("val", "test"):
        if split_name not in split_embeddings:
            continue
        X, y = split_embeddings[split_name]
        probs = clf.predict_proba(X)[:, 1]
        if len(np.unique(y)) < 2:
            print(f"  {split_name}: only one class present, skipping metrics")
            continue
        split_results[split_name] = (y, probs)

    if not split_results:
        print("  No results to save.")
        return

    # Derive PR-optimal threshold from val, applied blind to test
    if "val" in split_results:
        val_y, val_probs = split_results["val"]
        pr_opt_thresh_val, pr_opt_f1_val = _find_pr_optimal_threshold(val_y, val_probs)
    else:
        pr_opt_thresh_val, pr_opt_f1_val = 0.5, float("nan")

    # Print full metrics block for each split
    for split_name, (y, probs) in split_results.items():
        pr_auc     = average_precision_score(y, probs)
        roc_auc    = roc_auc_score(y, probs)
        prevalence = y.mean()
        n_pos      = int(y.sum())

        print()
        print(f'  {"="*60}')
        print(f'  {split_name.upper()} RESULTS')
        print(f'  {"="*60}')
        print(f'  Dataset:')
        print(f'    Total samples  : {len(y)}')
        print(f'    Positives      : {n_pos}  (flu-onset windows)')
        print(f'    Negatives      : {len(y) - n_pos}')
        print(f'    Prevalence     : {prevalence:.5f}  ({prevalence*100:.3f}%)')
        print(f'    Random PR-AUC  : {prevalence:.5f}  (baseline)')
        print()
        print(f'  Threshold-independent metrics:')
        print(f'    PR-AUC  (primary) : {pr_auc:.4f}  — {_interpret_pr_auc(pr_auc, prevalence)}')
        print(f'    ROC-AUC           : {roc_auc:.4f}')
        print()

        m_05 = _metrics_at_threshold(y, probs, threshold=0.5)
        _print_metrics_block('@0.5  (fixed reference)', m_05, threshold=0.5, n_pos=n_pos)

        if not np.isnan(pr_opt_thresh_val):
            print()
            if split_name == "val":
                print(f'  Val PR-optimal threshold : {pr_opt_thresh_val:.4f}  (best F1={pr_opt_f1_val:.4f} on val)')
                _print_metrics_block(
                    'PR-optimal threshold (max F1 on val PR curve)',
                    _metrics_at_threshold(y, probs, pr_opt_thresh_val),
                    threshold=pr_opt_thresh_val, n_pos=n_pos,
                )
            else:
                print(f'  Val PR-optimal threshold applied blind to test: {pr_opt_thresh_val:.4f}  (best F1={pr_opt_f1_val:.4f} on val)')
                _print_metrics_block(
                    'PR-optimal threshold from VAL (max F1 on val PR curve), applied to TEST',
                    _metrics_at_threshold(y, probs, pr_opt_thresh_val),
                    threshold=pr_opt_thresh_val, n_pos=n_pos,
                )
        print(f'  {"="*60}')

    # CSV — threshold-independent + @0.5 + @opt for each split
    row = {"val_pr_opt_threshold": pr_opt_thresh_val}
    for split_name, (y, probs) in split_results.items():
        row[f"{split_name}_pr_auc"]  = average_precision_score(y, probs)
        row[f"{split_name}_roc_auc"] = roc_auc_score(y, probs)
        for k, v in _metrics_at_threshold(y, probs, 0.5).items():
            row[f"{split_name}_@0.5_{k}"] = v
        if not np.isnan(pr_opt_thresh_val):
            for k, v in _metrics_at_threshold(y, probs, pr_opt_thresh_val).items():
                row[f"{split_name}_@opt_{k}"] = v
    csv_path = os.path.join(args.out_dir, f"{xgb_model_name}_results.csv")
    pd.DataFrame([row]).to_csv(csv_path, float_format="%.6f", index=False)
    print(f"\n  Results CSV saved → {csv_path}")

    # PR + ROC curves
    colors = {"val": "#ff7f0e", "test": "#2ca02c"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax_pr, ax_roc = axes

    for split_name, (y, probs) in split_results.items():
        prec, rec, _ = precision_recall_curve(y, probs)
        fpr, tpr, _  = roc_curve(y, probs)
        pr_auc  = average_precision_score(y, probs)
        roc_auc = roc_auc_score(y, probs)
        ax_pr.plot(rec, prec, color=colors[split_name], lw=1.8,
                   label=f"{split_name}  (PR-AUC={pr_auc:.3f})")
        ax_roc.plot(fpr, tpr, color=colors[split_name], lw=1.8,
                    label=f"{split_name}  (ROC-AUC={roc_auc:.3f})")

    if "test" in split_results:
        prevalence = split_results["test"][0].mean()
        ax_pr.axhline(prevalence, color="gray", lw=1, ls="--",
                      label=f"random baseline ({prevalence:.4f})")

    ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curves"); ax_pr.legend(fontsize=9)
    ax_pr.set_xlim([0, 1]); ax_pr.set_ylim([0, 1.02]); ax_pr.grid(alpha=0.3)

    ax_roc.plot([0, 1], [0, 1], "k--", lw=1, label="random baseline")
    ax_roc.set_xlabel("False Positive Rate"); ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curves"); ax_roc.legend(fontsize=9)
    ax_roc.set_xlim([0, 1]); ax_roc.set_ylim([0, 1.02]); ax_roc.grid(alpha=0.3)

    plt.tight_layout()
    curves_path = os.path.join(args.out_dir, f"{xgb_model_name}_pr_roc_curves.png")
    plt.savefig(curves_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Curves saved → {curves_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = build_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Mode: {'train XGBoost' if args.is_xgboost else 'test-only'}\n")

    model = build_model(args, device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    if args.pretrain:
        captured, hook = register_pretrain_embedding_hook(model)
        print("Embedding source: backbone output (last patch, flattened) — pretrain mode")
    else:
        captured, hook = register_finetune_embedding_hook(model)
        print("Embedding source: pre-classification dropout output — finetune mode")

    revin = RevIN(num_features=args.c_in, eps=1e-5, affine=False).to(device)

    print("\nLoading data …")
    dls = get_dls(make_data_args(args))
    split_map = {"train": dls.train, "val": dls.valid, "test": dls.test}

    # In test-only mode we only need val+test; training mode uses args.splits.
    splits_to_extract = args.splits if args.is_xgboost else ["val", "test"]

    split_data       = {}   # {split_name: (labels, z_2d)} for t-SNE combined plot
    split_embeddings = {}   # {split_name: (embeddings, labels)} for XGBoost

    for split_name in splits_to_extract:
        dl = split_map.get(split_name)

        if dl is None:
            print(f"\n{split_name}: no data available, skipping.")
            continue

        print(f"\n--- {split_name} ---")
        embeddings, labels, raw_feats = extract_embeddings(model, dl, revin, captured, args, device)

        if embeddings is None:
            print(f"  {split_name}: empty dataloader, skipping.")
            continue

        embeddings = embeddings.astype(np.float32)
        labels     = labels.astype(np.int64)

        if args.use_raw_features:
            embeddings = np.concatenate([embeddings, raw_feats.astype(np.float32)], axis=1)
            print(f"  Raw features appended: {raw_feats.shape[1]} stats (mean/std/min/max per channel)")

        n_pos = int((labels == 1).sum())
        n_neg = int((labels == 0).sum())
        print(f"  Samples: {len(labels)}  (pos={n_pos}, neg={n_neg})")
        print(f"  Feature dim: {embeddings.shape[1]}")

        split_embeddings[split_name] = (embeddings, labels)

        if not args.skip_tsne:
            z = run_tsne(embeddings, args, len(labels))
            out_path = os.path.join(args.out_dir, f"embedding_tsne_{split_name}.png")
            save_individual_plot(z, labels, split_name, args, out_path)
            split_data[split_name] = (labels, z)

    hook.remove()

    if not args.skip_tsne and len(split_data) > 1:
        combined_path = os.path.join(args.out_dir, "embedding_tsne_all_splits.png")
        save_combined_plot(split_data, args, combined_path)

    xgb_model_path = os.path.join(args.out_dir, f"xgboost_model_{args.xgb_model_id}.pkl")

    if args.is_xgboost:
        required = {"train", "val", "test"}
        missing  = required - set(split_embeddings.keys())
        if missing:
            print(f"\nSkipping XGBoost: missing splits {sorted(missing)}")
        else:
            clf = train_xgboost_from_split_embeddings(split_embeddings, args)
            joblib.dump(clf, xgb_model_path)
            print(f"\nXGBoost model saved → {xgb_model_path}")
            _save_xgb_results(clf, split_embeddings, args)
    else:
        print(f"\nTest-only mode — loading XGBoost model from: {xgb_model_path}")
        clf = joblib.load(xgb_model_path)
        _save_xgb_results(clf, split_embeddings, args)

    print("\nDone.")



if __name__ == "__main__":
    main()
