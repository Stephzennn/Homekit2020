# =============================================================================
# phase3_finetune.py — Phase 3: Fine-tuning ClusterPatchTST for Classification
# =============================================================================
#
# PIPELINE POSITION:
#   Phase 1  → discover task-driven patches via clustering
#   Phase 2  → masked self-supervised pretraining on those patches
#   Phase 3  → THIS FILE: fine-tune for flu positivity classification
#
# RELATION TO patchtst_finetune.py:
#   Mirrors patchtst_finetune.py but replaces:
#     PatchCB          → ClusterPatchCB (cluster-based patches, no masking)
#     PatchTST         → ClusterPatchTST (head_type='classification')
#     patch_len/stride → patch_learner_path/meta_path/pop_percentile
#
#   The pre-trained backbone (W_P + transformer layers) is loaded via
#   transfer_weights, which skips any weight whose shape does not match
#   (e.g. the pretrain head Linear(d_model, P*C) is automatically skipped
#   when the classification head is Linear(d_model, 1)).
#
# USAGE — linear probe (freeze backbone, train head only):
#   python phase3_finetune.py \
#     --is_linear_probe 1 \
#     --dset_finetune Wearable \
#     --context_points 10080 \
#     --patch_learner_path saved_models/.../model3_patch_learner.pth \
#     --meta_path         saved_models/.../model3_meta.json \
#     --pretrained_model  saved_models/.../clusterpatchtst_pretrained_....pth \
#     --n_epochs_finetune 20 \
#     --n_layers 6 --n_heads 8 --d_model 256 --d_ff 512 \
#     --model_type LearnedPatch_phase3
#
# USAGE — end-to-end fine-tuning:
#   python phase3_finetune.py \
#     --is_finetune 1 \
#     [same flags as above]
#
# USAGE — test-only (no training):
#   python phase3_finetune.py \
#     --pretrained_model  saved_models/.../clusterpatchtst_finetuned_....pth \
#     [same data/arch flags]
# =============================================================================

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score,
    recall_score, f1_score, fbeta_score, matthews_corrcoef,
    balanced_accuracy_score, confusion_matrix, roc_curve,
)

_PATCHTST_DIR = os.path.join(os.path.dirname(__file__), '..', 'PatchTST_self_supervised')
# Only add PatchTST_self_supervised — same pattern as phase2_pretrain.py.
# Python/torchrun automatically inserts the script's own directory (LearnedPatchTST/)
# at sys.path[0], so phase2_pretrain and patch_learner are importable without a manual insert.
# Adding _PATCHTST_DIR at [0] ensures its 'src' package resolves before any installed 'src'.
sys.path.insert(0, os.path.abspath(_PATCHTST_DIR))

from datautils import get_dls
from src.learner import Learner, transfer_weights
from src.callback.core import Callback
from src.callback.tracking import *
from src.callback.transforms import RevInCB
from src.basics import set_device

# Import shared components from Phase 2
from phase2_pretrain import (
    ClusterPatchTST,
    ClusterPatchCB,
    setup_ddp,
    load_patch_learner_and_compute_P,
    _verify_P_vs_checkpoint,
)


# =============================================================================
# ValidationROCAUCCB — copied from patchtst_finetune.py
# (cannot import from that file — it has module-level args = parser.parse_args())
# =============================================================================
class ValidationROCAUCCB(Callback):
    """Computes ROC-AUC on the full validation set after every epoch."""

    def before_fit(self):
        if self.run_finder:
            return
        if hasattr(self.learner, 'recorder'):
            self.learner.recorder['valid_roc_auc'] = []

    def before_epoch_valid(self):
        self._preds = []
        self._targs = []

    def after_batch_valid(self):
        self._preds.append(self.pred.detach().cpu())
        self._targs.append(self.yb.detach().cpu())

    def after_epoch_valid(self):
        if not self.learner.dls.valid:
            return

        preds = torch.cat(self._preds)
        targs = torch.cat(self._targs)

        if dist.is_initialized() and dist.get_world_size() > 1:
            world_size = dist.get_world_size()
            all_preds = [None] * world_size
            all_targs = [None] * world_size
            dist.all_gather_object(all_preds, preds)
            dist.all_gather_object(all_targs, targs)
            preds = torch.cat(all_preds)
            targs = torch.cat(all_targs)

        probs  = torch.sigmoid(preds).numpy().reshape(-1)
        y_true = targs.numpy().reshape(-1)

        if len(np.unique(y_true)) < 2:
            roc_auc = 0.0
        else:
            roc_auc = roc_auc_score(y_true, probs)

        self.learner.recorder['valid_roc_auc'].append(roc_auc)

        _rank = int(os.environ.get("RANK", 0))
        if _rank == 0:
            print(f"  [val ROC-AUC: {roc_auc:.4f}]")


# =============================================================================
# Argument parser
# =============================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Phase 3: Cluster-based fine-tuning')

    parser.add_argument('--is_finetune',      type=int, default=0,
                        help='run end-to-end fine-tuning (1=yes)')
    parser.add_argument('--is_linear_probe',  type=int, default=0,
                        help='run linear probe — freeze backbone, train head only (1=yes)')

    # Data
    parser.add_argument('--dset_finetune',   type=str, default='Wearable')
    parser.add_argument('--context_points',  type=int, default=10080)
    parser.add_argument('--target_points',   type=int, default=1,
                        help='number of output logits; 1 for binary classification')
    parser.add_argument('--batch_size',      type=int, default=16)
    parser.add_argument('--num_workers',     type=int, default=2)
    parser.add_argument('--scaler',          type=str, default='standard')
    parser.add_argument('--features',        type=str, default='M')
    parser.add_argument('--revin',           type=int, default=1,
                        help='reversible instance normalisation (denorm=False for classification)')

    # Phase 1 outputs
    parser.add_argument('--patch_learner_path', type=str, required=True,
                        help='path to *_patch_learner.pth from Phase 1')
    parser.add_argument('--meta_path',          type=str, required=True,
                        help='path to *_meta.json from Phase 1')

    # Approach C: cluster patch content size
    parser.add_argument('--pop_percentile', type=int, default=75,
                        help='percentile of natural cluster population used to set P '
                             '(must match Phase 2 value used to build the pretrained model)')

    # Model architecture — must match Phase 2 checkpoint
    parser.add_argument('--n_layers',      type=int,   default=6)
    parser.add_argument('--n_heads',       type=int,   default=8)
    parser.add_argument('--d_model',       type=int,   default=256)
    parser.add_argument('--d_ff',          type=int,   default=512)
    parser.add_argument('--dropout',       type=float, default=0.1)
    parser.add_argument('--head_dropout',  type=float, default=0.1)

    # Training
    parser.add_argument('--n_epochs_finetune', type=int,   default=20)
    parser.add_argument('--lr',                type=float, default=1e-4)
    parser.add_argument('--use_lr_finder',     type=int,   default=1,
                        help='run LR finder before training (1=yes, 0=use --lr directly)')

    # Checkpoints
    parser.add_argument('--pretrained_model', type=str, default=None,
                        help='path to Phase 2 pretrained checkpoint (.pth)')
    parser.add_argument('--linear_probe_model', type=str, default=None,
                        help='path to a linear probe checkpoint to warm-start full finetuning')
    parser.add_argument('--finetuned_model_id', type=int, default=1,
                        help='integer suffix for saved model filename')
    parser.add_argument('--model_type',         type=str, default='LearnedPatch_phase3')

    # Class imbalance
    parser.add_argument('--pos_weight_cap', type=float, default=-1.0,
                        help='cap on BCEWithLogitsLoss pos_weight (n_neg/n_pos); -1 = no cap')

    # Logging
    parser.add_argument('--use_wandb',       type=int, default=0)
    parser.add_argument('--wandb_project',   type=str, default='LearnedPatchTST')
    parser.add_argument('--wandb_run_name',  type=str, default=None)

    return parser


# =============================================================================
# Helpers
# =============================================================================
def compute_pos_weight(labels, device, pos_weight_cap):
    n_pos = int((labels == 1).sum().item())
    n_neg = int((labels == 0).sum().item())
    raw_ratio = n_neg / n_pos
    capped = raw_ratio if pos_weight_cap < 0 else min(raw_ratio, pos_weight_cap)
    print(f"pos_weight: raw={raw_ratio:.1f}, cap={pos_weight_cap}, using={capped:.1f}  "
          f"(pos={n_pos}, neg={n_neg})")
    return torch.tensor([capped], dtype=torch.float32, device=device)


def make_loss_func(pos_weight):
    """
    Wrap BCEWithLogitsLoss to reconcile pred shape [B,1] with label shape [B].
    """
    _bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def loss_func(pred, tgt):
        if pred.shape != tgt.shape:
            if pred.ndim == 2 and pred.shape[-1] == 1 and tgt.ndim == 1 and pred.shape[0] == tgt.shape[0]:
                pred = pred.squeeze(-1)
            else:
                raise ValueError(f"Cannot reconcile pred shape {pred.shape} with target shape {tgt.shape}")
        return _bce(pred, tgt.float())

    return loss_func


def get_model(K_eff, patch_content_size, args, rank):
    model = ClusterPatchTST(
        patch_content_size=patch_content_size,
        K_eff=K_eff,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        head_type='classification',
    )
    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'ClusterPatchTST [classification] | K_eff={K_eff} '
              f'| patch_content_size={patch_content_size} | params={n_params:,}')
    return model


def save_recorders(learn, save_path, model_name, rank):
    if rank != 0:
        return
    train_loss = learn.recorder['train_loss']
    valid_loss = learn.recorder['valid_loss']
    df = pd.DataFrame({'train_loss': train_loss, 'valid_loss': valid_loss})
    df.to_csv(save_path + model_name + '_losses.csv', float_format='%.6f', index=False)


# =============================================================================
# Metrics helpers (copied from patchtst_finetune.py)
# =============================================================================
def _metrics_at_threshold(y_true, probs, threshold):
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
        sensitivity=sensitivity, specificity=specificity,
        precision=precision, npv=npv,
        f1=f1, f2=f2, mcc=mcc, balanced_accuracy=bal_acc,
    )


def _interpret_sensitivity(s, n_pos):
    caught = round(s * n_pos); missed = n_pos - caught
    if s == 0.0:       return 'catching NO flu cases — model predicts all negative'
    elif s < 0.3:      return f'catching only {caught}/{n_pos} flu cases — very poor'
    elif s < 0.6:      return f'catching {caught}/{n_pos} flu cases — moderate'
    elif s < 0.8:      return f'catching {caught}/{n_pos} flu cases — reasonable'
    else:              return f'catching {caught}/{n_pos} flu cases — strong'


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
    sens_note = _interpret_sensitivity(m['sensitivity'], n_pos)
    print(f'  Sensitivity (Recall)  : {m["sensitivity"]:.4f}  — {sens_note}')
    print(f'  Specificity           : {m["specificity"]:.4f}')
    print(f'  Precision (PPV)       : {m["precision"]:.4f}')
    print(f'  NPV                   : {m["npv"]:.4f}')
    print(f'  {"─"*60}')
    print(f'  F1                    : {m["f1"]:.4f}')
    print(f'  F2 (recall-weighted)  : {m["f2"]:.4f}')
    print(f'  MCC                   : {m["mcc"]:.4f}  — {_interpret_mcc(m["mcc"])}')
    print(f'  Balanced Accuracy     : {m["balanced_accuracy"]:.4f}')


# =============================================================================
# LR Finder
# =============================================================================
def find_lr(K_eff, P, args, rank, is_distributed, device, dls=None):
    if dls is None:
        dls = get_dls(args)

    patch_content_size = P * dls.vars
    model = get_model(K_eff, patch_content_size, args, rank)

    weight_path = args.linear_probe_model or args.pretrained_model
    if weight_path is not None:
        model = transfer_weights(weight_path, model)
        if rank == 0:
            print(f'[find_lr] Loaded weights from: {weight_path}')

    pos_weight = compute_pos_weight(
        dls.train.dataset.data["label"], device, args.pos_weight_cap
    )
    loss_func = make_loss_func(pos_weight)

    cbs  = [RevInCB(dls.vars, denorm=False)] if args.revin else []
    cbs += [ClusterPatchCB(
        patch_learner_path=args.patch_learner_path,
        meta_path=args.meta_path,
        P=P,
        pop_percentile=args.pop_percentile,
    )]

    learn = Learner(dls, model, loss_func, lr=args.lr, cbs=cbs)
    suggested_lr = learn.lr_finder(end_lr=args.lr)
    if rank == 0:
        print('suggested_lr', suggested_lr)
    return suggested_lr, dls


# =============================================================================
# Fine-tuning (end-to-end)
# =============================================================================
def finetune_func(K_eff, P, args, lr, rank, is_distributed, device, dls=None):
    if rank == 0:
        print('end-to-end finetuning')

    if dls is None:
        dls = get_dls(args)

    patch_content_size = P * dls.vars
    model = get_model(K_eff, patch_content_size, args, rank)

    weight_path = args.linear_probe_model or args.pretrained_model
    if weight_path is not None:
        model = transfer_weights(weight_path, model)
        if rank == 0:
            src = 'linear probe checkpoint' if args.linear_probe_model else 'pretrained checkpoint'
            print(f'[rank {rank}] Loaded {src}: {weight_path}')

    pos_weight = compute_pos_weight(
        dls.train.dataset.data["label"], device, args.pos_weight_cap
    )
    loss_func = make_loss_func(pos_weight)

    cbs  = [RevInCB(dls.vars, denorm=False)] if args.revin else []
    cbs += [
        ClusterPatchCB(
            patch_learner_path=args.patch_learner_path,
            meta_path=args.meta_path,
            P=P,
            pop_percentile=args.pop_percentile,
        ),
        ValidationROCAUCCB(),
        SaveModelCB(monitor='valid_loss', fname=args.save_finetuned_model,
                    path=args.save_path),
    ]

    if args.use_wandb:
        cbs.append(WandbCB(
            project=args.wandb_project,
            run_name=args.wandb_run_name or args.save_finetuned_model,
            config=vars(args),
        ))

    learn = Learner(dls, model, loss_func, lr=lr, cbs=cbs, metrics=[])

    if is_distributed:
        learn.to_distributed()

    if is_distributed:
        learn.fit_one_cycle(args.n_epochs_finetune, lr_max=lr)
    else:
        learn.fine_tune(n_epochs=args.n_epochs_finetune, base_lr=lr, freeze_epochs=2)

    save_recorders(learn, args.save_path, args.save_finetuned_model, rank)


# =============================================================================
# Linear probe (backbone frozen)
# =============================================================================
def linear_probe_func(K_eff, P, args, lr, rank, is_distributed, device, dls=None):
    if rank == 0:
        print('linear probing')

    if dls is None:
        dls = get_dls(args)

    patch_content_size = P * dls.vars
    model = get_model(K_eff, patch_content_size, args, rank)
    model = transfer_weights(args.pretrained_model, model)

    pos_weight = compute_pos_weight(
        dls.train.dataset.data["label"], device, args.pos_weight_cap
    )
    loss_func = make_loss_func(pos_weight)

    cbs  = [RevInCB(dls.vars, denorm=False)] if args.revin else []
    cbs += [
        ClusterPatchCB(
            patch_learner_path=args.patch_learner_path,
            meta_path=args.meta_path,
            P=P,
            pop_percentile=args.pop_percentile,
        ),
        ValidationROCAUCCB(),
        SaveModelCB(monitor='valid_loss', fname=args.save_finetuned_model,
                    path=args.save_path),
    ]

    if args.use_wandb:
        cbs.append(WandbCB(
            project=args.wandb_project,
            run_name=args.wandb_run_name or args.save_finetuned_model,
            config=vars(args),
        ))

    learn = Learner(dls, model, loss_func, lr=lr, cbs=cbs, metrics=[])

    if is_distributed:
        learn.to_distributed()

    learn.linear_probe(n_epochs=args.n_epochs_finetune, base_lr=lr)
    save_recorders(learn, args.save_path, args.save_finetuned_model, rank)


# =============================================================================
# Test / evaluation
# =============================================================================
def test_func(K_eff, P, args, weight_path, n_features):
    dls = get_dls(args)

    print("val   labels:", dls.valid.dataset.data["label"].unique(return_counts=True))
    print("test  labels:", dls.test.dataset.data["label"].unique(return_counts=True))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    patch_content_size = P * n_features
    model = get_model(K_eff, patch_content_size, args, rank=0).to(device)

    cbs  = [RevInCB(dls.vars, denorm=False)] if args.revin else []
    cbs += [ClusterPatchCB(
        patch_learner_path=args.patch_learner_path,
        meta_path=args.meta_path,
        P=P,
        pop_percentile=args.pop_percentile,
    )]

    learn = Learner(dls, model, cbs=cbs)

    # Step 1 — val inference: find optimal threshold via Youden's J
    val_preds, val_targets = learn.test(dls.valid, weight_path=weight_path + '.pth')
    val_probs = (1 / (1 + np.exp(-np.array(val_preds).reshape(-1)))).astype(float)
    val_true  = np.array(val_targets).reshape(-1).astype(int)

    if len(np.unique(val_true)) >= 2:
        fpr_arr, tpr_arr, thresh_arr = roc_curve(val_true, val_probs)
        youden_idx    = int(np.argmax(tpr_arr - fpr_arr))
        opt_threshold = float(thresh_arr[youden_idx])
        val_j         = float(tpr_arr[youden_idx] - fpr_arr[youden_idx])
    else:
        opt_threshold = 0.5
        val_j         = float('nan')

    # Step 2 — test inference
    test_preds, test_targets = learn.test(dls.test, weight_path=weight_path + '.pth')
    probs  = (1 / (1 + np.exp(-np.array(test_preds).reshape(-1)))).astype(float)
    y_true = np.array(test_targets).reshape(-1).astype(int)

    prevalence = y_true.mean()
    n_pos      = int(y_true.sum())

    if len(np.unique(y_true)) >= 2:
        roc_auc = roc_auc_score(y_true, probs)
        pr_auc  = average_precision_score(y_true, probs)
    else:
        roc_auc = pr_auc = float('nan')

    m_05  = _metrics_at_threshold(y_true, probs, threshold=0.5)
    m_opt = _metrics_at_threshold(y_true, probs, threshold=opt_threshold) \
            if not np.isnan(opt_threshold) else None

    print()
    print(f'  {"="*60}')
    print(f'  TEST RESULTS  [ClusterPatchTST Phase 3]')
    print(f'  {"="*60}')
    print(f'  Dataset (test set):')
    print(f'    Total samples  : {len(y_true)}')
    print(f'    Positives      : {n_pos}')
    print(f'    Negatives      : {len(y_true) - n_pos}')
    print(f'    Prevalence     : {prevalence:.5f}  ({prevalence*100:.3f}%)')
    print(f'    Random PR-AUC  : {prevalence:.5f}')
    print()
    print(f'  Threshold selection (on VALIDATION set):')
    print(f'    Youden\'s J threshold : {opt_threshold:.4f}')
    print(f'    J statistic (val)    : {val_j:.4f}')
    print(f'    Val positives        : {val_true.sum()}  /  {len(val_true)}')
    print()
    print(f'  Threshold-independent metrics (test set):')
    print(f'    PR-AUC  (primary) : {pr_auc:.4f}  — {_interpret_pr_auc(pr_auc, prevalence)}')
    print(f'    ROC-AUC           : {roc_auc:.4f}')
    print()
    _print_metrics_block('@0.5  (fixed reference)', m_05, threshold=0.5, n_pos=n_pos)
    if m_opt is not None:
        print()
        _print_metrics_block(
            "optimal Youden's J  (threshold from VAL set, applied to TEST)",
            m_opt, threshold=opt_threshold, n_pos=n_pos,
        )
    print(f'  {"="*60}')

    row = dict(roc_auc=roc_auc, pr_auc=pr_auc,
               opt_threshold=opt_threshold, val_youden_j=val_j)
    for k, v in m_05.items():
        row[f'{k}@0.5'] = v
    if m_opt is not None:
        for k, v in m_opt.items():
            row[f'{k}@opt'] = v

    pd.DataFrame([row]).to_csv(
        args.save_path + args.save_finetuned_model + '_acc.csv',
        float_format='%.6f', index=False,
    )

    return test_preds, test_targets, [roc_auc, pr_auc, m_05['precision']]


# =============================================================================
# main
# =============================================================================
def main():
    parser = build_parser()
    args   = parser.parse_args()

    is_distributed, rank, world_size, local_rank, device = setup_ddp()

    args.dset = args.dset_finetune
    args.save_path = (
        'saved_models/' + args.dset_finetune
        + '/learned_patchtst/' + args.model_type + '/'
    )
    if rank == 0:
        os.makedirs(args.save_path, exist_ok=True)

    suffix = (
        '_cw' + str(args.context_points)
        + '_epochs-finetune' + str(args.n_epochs_finetune)
        + '_model' + str(args.finetuned_model_id)
    )
    if args.is_finetune:
        args.save_finetuned_model = args.dset_finetune + '_cluster_finetuned' + suffix
    elif args.is_linear_probe:
        args.save_finetuned_model = args.dset_finetune + '_cluster_linear-probe' + suffix
    else:
        args.save_finetuned_model = args.dset_finetune + '_cluster_finetuned' + suffix

    if not is_distributed and torch.cuda.is_available():
        set_device()

    if rank == 0:
        print('args:', args)

    # ------------------------------------------------------------------
    # Load meta, compute P
    # ------------------------------------------------------------------
    with open(args.meta_path, 'r') as f:
        meta = json.load(f)
    K_eff = meta['effective_k']

    # load_patch_learner_and_compute_P handles rank-0 computation + DDP broadcast
    _, P, cached_dls = load_patch_learner_and_compute_P(args, K_eff, device, rank)

    if is_distributed:
        # Broadcast P from rank 0 to all ranks
        p_tensor = torch.tensor([P], dtype=torch.int64, device=device)
        dist.broadcast(p_tensor, src=0)
        P = int(p_tensor.item())

    if rank == 0:
        print(f'[Phase 3] K_eff={K_eff}  P={P}  patch_content_size={P * cached_dls.vars}')

    # ------------------------------------------------------------------
    # Verify P vs checkpoint (if loading pretrained weights)
    # _verify_P_vs_checkpoint reads args.resume_from, so temporarily set it
    # to the relevant weight path so the check works for Phase 3 as well.
    # ------------------------------------------------------------------
    weight_path_to_check = args.linear_probe_model or args.pretrained_model
    if weight_path_to_check is not None and rank == 0:
        _orig_resume_from = getattr(args, 'resume_from', None)
        args.resume_from = weight_path_to_check
        _verify_P_vs_checkpoint(args, P, dls=cached_dls)
        args.resume_from = _orig_resume_from

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    if args.is_finetune:
        if is_distributed:
            lr_tensor = torch.zeros(1, dtype=torch.float32, device=device)
            if rank == 0:
                if args.use_lr_finder:
                    print("Running lr_finder on rank 0...")
                    suggested_lr, _ = find_lr(K_eff, P, args, rank, is_distributed, device, dls=cached_dls)
                    suggested_lr = float(suggested_lr)
                    print(f"[rank 0] Suggested LR: {suggested_lr:.6f}")
                else:
                    suggested_lr = args.lr
                    print(f"[rank 0] Using fixed LR: {suggested_lr:.6f}")
                lr_tensor[0] = suggested_lr
            dist.broadcast(lr_tensor, src=0)
            suggested_lr = lr_tensor.item()
        else:
            if args.use_lr_finder:
                suggested_lr, cached_dls = find_lr(K_eff, P, args, rank, is_distributed, device, dls=cached_dls)
            else:
                suggested_lr = args.lr
                if rank == 0:
                    print(f"Using fixed LR: {suggested_lr:.6f}")

        if rank == 0:
            print('About to start finetuning')
        finetune_func(K_eff, P, args, suggested_lr, rank, is_distributed, device,
                      dls=cached_dls if rank == 0 or not is_distributed else None)

        if rank == 0:
            print('finetune completed')
            n_features = meta.get('n_features', cached_dls.vars)
            test_func(K_eff, P, args,
                      weight_path=args.save_path + args.save_finetuned_model,
                      n_features=n_features)
            print('----------- Complete! -----------')

    elif args.is_linear_probe:
        if is_distributed:
            lr_tensor = torch.zeros(1, dtype=torch.float32, device=device)
            if rank == 0:
                if args.use_lr_finder:
                    print("Running lr_finder on rank 0...")
                    suggested_lr, _ = find_lr(K_eff, P, args, rank, is_distributed, device, dls=cached_dls)
                    suggested_lr = float(suggested_lr)
                    print(f"[rank 0] Suggested LR: {suggested_lr:.6f}")
                else:
                    suggested_lr = args.lr
                    print(f"[rank 0] Using fixed LR: {suggested_lr:.6f}")
                lr_tensor[0] = suggested_lr
            dist.broadcast(lr_tensor, src=0)
            suggested_lr = lr_tensor.item()
        else:
            if args.use_lr_finder:
                suggested_lr, cached_dls = find_lr(K_eff, P, args, rank, is_distributed, device, dls=cached_dls)
            else:
                suggested_lr = args.lr
                if rank == 0:
                    print(f"Using fixed LR: {suggested_lr:.6f}")

        if rank == 0:
            print('About to start linear probing')
        linear_probe_func(K_eff, P, args, suggested_lr, rank, is_distributed, device,
                          dls=cached_dls if rank == 0 or not is_distributed else None)

        if rank == 0:
            print('linear probe completed')
            n_features = meta.get('n_features', cached_dls.vars)
            test_func(K_eff, P, args,
                      weight_path=args.save_path + args.save_finetuned_model,
                      n_features=n_features)
            print('----------- Complete! -----------')

    else:
        # Test-only mode
        if args.linear_probe_model is not None:
            weight_path = args.linear_probe_model.replace('.pth', '')
        elif args.pretrained_model is not None:
            weight_path = args.pretrained_model.replace('.pth', '')
        else:
            weight_path = args.save_path + args.save_finetuned_model

        if rank == 0:
            print(f"Test-only mode — loading weights from: {weight_path}.pth")
            n_features = meta.get('n_features', cached_dls.vars)
            test_func(K_eff, P, args, weight_path=weight_path, n_features=n_features)
            print('----------- Complete! -----------')


if __name__ == '__main__':
    main()
