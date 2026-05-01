import numpy as np
import pandas as pd
import os
import torch
import torch.distributed as dist
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — safe for SLURM/headless nodes
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score,
    recall_score, f1_score, fbeta_score, matthews_corrcoef,
    balanced_accuracy_score, confusion_matrix, roc_curve,
    precision_recall_curve,
)
from src.callback.core import Callback

from src.models.patchTST import PatchTST
from src.learner import Learner, transfer_weights

from src.callback.core import *
from src.callback.tracking import *
from src.callback.patch_mask import *
from src.callback.transforms import *

from src.metrics import *
from src.basics import set_device
from datautils import *

import argparse


# ------------------------------------------------------------------
# Validation ROC-AUC callback
# Collects logits + labels during each validation epoch, computes
# ROC-AUC at the end, and stores it in learner.recorder['valid_roc_auc']
# so TrackerCB / SaveModelCB can monitor it.
# NOTE: currently wired up for visibility only — SaveModelCB still
# monitors valid_loss. Ask before switching the monitor to valid_roc_auc.
# ------------------------------------------------------------------
class ValidationROCAUCCB(Callback):
    """Computes ROC-AUC on the full validation set after every epoch."""

    def before_fit(self):
        if self.run_finder:
            return
        # Add the keys to the recorder so SaveModelCB can assert they exist
        # during its own before_fit. TrackTrainingCB already created the
        # recorder by this point (default cbs run before user cbs).
        if hasattr(self.learner, 'recorder'):
            self.learner.recorder['valid_roc_auc'] = []
            self.learner.recorder['valid_pr_auc'] = []

    def before_epoch_valid(self):
        self._preds = []
        self._targs = []

    def after_batch_valid(self):
        # Accumulate raw logits and binary labels from each validation batch.
        # Detach immediately to avoid holding computation graphs in memory.
        self._preds.append(self.pred.detach().cpu())
        self._targs.append(self.yb.detach().cpu())

    def after_epoch_valid(self):
        if not self.learner.dls.valid:
            return

        preds = torch.cat(self._preds)  # [N_local] — still a tensor for gathering
        targs = torch.cat(self._targs)

        # In DDP: each rank only saw its shard of the validation set.
        # all_gather_object collects tensors from every rank onto every rank
        # so all ranks compute metrics on the full validation set and store
        # the same value — making SaveModelCB consistent across ranks.
        if dist.is_initialized() and dist.get_world_size() > 1:
            world_size = dist.get_world_size()
            all_preds = [None] * world_size
            all_targs = [None] * world_size
            dist.all_gather_object(all_preds, preds)
            dist.all_gather_object(all_targs, targs)
            preds = torch.cat(all_preds)
            targs = torch.cat(all_targs)

        probs = torch.sigmoid(preds).numpy().reshape(-1)
        y_true = targs.numpy().reshape(-1)

        if len(np.unique(y_true)) < 2:
            # Validation split has only one class — metrics are undefined.
            roc_auc = 0.0
            pr_auc = 0.0
        else:
            roc_auc = roc_auc_score(y_true, probs)
            pr_auc = average_precision_score(y_true, probs)

        self.learner.recorder['valid_roc_auc'].append(roc_auc)
        self.learner.recorder['valid_pr_auc'].append(pr_auc)

        # Only rank 0 prints — all ranks have the same value so one print is enough
        _rank = int(os.environ.get("RANK", 0))
        if _rank == 0:
            print(f"  [val ROC-AUC: {roc_auc:.4f}  |  val PR-AUC: {pr_auc:.4f}]")


# ------------------------------------------------------------------
# Generalization-aware save callback
# Saves the model on best valid_loss, but ONLY while valid_loss <= train_loss.
# Once the gap inverts (model starts memorising), saving stops.
# ------------------------------------------------------------------
class GeneralizationSaveModelCB(SaveModelCB):
    """
    Extends SaveModelCB(monitor='valid_loss') with an extra guard:
    only save when valid_loss <= train_loss.

    This pins the saved checkpoint to the generalization peak —
    the last epoch where the model is still improving on held-out
    data without having started to memorise the training set.
    """

    def after_epoch(self):
        if self.run_finder:
            return
        recorder = self.learner.recorder
        # Need at least one value for both losses
        if not recorder.get('train_loss') or not recorder.get('valid_loss'):
            return
        train_loss = recorder['train_loss'][-1]
        valid_loss = recorder['valid_loss'][-1]
        if valid_loss <= train_loss:
            # Gap has not inverted — defer to normal SaveModelCB logic
            super().after_epoch()
        else:
            # Overfitting detected: update TrackerCB state without saving
            # so the epoch counter stays consistent, but skip the file write.
            _rank = int(os.environ.get("RANK", 0))
            if _rank == self.save_process_id:
                print(
                    f'Epoch {self.epoch}: valid_loss ({valid_loss:.6f}) > '
                    f'train_loss ({train_loss:.6f}) — skipping save (overfitting guard)'
                )


# ------------------------------------------------------------------
# Argument parser setup
# ------------------------------------------------------------------
parser = argparse.ArgumentParser()

parser.add_argument('--is_finetune', type=int, default=0, help='do finetuning or not')
parser.add_argument('--is_linear_probe', type=int, default=0, help='if linear_probe: only finetune the last layer')

parser.add_argument('--dset_finetune', type=str, default='etth1', help='dataset name')
parser.add_argument('--context_points', type=int, default=512, help='sequence length')
parser.add_argument('--target_points', type=int, default=1, help='number of output classes/logits; use 1 for binary classification')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers for DataLoader')
parser.add_argument('--scaler', type=str, default='standard', help='scale the input data')
parser.add_argument('--features', type=str, default='M', help='for multivariate model or univariate model')

parser.add_argument('--patch_len', type=int, default=12, help='patch length')
parser.add_argument('--stride', type=int, default=12, help='stride between patch')

parser.add_argument('--revin', type=int, default=1, help='reversible instance normalization')

parser.add_argument('--n_layers', type=int, default=3, help='number of Transformer layers')
parser.add_argument('--n_heads', type=int, default=16, help='number of Transformer heads')
parser.add_argument('--d_model', type=int, default=128, help='Transformer d_model')
parser.add_argument('--d_ff', type=int, default=256, help='Tranformer MLP dimension')
parser.add_argument('--dropout', type=float, default=0.2, help='Transformer dropout')
parser.add_argument('--head_dropout', type=float, default=0.2, help='head dropout')
parser.add_argument('--head_type', type=str, default='classification',
                    choices=['classification', 'resnet_classification'],
                    help='classification head architecture: '
                         '"classification" = linear probe on last patch (default); '
                         '"resnet_classification" = 1D ResNet over all patches')

parser.add_argument('--n_epochs_finetune', type=int, default=3, help='number of finetuning epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--use_lr_finder', type=int, default=1,
                    help='run LR finder before training (1=yes, 0=use --lr directly)')

parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained model name')
# Optional: path to a linear probe checkpoint to use as starting point for full finetuning.
# When provided, finetune_func loads this instead of --pretrained_model, giving it a
# backbone + already-trained classification head. This prevents catastrophic forgetting
# because the head is stable before the backbone starts receiving gradients.
# If not provided, finetune_func loads --pretrained_model as before (default behaviour).
parser.add_argument('--linear_probe_model', type=str, default=None,
                    help='path to a linear probe checkpoint to warm-start full finetuning (optional)')
parser.add_argument('--finetuned_model_id', type=int, default=1, help='id of the saved finetuned model')
parser.add_argument('--model_type', type=str, default='based_model', help='for multivariate model or univariate model')
# Cap the BCEWithLogitsLoss pos_weight to prevent gradient instability on
# severely imbalanced datasets. The raw ratio (n_neg/n_pos) can be >2000,
# which destabilises training. A cap of 50-100 is a safe starting point.
# Set to -1 to use the raw ratio with no cap.
parser.add_argument('--pos_weight_cap', type=float, default=-1.0,
                    help='maximum value for BCEWithLogitsLoss pos_weight (caps n_neg/n_pos ratio); -1 = no cap')
parser.add_argument('--neg_subsample_ratio', type=int, default=0,
                    help='keep this many negatives per positive in the training set; 0 = disabled (use all)')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed for negative undersampling')

parser.add_argument('--use_wandb', type=int, default=0, help='enable Weights & Biases logging (1=yes, 0=no)')
parser.add_argument('--wandb_project', type=str, default='PatchTST-Wearable', help='wandb project name')
parser.add_argument('--wandb_run_name', type=str, default=None, help='wandb run name (defaults to model save name)')


# ------------------------------------------------------------------
# Parse command-line arguments
# ------------------------------------------------------------------
args = parser.parse_args()


def compute_pos_weight(labels, device):
    """Compute BCEWithLogitsLoss pos_weight from label tensor, capped by args.pos_weight_cap.

    pos_weight = n_negatives / n_positives, but capped to avoid gradient
    instability when the imbalance ratio is extreme (e.g. 2000:1).
    Use --pos_weight_cap -1 to disable the cap and use the raw ratio.
    """
    n_pos = int((labels == 1).sum().item())
    n_neg = int((labels == 0).sum().item())
    raw_ratio = n_neg / n_pos
    capped = raw_ratio if args.pos_weight_cap < 0 else min(raw_ratio, args.pos_weight_cap)
    if rank == 0:
        print(f"pos_weight: raw={raw_ratio:.1f}, cap={args.pos_weight_cap}, using={capped:.1f}  "
              f"(pos={n_pos}, neg={n_neg})")
    return torch.tensor([capped], dtype=torch.float32, device=device)


# ------------------------------------------------------------------
# Distributed detection
# ------------------------------------------------------------------
is_distributed = "LOCAL_RANK" in os.environ
local_rank = int(os.environ.get("LOCAL_RANK", 0))
rank = int(os.environ.get("RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))

# ------------------------------------------------------------------
# Device setup + distributed process group init
# ------------------------------------------------------------------
if is_distributed:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Distributed run requested, but torch.cuda.is_available() is False. "
            "You are likely on a non-CUDA node (for example an AMD MI210 node) or using a non-CUDA PyTorch build. "
            "Run this on an NVIDIA CUDA node, or use normal python mode on CPU."
        )
    torch.cuda.set_device(local_rank)

    # Disable InfiniBand for intra-node jobs on PACE — IB discovery can take
    # ~20 min even when all GPUs are on one node and never use IB.
    os.environ.setdefault("NCCL_IB_DISABLE", "1")

    # Initialize the NCCL process group so dist.broadcast works in main().
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")

    # Warm up CUDA context now so the first model.to(device) is not slow.
    import time as _t
    _t0 = _t.time()
    print(f"[rank {rank}] CUDA warmup start...", flush=True)
    torch.zeros(1, device=torch.device(f"cuda:{local_rank}"))
    torch.cuda.synchronize()
    print(f"[rank {rank}] CUDA warmup done ({_t.time() - _t0:.1f}s)", flush=True)
else:
    set_device()

if rank == 0:
    print('args:', args)

# ------------------------------------------------------------------
# Construct the directory where finetuned models and outputs will be saved
# ------------------------------------------------------------------
args.save_path = 'saved_models/' + args.dset_finetune + '/masked_patchtst/' + args.model_type + '/'

if rank == 0 and not os.path.exists(args.save_path):
    os.makedirs(args.save_path, exist_ok=True)

# ------------------------------------------------------------------
# Build a suffix string that records the key finetuning hyperparameters
# ------------------------------------------------------------------
suffix_name = (
    '_cw' + str(args.context_points)
    + '_tw' + str(args.target_points)
    + '_patch' + str(args.patch_len)
    + '_stride' + str(args.stride)
    + '_epochs-finetune' + str(args.n_epochs_finetune)
    + '_model' + str(args.finetuned_model_id)
)

# ------------------------------------------------------------------
# Decide how the finetuned model should be named
# ------------------------------------------------------------------
if args.is_finetune:
    args.save_finetuned_model = args.dset_finetune + '_patchtst_finetuned' + suffix_name
elif args.is_linear_probe:
    args.save_finetuned_model = args.dset_finetune + '_patchtst_linear-probe' + suffix_name
else:
    args.save_finetuned_model = args.dset_finetune + '_patchtst_finetuned' + suffix_name


def get_model(c_in, args, head_type, weight_path=None):
    num_patch = (max(args.context_points, args.patch_len) - args.patch_len) // args.stride + 1
    if rank == 0:
        print('number of patches:', num_patch)

    model = PatchTST(
        c_in=c_in,
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
        head_type=head_type,
        res_attention=False
    )

    if weight_path:
        model = transfer_weights(weight_path, model)

    if rank == 0:
        print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))

    return model


def find_lr(head_type):
    dls = get_dls(args)
    model = get_model(dls.vars, args, head_type)
    # For full finetuning: load linear probe checkpoint if provided, else pretrained model.
    # For linear probing: always load pretrained model.
    if args.linear_probe_model is not None:
        model = transfer_weights(args.linear_probe_model, model)
    else:
        model = transfer_weights(args.pretrained_model, model)

    # Use the same weighted BCE as training so the LR finder sees a realistic
    # loss landscape rather than the collapsed all-negative trivial solution.
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    pos_weight = compute_pos_weight(dls.train.dataset.data["label"], device)
    # Same squeeze wrapper as finetune_func/linear_probe_func — the model
    # outputs [B, 1] but labels are [B], so we need to reconcile shapes.
    #
    # --- Loss options (swap in when ready to experiment) ---
    # Option A (active): weighted BCE — simple, stable, well-understood.
    _bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # Option B: Focal Loss — down-weights easy negatives so training focuses
    # on hard/rare positives. Drop-in replacement, no optimizer change needed.
    # gamma=2 is the standard value; increase to focus harder on rare cases.
    # from torchvision.ops import sigmoid_focal_loss
    # def focal_loss_func(pred, tgt):
    #     p = pred.squeeze(-1) if (pred.ndim==2 and pred.shape[-1]==1) else pred
    #     return sigmoid_focal_loss(p, tgt.float(), alpha=0.25, gamma=2, reduction='mean')
    # Option C: LibAUC AUCMLoss — directly maximises ROC-AUC as training
    # objective. Requires changing the optimizer to PESG (see finetune_func).
    # pip install libauc
    # from libauc.losses import AUCMLoss
    # _auc_loss = AUCMLoss()
    # -------------------------------------------------------
    def loss_func(pred, tgt):
        if pred.shape != tgt.shape:
            if pred.ndim == 2 and pred.shape[-1] == 1 and tgt.ndim == 1 and pred.shape[0] == tgt.shape[0]:
                pred = pred.squeeze(-1)
            else:
                raise ValueError(f"Cannot reconcile pred shape {pred.shape} with target shape {tgt.shape}")
        return _bce(pred, tgt.float())

    #cbs = [RevInCB(dls.vars)] if args.revin else []

    cbs = [RevInCB(dls.vars, denorm=False)] if args.revin else []
    cbs += [PatchCB(patch_len=args.patch_len, stride=args.stride)]

    learn = Learner(
        dls,
        model,
        loss_func,
        lr=args.lr,
        cbs=cbs,
    )

    suggested_lr = learn.lr_finder(end_lr=args.lr)
    if rank == 0:
        print('suggested_lr', suggested_lr)
    return suggested_lr


def save_recorders(learn):
    if rank != 0:
        return

    train_loss = learn.recorder['train_loss']
    valid_loss = learn.recorder['valid_loss']

    df = pd.DataFrame(data={'train_loss': train_loss, 'valid_loss': valid_loss})
    df.to_csv(
        args.save_path + args.save_finetuned_model + '_losses.csv',
        float_format='%.6f',
        index=False
    )


def finetune_func(lr=args.lr):
    if rank == 0:
        print('end-to-end finetuning')

    dls = get_dls(args)
    model = get_model(dls.vars, args, head_type=args.head_type)

    # If a linear probe checkpoint is provided, load it instead of the raw pretrain weights.
    # The linear probe checkpoint contains both the pretrained backbone AND a trained
    # classification head — so finetuning starts with a stable head, preventing the
    # large random-head gradients that cause catastrophic forgetting of pretrained features.
    # If no linear_probe_model is given, fall back to loading the pretrain weights as before.
    if args.linear_probe_model is not None:
        model = transfer_weights(args.linear_probe_model, model)
        if rank == 0:
            print(f"[rank 0] Loaded linear probe checkpoint: {args.linear_probe_model}")
    else:
        model = transfer_weights(args.pretrained_model, model)

    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    pos_weight = compute_pos_weight(dls.train.dataset.data["label"], device)

    # --- Loss options (swap in when ready to experiment) ---
    # Option A (active): weighted BCE — simple, stable, well-understood.
    _bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # Option B: Focal Loss — down-weights easy negatives so training focuses
    # on hard/rare positives. Drop-in replacement, no optimizer change needed.
    # gamma=2 is the standard value; increase to focus harder on rare cases.
    # from torchvision.ops import sigmoid_focal_loss
    # def focal_loss_func(pred, tgt):
    #     p = pred.squeeze(-1) if (pred.ndim==2 and pred.shape[-1]==1) else pred
    #     return sigmoid_focal_loss(p, tgt.float(), alpha=0.25, gamma=2, reduction='mean')
    # Option C: LibAUC AUCMLoss — directly maximises ROC-AUC as training
    # objective. Requires changing the optimizer to PESG (see below).
    # pip install libauc
    # from libauc.losses import AUCMLoss
    # from libauc.optimizers import PESG
    # _auc_loss = AUCMLoss()
    # (also replace opt_func=Adam in Learner with opt_func=PESG if using this)
    # -------------------------------------------------------
    def loss_func(pred, tgt):
        if pred.shape != tgt.shape:
            # Only fix the obvious case: pred is [B, 1] and tgt is [B]
            if pred.ndim == 2 and pred.shape[-1] == 1 and tgt.ndim == 1 and pred.shape[0] == tgt.shape[0]:
                pred = pred.squeeze(-1)
            else:
                raise ValueError(f"Cannot reconcile pred shape {pred.shape} with target shape {tgt.shape}")
        return _bce(pred, tgt.float())

    #cbs = [RevInCB(dls.vars, denorm=True)] if args.revin else []
    cbs = [RevInCB(dls.vars, denorm=False)] if args.revin else []

    cbs += [
        PatchCB(patch_len=args.patch_len, stride=args.stride),
        # ValidationROCAUCCB must come before SaveModelCB so it populates
        # recorder['valid_roc_auc'] before SaveModelCB.before_fit asserts it.
        ValidationROCAUCCB(),
        SaveModelCB(monitor='valid_loss', fname=args.save_finetuned_model, path=args.save_path)
        # SaveModelCB(monitor='valid_roc_auc', fname=args.save_finetuned_model, path=args.save_path)
        # GeneralizationSaveModelCB(monitor='valid_loss', fname=args.save_finetuned_model, path=args.save_path)
    ]

    if args.use_wandb:
        cbs.append(WandbCB(
            project=args.wandb_project,
            run_name=args.wandb_run_name or args.save_finetuned_model,
            config=vars(args)
        ))

    learn = Learner(
        dls,
        model,
        loss_func,
        lr=lr,
        cbs=cbs,
        metrics=[]
    )

    if is_distributed:
        learn.to_distributed()

    if is_distributed:
        learn.fit_one_cycle(args.n_epochs_finetune, lr_max=lr)
    else:
        learn.fine_tune(n_epochs=args.n_epochs_finetune, base_lr=lr, freeze_epochs=2)

    save_recorders(learn)


def linear_probe_func(lr=args.lr):
    if rank == 0:
        print('linear probing')

    dls = get_dls(args)
    model = get_model(dls.vars, args, head_type=args.head_type)
    model = transfer_weights(args.pretrained_model, model)

    # Same weighted BCE as finetune_func — prevents the model from collapsing
    # to all-negative predictions on the imbalanced dataset.
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    pos_weight = compute_pos_weight(dls.train.dataset.data["label"], device)

    # --- Loss options (swap in when ready to experiment) ---
    # Option A (active): weighted BCE — simple, stable, well-understood.
    _bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # Option B: Focal Loss — down-weights easy negatives so training focuses
    # on hard/rare positives. Drop-in replacement, no optimizer change needed.
    # gamma=2 is the standard value; increase to focus harder on rare cases.
    # from torchvision.ops import sigmoid_focal_loss
    # def focal_loss_func(pred, tgt):
    #     p = pred.squeeze(-1) if (pred.ndim==2 and pred.shape[-1]==1) else pred
    #     return sigmoid_focal_loss(p, tgt.float(), alpha=0.25, gamma=2, reduction='mean')
    # Option C: LibAUC AUCMLoss — directly maximises ROC-AUC as training
    # objective. Requires changing the optimizer to PESG (see finetune_func).
    # pip install libauc
    # from libauc.losses import AUCMLoss
    # _auc_loss = AUCMLoss()
    # -------------------------------------------------------
    def loss_func(pred, tgt):
        if pred.shape != tgt.shape:
            # Only fix the obvious case: pred is [B, 1] and tgt is [B]
            if pred.ndim == 2 and pred.shape[-1] == 1 and tgt.ndim == 1 and pred.shape[0] == tgt.shape[0]:
                pred = pred.squeeze(-1)
            else:
                raise ValueError(f"Cannot reconcile pred shape {pred.shape} with target shape {tgt.shape}")
        return _bce(pred, tgt.float())

    # denorm=False: RevIN must NOT inverse-normalize classification logits.
    # denorm=True is only correct for time series forecasting where pred has
    # the same shape as the input and needs to be back in the original scale.
    cbs = [RevInCB(dls.vars, denorm=False)] if args.revin else []
    cbs += [
        PatchCB(patch_len=args.patch_len, stride=args.stride),
        ValidationROCAUCCB(),
        SaveModelCB(monitor='valid_loss', fname=args.save_finetuned_model, path=args.save_path)
        # SaveModelCB(monitor='valid_roc_auc', fname=args.save_finetuned_model, path=args.save_path)
    ]

    if args.use_wandb:
        cbs.append(WandbCB(
            project=args.wandb_project,
            run_name=args.wandb_run_name or args.save_finetuned_model,
            config=vars(args)
        ))

    learn = Learner(
        dls,
        model,
        loss_func,
        lr=lr,
        cbs=cbs,
        metrics=[]
    )

    if is_distributed:
        learn.to_distributed()

    learn.linear_probe(n_epochs=args.n_epochs_finetune, base_lr=lr)
    save_recorders(learn)


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

    # Core rates
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # recall / TPR
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0   # TNR
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0   # PPV
    npv         = tn / (tn + fn) if (tn + fn) > 0 else 0.0   # NPV

    # Composite scores
    f1      = f1_score(y_true, y_pred, zero_division=0)
    f2      = fbeta_score(y_true, y_pred, beta=2, zero_division=0)  # recall-weighted
    mcc     = matthews_corrcoef(y_true, y_pred)                     # best for imbalance
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
    caught  = round(s * n_pos)
    missed  = n_pos - caught
    if s == 0.0:
        return f'catching NO flu cases — model predicts all negative at this threshold'
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
    """
    Find the threshold that maximises F1 on the Precision-Recall curve.

    Returns
    -------
    threshold : float
    best_f1   : float
    prec_arr  : ndarray  (full precision array from precision_recall_curve)
    rec_arr   : ndarray  (full recall array)
    thresh_arr: ndarray  (candidate thresholds — one fewer element than prec/rec)
    """
    prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_true, probs)
    # prec_arr / rec_arr have one extra element at the end (trivial point);
    # thresh_arr aligns with prec_arr[:-1] / rec_arr[:-1].
    denom = prec_arr[:-1] + rec_arr[:-1]
    f1_arr = np.where(denom > 0, 2 * prec_arr[:-1] * rec_arr[:-1] / denom, 0.0)
    best_idx = int(np.argmax(f1_arr))
    return float(thresh_arr[best_idx]), float(f1_arr[best_idx]), prec_arr, rec_arr, thresh_arr


def _save_curves(
    save_path, model_name,
    train_data, val_data, test_data,
    pr_opt_thresh_train, pr_opt_thresh_val,
    youden_thresh_val,
):
    """
    Plot and save PR curves and ROC curves for train / val / test splits.

    Each tuple in *_data: (y_true, probs, split_label)
    Optimal-threshold markers are drawn on the val curve only (where they
    were derived), so the plot makes the threshold-selection procedure clear.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = {'train': '#1f77b4', 'val': '#ff7f0e', 'test': '#2ca02c'}

    # ── PR curves ────────────────────────────────────────────────────────────
    ax_pr = axes[0]
    for y_true, probs, label in [train_data, val_data, test_data]:
        if len(np.unique(y_true)) < 2:
            continue
        prec, rec, _ = precision_recall_curve(y_true, probs)
        auc_val = average_precision_score(y_true, probs)
        ax_pr.plot(rec, prec, color=colors[label], lw=1.8,
                   label=f'{label}  (PR-AUC={auc_val:.3f})')

    # Mark PR-optimal threshold on val curve
    val_true, val_probs, _ = val_data
    if len(np.unique(val_true)) >= 2:
        val_probs_arr = np.array(val_probs)
        val_pred_val = (val_probs_arr >= pr_opt_thresh_val).astype(int)
        p_val = val_pred_val[val_true == 1].mean() if val_pred_val.sum() > 0 else 0
        r_val = val_pred_val[val_true == 1].mean() if val_pred_val.sum() > 0 else 0
        # Compute exact precision/recall at the threshold
        tp = ((val_pred_val == 1) & (val_true == 1)).sum()
        fp = ((val_pred_val == 1) & (val_true == 0)).sum()
        fn = ((val_pred_val == 0) & (val_true == 1)).sum()
        p_pt = tp / (tp + fp) if (tp + fp) > 0 else 0
        r_pt = tp / (tp + fn) if (tp + fn) > 0 else 0
        ax_pr.scatter([r_pt], [p_pt], color=colors['val'], s=120, zorder=5,
                      marker='*', label=f'val PR-opt thresh={pr_opt_thresh_val:.3f}')

    # Random baseline = prevalence of test set
    test_true, _, _ = test_data
    prevalence = test_true.mean()
    ax_pr.axhline(prevalence, color='gray', lw=1, ls='--',
                  label=f'random baseline ({prevalence:.4f})')
    ax_pr.set_xlabel('Recall', fontsize=12)
    ax_pr.set_ylabel('Precision', fontsize=12)
    ax_pr.set_title('Precision-Recall Curves', fontsize=13)
    ax_pr.legend(fontsize=9)
    ax_pr.set_xlim([0, 1])
    ax_pr.set_ylim([0, 1.02])
    ax_pr.grid(alpha=0.3)

    # ── ROC curves ───────────────────────────────────────────────────────────
    ax_roc = axes[1]
    for y_true, probs, label in [train_data, val_data, test_data]:
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, probs)
        auc_val = roc_auc_score(y_true, probs)
        ax_roc.plot(fpr, tpr, color=colors[label], lw=1.8,
                    label=f'{label}  (ROC-AUC={auc_val:.3f})')

    # Mark Youden's J threshold on val ROC curve
    if len(np.unique(val_true)) >= 2:
        fpr_v, tpr_v, thresh_v = roc_curve(val_true, val_probs_arr)
        j_idx = int(np.argmax(tpr_v - fpr_v))
        ax_roc.scatter([fpr_v[j_idx]], [tpr_v[j_idx]],
                       color=colors['val'], s=120, zorder=5, marker='*',
                       label=f"val Youden's J thresh={youden_thresh_val:.3f}")

    ax_roc.plot([0, 1], [0, 1], 'k--', lw=1, label='random baseline')
    ax_roc.set_xlabel('False Positive Rate', fontsize=12)
    ax_roc.set_ylabel('True Positive Rate', fontsize=12)
    ax_roc.set_title('ROC Curves', fontsize=13)
    ax_roc.legend(fontsize=9)
    ax_roc.set_xlim([0, 1])
    ax_roc.set_ylim([0, 1.02])
    ax_roc.grid(alpha=0.3)

    plt.tight_layout()
    out_path = save_path + model_name + '_pr_roc_curves.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Curves saved → {out_path}')


def _print_metrics_block(label, m, threshold, n_pos):
    """Print a formatted block of metrics with inline interpretations."""
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


def test_func(weight_path):
    # -------------------------------------------------------------------------
    # Load all splits — we need train + val to find the optimal threshold,
    # and test to evaluate it blind.  Correct procedure:
    #
    #   Train → PR-optimal threshold (max F1) — printed for reference
    #   Val   → PR-optimal threshold (max F1) → applied blind to test
    #   Test  → all metrics reported at @0.5 and @val-PR-optimal
    #
    # This avoids leakage of finding AND evaluating the threshold on the
    # same data (which produces optimistically biased results).
    # -------------------------------------------------------------------------
    dls = get_dls(args)

    print("val   labels:", dls.valid.dataset.data["label"].unique(return_counts=True))
    print("test  labels:", dls.test.dataset.data["label"].unique(return_counts=True))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = get_model(dls.vars, args, head_type=args.head_type).to(device)

    cbs  = [RevInCB(dls.vars, denorm=False)] if args.revin else []
    cbs += [PatchCB(patch_len=args.patch_len, stride=args.stride)]

    learn = Learner(dls, model, cbs=cbs)

    # -------------------------------------------------------------------------
    # Step 1 — Train + Val inference: find optimal threshold via PR curve
    # Optimal threshold = argmax F1 on the Precision-Recall curve.
    # PR-AUC is the primary metric here (dataset is severely imbalanced), so
    # the threshold is derived from the PR curve rather than ROC.
    #   Train → find PR-optimal threshold (reference / sanity check)
    #   Val   → find PR-optimal threshold → applied BLIND to test set
    # -------------------------------------------------------------------------
    train_preds, train_targets = learn.test(dls.train, weight_path=weight_path + '.pth')
    train_probs = (1 / (1 + np.exp(-np.array(train_preds).reshape(-1)))).astype(float)
    train_true  = np.array(train_targets).reshape(-1).astype(int)

    val_preds, val_targets = learn.test(dls.valid, weight_path=weight_path + '.pth')
    val_probs  = (1 / (1 + np.exp(-np.array(val_preds).reshape(-1)))).astype(float)
    val_true   = np.array(val_targets).reshape(-1).astype(int)

    if len(np.unique(train_true)) >= 2:
        pr_opt_thresh_train, pr_opt_f1_train, _, _, _ = _find_pr_optimal_threshold(train_true, train_probs)
    else:
        pr_opt_thresh_train, pr_opt_f1_train = 0.5, float('nan')

    if len(np.unique(val_true)) >= 2:
        pr_opt_thresh_val, pr_opt_f1_val, _, _, _ = _find_pr_optimal_threshold(val_true, val_probs)
        opt_threshold = pr_opt_thresh_val   # this is applied blind to test
    else:
        pr_opt_thresh_val, pr_opt_f1_val = 0.5, float('nan')
        opt_threshold = 0.5

    # -------------------------------------------------------------------------
    # Step 2 — Test inference: evaluate blind using val-derived threshold
    # -------------------------------------------------------------------------
    test_preds, test_targets = learn.test(dls.test, weight_path=weight_path + '.pth')
    probs  = (1 / (1 + np.exp(-np.array(test_preds).reshape(-1)))).astype(float)
    y_true = np.array(test_targets).reshape(-1).astype(int)

    prevalence = y_true.mean()
    n_pos      = int(y_true.sum())

    # -------------------------------------------------------------------------
    # Threshold-independent metrics (test set)
    # -------------------------------------------------------------------------
    if len(np.unique(y_true)) >= 2:
        roc_auc = roc_auc_score(y_true, probs)
        pr_auc  = average_precision_score(y_true, probs)
    else:
        roc_auc = pr_auc = float('nan')

    # -------------------------------------------------------------------------
    # Threshold-dependent metrics at two thresholds (both applied to test set)
    #   @0.5     — fixed reference point
    #   @optimal — PR-optimal threshold (max F1 on val PR curve) applied blind
    # -------------------------------------------------------------------------
    m_05  = _metrics_at_threshold(y_true, probs, threshold=0.5)
    m_opt = _metrics_at_threshold(y_true, probs, threshold=opt_threshold) \
            if not np.isnan(opt_threshold) else None

    # -------------------------------------------------------------------------
    # Print
    # -------------------------------------------------------------------------
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
    print(f'    Total samples  : {len(y_true)}')
    print(f'    Positives      : {n_pos}  (flu-onset windows)')
    print(f'    Negatives      : {len(y_true) - n_pos}')
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
            "PR-optimal threshold from VAL (max F1 on val PR curve), applied to TEST",
            m_opt, threshold=opt_threshold, n_pos=n_pos,
        )
    print(f'  {"="*60}')

    # -------------------------------------------------------------------------
    # Save PR + ROC curve plots
    # -------------------------------------------------------------------------
    _save_curves(
        save_path=args.save_path,
        model_name=args.save_finetuned_model,
        train_data=(train_true, train_probs, 'train'),
        val_data=(val_true, val_probs, 'val'),
        test_data=(y_true, probs, 'test'),
        pr_opt_thresh_train=pr_opt_thresh_train,
        pr_opt_thresh_val=pr_opt_thresh_val,
        youden_thresh_val=opt_threshold,
    )

    # -------------------------------------------------------------------------
    # Save CSV
    # -------------------------------------------------------------------------
    row = dict(
        roc_auc=roc_auc, pr_auc=pr_auc,
        pr_opt_threshold_train=pr_opt_thresh_train,
        pr_opt_f1_train=pr_opt_f1_train,
        pr_opt_threshold_val=pr_opt_thresh_val,
        pr_opt_f1_val=pr_opt_f1_val,
    )
    for k, v in m_05.items():
        row[f'{k}@0.5'] = v
    if m_opt is not None:
        for k, v in m_opt.items():
            row[f'{k}@opt'] = v

    pd.DataFrame([row]).to_csv(
        args.save_path + args.save_finetuned_model + '_acc.csv',
        float_format='%.6f',
        index=False,
    )

    scores = [roc_auc, pr_auc, m_05['precision']]
    return test_preds, test_targets, scores


if __name__ == '__main__':

    if args.is_finetune:
        args.dset = args.dset_finetune

        if is_distributed:
            device = torch.device(f'cuda:{local_rank}')
            lr_tensor = torch.zeros(1, dtype=torch.float32, device=device)

            if rank == 0:
                if args.use_lr_finder:
                    print("Running lr_finder on rank 0...")
                    suggested_lr = find_lr(head_type=args.head_type)
                    suggested_lr = float(suggested_lr)
                    print(f"[rank 0] Suggested LR: {suggested_lr:.6f}. Broadcasting to all ranks...")
                else:
                    suggested_lr = args.lr
                    print(f"[rank 0] Skipping LR finder, using fixed LR: {suggested_lr:.6f}. Broadcasting to all ranks...")
                lr_tensor[0] = suggested_lr

            dist.broadcast(lr_tensor, src=0)
            suggested_lr = lr_tensor.item()

            if rank == 0:
                print("About to start finetuning")
            finetune_func(suggested_lr)
        else:
            if args.use_lr_finder:
                suggested_lr = find_lr(head_type=args.head_type)
            else:
                suggested_lr = args.lr
                if rank == 0:
                    print(f"Skipping LR finder, using fixed LR: {suggested_lr:.6f}")
            finetune_func(suggested_lr)

        if rank == 0:
            print('finetune completed')
            out = test_func(args.save_path + args.save_finetuned_model)
            print('----------- Complete! -----------')

    elif args.is_linear_probe:
        args.dset = args.dset_finetune

        if is_distributed:
            device = torch.device(f'cuda:{local_rank}')
            lr_tensor = torch.zeros(1, dtype=torch.float32, device=device)

            if rank == 0:
                if args.use_lr_finder:
                    print("Running lr_finder on rank 0...")
                    suggested_lr = find_lr(head_type=args.head_type)
                    suggested_lr = float(suggested_lr)
                    print(f"[rank 0] Suggested LR: {suggested_lr:.6f}. Broadcasting to all ranks...")
                else:
                    suggested_lr = args.lr
                    print(f"[rank 0] Skipping LR finder, using fixed LR: {suggested_lr:.6f}. Broadcasting to all ranks...")
                lr_tensor[0] = suggested_lr

            dist.broadcast(lr_tensor, src=0)
            suggested_lr = lr_tensor.item()

            if rank == 0:
                print("About to start linear probing")
            linear_probe_func(suggested_lr)
        else:
            if args.use_lr_finder:
                suggested_lr = find_lr(head_type=args.head_type)
            else:
                suggested_lr = args.lr
                if rank == 0:
                    print(f"Skipping LR finder, using fixed LR: {suggested_lr:.6f}")
            linear_probe_func(suggested_lr)

        if rank == 0:
            print('finetune completed')
            out = test_func(args.save_path + args.save_finetuned_model)
            print('----------- Complete! -----------')

    else:
        # Test-only mode: no training, just evaluate a saved checkpoint on the test set.
        # Priority: --linear_probe_model > --pretrained_model > default finetuned path.
        args.dset = args.dset_finetune
        if args.linear_probe_model is not None:
            weight_path = args.linear_probe_model.replace('.pth', '')
        elif args.pretrained_model is not None:
            weight_path = args.pretrained_model.replace('.pth', '')
        else:
            weight_path = args.save_path + args.dset_finetune + '_patchtst_finetuned' + suffix_name

        if rank == 0:
            print(f"Test-only mode — loading weights from: {weight_path}.pth")
            out = test_func(weight_path)
            print('----------- Complete! -----------')