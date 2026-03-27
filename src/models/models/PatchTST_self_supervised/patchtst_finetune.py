import numpy as np
import pandas as pd
import os
import torch
import torch.distributed as dist
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score
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
        # Add the key to the recorder so SaveModelCB can assert it exists
        # during its own before_fit. TrackTrainingCB already created the
        # recorder by this point (default cbs run before user cbs).
        if hasattr(self.learner, 'recorder'):
            self.learner.recorder['valid_roc_auc'] = []

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

        preds = torch.cat(self._preds).numpy()
        targs = torch.cat(self._targs).numpy()

        # Convert raw logits → probabilities via sigmoid
        probs = 1 / (1 + np.exp(-preds.reshape(-1)))
        y_true = targs.reshape(-1)

        if len(np.unique(y_true)) < 2:
            # Validation split has only one class — ROC-AUC is undefined.
            # Store 0.0 so the recorder key always has a value this epoch.
            roc_auc = 0.0
        else:
            roc_auc = roc_auc_score(y_true, probs)

        self.learner.recorder['valid_roc_auc'].append(roc_auc)

        # Only rank 0 prints to avoid duplicate lines in DDP runs
        _rank = int(os.environ.get("RANK", 0))
        if _rank == 0:
            print(f"  [val ROC-AUC: {roc_auc:.4f}]")


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

parser.add_argument('--n_epochs_finetune', type=int, default=3, help='number of finetuning epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')

parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained model name')
parser.add_argument('--finetuned_model_id', type=int, default=1, help='id of the saved finetuned model')
parser.add_argument('--model_type', type=str, default='based_model', help='for multivariate model or univariate model')
# Cap the BCEWithLogitsLoss pos_weight to prevent gradient instability on
# severely imbalanced datasets. The raw ratio (n_neg/n_pos) can be >2000,
# which destabilises training. A cap of 50-100 is a safe starting point.
# Set to -1 to use the raw ratio with no cap.
parser.add_argument('--pos_weight_cap', type=float, default=100.0,
                    help='maximum value for BCEWithLogitsLoss pos_weight (caps n_neg/n_pos ratio); -1 = no cap')


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

    suggested_lr = learn.lr_finder()
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
    model = get_model(dls.vars, args, head_type='classification')
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
        # SaveModelCB(monitor='valid_loss', fname=args.save_finetuned_model, path=args.save_path)
        SaveModelCB(monitor='valid_roc_auc', fname=args.save_finetuned_model, path=args.save_path)
    ]

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
    model = get_model(dls.vars, args, head_type='classification')
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
        # SaveModelCB(monitor='valid_loss', fname=args.save_finetuned_model, path=args.save_path)
        SaveModelCB(monitor='valid_roc_auc', fname=args.save_finetuned_model, path=args.save_path)
    ]

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


def test_func(weight_path):
    dls = get_dls(args)

    print("train labels:", dls.train.dataset.data["label"].unique(return_counts=True))
    print("test  labels:", dls.test.dataset.data["label"].unique(return_counts=True))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_model(dls.vars, args, head_type='classification').to(device)

    #cbs = [RevInCB(dls.vars, denorm=True)] if args.revin else []
    cbs = [RevInCB(dls.vars, denorm=False)] if args.revin else []
    cbs += [PatchCB(patch_len=args.patch_len, stride=args.stride)]

    learn = Learner(dls, model, cbs=cbs)

    preds, targets = learn.test(dls.test, weight_path=weight_path + '.pth')

    preds = np.array(preds)
    targets = np.array(targets)

    probs = 1 / (1 + np.exp(-preds.reshape(-1)))
    y_true = targets.reshape(-1)

    y_pred = (probs >= 0.5).astype(int)

    if len(np.unique(y_true)) >= 2:
        roc_auc = roc_auc_score(y_true, probs)
        pr_auc = average_precision_score(y_true, probs)
    else:
        roc_auc = np.nan
        pr_auc = np.nan

    precision = precision_score(y_true, y_pred, zero_division=0)

    scores = [roc_auc, pr_auc, precision]

    # PR-AUC is the primary metric for imbalanced binary classification:
    # it is not inflated by the large number of true negatives, unlike ROC-AUC.
    # A random baseline achieves PR-AUC ≈ prevalence (fraction of positives).
    print(f"--- Test results ---")
    print(f"  PR-AUC   (primary): {pr_auc:.4f}")
    print(f"  ROC-AUC           : {roc_auc:.4f}")
    print(f"  Precision @0.5    : {precision:.4f}")

    pd.DataFrame(
        np.array(scores).reshape(1, -1),
        columns=['roc_auc', 'pr_auc', 'precision']
    ).to_csv(
        args.save_path + args.save_finetuned_model + '_acc.csv',
        float_format='%.6f',
        index=False
    )

    return preds, targets, scores


if __name__ == '__main__':

    if args.is_finetune:
        args.dset = args.dset_finetune

        if is_distributed:
            # Run lr_finder on rank 0 only, then broadcast the result so all
            # ranks train with the same optimal LR (same pattern as pretrain).
            device = torch.device(f'cuda:{local_rank}')
            lr_tensor = torch.zeros(1, dtype=torch.float32, device=device)

            if rank == 0:
                print("Running lr_finder on rank 0...")
                suggested_lr = find_lr(head_type='classification')
                suggested_lr = float(suggested_lr)
                print(f"[rank 0] Suggested LR: {suggested_lr:.6f}. Broadcasting to all ranks...")
                lr_tensor[0] = suggested_lr

            dist.broadcast(lr_tensor, src=0)
            suggested_lr = lr_tensor.item()

            if rank == 0:
                print("About to start finetuning")
            finetune_func(suggested_lr)
        else:
            suggested_lr = find_lr(head_type='classification')
            finetune_func(suggested_lr)

        if rank == 0:
            print('finetune completed')
            out = test_func(args.save_path + args.save_finetuned_model)
            print('----------- Complete! -----------')

    elif args.is_linear_probe:
        args.dset = args.dset_finetune

        if is_distributed:
            # Same LR broadcast pattern as is_finetune above.
            device = torch.device(f'cuda:{local_rank}')
            lr_tensor = torch.zeros(1, dtype=torch.float32, device=device)

            if rank == 0:
                print("Running lr_finder on rank 0...")
                suggested_lr = find_lr(head_type='classification')
                suggested_lr = float(suggested_lr)
                print(f"[rank 0] Suggested LR: {suggested_lr:.6f}. Broadcasting to all ranks...")
                lr_tensor[0] = suggested_lr

            dist.broadcast(lr_tensor, src=0)
            suggested_lr = lr_tensor.item()

            if rank == 0:
                print("About to start linear probing")
            linear_probe_func(suggested_lr)
        else:
            suggested_lr = find_lr(head_type='classification')
            linear_probe_func(suggested_lr)

        if rank == 0:
            print('finetune completed')
            out = test_func(args.save_path + args.save_finetuned_model)
            print('----------- Complete! -----------')

    else:
        args.dset = args.dset_finetune
        weight_path = args.save_path + args.dset_finetune + '_patchtst_finetuned' + suffix_name

        if rank == 0:
            out = test_func(weight_path)
            print('----------- Complete! -----------')