import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

from torch import nn

# Import the PatchTST model architecture
from src.models.patchTST import PatchTST

# Import the learner class and utility for weight transfer
from src.learner import Learner, transfer_weights

# Import callbacks used for tracking, saving, masking, etc.
from src.callback.tracking import *
from src.callback.patch_mask import *
from src.callback.transforms import *

# Import evaluation metrics
from src.metrics import *

# Import utility to automatically select and set the compute device
from src.basics import set_device

# Import dataset / dataloader helper functions
from datautils import *


def build_parser() -> argparse.ArgumentParser:
    """
    Build and return the CLI argument parser.

    Keeping parser creation inside a function avoids top-level side effects
    and is safer with multiprocessing / torchrun.
    """
    parser = argparse.ArgumentParser()

    # Dataset and dataloader arguments
    parser.add_argument('--dset_pretrain', type=str, default='etth1', help='dataset name')
    parser.add_argument('--context_points', type=int, default=512, help='sequence length')
    parser.add_argument('--target_points', type=int, default=96, help='forecast horizon')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for DataLoader')
    parser.add_argument('--scaler', type=str, default='standard', help='scale the input data')
    parser.add_argument('--features', type=str, default='M', help='for multivariate model or univariate model')

    # Patch-related arguments
    parser.add_argument('--patch_len', type=int, default=12, help='patch length')
    parser.add_argument('--stride', type=int, default=12, help='stride between patch')

    # RevIN argument
    parser.add_argument('--revin', type=int, default=1, help='reversible instance normalization')

    # Model architecture arguments
    parser.add_argument('--n_layers', type=int, default=3, help='number of Transformer layers')
    parser.add_argument('--n_heads', type=int, default=16, help='number of Transformer heads')
    parser.add_argument('--d_model', type=int, default=128, help='Transformer d_model')
    parser.add_argument('--d_ff', type=int, default=512, help='Transformer MLP dimension')
    parser.add_argument('--dropout', type=float, default=0.2, help='Transformer dropout')
    parser.add_argument('--head_dropout', type=float, default=0.2, help='head dropout')

    # Pretraining mask argument
    parser.add_argument('--mask_ratio', type=float, default=0.4, help='masking ratio for the input')

    # Optimization arguments
    parser.add_argument('--n_epochs_pretrain', type=int, default=10, help='number of pre-training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')

    # Identifier used to distinguish saved pretrained models
    parser.add_argument('--pretrained_model_id', type=int, default=1, help='id of the saved pretrained model')

    # Argument specifying model type for organizing save paths
    parser.add_argument('--model_type', type=str, default='based_model', help='for multivariate model or univariate model')

    return parser


def setup_ddp():
    """
    Detect whether we are running under torchrun and initialize DDP if needed.

    Returns
    -------
    is_distributed : bool
    rank : int
    world_size : int
    local_rank : int
    device : torch.device
    """
    using_torchrun = "RANK" in os.environ and "WORLD_SIZE" in os.environ

    if using_torchrun:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")

        if world_size > 1:
            # Disable InfiniBand for intra-node multi-GPU jobs: IB topology
            # discovery takes ~20 min on PACE even when all GPUs are on one
            # node and never use IB.  NVLink/PCIe is used instead, which is
            # faster for single-node anyway.
            os.environ.setdefault("NCCL_IB_DISABLE", "1")
            # Multi-GPU: use NCCL (fast collective comms, worth the init cost).
            backend = "nccl"
            if not dist.is_initialized():
                dist.init_process_group(backend=backend, init_method="env://")
            is_distributed = True
        else:
            # Single process launched via torchrun: skip the process group
            # entirely — NCCL init on HPC nodes can take ~20 minutes even for
            # 1 GPU due to InfiniBand topology discovery.
            is_distributed = False
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_distributed = False

    # Trigger CUDA context creation NOW so the first model.to(device) inside
    # SetupLearnerCB.before_fit does not pay the cold-start penalty (~20 min
    # on PACE/HPC nodes).  This is a one-time cost per process.
    if torch.cuda.is_available():
        import time as _t
        _t0 = _t.time()
        _r = int(os.environ.get("RANK", 0))
        print(f"[rank {_r}] CUDA warmup start...", flush=True)
        torch.zeros(1, device=device)
        torch.cuda.synchronize(device)
        print(f"[rank {_r}] CUDA warmup done ({_t.time()-_t0:.1f}s)", flush=True)

    return is_distributed, rank, world_size, local_rank, device


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    """
    Populate derived arguments that were previously being created at top level.
    """
    args.dset = args.dset_pretrain

    args.save_pretrained_model = (
        'patchtst_pretrained_cw' + str(args.context_points)
        + '_patch' + str(args.patch_len)
        + '_stride' + str(args.stride)
        + '_epochs-pretrain' + str(args.n_epochs_pretrain)
        + '_mask' + str(args.mask_ratio)
        + '_model' + str(args.pretrained_model_id)
    )

    args.save_path = (
        'saved_models/'
        + args.dset_pretrain
        + '/masked_patchtst/'
        + args.model_type
        + '/'
    )

    return args


def ensure_save_path(save_path: str, rank: int) -> None:
    """
    Create save directory only from rank 0.
    """
    if rank == 0:
        os.makedirs(save_path, exist_ok=True)


def get_model(c_in: int, args: argparse.Namespace, rank: int):
    """
    Build and return the PatchTST model.
    """
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
        head_type='pretrain',
        res_attention=False
    )

    if rank == 0:
        print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))

    return model


def find_lr(args: argparse.Namespace, rank: int):
    """
    Run the learning rate finder to automatically suggest a good learning rate.

    This should only be used in non-distributed mode.
    """
    dls = get_dls(args)
    model = get_model(dls.vars, args, rank)

    loss_func = torch.nn.MSELoss(reduction='mean')

    cbs = [RevInCB(dls.vars, denorm=False)] if args.revin else []
    cbs += [PatchMaskCB(patch_len=args.patch_len, stride=args.stride, mask_ratio=args.mask_ratio)]

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

    # Return dls alongside lr so rank 0 can reuse it in pretrain_func without
    # loading the full dataset a second time. DictDataset is stateless (pure tensors
    # in RAM) and the DataLoader creates a fresh iterator each epoch, so reusing dls
    # here is safe. If this causes unexpected behaviour, revert to returning only
    # suggested_lr and calling get_dls(args) again inside pretrain_func.
    return suggested_lr, dls


def pretrain_func(
    args: argparse.Namespace,
    lr: float,
    rank: int,
    is_distributed: bool,
    dls=None,  # optional: pass pre-loaded dls to avoid loading the dataset twice on rank 0
):
    """
    Perform masked pretraining of the PatchTST model using the given learning rate.
    """
    if rank == 0:
        print(1)

    # If dls was pre-loaded (e.g. reused from find_lr on rank 0), skip reloading.
    # Otherwise load fresh — this is always the case for rank 1.
    if dls is None:
        dls = get_dls(args)
    if rank == 0:
        print(2)

    model = get_model(dls.vars, args, rank)
    if rank == 0:
        print(3)

    loss_func = torch.nn.MSELoss(reduction='mean')
    if rank == 0:
        print(4)

    cbs = [RevInCB(dls.vars, denorm=False)] if args.revin else []
    cbs += [
        PatchMaskCB(
            patch_len=args.patch_len,
            stride=args.stride,
            mask_ratio=args.mask_ratio,
        ),
        SaveModelCB(
            monitor='valid_loss',
            fname=args.save_pretrained_model,
            path=args.save_path
        )
    ]
    if rank == 0:
        print(5)

    learn = Learner(
        dls,
        model,
        loss_func,
        lr=lr,
        cbs=cbs,
        # Removed mse from metrics: the tracking callback accumulates ALL validation
        # predictions in CPU RAM before computing metrics (preds shape: [n_val, num_patch, c_in, patch_len]).
        # With a large dataset this causes ~180GB RAM usage during validation and OOM-kills the process.
        # valid_loss already tracks MSE since the loss function is MSELoss — no information is lost.
        metrics=[]
    )

    if is_distributed:
        learn.to_distributed()

    if rank == 0:
        print(6)

    learn.fit_one_cycle(args.n_epochs_pretrain, lr_max=lr)

    if rank == 0:
        print(7)

    if rank == 0:
        train_loss = learn.recorder['train_loss']
        valid_loss = learn.recorder['valid_loss']
        print(8)

        df = pd.DataFrame(data={'train_loss': train_loss, 'valid_loss': valid_loss})
        print(9)

        df.to_csv(
            args.save_path + args.save_pretrained_model + '_losses.csv',
            float_format='%.6f',
            index=False
        )
        print(10)


def main():
    parser = build_parser()
    args = parser.parse_args()

    is_distributed, rank, world_size, local_rank, device = setup_ddp()
    args = finalize_args(args)

    ensure_save_path(args.save_path, rank)

    # Keep device selection as a runtime side effect, not an import-time side effect.
    if not is_distributed:
        set_device()

    if rank == 0:
        print("Entered")
        print("args:", args)
        print("rank:", rank, "local_rank:", local_rank, "world_size:", world_size)
        print("dist.is_initialized():", dist.is_initialized())

    if is_distributed:
        # Run lr_finder on rank 0 only, then broadcast result to all ranks.
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        lr_tensor = torch.zeros(1, dtype=torch.float32, device=device)

        if rank == 0:
            print("Running lr_finder on rank 0...")
            suggested_lr, cached_dls = find_lr(args, rank)
            suggested_lr = float(suggested_lr)
            print(f"[rank 0] Finished finding optimum learning rate. Suggested LR: {suggested_lr:.6f}")
            print(f"[rank 0] Broadcasting lr={suggested_lr:.6f} to all ranks...")
            lr_tensor[0] = suggested_lr

        dist.broadcast(lr_tensor, src=0)
        suggested_lr = lr_tensor.item()

        if rank == 0:
            print("About to start training")

        pretrain_func(
            args=args,
            lr=suggested_lr,
            rank=rank,
            is_distributed=is_distributed,
            dls=cached_dls if rank == 0 else None,  # rank 0 reuses dls, rank 1 loads fresh
        )
    else:
        suggested_lr, cached_dls = find_lr(args, rank)
        suggested_lr = float(suggested_lr)

        if rank == 0:
            print("About to start training")

        pretrain_func(
            args=args,
            lr=suggested_lr,
            rank=rank,
            is_distributed=is_distributed,
            dls=cached_dls,  # reuse dls from find_lr — no distributed split needed here
        )

    if rank == 0:
        print('pretraining completed')


if __name__ == '__main__':
    main()