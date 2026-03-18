
import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch import nn

# ------------------------------------------------------------
# Model / learner imports
# ------------------------------------------------------------
from src.models.patchTST import PatchTST
from src.learner import Learner, transfer_weights

# ------------------------------------------------------------
# Callback imports
# ------------------------------------------------------------
from src.callback.core import *
from src.callback.tracking import *
from src.callback.scheduler import *
from src.callback.patch_mask import *
from src.callback.transforms import *

# ------------------------------------------------------------
# Metrics and dataloader helper
# ------------------------------------------------------------
from src.metrics import *
from datautils import get_dls


# ============================================================
# Argument parser
# ============================================================
parser = argparse.ArgumentParser()

# ------------------------------------------------------------
# Dataset and dataloader arguments
# ------------------------------------------------------------
parser.add_argument('--dset', type=str, default='etth1', help='dataset name')
parser.add_argument('--context_points', type=int, default=336, help='input sequence length')
parser.add_argument('--target_points', type=int, default=96, help='forecast horizon')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--num_workers', type=int, default=1, help='number of dataloader workers')
parser.add_argument('--scaler', type=str, default='standard', help='data scaling method')
parser.add_argument('--features', type=str, default='M', help='M for multivariate, S for univariate')
parser.add_argument('--use_time_features', type=int, default=0, help='whether to return time features')

# ------------------------------------------------------------
# Patch arguments
# ------------------------------------------------------------
parser.add_argument('--patch_len', type=int, default=32, help='patch length')
parser.add_argument('--stride', type=int, default=16, help='patch stride')

# ------------------------------------------------------------
# RevIN argument
# ------------------------------------------------------------
parser.add_argument('--revin', type=int, default=1, help='use reversible instance normalization')

# ------------------------------------------------------------
# Model hyperparameters
# ------------------------------------------------------------
parser.add_argument('--n_layers', type=int, default=3, help='number of Transformer layers')
parser.add_argument('--n_heads', type=int, default=16, help='number of attention heads')
parser.add_argument('--d_model', type=int, default=128, help='Transformer hidden dimension')
parser.add_argument('--d_ff', type=int, default=256, help='feedforward hidden dimension')
parser.add_argument('--dropout', type=float, default=0.2, help='Transformer dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='prediction head dropout')

# ------------------------------------------------------------
# Optimization arguments
# ------------------------------------------------------------
parser.add_argument('--n_epochs', type=int, default=20, help='number of training epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')

# ------------------------------------------------------------
# Save / model naming arguments
# ------------------------------------------------------------
parser.add_argument('--model_id', type=int, default=1, help='identifier for saved supervised model')
parser.add_argument('--model_type', type=str, default='based_model', help='subfolder name for organizing models')

# ------------------------------------------------------------
# Mode arguments
# ------------------------------------------------------------
parser.add_argument('--is_train', type=int, default=1, help='1=train, 0=test')

# ------------------------------------------------------------
# New argument:
# Optional checkpoint path for initializing supervised training
# from a fine-tuned or pretrained model.
# ------------------------------------------------------------
parser.add_argument(
    '--pretrained_model',
    type=str,
    default=None,
    help='optional path to a pretrained / finetuned .pth model to initialize from'
)

# ------------------------------------------------------------
# Optional switch for LR finder
# You may want to skip LR finder when starting from a pretrained
# model or when you already know the LR.
# ------------------------------------------------------------
parser.add_argument(
    '--use_lr_finder',
    type=int,
    default=1,
    help='1=use lr_finder before training, 0=train directly with --lr'
)

args = parser.parse_args()
print('args:', args)


# ============================================================
# Save path / naming setup
# ============================================================
args.save_model_name = (
    'patchtst_supervised'
    + '_cw' + str(args.context_points)
    + '_tw' + str(args.target_points)
    + '_patch' + str(args.patch_len)
    + '_stride' + str(args.stride)
    + '_epochs' + str(args.n_epochs)
    + '_model' + str(args.model_id)
)

args.save_path = 'saved_models/' + args.dset + '/patchtst_supervised/' + args.model_type + '/'

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)


# ============================================================
# Model builder
# ============================================================
def get_model(c_in, args):
    """
    Build a PatchTST model for supervised forecasting.

    Parameters
    ----------
    c_in : int
        Number of input variables / channels.
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    model : PatchTST
        Fresh PatchTST model with a prediction head.
    """
    # --------------------------------------------------------
    # Compute how many patches each input sequence will produce
    # --------------------------------------------------------
    num_patch = (max(args.context_points, args.patch_len) - args.patch_len) // args.stride + 1
    print('number of patches:', num_patch)

    # --------------------------------------------------------
    # Build a prediction-head PatchTST model
    # --------------------------------------------------------
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
        head_type='prediction',
        res_attention=False
    )

    print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model


# ============================================================
# Optional weight initialization from a checkpoint
# ============================================================
def maybe_load_pretrained(model, checkpoint_path):
    """
    If a checkpoint path is provided, initialize the current
    supervised model from that checkpoint.

    This is useful when:
    - you already have a pretrained masked PatchTST model
    - you already have a fine-tuned model
    - you want supervised training to start from existing weights

    Parameters
    ----------
    model : nn.Module
        Freshly built model.
    checkpoint_path : str or None
        Path to checkpoint file.

    Returns
    -------
    model : nn.Module
        Model, optionally initialized from checkpoint.
    """
    if checkpoint_path is None:
        print('No pretrained_model provided. Training will start from scratch.')
        return model

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    print(f'Loading initialization weights from: {checkpoint_path}')
    model = transfer_weights(checkpoint_path, model)
    return model


# ============================================================
# Learning rate finder
# ============================================================
def find_lr():
    """
    Run LR finder on the supervised setup.

    This builds:
    - dataloaders
    - model
    - optional checkpoint initialization
    - loss
    - callbacks
    - learner

    Then it returns the suggested learning rate.
    """
    # --------------------------------------------------------
    # Build dataloaders
    # --------------------------------------------------------
    dls = get_dls(args)

    # --------------------------------------------------------
    # Build and optionally initialize model
    # --------------------------------------------------------
    model = get_model(dls.vars, args)
    model = maybe_load_pretrained(model, args.pretrained_model)

    # --------------------------------------------------------
    # Supervised forecasting loss
    # --------------------------------------------------------
    loss_func = torch.nn.MSELoss(reduction='mean')

    # --------------------------------------------------------
    # Callbacks:
    # - RevInCB: normalization / denormalization support
    # - PatchCB: patchifies the input sequence for PatchTST
    # --------------------------------------------------------
    cbs = [RevInCB(dls.vars)] if args.revin else []
    cbs += [PatchCB(patch_len=args.patch_len, stride=args.stride)]

    # --------------------------------------------------------
    # Learner object
    # --------------------------------------------------------
    learn = Learner(dls, model, loss_func, cbs=cbs)

    # --------------------------------------------------------
    # Run LR finder
    # --------------------------------------------------------
    return learn.lr_finder()


# ============================================================
# Training function
# ============================================================
def train_func(lr=args.lr):
    """
    Run supervised training.

    Steps:
    1. Build dataloaders
    2. Build model
    3. Optionally initialize from checkpoint
    4. Define loss
    5. Build callbacks
    6. Create learner
    7. Train with one-cycle schedule
    """
    # --------------------------------------------------------
    # Build dataloaders
    # --------------------------------------------------------
    dls = get_dls(args)
    print('dls summary:', dls.vars, dls.c, dls.len)

    # --------------------------------------------------------
    # Build fresh model
    # --------------------------------------------------------
    model = get_model(dls.vars, args)

    # --------------------------------------------------------
    # Optionally initialize from pretrained / finetuned checkpoint
    # --------------------------------------------------------
    model = maybe_load_pretrained(model, args.pretrained_model)

    # --------------------------------------------------------
    # Loss for supervised forecasting
    # --------------------------------------------------------
    loss_func = torch.nn.MSELoss(reduction='mean')

    # --------------------------------------------------------
    # Callbacks
    # - RevInCB handles normalization
    # - PatchCB patchifies the input
    # - SaveModelCB saves the best model according to valid_loss
    # --------------------------------------------------------
    cbs = [RevInCB(dls.vars)] if args.revin else []
    cbs += [
        PatchCB(patch_len=args.patch_len, stride=args.stride),
        SaveModelCB(
            monitor='valid_loss',
            fname=args.save_model_name,
            path=args.save_path
        )
    ]

    # --------------------------------------------------------
    # Learner
    # --------------------------------------------------------
    learn = Learner(
        dls,
        model,
        loss_func,
        lr=lr,
        cbs=cbs,
        metrics=[mse]
    )

    # --------------------------------------------------------
    # Train using one-cycle learning rate schedule
    # --------------------------------------------------------
    learn.fit_one_cycle(n_epochs=args.n_epochs, lr_max=lr, pct_start=0.2)


# ============================================================
# Test function
# ============================================================
def test_func():
    """
    Test the supervised model.

    By default, this tests the checkpoint that was saved by this
    supervised script, not the initialization checkpoint.

    Steps:
    1. Build the expected saved supervised checkpoint path
    2. Build dataloaders
    3. Build model
    4. Build callbacks
    5. Create learner
    6. Load supervised checkpoint and evaluate on test set
    """
    # --------------------------------------------------------
    # This is the checkpoint saved by the supervised training run
    # --------------------------------------------------------
    weight_path = args.save_path + args.save_model_name + '.pth'
    
    

    # --------------------------------------------------------
    # Build dataloaders
    # --------------------------------------------------------
    dls = get_dls(args)

    # --------------------------------------------------------
    # Build model architecture
    # --------------------------------------------------------
    #model = get_model(dls.vars, args)
    
    model = maybe_load_pretrained(model, args.pretrained_model)
    
    # --------------------------------------------------------
    # Test-time callbacks
    # --------------------------------------------------------
    cbs = [RevInCB(dls.vars)] if args.revin else []
    cbs += [PatchCB(patch_len=args.patch_len, stride=args.stride)]

    # --------------------------------------------------------
    # Learner for testing
    # --------------------------------------------------------
    learn = Learner(dls, model, cbs=cbs)

    # --------------------------------------------------------
    # Evaluate saved supervised checkpoint on test split
    # --------------------------------------------------------
    out = learn.test(dls.test, weight_path=weight_path, scores=[mse, mae])

    return out


# ============================================================
# Main execution
# ============================================================
if __name__ == '__main__':

    # --------------------------------------------------------
    # Training mode
    # --------------------------------------------------------
    if args.is_train:
        if args.use_lr_finder:
            suggested_lr = find_lr()
            print('suggested lr:', suggested_lr)
            train_func(suggested_lr)
        else:
            print('Skipping lr_finder. Using provided lr:', args.lr)
            train_func(args.lr)

    # --------------------------------------------------------
    # Test mode
    # --------------------------------------------------------
    else:
        out = test_func()
        print('score:', out[2])
        print('shape:', out[0].shape)

    print('----------- Complete! -----------')


"""
import numpy as np
import pandas as pd
import os
import torch
from torch import nn

from src.models.patchTST import PatchTST
from src.learner import Learner
from src.callback.core import *
from src.callback.tracking import *
from src.callback.scheduler import *
from src.callback.patch_mask import *
from src.callback.transforms import *
from src.metrics import *
from datautils import get_dls


import argparse

parser = argparse.ArgumentParser()
# Dataset and dataloader
parser.add_argument('--dset', type=str, default='etth1', help='dataset name')
parser.add_argument('--context_points', type=int, default=336, help='sequence length')
parser.add_argument('--target_points', type=int, default=96, help='forecast horizon')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--num_workers', type=int, default=1, help='number of workers for DataLoader')
parser.add_argument('--scaler', type=str, default='standard', help='scale the input data')
parser.add_argument('--features', type=str, default='M', help='for multivariate model or univariate model')
parser.add_argument('--use_time_features', type=int, default=0, help='whether to use time features or not')
# Patch
parser.add_argument('--patch_len', type=int, default=32, help='patch length')
parser.add_argument('--stride', type=int, default=16, help='stride between patch')
# RevIN
parser.add_argument('--revin', type=int, default=1, help='reversible instance normalization')
# Model args
parser.add_argument('--n_layers', type=int, default=3, help='number of Transformer layers')
parser.add_argument('--n_heads', type=int, default=16, help='number of Transformer heads')
parser.add_argument('--d_model', type=int, default=128, help='Transformer d_model')
parser.add_argument('--d_ff', type=int, default=256, help='Tranformer MLP dimension')
parser.add_argument('--dropout', type=float, default=0.2, help='Transformer dropout')
parser.add_argument('--head_dropout', type=float, default=0, help='head dropout')
# Optimization args
parser.add_argument('--n_epochs', type=int, default=20, help='number of training epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
# model id to keep track of the number of models saved
parser.add_argument('--model_id', type=int, default=1, help='id of the saved model')
parser.add_argument('--model_type', type=str, default='based_model', help='for multivariate model or univariate model')
# training
parser.add_argument('--is_train', type=int, default=1, help='training the model')


args = parser.parse_args()
print('args:', args)
args.save_model_name = 'patchtst_supervised'+'_cw'+str(args.context_points)+'_tw'+str(args.target_points) + '_patch'+str(args.patch_len) + '_stride'+str(args.stride)+'_epochs'+str(args.n_epochs) + '_model' + str(args.model_id)
args.save_path = 'saved_models/' + args.dset + '/patchtst_supervised/' + args.model_type + '/'
if not os.path.exists(args.save_path): os.makedirs(args.save_path)


def get_model(c_in, args):
    
    #c_in: number of input variables
    
    # get number of patches
    num_patch = (max(args.context_points, args.patch_len)-args.patch_len) // args.stride + 1    
    print('number of patches:', num_patch)
    
    # get model
    model = PatchTST(c_in=c_in,
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
                head_type='prediction',
                res_attention=False
                )    
    return model


def find_lr():
    # get dataloader
    dls = get_dls(args)    
    model = get_model(dls.vars, args)
    # get loss
    loss_func = torch.nn.MSELoss(reduction='mean')
    # get callbacks
    cbs = [RevInCB(dls.vars)] if args.revin else []
    cbs += [PatchCB(patch_len=args.patch_len, stride=args.stride)]
    # define learner
    learn = Learner(dls, model, loss_func, cbs=cbs)                        
    
    # fit the data to the model
    return learn.lr_finder()


def train_func(lr=args.lr):
    # get dataloader
    dls = get_dls(args)
    print('in out', dls.vars, dls.c, dls.len)
    
    # get model
    model = get_model(dls.vars, args)

    # get loss
    loss_func = torch.nn.MSELoss(reduction='mean')

    # get callbacks
    cbs = [RevInCB(dls.vars)] if args.revin else []
    cbs += [
         PatchCB(patch_len=args.patch_len, stride=args.stride),
         SaveModelCB(monitor='valid_loss', fname=args.save_model_name, 
                     path=args.save_path )
        ]

    # define learner
    learn = Learner(dls, model, 
                        loss_func, 
                        lr=lr, 
                        cbs=cbs,
                        metrics=[mse]
                        )
                        
    # fit the data to the model
    learn.fit_one_cycle(n_epochs=args.n_epochs, lr_max=lr, pct_start=0.2)


def test_func():
    weight_path = args.save_path + args.save_model_name + '.pth'
    # get dataloader
    dls = get_dls(args)
    model = get_model(dls.vars, args)
    #model = torch.load(weight_path)
    # get callbacks
    cbs = [RevInCB(dls.vars)] if args.revin else []
    cbs += [PatchCB(patch_len=args.patch_len, stride=args.stride)]
    learn = Learner(dls, model,cbs=cbs)
    out  = learn.test(dls.test, weight_path=weight_path, scores=[mse,mae])         # out: a list of [pred, targ, score_values]
    return out


if __name__ == '__main__':

    if args.is_train:   # training mode
        suggested_lr = find_lr()
        print('suggested lr:', suggested_lr)
        train_func(suggested_lr)
    else:   # testing mode
        out = test_func()
        print('score:', out[2])
        print('shape:', out[0].shape)
   
    print('----------- Complete! -----------')


"""