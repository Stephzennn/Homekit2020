
import numpy as np
import pandas as pd
import os
import torch
from torch import nn

# Import the PatchTST model architecture
from src.models.patchTST import PatchTST

# Import the learner class and utility for loading / transferring pretrained weights
from src.learner import Learner, transfer_weights

# Import core callbacks and utilities used during training and evaluation
from src.callback.core import *
from src.callback.tracking import *
from src.callback.patch_mask import *
from src.callback.transforms import *

# Import evaluation metrics
from src.metrics import *

# Import utility to automatically select the available device
from src.basics import set_device

# Import dataset / dataloader helper functions
from datautils import *

# Import argparse for command-line configuration
import argparse


# ------------------------------------------------------------------
# Argument parser setup
# This section defines arguments for:
# 1. Whether to finetune or do linear probing
# 2. Dataset / dataloader settings
# 3. Patch settings
# 4. Model hyperparameters
# 5. Optimization settings
# 6. Pretrained / finetuned model naming
# ------------------------------------------------------------------
parser = argparse.ArgumentParser()

# Pretraining and finetuning mode selection
parser.add_argument('--is_finetune', type=int, default=0, help='do finetuning or not')
parser.add_argument('--is_linear_probe', type=int, default=0, help='if linear_probe: only finetune the last layer')

# Dataset and dataloader arguments
parser.add_argument('--dset_finetune', type=str, default='etth1', help='dataset name')
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
parser.add_argument('--d_ff', type=int, default=256, help='Tranformer MLP dimension')
parser.add_argument('--dropout', type=float, default=0.2, help='Transformer dropout')
parser.add_argument('--head_dropout', type=float, default=0.2, help='head dropout')

# Optimization arguments
parser.add_argument('--n_epochs_finetune', type=int, default=20, help='number of finetuning epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')

# Name / path of the pretrained model to load before finetuning or linear probing
parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained model name')

# Identifier used to distinguish saved finetuned models
parser.add_argument('--finetuned_model_id', type=int, default=1, help='id of the saved finetuned model')

# Model type string used for organizing output directories
parser.add_argument('--model_type', type=str, default='based_model', help='for multivariate model or univariate model')


# ------------------------------------------------------------------
# Parse command-line arguments and print them for logging
# ------------------------------------------------------------------
args = parser.parse_args()
print('args:', args)

# ------------------------------------------------------------------
# Construct the directory where finetuned models and outputs will be saved
# ------------------------------------------------------------------
args.save_path = 'saved_models/' + args.dset_finetune + '/masked_patchtst/' + args.model_type + '/'

# Create the save directory if it does not already exist
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

# ------------------------------------------------------------------
# Build a suffix string that records the key finetuning hyperparameters
# to keep saved model names organized and interpretable
# ------------------------------------------------------------------
# args.save_finetuned_model = '_cw'+str(args.context_points)+'_tw'+str(args.target_points) + '_patch'+str(args.patch_len) + '_stride'+str(args.stride) + '_epochs-finetune' + str(args.n_epochs_finetune) + '_mask' + str(args.mask_ratio)  + '_model' + str(args.finetuned_model_id)
suffix_name = (
    '_cw' + str(args.context_points)
    + '_tw' + str(args.target_points)
    + '_patch' + str(args.patch_len)
    + '_stride' + str(args.stride)
    + '_epochs-finetune' + str(args.n_epochs_finetune)
    + '_model' + str(args.finetuned_model_id)
)

# ------------------------------------------------------------------
# Decide how the finetuned model should be named depending on whether
# we are doing full finetuning or linear probing
# ------------------------------------------------------------------
if args.is_finetune:
    args.save_finetuned_model = args.dset_finetune + '_patchtst_finetuned' + suffix_name
elif args.is_linear_probe:
    args.save_finetuned_model = args.dset_finetune + '_patchtst_linear-probe' + suffix_name
else:
    args.save_finetuned_model = args.dset_finetune + '_patchtst_finetuned' + suffix_name

# ------------------------------------------------------------------
# Automatically detect and set the available compute device
# ------------------------------------------------------------------
set_device()


def get_model(c_in, args, head_type, weight_path=None):
    """
    Build and return a PatchTST model configured for either prediction
    or another head type, with optional weight transfer.

    Parameters
    ----------
    c_in : int
        Number of input variables / channels.
    args : argparse.Namespace
        Parsed command-line arguments containing model configuration.
    head_type : str
        Specifies which head the model should use (e.g., 'prediction').
    weight_path : str, optional
        Path to pretrained weights to load into the model.
    """
    # --------------------------------------------------------------
    # Compute the number of patches extracted from the input sequence
    # based on context length, patch length, and stride
    # --------------------------------------------------------------
    num_patch = (max(args.context_points, args.patch_len) - args.patch_len) // args.stride + 1
    print('number of patches:', num_patch)

    # --------------------------------------------------------------
    # Initialize the PatchTST model
    # --------------------------------------------------------------
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

    # --------------------------------------------------------------
    # If a pretrained weight path is provided, load / transfer the
    # pretrained weights into the current model
    # --------------------------------------------------------------
    if weight_path:
        model = transfer_weights(weight_path, model)

    # --------------------------------------------------------------
    # Print the total number of trainable parameters
    # --------------------------------------------------------------
    print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))

    return model


def find_lr(head_type):
    """
    Run a learning-rate finder on the finetuning setup to determine
    a suggested learning rate.
    """
    # --------------------------------------------------------------
    # Build dataloaders for the finetuning dataset
    # --------------------------------------------------------------
    dls = get_dls(args)

    # --------------------------------------------------------------
    # Build the model with the requested head type
    # --------------------------------------------------------------
    model = get_model(dls.vars, args, head_type)

    # --------------------------------------------------------------
    # Load the pretrained weights into the model before LR search
    # --------------------------------------------------------------
    # weight_path = args.save_path + args.pretrained_model + '.pth'
    model = transfer_weights(args.pretrained_model, model)

    # --------------------------------------------------------------
    # Define the loss function for supervised forecasting
    # --------------------------------------------------------------
    loss_func = torch.nn.MSELoss(reduction='mean')

    # --------------------------------------------------------------
    # Build callbacks:
    # - RevInCB applies reversible normalization if enabled
    # - PatchCB converts input sequences into patches during finetuning
    # --------------------------------------------------------------
    cbs = [RevInCB(dls.vars)] if args.revin else []
    cbs += [PatchCB(patch_len=args.patch_len, stride=args.stride)]

    # --------------------------------------------------------------
    # Create the Learner object
    # --------------------------------------------------------------
    learn = Learner(
        dls,
        model,
        loss_func,
        lr=args.lr,
        cbs=cbs,
    )

    # --------------------------------------------------------------
    # Run the learning-rate finder and return the suggestion
    # --------------------------------------------------------------
    suggested_lr = learn.lr_finder()
    print('suggested_lr', suggested_lr)
    return suggested_lr


def save_recorders(learn):
    """
    Save the training and validation loss history from the learner
    into a CSV file.
    """
    # --------------------------------------------------------------
    # Extract recorded train and validation losses
    # --------------------------------------------------------------
    train_loss = learn.recorder['train_loss']
    valid_loss = learn.recorder['valid_loss']

    # Convert losses into a DataFrame
    df = pd.DataFrame(data={'train_loss': train_loss, 'valid_loss': valid_loss})

    # Save the DataFrame to disk
    df.to_csv(args.save_path + args.save_finetuned_model + '_losses.csv', float_format='%.6f', index=False)


def finetune_func(lr=args.lr):
    """
    Perform end-to-end finetuning of the pretrained PatchTST model.
    """
    print('end-to-end finetuning')

    # --------------------------------------------------------------
    # Build dataloaders
    # --------------------------------------------------------------
    dls = get_dls(args)

    # --------------------------------------------------------------
    # Build the model with a prediction head
    # --------------------------------------------------------------
    model = get_model(dls.vars, args, head_type='prediction')

    # --------------------------------------------------------------
    # Load pretrained weights into the prediction model
    # --------------------------------------------------------------
    # weight_path = args.pretrained_model + '.pth'
    model = transfer_weights(args.pretrained_model, model)

    # --------------------------------------------------------------
    # Define the supervised forecasting loss
    # --------------------------------------------------------------
    loss_func = torch.nn.MSELoss(reduction='mean')

    # --------------------------------------------------------------
    # Build callbacks:
    # - RevInCB with denorm=True reverses normalization for outputs
    # - PatchCB applies patching to the inputs
    # - SaveModelCB saves the best model based on validation loss
    # --------------------------------------------------------------
    cbs = [RevInCB(dls.vars, denorm=True)] if args.revin else []
    cbs += [
        PatchCB(patch_len=args.patch_len, stride=args.stride),
        SaveModelCB(monitor='valid_loss', fname=args.save_finetuned_model, path=args.save_path)
    ]

    # --------------------------------------------------------------
    # Create the Learner object with mse metric tracking
    # --------------------------------------------------------------
    learn = Learner(
        dls,
        model,
        loss_func,
        lr=lr,
        cbs=cbs,
        metrics=[mse]
    )

    # --------------------------------------------------------------
    # Run full finetuning:
    # - The commented line shows an alternative one-cycle fit
    # - The active call uses fine_tune with a freeze/unfreeze schedule
    # --------------------------------------------------------------
    # learn.fit_one_cycle(n_epochs=args.n_epochs_finetune, lr_max=lr)
    learn.fine_tune(n_epochs=args.n_epochs_finetune, base_lr=lr, freeze_epochs=10)

    # --------------------------------------------------------------
    # Save recorded train / validation losses
    # --------------------------------------------------------------
    save_recorders(learn)


def linear_probe_func(lr=args.lr):
    """
    Perform linear probing, i.e., only train the final prediction head
    while keeping the pretrained backbone frozen.
    """
    print('linear probing')

    # --------------------------------------------------------------
    # Build dataloaders
    # --------------------------------------------------------------
    dls = get_dls(args)

    # --------------------------------------------------------------
    # Build the model with a prediction head
    # --------------------------------------------------------------
    model = get_model(dls.vars, args, head_type='prediction')

    # --------------------------------------------------------------
    # Load pretrained weights
    # --------------------------------------------------------------
    # weight_path = args.save_path + args.pretrained_model + '.pth'
    model = transfer_weights(args.pretrained_model, model)

    # --------------------------------------------------------------
    # Define the supervised forecasting loss
    # --------------------------------------------------------------
    loss_func = torch.nn.MSELoss(reduction='mean')

    # --------------------------------------------------------------
    # Build callbacks
    # --------------------------------------------------------------
    cbs = [RevInCB(dls.vars, denorm=True)] if args.revin else []
    cbs += [
        PatchCB(patch_len=args.patch_len, stride=args.stride),
        SaveModelCB(monitor='valid_loss', fname=args.save_finetuned_model, path=args.save_path)
    ]

    # --------------------------------------------------------------
    # Create the Learner object
    # --------------------------------------------------------------
    learn = Learner(
        dls,
        model,
        loss_func,
        lr=lr,
        cbs=cbs,
        metrics=[mse]
    )

    # --------------------------------------------------------------
    # Run linear probing, training only the final layer/head
    # --------------------------------------------------------------
    learn.linear_probe(n_epochs=args.n_epochs_finetune, base_lr=lr)

    # --------------------------------------------------------------
    # Save recorded train / validation losses
    # --------------------------------------------------------------
    save_recorders(learn)


def test_func(weight_path):
    """
    Evaluate a trained finetuned model on the test set and save the
    resulting metrics.
    """
    # --------------------------------------------------------------
    # Build dataloaders
    # --------------------------------------------------------------
    dls = get_dls(args)

    # --------------------------------------------------------------
    # Build the prediction model and move it to GPU
    # --------------------------------------------------------------
    model = get_model(dls.vars, args, head_type='prediction').to('cuda')

    # --------------------------------------------------------------
    # Build callbacks for test-time preprocessing / denormalization
    # --------------------------------------------------------------
    cbs = [RevInCB(dls.vars, denorm=True)] if args.revin else []
    cbs += [PatchCB(patch_len=args.patch_len, stride=args.stride)]

    # --------------------------------------------------------------
    # Create the Learner object for evaluation
    # --------------------------------------------------------------
    learn = Learner(dls, model, cbs=cbs)

    # --------------------------------------------------------------
    # Run testing using the provided weight path and compute mse / mae
    # out is expected to be [pred, targ, score]
    # --------------------------------------------------------------
    out = learn.test(dls.test, weight_path=weight_path + '.pth', scores=[mse, mae])  # out: a list of [pred, targ, score]

    print('score:', out[2])

    # --------------------------------------------------------------
    # Save the test metrics as a CSV file
    # --------------------------------------------------------------
    pd.DataFrame(
        np.array(out[2]).reshape(1, -1),
        columns=['mse', 'mae']
    ).to_csv(
        args.save_path + args.save_finetuned_model + '_acc.csv',
        float_format='%.6f',
        index=False
    )

    return out


if __name__ == '__main__':

    # --------------------------------------------------------------
    # Case 1: Full finetuning
    # --------------------------------------------------------------
    if args.is_finetune:
        args.dset = args.dset_finetune

        # Find a suggested learning rate for prediction finetuning
        suggested_lr = find_lr(head_type='prediction')

        # Run end-to-end finetuning
        finetune_func(suggested_lr)

        print('finetune completed')

        # Evaluate the best saved finetuned model on the test set
        out = test_func(args.save_path + args.save_finetuned_model)

        print('----------- Complete! -----------')

    # --------------------------------------------------------------
    # Case 2: Linear probing
    # --------------------------------------------------------------
    elif args.is_linear_probe:
        args.dset = args.dset_finetune

        # Find a suggested learning rate
        suggested_lr = find_lr(head_type='prediction')

        # Run linear probing
        linear_probe_func(suggested_lr)

        print('finetune completed')

        # Evaluate the saved linear-probe model
        out = test_func(args.save_path + args.save_finetuned_model)

        print('----------- Complete! -----------')

    # --------------------------------------------------------------
    # Case 3: Test-only mode
    # --------------------------------------------------------------
    else:
        args.dset = args.dset_finetune

        # Build the expected weight path for a previously finetuned model
        weight_path = args.save_path + args.dset_finetune + '_patchtst_finetuned' + suffix_name

        # Run test only
        out = test_func(weight_path)

        print('----------- Complete! -----------')
        
        
"""

import numpy as np
import pandas as pd
import os
import torch
from torch import nn

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

parser = argparse.ArgumentParser()
# Pretraining and Finetuning
parser.add_argument('--is_finetune', type=int, default=0, help='do finetuning or not')
parser.add_argument('--is_linear_probe', type=int, default=0, help='if linear_probe: only finetune the last layer')
# Dataset and dataloader
parser.add_argument('--dset_finetune', type=str, default='etth1', help='dataset name')
parser.add_argument('--context_points', type=int, default=512, help='sequence length')
parser.add_argument('--target_points', type=int, default=96, help='forecast horizon')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers for DataLoader')
parser.add_argument('--scaler', type=str, default='standard', help='scale the input data')
parser.add_argument('--features', type=str, default='M', help='for multivariate model or univariate model')
# Patch
parser.add_argument('--patch_len', type=int, default=12, help='patch length')
parser.add_argument('--stride', type=int, default=12, help='stride between patch')
# RevIN
parser.add_argument('--revin', type=int, default=1, help='reversible instance normalization')
# Model args
parser.add_argument('--n_layers', type=int, default=3, help='number of Transformer layers')
parser.add_argument('--n_heads', type=int, default=16, help='number of Transformer heads')
parser.add_argument('--d_model', type=int, default=128, help='Transformer d_model')
parser.add_argument('--d_ff', type=int, default=256, help='Tranformer MLP dimension')
parser.add_argument('--dropout', type=float, default=0.2, help='Transformer dropout')
parser.add_argument('--head_dropout', type=float, default=0.2, help='head dropout')
# Optimization args
parser.add_argument('--n_epochs_finetune', type=int, default=20, help='number of finetuning epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
# Pretrained model name
parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained model name')
# model id to keep track of the number of models saved
parser.add_argument('--finetuned_model_id', type=int, default=1, help='id of the saved finetuned model')
parser.add_argument('--model_type', type=str, default='based_model', help='for multivariate model or univariate model')


args = parser.parse_args()
print('args:', args)
args.save_path = 'saved_models/' + args.dset_finetune + '/masked_patchtst/' + args.model_type + '/'
if not os.path.exists(args.save_path): os.makedirs(args.save_path)

# args.save_finetuned_model = '_cw'+str(args.context_points)+'_tw'+str(args.target_points) + '_patch'+str(args.patch_len) + '_stride'+str(args.stride) + '_epochs-finetune' + str(args.n_epochs_finetune) + '_mask' + str(args.mask_ratio)  + '_model' + str(args.finetuned_model_id)
suffix_name = '_cw'+str(args.context_points)+'_tw'+str(args.target_points) + '_patch'+str(args.patch_len) + '_stride'+str(args.stride) + '_epochs-finetune' + str(args.n_epochs_finetune) + '_model' + str(args.finetuned_model_id)
if args.is_finetune: args.save_finetuned_model = args.dset_finetune+'_patchtst_finetuned'+suffix_name
elif args.is_linear_probe: args.save_finetuned_model = args.dset_finetune+'_patchtst_linear-probe'+suffix_name
else: args.save_finetuned_model = args.dset_finetune+'_patchtst_finetuned'+suffix_name

# get available GPU devide
set_device()

def get_model(c_in, args, head_type, weight_path=None):
    
    #c_in: number of variables
    
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
                head_type=head_type,
                res_attention=False
                )    
    if weight_path: model = transfer_weights(weight_path, model)
    # print out the model size
    print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model



def find_lr(head_type):
    # get dataloader
    dls = get_dls(args)    
    model = get_model(dls.vars, args, head_type)
    # transfer weight
    # weight_path = args.save_path + args.pretrained_model + '.pth'
    model = transfer_weights(args.pretrained_model, model)
    # get loss
    loss_func = torch.nn.MSELoss(reduction='mean')
    # get callbacks
    cbs = [RevInCB(dls.vars)] if args.revin else []
    cbs += [PatchCB(patch_len=args.patch_len, stride=args.stride)]
        
    # define learner
    learn = Learner(dls, model, 
                        loss_func, 
                        lr=args.lr, 
                        cbs=cbs,
                        )                        
    # fit the data to the model
    suggested_lr = learn.lr_finder()
    print('suggested_lr', suggested_lr)
    return suggested_lr


def save_recorders(learn):
    train_loss = learn.recorder['train_loss']
    valid_loss = learn.recorder['valid_loss']
    df = pd.DataFrame(data={'train_loss': train_loss, 'valid_loss': valid_loss})
    df.to_csv(args.save_path + args.save_finetuned_model + '_losses.csv', float_format='%.6f', index=False)


def finetune_func(lr=args.lr):
    print('end-to-end finetuning')
    # get dataloader
    dls = get_dls(args)
    # get model 
    model = get_model(dls.vars, args, head_type='prediction')
    # transfer weight
    # weight_path = args.pretrained_model + '.pth'
    model = transfer_weights(args.pretrained_model, model)
    # get loss
    loss_func = torch.nn.MSELoss(reduction='mean')   
    # get callbacks
    cbs = [RevInCB(dls.vars, denorm=True)] if args.revin else []
    cbs += [
         PatchCB(patch_len=args.patch_len, stride=args.stride),
         SaveModelCB(monitor='valid_loss', fname=args.save_finetuned_model, path=args.save_path)
        ]
    # define learner
    learn = Learner(dls, model, 
                        loss_func, 
                        lr=lr, 
                        cbs=cbs,
                        metrics=[mse]
                        )                            
    # fit the data to the model
    #learn.fit_one_cycle(n_epochs=args.n_epochs_finetune, lr_max=lr)
    learn.fine_tune(n_epochs=args.n_epochs_finetune, base_lr=lr, freeze_epochs=10)
    save_recorders(learn)


def linear_probe_func(lr=args.lr):
    print('linear probing')
    # get dataloader
    dls = get_dls(args)
    # get model 
    model = get_model(dls.vars, args, head_type='prediction')
    # transfer weight
    # weight_path = args.save_path + args.pretrained_model + '.pth'
    model = transfer_weights(args.pretrained_model, model)
    # get loss
    loss_func = torch.nn.MSELoss(reduction='mean')    
    # get callbacks
    cbs = [RevInCB(dls.vars, denorm=True)] if args.revin else []
    cbs += [
         PatchCB(patch_len=args.patch_len, stride=args.stride),
         SaveModelCB(monitor='valid_loss', fname=args.save_finetuned_model, path=args.save_path)
        ]
    # define learner
    learn = Learner(dls, model, 
                        loss_func, 
                        lr=lr, 
                        cbs=cbs,
                        metrics=[mse]
                        )                            
    # fit the data to the model
    learn.linear_probe(n_epochs=args.n_epochs_finetune, base_lr=lr)
    save_recorders(learn)


def test_func(weight_path):
    # get dataloader
    dls = get_dls(args)
    model = get_model(dls.vars, args, head_type='prediction').to('cuda')
    # get callbacks
    cbs = [RevInCB(dls.vars, denorm=True)] if args.revin else []
    cbs += [PatchCB(patch_len=args.patch_len, stride=args.stride)]
    learn = Learner(dls, model,cbs=cbs)
    out  = learn.test(dls.test, weight_path=weight_path+'.pth', scores=[mse,mae])         # out: a list of [pred, targ, score]
    print('score:', out[2])
    # save results
    pd.DataFrame(np.array(out[2]).reshape(1,-1), columns=['mse','mae']).to_csv(args.save_path + args.save_finetuned_model + '_acc.csv', float_format='%.6f', index=False)
    return out



if __name__ == '__main__':
        
    if args.is_finetune:
        args.dset = args.dset_finetune
        # Finetune
        suggested_lr = find_lr(head_type='prediction')        
        finetune_func(suggested_lr)        
        print('finetune completed')
        # Test
        out = test_func(args.save_path+args.save_finetuned_model)         
        print('----------- Complete! -----------')

    elif args.is_linear_probe:
        args.dset = args.dset_finetune
        # Finetune
        suggested_lr = find_lr(head_type='prediction')        
        linear_probe_func(suggested_lr)        
        print('finetune completed')
        # Test
        out = test_func(args.save_path+args.save_finetuned_model)        
        print('----------- Complete! -----------')

    else:
        args.dset = args.dset_finetune
        weight_path = args.save_path+args.dset_finetune+'_patchtst_finetuned'+suffix_name
        # Test
        out = test_func(weight_path)        
        print('----------- Complete! -----------')
"""