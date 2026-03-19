import numpy as np
import pandas as pd
import os
import torch
from torch import nn
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score

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


# ------------------------------------------------------------------
# Parse command-line arguments
# ------------------------------------------------------------------
args = parser.parse_args()

# ------------------------------------------------------------------
# Distributed detection
# ------------------------------------------------------------------
is_distributed = "LOCAL_RANK" in os.environ
local_rank = int(os.environ.get("LOCAL_RANK", 0))
rank = int(os.environ.get("RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))

# ------------------------------------------------------------------
# Device setup
# ------------------------------------------------------------------
if is_distributed:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Distributed run requested, but torch.cuda.is_available() is False. "
            "You are likely on a non-CUDA node (for example an AMD MI210 node) or using a non-CUDA PyTorch build. "
            "Run this on an NVIDIA CUDA node, or use normal python mode on CPU."
        )
    torch.cuda.set_device(local_rank)
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

    loss_func = torch.nn.BCEWithLogitsLoss()

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

    loss_func = torch.nn.BCEWithLogitsLoss()

    #cbs = [RevInCB(dls.vars, denorm=True)] if args.revin else []
    cbs = [RevInCB(dls.vars, denorm=False)] if args.revin else []
    
    cbs += [
        PatchCB(patch_len=args.patch_len, stride=args.stride),
        SaveModelCB(monitor='valid_loss', fname=args.save_finetuned_model, path=args.save_path)
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

    loss_func = torch.nn.BCEWithLogitsLoss()

    cbs = [RevInCB(dls.vars, denorm=True)] if args.revin else []
    
    #cbs = [RevInCB(dls.vars, denorm=False)] if args.revin else []
    cbs += [
        PatchCB(patch_len=args.patch_len, stride=args.stride),
        SaveModelCB(monitor='valid_loss', fname=args.save_finetuned_model, path=args.save_path)
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

    print('score:', scores)

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
            if rank == 0:
                print("Distributed run detected. Skipping lr_finder.")
                print("About to start finetuning")
            finetune_func(args.lr)
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
            if rank == 0:
                print("Distributed run detected. Skipping lr_finder.")
                print("About to start linear probing")
            linear_probe_func(args.lr)
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