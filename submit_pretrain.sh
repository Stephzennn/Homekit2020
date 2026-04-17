#!/bin/bash
#SBATCH --job-name=patchtst_pretrain_positive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=500G
#SBATCH --account=coc
#SBATCH --qos=coc-ice
#SBATCH --partition=coc-gpu,ice-gpu
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:2
#SBATCH --constraint=nvidia-gpu
#SBATCH --output=/home/hice1/ezg6/projects/Homekit2020/logs/%x-%j.out
#SBATCH --error=/home/hice1/ezg6/projects/Homekit2020/logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ezg6@gatech.edu

# ── Working directory ────────────────────────────────────────────────────────
# saved_models/ lives at /home/hice1/ezg6/projects/saved_models/
# so CWD must be the projects root (not inside Homekit2020).
cd /home/hice1/ezg6/projects

# ── Redirect ALL output into the model save folder ───────────────────────────
# #SBATCH --output can't handle directory names with spaces, so we exec-redirect
# here.  Everything printed after this line goes to the model folder.
# The #SBATCH lines above act as an early fallback for SLURM's own preamble.
SAVE_DIR="/home/hice1/ezg6/projects/saved_models/Wearable/masked_patchtst/Positive only training 4132026"
exec > "${SAVE_DIR}/slurm-${SLURM_JOB_ID}.out" 2>&1

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"

# ── Environment ──────────────────────────────────────────────────────────────
source ~/.bashrc
conda activate Homekit2020

# ── DDP / thread settings ────────────────────────────────────────────────────
export OMP_NUM_THREADS=8
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# Disable InfiniBand for intra-node jobs (avoids ~20-min NCCL init on PACE)
export NCCL_IB_DISABLE=1

# ── Launch ───────────────────────────────────────────────────────────────────
torchrun --nproc_per_node=2 \
  /home/hice1/ezg6/projects/Homekit2020/src/models/models/PatchTST_self_supervised/patchtst_pretrain.py \
  --dset_pretrain Wearable \
  --context_points 10080 \
  --target_points 1 \
  --batch_size 32 \
  --num_workers 2 \
  --scaler standard \
  --features M \
  --patch_len 1440 \
  --stride 180 \
  --revin 1 \
  --n_layers 4 \
  --n_heads 8 \
  --d_model 256 \
  --d_ff 512 \
  --dropout 0.1 \
  --head_dropout 0.1 \
  --mask_ratio 0.5 \
  --n_epochs_pretrain 100 \
  --lr 0.005 \
  --pretrained_model_id 12 \
  --model_type "Positive only training 4132026" \
  --label_filter positive \
  --resume_from "/home/hice1/ezg6/projects/saved_models/Wearable/masked_patchtst/Positive only training 4132026/patchtst_pretrained_cw10080_patch1440_stride180_epochs-pretrain100_mask0.5_model12.pth"

echo "Job finished: $(date)"
