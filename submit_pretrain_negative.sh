#!/bin/bash
#SBATCH --job-name=patchtst_pretrain_negative
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --account=coc
#SBATCH --qos=coc-ice
#SBATCH --partition=coc-gpu,ice-gpu
#SBATCH --time=11:00:00
#SBATCH --gres=gpu:2
#SBATCH --constraint=nvidia-gpu
#SBATCH --output=/home/hice1/ezg6/projects/Homekit2020/logs/%x-%j.out
#SBATCH --error=/home/hice1/ezg6/projects/Homekit2020/logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ezg6@gatech.edu

# ── Working directory ────────────────────────────────────────────────────────
cd /home/hice1/ezg6/projects

# ── Redirect ALL output into the model save folder ───────────────────────────
SAVE_DIR="/home/hice1/ezg6/projects/saved_models/Wearable/masked_patchtst/Negative only training 4132026"
exec > "${SAVE_DIR}/slurm-${SLURM_JOB_ID}.out" 2>&1

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"

# ── Environment ──────────────────────────────────────────────────────────────
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Homekit2020

# ── DDP / thread settings ────────────────────────────────────────────────────
export OMP_NUM_THREADS=8
export CUDA_DEVICE_ORDER=PCI_BUS_ID
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
  --model_type "Negative only training 4132026" \
  --label_filter negative \
  --resume_from "/home/hice1/ezg6/projects/saved_models/Wearable/masked_patchtst/Negative only training 4132026/patchtst_pretrained_cw10080_patch1440_stride180_epochs-pretrain30_mask0.5_model12.pth"

echo "Job finished: $(date)"
