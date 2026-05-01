#!/bin/bash
#SBATCH --job-name=patchtst_finetune_undersample_noweight
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

# ── Working directory ─────────────────────────────────────────────────────────
cd /home/hice1/ezg6/projects

# ── Redirect ALL output into the model save folder ───────────────────────────
SAVE_DIR="/home/hice1/ezg6/projects/saved_models/Wearable/masked_patchtst/Two_Full_base_model_patch1440_stride180"
exec > "${SAVE_DIR}/slurm-finetune-undersample-noweight-${SLURM_JOB_ID}.out" 2>&1

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"

# ── Environment ───────────────────────────────────────────────────────────────
source ~/.bashrc
conda activate Homekit2020

# ── DDP / thread settings ─────────────────────────────────────────────────────
export OMP_NUM_THREADS=8
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_IB_DISABLE=1

# ── Launch ────────────────────────────────────────────────────────────────────
torchrun --nproc_per_node=2 \
  Homekit2020/src/models/models/PatchTST_self_supervised/patchtst_finetune.py \
  --is_finetune 1 \
  --dset_finetune Wearable \
  --context_points 10080 \
  --target_points 1 \
  --batch_size 32 \
  --num_workers 2 \
  --scaler standard \
  --features M \
  --patch_len 1440 \
  --stride 180 \
  --revin 1 \
  --n_layers 6 \
  --n_heads 8 \
  --d_model 256 \
  --d_ff 512 \
  --dropout 0.2 \
  --head_dropout 0.2 \
  --n_epochs_finetune 20 \
  --neg_subsample_ratio 10 \
  --pos_weight_cap 1 \
  --finetuned_model_id 4 \
  --model_type "Two_Full_base_model_patch1440_stride180" \
  --pretrained_model "/home/hice1/ezg6/projects/saved_models/Wearable/masked_patchtst/Two_Full_base_model_patch1440_stride180/patchtst_pretrained_cw10080_patch1440_stride180_epochs-pretrain100_mask0.5_model492026.pth"

echo "Job finished: $(date)"
