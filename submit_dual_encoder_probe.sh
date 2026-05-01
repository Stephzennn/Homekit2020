#!/bin/bash
#SBATCH --job-name=patchtst_dual_encoder_probe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=500G
#SBATCH --account=coc
#SBATCH --qos=coc-ice
#SBATCH --partition=coc-gpu,ice-gpu
#SBATCH --time=03:30:00
#SBATCH --gres=gpu:2
#SBATCH --constraint=nvidia-gpu
#SBATCH --output=/home/hice1/ezg6/projects/Homekit2020/logs/%x-%j.out
#SBATCH --error=/home/hice1/ezg6/projects/Homekit2020/logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ezg6@gatech.edu

# ── Working directory ────────────────────────────────────────────────────────
cd /home/hice1/ezg6/projects

# ── Redirect ALL output into the model save folder ───────────────────────────
SAVE_DIR="/home/hice1/ezg6/projects/saved_models/Wearable/masked_patchtst/Dual Encoder Probe 4202026"
mkdir -p "${SAVE_DIR}"
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
export NCCL_IB_DISABLE=1

# ── Paths ────────────────────────────────────────────────────────────────────
POS_MODEL="/home/hice1/ezg6/projects/saved_models/Wearable/masked_patchtst/Positive only training 4132026/patchtst_pretrained_cw10080_patch1440_stride180_epochs-pretrain100_mask0.5_model12.pth"
NEG_MODEL="/home/hice1/ezg6/projects/saved_models/Wearable/masked_patchtst/Negative only training 4132026/patchtst_pretrained_cw10080_patch1440_stride180_epochs-pretrain100_mask0.5_model12.pth"
PROBE_SCRIPT="/home/hice1/ezg6/projects/Homekit2020/src/models/models/PatchTST_self_supervised/dual_encoder_probe.py"

# ── Train and evaluate dual-encoder MLP probe ────────────────────────────────
echo ""
echo "=== DUAL ENCODER MLP PROBE ==="
torchrun --nproc_per_node=2 "${PROBE_SCRIPT}" \
  --positive_model "${POS_MODEL}" \
  --negative_model "${NEG_MODEL}" \
  --c_in 8 \
  --d_model 256 \
  --n_layers 4 \
  --n_heads 8 \
  --d_ff 512 \
  --dropout 0.1 \
  --head_dropout 0.1 \
  --patch_len 1440 \
  --stride 180 \
  --context_points 10080 \
  --revin 1 \
  --n_hidden_layers 2 \
  --hidden_dim 256 \
  --mlp_dropout 0.2 \
  --n_epochs_finetune 50 \
  --backbone_batch_size 64 \
  --batch_size 128 \
  --num_workers 6 \
  --use_lr_finder 1 \
  --use_scheduler 1 \
  --model_type "Dual Encoder Probe 4202026" \
  --finetuned_model_id 1

echo ""
echo "Job finished: $(date)"
