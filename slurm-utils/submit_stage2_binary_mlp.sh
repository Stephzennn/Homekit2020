#!/bin/bash
#SBATCH --job-name=stage2_binary_mlp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --account=coc
#SBATCH --qos=coc-ice
#SBATCH --partition=coc-gpu,ice-gpu
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=nvidia-gpu
#SBATCH --output=/home/hice1/ezg6/projects/Homekit2020/logs/%x-%j.out
#SBATCH --error=/home/hice1/ezg6/projects/Homekit2020/logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ezg6@gatech.edu

# ── Working directory ─────────────────────────────────────────────────────────
cd /home/hice1/ezg6/projects

# ── Redirect ALL output into save folder ─────────────────────────────────────
SAVE_DIR="/home/hice1/ezg6/projects/saved_models/Wearable_3class/masked_patchtst/Stage2_BinaryMLP"
mkdir -p "${SAVE_DIR}"
exec > "${SAVE_DIR}/slurm-stage2-${SLURM_JOB_ID}.out" 2>&1

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"

# ── Environment ───────────────────────────────────────────────────────────────
source ~/.bashrc
conda activate Homekit2020

# ── Thread settings ───────────────────────────────────────────────────────────
export OMP_NUM_THREADS=8
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# ── Launch ────────────────────────────────────────────────────────────────────
# Stage 2: train binary MLP on 3-class PatchTST softmax outputs.
#   Input: [p0, p1, p2] from frozen stage-1 checkpoint (3 features per window)
#   Output: P(flu positive) — binary sigmoid
#   Undersampling: 20:1 on train negatives (val/test untouched)
#   pos_weight_cap=1: no extra upweighting — undersampling is enough (matches best binary setup)
python \
  Homekit2020/src/models/models/PatchTST_self_supervised/stage2_binary_mlp.py \
  --checkpoint "/home/hice1/ezg6/projects/saved_models/Wearable_3class/masked_patchtst/Three_Class_finetune_patch1440_stride180/Wearable_3class_patchtst_finetuned_cw10080_tw3_patch1440_stride180_epochs-finetune60_model35032026.pth" \
  --context_points 10080 \
  --target_points 3 \
  --patch_len 1440 \
  --stride 180 \
  --revin 1 \
  --n_layers 6 \
  --n_heads 8 \
  --d_model 256 \
  --d_ff 512 \
  --dropout 0.2 \
  --head_dropout 0.2 \
  --batch_size 32 \
  --num_workers 2 \
  --neg_subsample_ratio 20 \
  --pos_weight_cap 1.0 \
  --mlp_hidden 64 32 \
  --mlp_dropout 0.3 \
  --mlp_lr 1e-3 \
  --mlp_wd 1e-4 \
  --mlp_epochs 300 \
  --mlp_batch_size 256 \
  --mlp_patience 30 \
  --seed 42 \
  --save_path "${SAVE_DIR}/" \
  --model_id "stage2_mlp_20_1_35022026"

echo "Job finished: $(date)"
