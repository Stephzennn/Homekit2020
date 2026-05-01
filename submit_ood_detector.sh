#!/bin/bash
#SBATCH --job-name=patchtst_ood_detector
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --account=coc
#SBATCH --qos=coc-ice
#SBATCH --partition=coc-gpu,ice-gpu
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=nvidia-gpu
#SBATCH --output=/home/hice1/ezg6/projects/Homekit2020/logs/%x-%j.out
#SBATCH --error=/home/hice1/ezg6/projects/Homekit2020/logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ezg6@gatech.edu

# ── Working directory ────────────────────────────────────────────────────────
cd /home/hice1/ezg6/projects

# ── Redirect ALL output into the model save folder ───────────────────────────
# #SBATCH --output can't handle directory names with spaces, so we exec-redirect
# here.  Everything printed after this line goes to the model folder.
# The #SBATCH lines above act as an early fallback for SLURM's own preamble.
SAVE_DIR="/home/hice1/ezg6/projects/saved_models/Wearable/masked_patchtst/OOD Detector 4182026"
mkdir -p "${SAVE_DIR}"
exec > "${SAVE_DIR}/slurm-${SLURM_JOB_ID}.out" 2>&1

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"

# ── Environment ──────────────────────────────────────────────────────────────
source ~/.bashrc
conda activate Homekit2020

# ── Thread settings ──────────────────────────────────────────────────────────
export OMP_NUM_THREADS=8
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# Disable InfiniBand for intra-node jobs (avoids ~20-min NCCL init on PACE)
export NCCL_IB_DISABLE=1

# ── Model checkpoints ────────────────────────────────────────────────────────
POS_MODEL="/home/hice1/ezg6/projects/saved_models/Wearable/masked_patchtst/Positive only training 4132026/patchtst_pretrained_cw10080_patch1440_stride180_epochs-pretrain100_mask0.5_model12.pth"
NEG_MODEL="/home/hice1/ezg6/projects/saved_models/Wearable/masked_patchtst/Negative only training 4132026/patchtst_pretrained_cw10080_patch1440_stride180_epochs-pretrain100_mask0.5_model12.pth"
DETECTOR_PKL="${SAVE_DIR}/ood_detector_both.pkl"
OOD_SCRIPT="/home/hice1/ezg6/projects/Homekit2020/src/models/models/PatchTST_self_supervised/ood_detector.py"

# ── Step 1: Fit detector on training embeddings (both models) ────────────────
echo ""
echo "=== STEP 1: FIT ==="
python "${OOD_SCRIPT}" \
  --mode fit \
  --model_paths "${POS_MODEL}" "${NEG_MODEL}" \
  --save_detector "${DETECTOR_PKL}" \
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
  --batch_size 32 \
  --num_workers 2 \
  --revin 1 \
  --combine bayes \
  --out_dir "${SAVE_DIR}/plots"

# ── Step 2: Evaluate on test split ──────────────────────────────────────────
echo ""
echo "=== STEP 2: EVALUATE (test) ==="
python "${OOD_SCRIPT}" \
  --mode evaluate \
  --model_paths "${POS_MODEL}" \
  --load_detector "${DETECTOR_PKL}" \
  --eval_split test \
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
  --batch_size 32 \
  --num_workers 2 \
  --revin 1 \
  --out_dir "${SAVE_DIR}/plots"

echo ""
echo "Job finished: $(date)"
