#!/bin/bash
#SBATCH --job-name=xgboost_patchtst
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --account=coc
#SBATCH --qos=coc-ice
#SBATCH --partition=coc-gpu,ice-gpu
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=nvidia-gpu
#SBATCH --output=/home/hice1/ezg6/projects/Homekit2020/logs/%x-%j.out
#SBATCH --error=/home/hice1/ezg6/projects/Homekit2020/logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ezg6@gatech.edu

# ── Working directory ─────────────────────────────────────────────────────────
cd /home/hice1/ezg6/projects

# ── Redirect ALL output into the model save folder ───────────────────────────
SAVE_DIR="/home/hice1/ezg6/projects/saved_models/Wearable/masked_patchtst/Two_Full_base_model_patch1440_stride180"
exec > "${SAVE_DIR}/slurm-${SLURM_JOB_ID}.out" 2>&1

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
python Homekit2020/src/models/models/PatchTST_self_supervised/XgBoost.py \
    --is_xgboost 1 \
    --skip_tsne \
    --neg_subsample_ratio 30 \
    --checkpoint /home/hice1/ezg6/projects/saved_models/Wearable/masked_patchtst/Two_Full_base_model_patch1440_stride180/Wearable_patchtst_finetuned_cw10080_tw1_patch1440_stride180_epochs-finetune20_model1.pth \
    --out_dir    /home/hice1/ezg6/projects/saved_models/Wearable/masked_patchtst/Two_Full_base_model_patch1440_stride180 \
    --xgb_model_id 3014282026 \
    --c_in 8 --d_model 256 --n_layers 6 --n_heads 8 --d_ff 512 \
    --patch_len 1440 --stride 180 --context_points 10080 \
    --batch_size 64 --num_workers 4

echo "Job finished: $(date)"
