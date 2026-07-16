#!/bin/bash
#SBATCH --job-name=seam
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-2                 # one job per CV fold
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --exclude=node049

# THE SEAM (f_place -> f_cts). TWO runs per fold off ONE f_place, so they are directly comparable:
#   REAL     : f_cts trained on the REAL placement state (from EDA-Schema)  -> perfect-upstream ceiling
#   IMAGINED : f_cts trained on f_place's PREDICTED placement state         -> the real pipeline
# Each trained f_cts is then DUAL-EVALUATED (fed real AND imagined state at test). The gap
# REAL-input minus IMAGINED-input on the SAME model is the COMPOUNDING we are here to measure.
#
# Self-contained per fold, no leakage: f_place, f_cts-real and f_cts-imagined all use
# TF.make_folds()/same SEED, so fold F holds out the SAME designs everywhere. f_place never
# trained on the designs f_cts is tested on -> the imagined state fed at test is a genuine
# cross-design prediction (the deployed regime), not something f_place memorised.

set -euo pipefail
mkdir -p logs runs
module purge
module load python/2025.12-2
module load cuda12.8/toolkit/12.8.1
source "${HOME}/PlaceDreamer/venv/bin/activate"
cd "${HOME}/PlaceDreamer"

export ENCODER=dehnn VN_KNOBS=0 AGGR=multi FUSE=concat   # f_place's winning encoder config
FOLD=${SLURM_ARRAY_TASK_ID}

python -c "import torch; assert torch.cuda.is_available(), 'CUDA NOT VISIBLE'; print('GPU:', torch.cuda.get_device_name(0))"

echo "############ [fold ${FOLD}] 1/3: train f_place (winning config) -> runs/fplace_seam ############"
FOLD=${FOLD} OUT=runs/fplace_seam python src/train_fplace.py

echo "############ [fold ${FOLD}] 2/3: f_cts SEAM=real (real placement state; dual-eval) ############"
FOLD=${FOLD} OUT=runs/fcts_real SEAM_MODE=real FPLACE_CKPT=runs/fplace_seam \
    python src/train_fcts.py

echo "############ [fold ${FOLD}] 3/3: f_cts SEAM=imagined (f_place's prediction; dual-eval) ############"
FOLD=${FOLD} OUT=runs/fcts_imagined SEAM_MODE=imagined FPLACE_CKPT=runs/fplace_seam \
    python src/train_fcts.py
