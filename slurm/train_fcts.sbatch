#!/bin/bash
#SBATCH --job-name=fcts
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --exclude=node049
set -euo pipefail
mkdir -p logs runs
module purge; module load python/2025.12-2; module load cuda12.8/toolkit/12.8.1
source "${HOME}/PlaceDreamer/venv/bin/activate"; cd "${HOME}/PlaceDreamer"
export ENCODER=dehnn VN_KNOBS=0 AGGR=multi FUSE=concat   # f_place's winning encoder config
export FOLD=${SLURM_ARRAY_TASK_ID} OUT=runs/fcts
python -c "import torch; assert torch.cuda.is_available(); print('GPU:',torch.cuda.get_device_name(0))"
python src/train_fcts.py
