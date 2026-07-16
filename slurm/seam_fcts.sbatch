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

# THE SEAM (imagined) — f_cts consumes f_place's PREDICTION, not the real placement state.
# This is the ACTUAL pipeline number: what f_cts scores when the upstream stage is a model.
#
# Self-contained per fold so there is no dependency on stale runs and NO LEAKAGE:
#   step 1) train f_place  (winning config) on fold F  -> runs/fplace_seam/foldF.pt
#   step 2) train f_cts    SEAM_MODE=imagined on fold F, feeding runs/fplace_seam/foldF.pt
# Both use TF.make_folds() with the same SEED, so fold F holds out the SAME designs in both
# stages. f_place never trained on the designs f_cts is tested on -> the placement state it
# feeds at test time is a genuine cross-design PREDICTION, exactly the deployed regime.
#
# Compare later against SEAM_MODE=real (teacher-forced) to read off the compounding gap.

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

echo "=== [fold ${FOLD}] step 1/2: train f_place (winning config) -> runs/fplace_seam ==="
FOLD=${FOLD} OUT=runs/fplace_seam python src/train_fplace.py

echo "=== [fold ${FOLD}] step 2/2: train f_cts SEAM=imagined (feeds f_place's prediction) ==="
FOLD=${FOLD} OUT=runs/fcts_imagined SEAM_MODE=imagined FPLACE_CKPT=runs/fplace_seam \
    python src/train_fcts.py
