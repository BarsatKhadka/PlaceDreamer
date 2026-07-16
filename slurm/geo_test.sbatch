#!/bin/bash
#SBATCH --job-name=geo
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-2                 # one job per CV fold
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --exclude=node049

# CAN f_place LEARN PLACEMENT GEOMETRY? The one question, at full scale.
#
# The seam showed f_cts ignores placement summary metrics -- CTS is driven by WHERE the sinks
# landed, and no scalar expresses that. So f_place must emit geometry. Two heads are in:
#   cell_pos : per-cell normalized (x,y)             -- ~20k targets/flow
#   vn_box   : per-METIS-cluster bbox, die-normalized -- ~40-350 targets/flow, ~110 cells each
# Boxes are normalized by the die, so every design sits on the same [0,1] scale.
#
# The runs are scored against the predictor they must BEAT, not against themselves:
#   vn_box   baseline = the train-only MEAN BOX
#   cell_pos baseline = the die CENTRE
#   skill = 1 - err/baseline.   skill > 0 -> learned real geometry.  skill <= 0 -> learned nothing.
# Watch "GEO skill" in the per-epoch line. Short d=48/3-epoch probes gave skill ~= -0.03 (i.e.
# converged to the mean, no geometry) -- but they were far too small to conclude. THIS is the
# honest test: full config, 200 epochs, the real stratified folds.
#
# Everything else is the winning f_place config, unchanged, so geometry is the only new variable.

set -euo pipefail
mkdir -p logs runs
module purge
module load python/2025.12-2
module load cuda12.8/toolkit/12.8.1
source "${HOME}/PlaceDreamer/venv/bin/activate"
cd "${HOME}/PlaceDreamer"

export ENCODER=dehnn VN_KNOBS=0 AGGR=multi FUSE=concat   # f_place's winning encoder config
export FOLD=${SLURM_ARRAY_TASK_ID} OUT=runs/fplace_geo

python -c "import torch; assert torch.cuda.is_available(), 'CUDA NOT VISIBLE'; print('GPU:', torch.cuda.get_device_name(0))"
echo "=== fold ${FOLD}: f_place + geometry (cell_pos + vn_box). Watch GEO skill. ==="
python src/train_fplace.py
