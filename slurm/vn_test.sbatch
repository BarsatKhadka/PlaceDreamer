#!/bin/bash
#SBATCH --job-name=fpvn
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-8                 # 3 VN configs x 3 CV folds
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --exclude=node049

# THE VN QUESTION — decisive test.
# The encoder sweep showed dehnn_novn (no virtual nodes) beat dehnn. But the VN does TWO jobs:
#   1. long-range STRUCTURE (pool clusters so distant cells share info in a few hops) — useful
#   2. KNOB delivery (ctx rides into the VN and broadcasts) — LOSTIN says this DILUTES the knobs
# So does the VN itself hurt, or only the knobs-through-VN path? Decouple it:
#
#   A) dehnn      VN_KNOBS=1   VN + knobs-through-VN   (the current baseline)
#   B) dehnn      VN_KNOBS=0   VN structure only, knobs via the direct ctx skip only  <- hypothesis
#   C) dehnn_novn VN_KNOBS=1   no VN at all            (the sweep winner)
#
# If B >= C: the VN's STRUCTURE helps; it was the knob delivery hurting. Keep VN, drop knob-in-VN.
# If B <  C: the VN genuinely does not earn its keep here.
#
#   task 0,1,2 -> A  folds 0,1,2
#   task 3,4,5 -> B  folds 0,1,2
#   task 6,7,8 -> C  folds 0,1,2
#
# ALL on the WINNING config from the sweeps: AGGR=multi, FUSE=concat, on the SIZE-STRATIFIED
# folds (the sweep RESULTS.md used the old random folds — fold 1 was the ethernet extrapolation).

set -euo pipefail
mkdir -p logs runs

module purge
module load python/2025.12-2
module load cuda12.8/toolkit/12.8.1
source "${HOME}/PlaceDreamer/venv/bin/activate"
cd "${HOME}/PlaceDreamer"

IDX=${SLURM_ARRAY_TASK_ID}
case $((IDX / 3)) in
  0) ENCODER=dehnn;      VN_KNOBS=1; TAG=vnA_dehnn_knobvn   ;;
  1) ENCODER=dehnn;      VN_KNOBS=0; TAG=vnB_dehnn_structonly ;;
  2) ENCODER=dehnn_novn; VN_KNOBS=1; TAG=vnC_novn          ;;
  *) echo "bad array index $IDX (expected 0-8)" >&2; exit 1 ;;
esac
export ENCODER VN_KNOBS
export AGGR=multi
export FUSE=concat
export FOLD=$((IDX % 3))
export OUT="runs/${TAG}"

python -c "import torch; assert torch.cuda.is_available(), 'CUDA NOT VISIBLE'; print('GPU:', torch.cuda.get_device_name(0))"
echo "=== task ${IDX}: ${TAG}  (ENCODER=${ENCODER} VN_KNOBS=${VN_KNOBS} AGGR=multi FUSE=concat)  FOLD=${FOLD} ==="
python src/train_fplace.py
