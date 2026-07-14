#!/bin/bash
#SBATCH --job-name=fpaggr
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-8                 # 3 aggregations x 3 CV folds
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --exclude=node049

# AGGREGATION ABLATION — is the cells->nets aggregator the right one?
#
#   sum   a net's message scales with its fanout. Measured: the clock net sends a message 97x
#         stronger than a 2-pin net, and it feeds psi BEFORE any LayerNorm.
#   mean  normalized — BUT HPWL is a half-perimeter = a MAX-MIN SPAN, and a mean provably CANNOT
#         compute an extremal statistic of its neighbours. Structurally mismatched to the target.
#   multi [mean || max] concatenated: the net sees CENTRE and EXTREME, so it can express a span.
#
#   task 0,1,2 -> AGGR=sum    folds 0,1,2
#   task 3,4,5 -> AGGR=mean   folds 0,1,2
#   task 6,7,8 -> AGGR=multi  folds 0,1,2
#
# Everything else identical (dehnn encoder, same features, same loss, same folds).
#
# After: bash slurm/collect_results.sh   -> writes RESULTS.md (one pasteable file)

set -euo pipefail
mkdir -p logs runs

module purge
module load python/2025.12-2
module load cuda12.8/toolkit/12.8.1
source "${HOME}/PlaceDreamer/venv/bin/activate"
cd "${HOME}/PlaceDreamer"

IDX=${SLURM_ARRAY_TASK_ID}
case $((IDX / 3)) in
  0) AGGR=sum   ;;
  1) AGGR=mean  ;;
  2) AGGR=multi ;;
  *) echo "bad array index $IDX (expected 0-8)" >&2; exit 1 ;;
esac
export AGGR
export FOLD=$((IDX % 3))
export ENCODER=dehnn
export OUT="runs/aggr_${AGGR}"

python -c "import torch; assert torch.cuda.is_available(), 'CUDA NOT VISIBLE'; print('GPU:', torch.cuda.get_device_name(0))"
echo "=== task ${IDX}: AGGR=${AGGR}  FOLD=${FOLD}  ->  ${OUT} ==="
python src/train_fplace.py
