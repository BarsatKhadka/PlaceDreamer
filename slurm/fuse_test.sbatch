#!/bin/bash
#SBATCH --job-name=fpfuse
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-5                 # 2 fusions x 3 CV folds
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --exclude=node049

# FUSE A/B — the ONE architecture choice we never actually measured in isolation.
# concat became the default because it "should" help (keep library/type/PE in separate
# subspaces instead of superimposing them), was verified to BUILD, and got baked into the VN
# test. But the VN test also changed AGGR and the folds, so concat-vs-sum was never controlled.
#
#   cell encoder:  SUM     h = MLP(lib) + type_emb + SignNet(PE)   (superimposed in 64 dims)
#                  CONCAT  h = fuse_mlp([ MLP(lib) || type_emb || SignNet(PE) ])
#
#   task 0,1,2 -> FUSE=sum     folds 0,1,2
#   task 3,4,5 -> FUSE=concat  folds 0,1,2
#
# EVERYTHING else pinned to the winning config from the sweep + VN test:
#   ENCODER=dehnn  VN_KNOBS=0 (VN structure only)  AGGR=multi  + the SIZE-STRATIFIED folds.
# So any difference is attributable to FUSE alone.

set -euo pipefail
mkdir -p logs runs

module purge
module load python/2025.12-2
module load cuda12.8/toolkit/12.8.1
source "${HOME}/PlaceDreamer/venv/bin/activate"
cd "${HOME}/PlaceDreamer"

IDX=${SLURM_ARRAY_TASK_ID}
case $((IDX / 3)) in
  0) FUSE=sum    ;;
  1) FUSE=concat ;;
  *) echo "bad array index $IDX (expected 0-5)" >&2; exit 1 ;;
esac
export FUSE
export ENCODER=dehnn
export VN_KNOBS=0        # VN does structure only — the VN-test winner
export AGGR=multi        # the aggregation-sweep winner
export FOLD=$((IDX % 3))
export OUT="runs/fuse_${FUSE}"

python -c "import torch; assert torch.cuda.is_available(), 'CUDA NOT VISIBLE'; print('GPU:', torch.cuda.get_device_name(0))"
echo "=== task ${IDX}: FUSE=${FUSE}  (ENCODER=dehnn VN_KNOBS=0 AGGR=multi)  FOLD=${FOLD} ==="
python src/train_fplace.py
