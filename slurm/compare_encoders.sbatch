#!/bin/bash
#SBATCH --job-name=fpcmp
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-11                # 4 variants x 3 CV folds, all in parallel
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --exclude=node049

# ENCODER ABLATION — two questions we have ASSUMED the answer to and never tested:
#   Q1: does DE-HNN's directed-hypergraph conv actually beat a plain GNN on OUR task?
#   Q2: do the virtual nodes earn their keep?
#
#   task  0,1,2 -> dehnn        driver/sink split + psi/mlp   + VN     folds 0,1,2
#   task  3,4,5 -> sage         GraphSAGE                     + VN     folds 0,1,2
#   task  6,7,8 -> gat          GATv2                         + VN     folds 0,1,2
#   task 9,10,11-> dehnn_novn   driver/sink split + psi/mlp   NO VN    folds 0,1,2
#
# EVERYTHING else is identical — same graph, same node features (SignNet PE, type_emb), same
# knob/ctx conditioning, same heads, same loss, same folds, same normalization. Only the
# message-passing operator changes (and, for novn, the virtual nodes). So any difference is
# attributable to that one thing.
#
# Results -> runs/cmp_<variant>/. Bring them home with git (checkpoints are gitignored):
#   for E in dehnn sage gat dehnn_novn; do OUT=runs/cmp_$E python src/train_fplace.py --aggregate; done
#   git add -A runs/ logs/ && git commit -m "results: encoder ablation" && git push

set -euo pipefail
mkdir -p logs runs

module purge
module load python/2025.12-2
module load cuda12.8/toolkit/12.8.1
source "${HOME}/PlaceDreamer/venv/bin/activate"
cd "${HOME}/PlaceDreamer"

ENCODERS=(dehnn sage gat dehnn_novn)
IDX=${SLURM_ARRAY_TASK_ID}
export ENCODER=${ENCODERS[$((IDX / 3))]}
export FOLD=$((IDX % 3))
export OUT="runs/cmp_${ENCODER}"

python -c "import torch; assert torch.cuda.is_available(), 'CUDA NOT VISIBLE'; print('GPU:', torch.cuda.get_device_name(0))"
echo "=== task ${IDX}: ENCODER=${ENCODER}  FOLD=${FOLD}  ->  ${OUT} ==="
python src/train_fplace.py
