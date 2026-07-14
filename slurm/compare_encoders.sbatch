#!/bin/bash
#SBATCH --job-name=fpcmp
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-8                 # 3 encoders x 3 CV folds, all in parallel
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --exclude=node049

# ENCODER COMPARISON — does DE-HNN's directed-hypergraph machinery actually beat a plain GNN?
# We have ASSUMED it does and never tested it. Now we test it.
#
#   task 0,1,2 -> dehnn   (bipartite cell<->net, driver/sink split, psi/mlp)  folds 0,1,2
#   task 3,4,5 -> sage    (GraphSAGE on the same bipartite graph)             folds 0,1,2
#   task 6,7,8 -> gat     (GATv2   on the same bipartite graph)               folds 0,1,2
#
# Same data, same targets, same loss, same everything else. Only the message passing differs.
#
# Results land in runs/cmp_<encoder>/ — pull the whole dir off the cluster:
#   tar czf results.tar.gz runs/cmp_* logs/fpcmp_*
#
# After all 9 finish:
#   for E in dehnn sage gat; do OUT=runs/cmp_$E python src/train_fplace.py --aggregate; done

set -euo pipefail
mkdir -p logs runs

module purge
module load python/2025.12-2
module load cuda12.8/toolkit/12.8.1
source "${HOME}/PlaceDreamer/venv/bin/activate"
cd "${HOME}/PlaceDreamer"

ENCODERS=(dehnn sage gat)
IDX=${SLURM_ARRAY_TASK_ID}
export ENCODER=${ENCODERS[$((IDX / 3))]}
export FOLD=$((IDX % 3))
export OUT="runs/cmp_${ENCODER}"

python -c "import torch; assert torch.cuda.is_available(), 'CUDA NOT VISIBLE'; print('GPU:', torch.cuda.get_device_name(0))"
echo "=== task ${IDX}: ENCODER=${ENCODER}  FOLD=${FOLD}  ->  ${OUT} ==="
python src/train_fplace.py
