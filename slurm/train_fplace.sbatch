#!/bin/bash
#SBATCH --job-name=fplace
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --exclude=node049   # bad node (uncorrectable ECC) — same as MechRL

# Magnolia (Ole Miss MCSR) — L40S, 46GB VRAM.
# f_place: DE-HNN + knob-conditioning on EDA-Schema-V2 sky130hd.
#   5 LOCKED OOD designs (jpeg/aes_core/tv80/wb_dma/i2c) — never trained/tuned on.
#   13 dev designs -> 3 CV folds. Final OOD eval: python src/train_fplace.py --eval-ood
#
# Env-var knobs (all optional):
#   FOLD=0|1|2|all     which CV fold (dev designs)                  (default all)
#   ENCODER=dehnn|dehnn_novn|dehnn_undirected|sage|gat  ablation    (default dehnn)
#   EPOCHS=30  LR=1e-3  DIM=64  LAYERS=4  ACCUM=8  SEED=0
#   W_NETDEM=2         up-weight congestion (the headline target)   (default 1)
#   OUT=runs/<name>    output dir                                   (default runs/<encoder>)
#
# Examples:
#   sbatch slurm/train_fplace.sbatch                                        # full CV, DE-HNN
#   sbatch --export=ALL,ENCODER=sage slurm/train_fplace.sbatch              # GraphSAGE ablation
#   sbatch --export=ALL,ENCODER=dehnn_novn slurm/train_fplace.sbatch        # no-VN ablation
#   sbatch --export=ALL,FOLD=0,EPOCHS=5 slurm/train_fplace.sbatch           # quick single-fold check

set -euo pipefail
mkdir -p logs runs

module purge
module load python/2025.12-2
module load cuda12.8/toolkit/12.8.1

source "${HOME}/PlaceDreamer/venv/bin/activate"
cd "${HOME}/PlaceDreamer"

# sanity: torch MUST see the GPU. Default PyPI torch silently falls back to CPU on
# these nodes — the venv must have been built with the cu128 index (see slurm/README.md).
python -c "import torch; assert torch.cuda.is_available(), 'CUDA NOT VISIBLE — reinstall torch with --index-url https://download.pytorch.org/whl/cu128'; print('GPU:', torch.cuda.get_device_name(0))"

echo "encoder=${ENCODER:-dehnn} fold=${FOLD:-all} epochs=${EPOCHS:-30} lr=${LR:-1e-3}"
python src/train_fplace.py
