#!/bin/bash
#SBATCH --job-name=fplace
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-2                 # one job per CV fold — folds are independent, run them in PARALLEL
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --exclude=node049   # bad node (uncorrectable ECC) — same as MechRL

# Magnolia (Ole Miss MCSR) — L40S, 46GB VRAM.
# f_place: DE-HNN + knob-conditioning on EDA-Schema-V2 sky130hd.
#   5 LOCKED OOD designs (jpeg/aes_core/tv80/wb_dma/i2c) — never trained/tuned on.
#   13 dev designs -> 3 CV folds. Each fold = one array task = its own GPU.
#
# Measured: ~49 s/epoch (fold 0, DE-HNN, dim 64, 4 layers, 778 train flows).
#   200 epochs ~= 2.7 h/fold; early stopping (patience 25) usually cuts it well short.
#   Folds run concurrently, so wall-clock ~= ONE fold, not three.
#
# Env-var knobs (all optional):
#   FOLD=0|1|2|all     override the array's fold (default: $SLURM_ARRAY_TASK_ID)
#   ENCODER=dehnn|dehnn_novn|dehnn_undirected|sage|gat  ablation    (default dehnn)
#   EPOCHS=200  LR=1e-3  DIM=64  LAYERS=4  ACCUM=8  SEED=0
#   PATIENCE=0         early stopping OFF (default) — trains all EPOCHS, keeps best-val ckpt
#   W_NETDEM=2         up-weight congestion (the headline target)   (default 1)
#   OUT=runs/<name>    output dir                                   (default runs/<encoder>)
#
# Examples:
#   sbatch slurm/train_fplace.sbatch                                        # full CV, DE-HNN, 3 GPUs
#   sbatch --export=ALL,ENCODER=sage,OUT=runs/sage slurm/train_fplace.sbatch
#   sbatch --export=ALL,W_NETDEM=4,OUT=runs/dehnn_w4 slurm/train_fplace.sbatch
#   sbatch --array=0 --export=ALL,EPOCHS=1 slurm/train_fplace.sbatch        # quick smoke test
#
# After all folds finish:
#   python src/train_fplace.py --aggregate            # OUT=runs/<name> to pick which

set -euo pipefail
mkdir -p logs runs

module purge
module load python/2025.12-2
module load cuda12.8/toolkit/12.8.1

source "${HOME}/PlaceDreamer/venv/bin/activate"
cd "${HOME}/PlaceDreamer"

# each array task trains exactly one fold (unless FOLD was explicitly exported)
export FOLD="${FOLD:-${SLURM_ARRAY_TASK_ID:-all}}"

# sanity: torch MUST see the GPU. Default PyPI torch silently falls back to CPU on
# these nodes — the venv must have been built with the cu128 index (see docs/hpc.md).
# NOTE: this is False on the LOGIN node (no GPU there) — that's expected, not an error.
python -c "import torch; assert torch.cuda.is_available(), 'CUDA NOT VISIBLE — reinstall torch with --index-url https://download.pytorch.org/whl/cu128'; print('GPU:', torch.cuda.get_device_name(0))"

echo "encoder=${ENCODER:-dehnn} fold=${FOLD} epochs=${EPOCHS:-200} patience=${PATIENCE:-0} lr=${LR:-1e-3}"
python src/train_fplace.py
