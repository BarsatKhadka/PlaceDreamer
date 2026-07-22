#!/bin/bash
#SBATCH --job-name=muzero_probe
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%A.out
#SBATCH --error=logs/%x_%A.err
#SBATCH --exclude=node049

# THE PROBE (src/muzero_probe.py) — ONE run, look at the table at the end of the log.
# Does a value-equivalent latent ROLLOUT (MuZero model-half / VPN) predict final PPA better
# than DIRECT prediction, on data we already have? Three arms share f_place's OWN encoder
# (FPlace.encode) + the engineered 6-knob action + per-stage PPA reward heads; only the head
# differs. NO reconstruction anywhere — latents grounded ONLY through reward heads (value equiv.):
#   direct   z0            -> route metrics       (predict final PPA directly)
#   flat_mt  z0            -> ALL stage metrics    (multitask, NO rollout = honest control)
#   rollout  z1=g(z0,k)... -> per-stage reward heads (the VPN)
#
# READ (within-design Spearman on rt_wl, leave-designs-out, mean over TEST):
#   rollout ~ flat_mt  => staged latent adds nothing WITHOUT actions -> generate per-stage knobs.
#   rollout > flat_mt  => causal staging transfers -> build it out.
#   flat_mt  > direct  => the gain is EXTRA SUPERVISION, not structure (expected with one action).
# WFINAL=4 up-weights the route metrics so rollout's rt_wl gradient isn't diluted 1/15 vs 1/4.

set -euo pipefail
mkdir -p logs runs/muzero_probe
module purge
module load python/2025.12-2
module load cuda12.8/toolkit/12.8.1
source "${HOME}/PlaceDreamer/venv/bin/activate"
cd "${HOME}/PlaceDreamer"

export TEST="aes_core,i2c,sasc,ethernet"
export EPOCHS=300 D=128 LR=1e-3 SEED=0 SCALE=0 GHALF=0 WFINAL=4
export RESULT="runs/muzero_probe/probe.json"
export ENCODER=dehnn VN_KNOBS=0 AGGR=multi FUSE=concat CRIT_KNOB=1   # shipped encoder, pinned

NG=$(ls cache/graphs/*.npz 2>/dev/null | wc -l)
if [ "$NG" -eq 0 ]; then echo "FATAL: cache/graphs empty — sync cache first" >&2; exit 1; fi

python -c "import torch; assert torch.cuda.is_available(), 'CUDA NOT VISIBLE'; print('GPU:', torch.cuda.get_device_name(0))"
echo "=== muzero probe  TEST=${TEST}  WFINAL=${WFINAL}  EPOCHS=${EPOCHS} ==="
PYTHONPATH=src python src/muzero_probe.py
