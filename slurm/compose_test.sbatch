#!/bin/bash
#SBATCH --job-name=compose
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-5                 # 2 compositions x 3 CV folds
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --exclude=node049

# THE ARCHITECTURE A/B: how should a GLOBAL scalar be produced from a graph model?
#
#   HPWL_COMPOSE=pool : mean/max pooled readout -> MLP -> level+dev heads      (what we ship)
#   HPWL_COMPOSE=sum  : tot_hpwl = SUM_net HPWL_net -- an IDENTITY, not a model.
#                       The GNN predicts only the per-net unknown (where it has ~1e6 labels and
#                       scores AUC 0.912); an exact formula aggregates. GRANNITE's pattern
#                       (DAC'20): keep P = sum(a*C*V^2*f) exact, learn only the per-gate alpha.
#
# WHY THIS IS THE RIGHT EXPERIMENT (all measured or cited, see learning.md):
#  - The input graph is IDENTICAL across a design's 108 knob configs (cell_x drift 0.0000), so
#    the pooled LEVEL task has 18 effective examples -- one per distinct graph -- not 1944.
#  - tot_hpwl is a SUM of local quantities => inside the 1-WL-decomposable class (Xu GIN ICLR'19;
#    Chen NeurIPS'20 Cor 3.4). Theory says the GNN should be FINE at it, so our 0.912-per-net but
#    0.654-total is an AGGREGATION failure, not a representation failure.
#  - 182-flow probe: the pooled head DIVERGES (rel-err 29%->170%, knob-R2 -1.0->-9.6) while the
#    composed sum holds ~15-20% rel-err and wins at EVERY checkpoint. Full-data pooled head is
#    52% rel-err; the composed sum hit 17.4% on a toy set.
#  - MasterRTL (ICCAD'23) built exactly our pooled-GCN graph-level regressor for WNS/TNS/power/
#    area, found XGBoost more accurate, and shipped trees. We reproduced a published negative.
#  - DE-HNN predicts NO global scalars; CircuitGNN's readout taxonomy omits graph-level entirely.
#
# Also live in this run (first full-scale test of both):
#  - wns_g/tns_g as REAL heads instead of readouts off the broken endpt head (-1.102 -> +0.586 on
#    a probe, vs a design-informed ceiling of 0.693).
#  - die_area + sqrt(n*A) floorplan features (leak-free: die = cell_area/utilization, R2 0.9961
#    vs the placed-cell bbox). Lifts the tot_hpwl OLS ceiling 0.719 -> 0.857.
#
# READ: "GEO skill" (>0 = beat the trivial predictor) and the TEST block's tot_hpwl rel-err +
# knob-R2. Fold-matched, so any difference is attributable to the COMPOSITION alone.

set -euo pipefail
mkdir -p logs runs
module purge
module load python/2025.12-2
module load cuda12.8/toolkit/12.8.1
source "${HOME}/PlaceDreamer/venv/bin/activate"
cd "${HOME}/PlaceDreamer"

IDX=${SLURM_ARRAY_TASK_ID}
case $((IDX / 3)) in
  0) HPWL_COMPOSE=pool ;;
  1) HPWL_COMPOSE=sum  ;;
  *) echo "bad array index $IDX (expected 0-5)" >&2; exit 1 ;;
esac
export HPWL_COMPOSE
export ENCODER=dehnn VN_KNOBS=0 AGGR=multi FUSE=concat   # f_place's winning encoder, unchanged
export FOLD=$((IDX % 3))
export OUT="runs/compose_${HPWL_COMPOSE}"

NG=$(ls cache/graphs/*.npz 2>/dev/null | wc -l); NC=$(ls cache/coords/*.npz 2>/dev/null | wc -l)
if [ "$NC" -ne "$NG" ] || [ "$NC" -eq 0 ]; then
  echo "FATAL: cache/coords has ${NC} flows, cache/graphs has ${NG}. git pull brings coords." >&2
  exit 1
fi

python -c "import torch; assert torch.cuda.is_available(), 'CUDA NOT VISIBLE'; print('GPU:', torch.cuda.get_device_name(0))"
echo "=== task ${IDX}: HPWL_COMPOSE=${HPWL_COMPOSE}  FOLD=${FOLD} ==="
python src/train_fplace.py
