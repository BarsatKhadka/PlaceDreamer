#!/bin/bash
#SBATCH --job-name=ablate
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-14                # 5 configs x 3 CV folds
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --exclude=node049

# THE ABLATION. One factor at a time from the SHIPPED baseline, so every number is attributable.
# Local probes turned out to be inadmissible (4.2x less data, 2x smaller model, 10x fewer epochs,
# a different split, and checkpoint selection ON TEST — a 1.36 R2 swing from setup alone, and they
# trained on 2 of the 5 LOCKED OOD designs). So nothing below is decided locally: the cluster,
# on the real folds with the real protocol, decides.
#
#   cfg 0  base   WNS_HEAD=0 GEO_FEATS=0 CRIT_KNOB=0 HPWL_COMPOSE=pool   <- reproduces what we ship
#   cfg 1  +wns   WNS_HEAD=1 ...                                          <- real wns/tns heads
#   cfg 2  +sum   ...                          HPWL_COMPOSE=sum           <- analytic composition
#   cfg 3  +geo   GEO_FEATS=1 CRIT_KNOB=1 ...                             <- new features
#   cfg 4  all    everything on
#
# WHAT EACH IS EXPECTED TO MOVE, and the FAIR baseline it must beat (cluster fold-0 split, raw
# knobs, no OOD contamination — see learning.md 4h):
#
#   target        fair baseline   shipped GNN   verdict
#   total_hpwl        0.702          +0.654     wash        -> cfg 2 (+sum) targets this
#   buffer_area       0.386          +0.474     GNN wins    -> should hold
#   buffer_count      0.447          +0.595     GNN wins    -> should hold
#   wns               0.091          -1.102     CATASTROPHE -> cfg 1 (+wns) targets this
#   tns               0.118          -0.126     bad         -> cfg 1
#
# cfg 1 is the highest-value test: the shipped model is WORSE THAN A 3-KNOB LINEAR MODEL on wns
# (-1.102 vs 0.091) because WNS/TNS exist only as READOUTS off the per-cell endpt head, which is
# our worst prediction (pooled R2 -0.508). Real heads + the DIRECT_KNOB path are the fix.
#
# cfg 2: tot_hpwl = SUM_net HPWL_net is an IDENTITY, not a model. The GNN predicts the per-net
# unknown (AUC 0.912, ~1e6 labels) and an exact formula aggregates — GRANNITE's pattern (DAC'20).
# In-probe A/B (admissible, same setup both arms): sum held 17.4% rel-err while the pooled head
# diverged to 112.7%. Theory agrees the failure is aggregation, not representation: tot_hpwl is a
# sum of LOCAL quantities, inside the 1-WL class (Xu GIN ICLR'19; Chen NeurIPS'20 Cor 3.4).
#
# cfg 3 is expected to do ~NOTHING and is here to be falsified: die_area is provably collinear
# with utilization within a design (log die = log cell_area - log util, cell_area design-constant)
# and measured 0.702 -> 0.702; crit_path's fair gain is only 0.091 -> 0.118 and a probe showed no
# model gain. If cfg 3 == cfg 0, we DELETE both features rather than carry them.

set -euo pipefail
mkdir -p logs runs
module purge
module load python/2025.12-2
module load cuda12.8/toolkit/12.8.1
source "${HOME}/PlaceDreamer/venv/bin/activate"
cd "${HOME}/PlaceDreamer"

IDX=${SLURM_ARRAY_TASK_ID}
CFG=$((IDX / 3)); FOLD=$((IDX % 3))
case $CFG in
  0) NAME=base; WNS_HEAD=0; GEO_FEATS=0; CRIT_KNOB=0; HPWL_COMPOSE=pool ;;
  1) NAME=wns;  WNS_HEAD=1; GEO_FEATS=0; CRIT_KNOB=0; HPWL_COMPOSE=pool ;;
  2) NAME=sum;  WNS_HEAD=0; GEO_FEATS=0; CRIT_KNOB=0; HPWL_COMPOSE=sum  ;;
  3) NAME=geo;  WNS_HEAD=0; GEO_FEATS=1; CRIT_KNOB=1; HPWL_COMPOSE=pool ;;
  4) NAME=all;  WNS_HEAD=1; GEO_FEATS=1; CRIT_KNOB=1; HPWL_COMPOSE=sum  ;;
  *) echo "bad array index $IDX (expected 0-14)" >&2; exit 1 ;;
esac
export WNS_HEAD GEO_FEATS CRIT_KNOB HPWL_COMPOSE FOLD
export ENCODER=dehnn VN_KNOBS=0 AGGR=multi FUSE=concat   # winning encoder, pinned in every arm
export OUT="runs/ab_${NAME}"

NG=$(ls cache/graphs/*.npz 2>/dev/null | wc -l); NC=$(ls cache/coords/*.npz 2>/dev/null | wc -l)
if [ "$NC" -ne "$NG" ] || [ "$NC" -eq 0 ]; then
  echo "FATAL: cache/coords has ${NC} flows, cache/graphs has ${NG}. 'git pull' brings coords." >&2
  exit 1
fi

python -c "import torch; assert torch.cuda.is_available(), 'CUDA NOT VISIBLE'; print('GPU:', torch.cuda.get_device_name(0))"
echo "=== cfg ${CFG} (${NAME})  FOLD=${FOLD}  WNS_HEAD=${WNS_HEAD} GEO_FEATS=${GEO_FEATS} CRIT_KNOB=${CRIT_KNOB} HPWL_COMPOSE=${HPWL_COMPOSE} ==="
python src/train_fplace.py
