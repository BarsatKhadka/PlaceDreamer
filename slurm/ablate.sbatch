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
#   cfg 0  base   everything off, HPWL_COMPOSE=pool, ENDPT_TARGET=slack  <- reproduces what we ship
#   cfg 1  arriv  ENDPT_TARGET=arrival                                    <- MasterRTL's decomposition
#   cfg 2  +sum   HPWL_COMPOSE=sum                                        <- analytic composition
#   cfg 3  +geo   GEO_FEATS=1 CRIT_KNOB=1                                 <- expected to do NOTHING
#   cfg 4  all    arrival + sum (the two with real grounding)
#
# WHAT EACH IS EXPECTED TO MOVE, and the FAIR baseline it must beat (cluster fold-0 split, raw
# knobs, no OOD contamination — see learning.md 4h):
#
#   target        fair baseline   shipped GNN   verdict
#   total_hpwl        0.702          +0.654     wash        -> cfg 2 (+sum) targets this
#   buffer_area       0.386          +0.474     GNN WINS    -> should hold
#   buffer_count      0.447          +0.595     GNN WINS    -> should hold
#   wns/tns           WITHDRAWN      -1.102     -> cfg 1 targets this, as a clean A/B
# (the wns/tns BARS are withdrawn: one baseline un-standardized with the TEST design's own
#  mean/std, the other had no design term and scored -4.647. cfg0-vs-cfg1 needs neither.)
#
# cfg 1 (arrival) is the highest-value test, and it REPLACED an earlier "+wns real heads" arm that
# was BACKWARDS. Reading MasterRTL's source (paperCodes/MasterRTL) showed our WNS readout IS their
# architecture: graph_stat.cal_timing does slack = require_time - arrival (require_time = the CLOCK
# PERIOD), then wns = min(slack), tns = sum(slack) — identities, exactly like ours. Their level-1
# per-path RandomForest is accurate; OUR level-1 (endpt head) is broken (pooled R2 -0.508). So the
# defect is the TARGET, not the readout: slack carries the knob, arrival carries the structure.
# MEASURED per-endpoint CV across a design's knob configs (std/|mean|, lower = more structural):
#   slack   1.364 / 2.028 / 0.492 / 0.181      (ac97 / usb_funct / sasc / systemcdes)
#   arrival 0.076 / 0.329 / 0.103 / 0.191      -> up to 18x more stable
# Predicting arrival leaves the head a purely STRUCTURAL job and recovers slack by exact
# arithmetic, so the GNN never learns a timing knob response at all.
# NOTE: every wns BAR I previously quoted is withdrawn — both baselines were invalid (one
# un-standardized using the TEST design's own stats; the other had no design term). cfg 0 vs
# cfg 1 is a clean A/B and does not depend on any of those numbers.
#
# cfg 2: tot_hpwl = SUM_net HPWL_net is an IDENTITY, not a model. The GNN predicts the per-net
# unknown (AUC 0.912, ~1e6 labels) and an exact formula aggregates — GRANNITE's pattern (DAC'20).
# In-probe A/B (admissible, same setup both arms): sum held 17.4% rel-err while the pooled head
# diverged to 112.7%. Theory agrees the failure is aggregation, not representation: tot_hpwl is a
# sum of LOCAL quantities, inside the 1-WL class (Xu GIN ICLR'19; Chen NeurIPS'20 Cor 3.4).
#
# cfg 3 = CAN WE PREDICT GEOMETRY AT ALL? This is NOT a throwaway arm — CTS NEEDS it.
# In the SwiftCTS data (CTS knobs varied, unlike EDA-Schema) placement adds +0.124 to
# clock_buffers, so the seam is only load-bearing if f_place can emit geometry. If we cannot
# predict pos/vn_box, the whole f_place -> f_cts seam has nothing real to carry.
#
# Why the first launch had to be killed: the geometry losses were gated by `if "y_pos_x" in g`
# (ALWAYS TRUE), not by a flag — so they trained in EVERY arm including base, meaning cfg 0 was
# NOT the shipped model (which has no geometry heads) and the control validated nothing.
# And W_POS=5 / W_VNBOX=5 — numbers I picked with NO justification — made geometry 68.2% OF THE
# WHOLE LOSS while net_hpwl (AUC 0.912, the head that works) got 11.4%.
# Now W_POS=1 / W_VNBOX=0.5 -> geometry 22.2%, net_hpwl 27.8%. It can learn without steering
# the encoder.
#
# Read "GEO skill" (1 - err/trivial-baseline) OVER THE FULL 200 EPOCHS:
#   skill > 0 by ep 200  => geometry IS learnable -> the seam can carry real placement state.
#   skill <= 0 at ep 200 => this FORMULATION fails, and the literature says why: NOBODY does
#       plain L2 on absolute cross-design coordinates. Every group that generates coordinates
#       uses a distributional objective (GraphPlanner VAE'22 -> diffusion ICML'25 -> flow'26),
#       and TransPlace's worked answer is to predict RELATIVE (rho, dtheta) and decode from
#       FIXED I/O PADS. Anchor, do not canonicalize. That is the next formulation to try —
#       NOT deletion.
# (An earlier read of "GEO skill < 0" at EPOCH 26 of 200 was called dead prematurely. It is not
#  a verdict until the run ends.)

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
  0) NAME=base;  GEO_HEADS=0; GEO_FEATS=0; CRIT_KNOB=0; HPWL_COMPOSE=pool; ENDPT_TARGET=slack   ;;
  1) NAME=arriv; GEO_HEADS=0; GEO_FEATS=0; CRIT_KNOB=0; HPWL_COMPOSE=pool; ENDPT_TARGET=arrival ;;
  2) NAME=sum;   GEO_HEADS=0; GEO_FEATS=0; CRIT_KNOB=0; HPWL_COMPOSE=sum;  ENDPT_TARGET=slack   ;;
  3) NAME=geo;   GEO_HEADS=1; GEO_FEATS=1; CRIT_KNOB=1; HPWL_COMPOSE=pool; ENDPT_TARGET=slack   ;;
  4) NAME=all;   GEO_HEADS=0; GEO_FEATS=0; CRIT_KNOB=0; HPWL_COMPOSE=sum;  ENDPT_TARGET=arrival ;;
  *) echo "bad array index $IDX (expected 0-14)" >&2; exit 1 ;;
esac
export GEO_HEADS GEO_FEATS CRIT_KNOB HPWL_COMPOSE ENDPT_TARGET FOLD
export WNS_HEAD=0
export ENCODER=dehnn VN_KNOBS=0 AGGR=multi FUSE=concat   # winning encoder, pinned in every arm
export OUT="runs/ab_${NAME}"

NG=$(ls cache/graphs/*.npz 2>/dev/null | wc -l); NC=$(ls cache/coords/*.npz 2>/dev/null | wc -l)
if [ "$NC" -ne "$NG" ] || [ "$NC" -eq 0 ]; then
  echo "FATAL: cache/coords has ${NC} flows, cache/graphs has ${NG}. 'git pull' brings coords." >&2
  exit 1
fi

python -c "import torch; assert torch.cuda.is_available(), 'CUDA NOT VISIBLE'; print('GPU:', torch.cuda.get_device_name(0))"
echo "=== cfg ${CFG} (${NAME}) FOLD=${FOLD} ENDPT_TARGET=${ENDPT_TARGET} HPWL_COMPOSE=${HPWL_COMPOSE} GEO_FEATS=${GEO_FEATS} CRIT_KNOB=${CRIT_KNOB} ==="
python src/train_fplace.py
