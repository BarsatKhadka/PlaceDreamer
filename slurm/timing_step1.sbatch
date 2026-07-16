#!/bin/bash
#SBATCH --job-name=tstep1
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-5                 # 2 TIMING_HEADS x 3 folds
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --exclude=node049

# ============================================================================
# TIMING REBUILD, STEP 1 — THE FAIL-FAST HEAD. Designed to DIE CHEAPLY.
#
# THE PROBLEM (fold-0 TEST block, measured, not recalled):
#     endpt  r2 -0.382
#     wns    pooled r2 +0.332   BUT   within_r2 -0.720
# within_r2 is the honest one: WITHIN a design, as knobs move, we are worse than that design's own
# mean => ZERO knob signal on timing. The pooled +0.332 is design identity (big design -> bad
# slack), which is worthless to an RL agent that works inside ONE design.
# And the opportunity is real: within-design corr(place_wns, FINAL wns) = +0.796 (~63% of the knob
# variance). corr(place_HPWL, FINAL wns) = -0.105 (~1%) — so our GOOD head cannot carry timing.
# There is no bypass. Timing needs a timing path.
#
# WHAT THIS TRAINS: two heads, nothing else. h_ndly (per-NET delay) off the same embedding as our
# AUC-0.912 net_hpwl head, and h_cdly (per-GATE delay). No schedule, no max-channel, no
# propagation. This is deliberately the DUMBEST version of the rebuild.
#
# WHY: to kill the 2-3 week build in one day if it deserves killing.
#
# *** THE BAR. Beat the LINEAR floor or the chain is dead. ***
# Linear R2 on log net delay from [log hpwl, log fanout] — everything we know or predict:
#     ac97 +0.343 | ethernet +0.184 | usb_funct +0.080 | des3_area +0.214   -> median ~0.20
# Even ADDING post-placement cap+slew LABELS only reaches ~0.38. A GNN sees driver type and local
# topology a linear probe cannot, so 0.20 is a FLOOR — but if ndly r2 lands at ~0.20 the GNN is
# adding NOTHING over two columns of arithmetic, and the levelized chain cannot save it: a schedule
# propagates delays, it does not invent them. Kill it.
#
# WHAT TO LOOK FOR — the last TEST block:
#     ndly r2         <- THE number. Must clear ~0.20 by a real margin to justify step 2.
#     cdly r2         <- secondary. cell delay is 414x SMALLER than net delay pre-route.
#     net_hpwl AUC    <- MUST NOT REGRESS from 0.912. Two new heads share that embedding; if they
#                        damage our only good head, the trade is off regardless of ndly.
#     endpt r2 / wns within_r2  <- does having delay heads help the EXISTING timing head at all?
#
# ---------------------------------------------------------------------------
# THREE SECTION-6 CLAIMS THAT ARE FALSE. Measured before building, not after:
#  1. "33,617 arcs/flow, 48x more supervision" -> net_arcs/cell_arcs are PATH-indexed, not
#     arc-indexed. Real gain is 1.2x-9.3x (median ~2x): ac97 3238 labels vs 1764 endpoints = 1.8x.
#  2. "wire delay is a function of net length -> the chain closes on our strength" -> corr(log
#     hpwl, log delay) median ~+0.38, i.e. length explains ~14%. It does NOT close on our strength.
#  3. "the free floorplan-arrival prior beats our head by +0.98 R2" -> RETRACTED: T7b fit its shift
#     ON THE TEST DESIGN'S OWN TRUTH. Honest (train-fit) shift scores -653. Our head BEATS it.
# What survives: our head is broken, the signal is there, net/cell delay ARE well-defined per
# net/gate (within-net std across paths 0.0000 ns), and TimingGCN's levelized schedule is real
# (-0.84 synchronous -> +0.8957) — though they run POST-PLACEMENT and MEASURE net length, while we
# predict it at 45% rel-err. Do not promise their number.
# ============================================================================

set -euo pipefail
mkdir -p logs runs
module purge
module load python/2025.12-2
module load cuda12.8/toolkit/12.8.1
source "${HOME}/PlaceDreamer/venv/bin/activate"
cd "${HOME}/PlaceDreamer"

export ENCODER=dehnn VN_KNOBS=0 AGGR=multi FUSE=concat
export GEO_HEADS=0 GEO_FEATS=0 CRIT_KNOB=0
export WNS_HEAD=1 HPWL_COMPOSE=sum
export ENDPT_TARGET=${ENDPT_TARGET:-slack}     # the arm winning endpt_ablate. NOT delta (dead).
export NET_TARGET=${NET_TARGET:-own}           # set to family if net_family wins

I=${SLURM_ARRAY_TASK_ID}
TH=$((I / 3))          # 0 = today's model, 1 = + delay heads
FOLD=$((I % 3))

NG=$(ls cache/graphs/*.npz 2>/dev/null | wc -l)
if [ "$NG" -eq 0 ]; then echo "FATAL: cache/graphs is empty." >&2; exit 1; fi
for C in coords fp_arrival net_family arc_delay; do
  NC=$(ls cache/$C/*.npz 2>/dev/null | wc -l)
  if [ "$NC" -ne "$NG" ]; then
    echo "FATAL: cache/$C has ${NC} flows, cache/graphs has ${NG}. All ship via git: git pull" >&2
    exit 1
  fi
done
python -c "import torch; assert torch.cuda.is_available(), 'CUDA NOT VISIBLE'; print('GPU:', torch.cuda.get_device_name(0))"

echo "############ TIMING_HEADS=${TH}  fold=${FOLD}  ENDPT_TARGET=${ENDPT_TARGET} ############"
FOLD=${FOLD} TIMING_HEADS=${TH} OUT=runs/tstep1_${TH} python src/train_fplace.py
