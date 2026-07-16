#!/bin/bash
#SBATCH --job-name=wnshead
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-5                 # 2 WNS_HEAD x 3 folds
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --exclude=node049

# ============================================================================
# DIRECT POOLED HEAD  vs  IDENTITY READOUT.  Turns an argument into a number.
#
# docs/model_architecture.md section 1 says, in bold: "we don't predict PPA. We predict the
# state; PPA is arithmetic." The code does the opposite:
#     GLOBAL_TARGETS = ("tot_hpwl","buf_area","buf_cnt") + (("wns_g","tns_g") if WNS_HEAD else ())
#     self.h_lvl = ModuleDict({k: Linear(256, 2) for k in GLOBAL_TARGETS})
# wns_g/tns_g are POOLED GRAPH-LEVEL REGRESSION HEADS AT n=18 DESIGNS. That is exactly
# MasterRTL's architecture — which MasterRTL measured losing to XGBoost, and which our own doc
# says walls out, four paragraphs above the code that does it. WNS_HEAD defaults to 1.
#
#   WNS_HEAD=1   direct pooled head. What every run has silently shipped.
#   WNS_HEAD=0   WNS/TNS exist ONLY as a readout off the per-cell endpt head:
#                    slack = (T + c) - arrival ;  wns = min(slack) ;  tns = sum(min(slack,0))
#
# HONEST EXPECTATION — I expect WNS_HEAD=1 to WIN today, and that is not a defence of it:
# the readout reads off a per-cell head that is currently -0.508, and `min` is an ORDER
# STATISTIC — it reads the error TAIL, not the average, so it amplifies exactly that head's
# worst cells. The readout cannot be good until the timing chain (per-arc net/cell delay +
# levelized propagation, docs section 6) exists, and NONE of it is built: `grep net_arcs src/`
# returns 0 hits. So this job measures THE SIZE OF THE DEBT, not which design is right.
#
# WHAT THE NUMBER MEANS:
#   gap small  -> the direct head is carrying little; build the readout chain, it's nearly free.
#   gap large  -> the direct head is doing real work at n=18, and we must explain to a reviewer
#                 why that is not just MasterRTL with extra steps. That is a paper-level problem,
#                 not a code problem, and better to learn it now than in review.
#
# RUN THIS AFTER endpt_ablate — pin ENDPT_TARGET to whichever arm won there, because the readout
# is computed off the endpt head and its quality is exactly what is being tested.
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
export HPWL_COMPOSE=sum
# EDIT ME after endpt_ablate reports: set to the winning arm (slack | arrival | delta).
export ENDPT_TARGET=${ENDPT_TARGET:-delta}

I=${SLURM_ARRAY_TASK_ID}
WH=$((I / 3))          # 0 = readout only, 1 = direct head
FOLD=$((I % 3))

NG=$(ls cache/graphs/*.npz 2>/dev/null | wc -l)
if [ "$NG" -eq 0 ]; then echo "FATAL: cache/graphs is empty." >&2; exit 1; fi
for C in coords fp_arrival net_family; do
  NC=$(ls cache/$C/*.npz 2>/dev/null | wc -l)
  if [ "$NC" -ne "$NG" ]; then
    echo "FATAL: cache/$C has ${NC} flows, cache/graphs has ${NG}. Run: git pull" >&2; exit 1
  fi
done
python -c "import torch; assert torch.cuda.is_available(), 'CUDA NOT VISIBLE'; print('GPU:', torch.cuda.get_device_name(0))"

echo "############ WNS_HEAD=${WH}  ENDPT_TARGET=${ENDPT_TARGET}  fold=${FOLD} ############"
FOLD=${FOLD} WNS_HEAD=${WH} OUT=runs/wnshead_${WH} python src/train_fplace.py
