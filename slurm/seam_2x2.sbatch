#!/bin/bash
#SBATCH --job-name=seam2x2
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-2                 # one job per CV fold; each runs the whole matrix
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --exclude=node049

# THE SEAM EXPERIMENT — the 2x2, plus the arm the literature says WINS.
#
# ⚠️ READ THIS BEFORE READING THE RESULTS. We spent this project assuming the imagined seam is
# WORSE than real and calling the gap "compounding". TWO PAPERS MEASURED THE OPPOSITE:
#
#  PowPrediCT (DAC'24, Yibo Lin) — placement -> post-route power with a CTS-surrogate phase.
#    Phase1 pretrain on post-route graphs -> Phase3 SWAP the input to placement graphs and
#    FINE-TUNE. Cross-design leave-one-out total-power rel-err:
#       Innovus 9.652% | vanilla GNN 14.149% (WORSE than the tool) | Phase1-only 5.106% | full 1.981%
#    The 1->3 fine-tune MORE THAN HALVES the error. That is teacher forcing -> student forcing;
#    they never name it.
#  MasterRTL IV.A (ICCAD'23) — chains a tree on its OWN predictions. vs post-place truth:
#       REAL netlist:            TNS MAPE 62%
#       CHAINED-ON-PREDICTED:    TNS MAPE  4%
#    "chained predictions achieve similar or even higher accuracy than ground-truth."
#
# MECHANISM: the downstream model learns to INVERT the upstream model's systematic bias — which it
# cannot do if you only ever hand it clean input it will never see at test time.
# => IMAGINATION IS A CALIBRATION CHANNEL, NOT ONLY A TAX. `imagined > real` is a FINDING here,
#    not a bug. Do not read this table expecting degradation.
#
# THE MATRIX (each trained f_cts is dual-evaluated, so one training fills a row):
#
#                       eval REAL            eval IMAGINED
#   train REAL      teacher-forced ceiling   the naive seam (what everyone fears)
#   train IMAGINED           —               train-from-scratch-on-imagined
#   real -> FINE-TUNE        —               *** PowPrediCT's schedule — predicted to WIN ***
#
# The 3rd row is the one we did not have. PowPrediCT's ordering (pretrain teacher-forced FIRST,
# then fine-tune) is LOAD-BEARING — do not train from scratch on imagined and call it the same
# thing.
#
# Per fold: train f_place once, then f_cts three ways off that same f_place. Same folds/SEED
# throughout => f_place never saw f_cts's test designs; the imagined state at test is a genuine
# cross-design prediction.

set -euo pipefail
mkdir -p logs runs
module purge
module load python/2025.12-2
module load cuda12.8/toolkit/12.8.1
source "${HOME}/PlaceDreamer/venv/bin/activate"
cd "${HOME}/PlaceDreamer"

export ENCODER=dehnn VN_KNOBS=0 AGGR=multi FUSE=concat   # f_place's winning encoder, pinned
export GEO_HEADS=0 GEO_FEATS=0 CRIT_KNOB=0               # dead/retracted — off (see learning.md)
FOLD=${SLURM_ARRAY_TASK_ID}

NG=$(ls cache/graphs/*.npz 2>/dev/null | wc -l)
for C in coords fp_arrival; do
  NC=$(ls cache/$C/*.npz 2>/dev/null | wc -l)
  if [ "$NC" -ne "$NG" ]; then
    echo "FATAL: cache/$C has ${NC} flows, cache/graphs has ${NG}." >&2
    echo "       coords ships via git pull; fp_arrival: python scripts/add_fp_arrival.py (~15s)" >&2
    exit 1
  fi
done
python -c "import torch; assert torch.cuda.is_available(), 'CUDA NOT VISIBLE'; print('GPU:', torch.cuda.get_device_name(0))"

echo "############ [fold ${FOLD}] 1/4: f_place (the upstream model the seam will imagine from) ############"
FOLD=${FOLD} OUT=runs/fplace_seam ENDPT_TARGET=delta HPWL_COMPOSE=sum python src/train_fplace.py

echo "############ [fold ${FOLD}] 2/4: f_cts SEAM=real  (row 1 — teacher forced) ############"
FOLD=${FOLD} OUT=runs/fcts_real SEAM_MODE=real FPLACE_CKPT=runs/fplace_seam python src/train_fcts.py

echo "############ [fold ${FOLD}] 3/4: f_cts SEAM=imagined FROM SCRATCH (row 2) ############"
FOLD=${FOLD} OUT=runs/fcts_imag SEAM_MODE=imagined FPLACE_CKPT=runs/fplace_seam python src/train_fcts.py

echo "############ [fold ${FOLD}] 4/4: f_cts real -> FINE-TUNE on imagined (row 3 — PowPrediCT) ############"
FOLD=${FOLD} OUT=runs/fcts_ft SEAM_MODE=imagined FPLACE_CKPT=runs/fplace_seam \
    WARM_START=runs/fcts_real FT_LR_MULT=0.1 python src/train_fcts.py
