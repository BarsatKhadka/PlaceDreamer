#!/bin/bash
#SBATCH --job-name=endpt
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-8                 # 3 ENDPT_TARGET x 3 folds, all parallel
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --exclude=node049

# ============================================================================
# THE ONE QUESTION THIS ANSWERS
#
# stress_test T7b MEASURED, fold-0 test designs:
#     arrival_place := arrival_floorplan + one global shift   ->  R2  +0.476   (ZERO params)
#     our trained 680k-param endpt head                       ->  R2  -0.508
# The FREE prior — which is our own INPUT stage — beats our trained head by +0.98 R2.
#
# Everything in docs/model_architecture.md section 2 and 6 rests on that one number, and on the
# claim that predicting a RESIDUAL on the prior fixes it. THAT HAS NEVER BEEN TRAINED. Not once.
#
#   ENDPT_TARGET=slack     what we shipped. slack carries the knob AND the structure.
#   ENDPT_TARGET=arrival   predict arrival directly ( = (T + c) - slack ). Structure only.
#   ENDPT_TARGET=delta     predict arrival MINUS the free floorplan prior. The residual.
#
# If delta does not beat slack, section 2 + section 6 are WRONG and seam_2x2 is meaningless —
# it would train f_cts three ways off a broken f_place and we would learn nothing about the seam.
# That is why this runs FIRST and seam_2x2 does not.
#
# WHY THIS IS CHEAP: f_place only. No f_cts, no f_route, no seam, no imagination. 9 short runs.
#
# ---------------------------------------------------------------------------
# THE IDENTITY THIS DEPENDS ON — fixed today, and it was wrong before:
#     arrival = required - slack = (clock_period + c) - slack
#   NOT  arrival = clock_period - slack   <- what the code did. Median error 0.2735 ns.
#   c = required - clock_period is the capture flop's SETUP TIME + clock uncertainty. It is
#   STRUCTURAL (std across knob configs 0.019 ns vs spread across endpoints 0.180 ns, 9.4x), so
#   it is cached per-endpoint from the FLOORPLAN stage and the knob still enters by arithmetic.
#   Validated locally, untrained. `python scripts/stress_test.py T10` re-checks it in ~1 min.
#
# WHAT TO LOOK FOR IN THE LOG — the LAST TEST block of each run:
#     endpt   abs-R2   <- THE NUMBER. delta should beat slack. If it doesn't, tell me.
#     wns     med_ae   <- real ns. THE BAR IS THE FREE PRIOR (predict delta=0, ZERO params).
#                         MEASURED on fold 0, and IT IS PER-SPLIT — do not compare across splits:
#                            VAL  (des3_area, spi)                    wns med_ae 2.575 ns
#                                  des3_area 10.393 | spi 2.545       <- des3_area dominates
#                            TEST (ac97_ctrl, sasc, systemcdes, usb_funct)  wns med_ae 0.970 ns
#                                  ac97 0.744 | sasc 0.187 | systemcdes 3.295 | usb_funct 1.476
#                         The per-epoch line prints VAL. The last TEST block prints TEST. I once
#                         quoted "0.16 ns" here from memory and then compared a VAL log against a
#                         TEST baseline and called a healthy run broken. The bar is 2.575 on VAL.
#                         Training STARTS above the prior and descends: ep0 6.85 -> ep2 4.33.
#                         That is normal. The question is whether it lands BELOW the prior by ep200.
#     net_hpwl AUC     <- MUST NOT REGRESS. It is 0.912 and it is the head that works.
#                         If changing the timing target damages HPWL, the heads are fighting.
#     calib z2 -> 1.0  <- has NEVER been reported by any run. First look at whether sigma is honest.
# ============================================================================

set -euo pipefail
mkdir -p logs runs
module purge
module load python/2025.12-2
module load cuda12.8/toolkit/12.8.1
source "${HOME}/PlaceDreamer/venv/bin/activate"
cd "${HOME}/PlaceDreamer"

# f_place's winning encoder, pinned. GEO/CRIT are dead or retracted — see learning.md.
export ENCODER=dehnn VN_KNOBS=0 AGGR=multi FUSE=concat
export GEO_HEADS=0 GEO_FEATS=0 CRIT_KNOB=0
# WNS_HEAD PINNED ON so it is held CONSTANT across the three arms. It defaulted to 1 silently in
# seam_2x2, which meant we were shipping the direct pooled WNS head — MasterRTL's architecture,
# the one our own docs say walls out — as an unmade decision. Whether it should be 0 is a
# SEPARATE question, and it has its own job: slurm/wns_head.sbatch. Not confounded with this one.
export WNS_HEAD=1
export HPWL_COMPOSE=sum        # the one PPA identity that is actually built + supervised

TARGETS=(slack arrival delta)
I=${SLURM_ARRAY_TASK_ID}
T=${TARGETS[$((I / 3))]}
FOLD=$((I % 3))

# ---- GUARDS. seam_2x2's guard is what caught that fp_arrival never shipped; keep it everywhere.
NG=$(ls cache/graphs/*.npz 2>/dev/null | wc -l)
if [ "$NG" -eq 0 ]; then echo "FATAL: cache/graphs is empty." >&2; exit 1; fi
for C in coords fp_arrival net_family; do
  NC=$(ls cache/$C/*.npz 2>/dev/null | wc -l)
  if [ "$NC" -ne "$NG" ]; then
    echo "FATAL: cache/$C has ${NC} flows, cache/graphs has ${NG}." >&2
    echo "       Both ship via git now (fp_arrival is 2.5MB). Run: git pull" >&2
    echo "       Do NOT run scripts/add_fp_arrival.py here — it needs the 30GB timing_paths" >&2
    echo "       table, which is not on this cluster. That is why the cache is committed." >&2
    exit 1
  fi
done
# req_offset is what makes the corrected identity work. An OLD cache silently lacks it and would
# fall back to the WRONG arrival (off by a median 0.2735 ns) with no error. Check explicitly.
python - <<'PY'
import numpy as np, glob, sys
f = sorted(glob.glob("cache/fp_arrival/*.npz"))[0]
k = list(np.load(f).keys())
if "req_offset" not in k:
    sys.exit(f"FATAL: {f} lacks req_offset -> STALE cache (pre-identity-fix). Has {k}. git pull.")
print(f"[guard] fp_arrival OK: {k}")
PY
python -c "import torch; assert torch.cuda.is_available(), 'CUDA NOT VISIBLE'; print('GPU:', torch.cuda.get_device_name(0))"

echo "############ ENDPT_TARGET=${T}  fold=${FOLD}  (array ${I}) ############"
FOLD=${FOLD} ENDPT_TARGET=${T} OUT=runs/endpt_${T} python src/train_fplace.py
