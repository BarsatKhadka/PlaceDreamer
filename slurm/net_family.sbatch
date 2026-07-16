#!/bin/bash
#SBATCH --job-name=netfam
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-5                 # 2 NET_TARGET x 3 folds
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --exclude=node049

# ============================================================================
# NET_TARGET=own  vs  family  — the net-family rollup.
#
# "the graph grows" was docs section 10's UNDECIDED. It is now decided, and it cost no model:
# a resizer split does not CREATE wire, it REDISTRIBUTES one net's span across a family. So roll
# the LABEL up to the floorplan net. The label changes; the node set does not.
#
# MEASURED, all 1944 flows, at LABEL_STAGE=global_place:
#   Sigma_i fam_hpwl(i) / meta.total_hpwl = 1.000000       <- AN EXACT IDENTITY (own: 0.8963)
#   unattributed HPWL: median 0.00%, max 0.00%
# Three things follow, and they are why this is worth a job:
#   1. HPWL_COMPOSE=sum becomes EXACT. T1 FAILED at ratio 0.9469 and forced us to supervise
#      Sigma(our nets) — a weaker proxy — because supervising meta.total_hpwl would have inflated
#      every per-net prediction 13-30%. There is no gap left to inflate. T1 FAIL -> PASS.
#   2. cache/netmask is RETIRED (not applied in family mode). It discarded 0.39% of nets that
#      carry 4-6% of HPWL at a median 20x the rest — the highest-leverage nets in our best head.
#      The rollup explains 323/323 = 100% of them. ethernet-000042: 99.3% -> 100.0% of nets kept.
#   3. fam_n (family size) is a FREE per-net buffer-insertion label, 57,128 split families. That
#      is what MLBuf and BufFormer predict — except both need placement first and we do not.
#      NOT a feature: it is a global_place property; feeding it would leak. Predict it, don't eat it.
#
# WATCH: net_hpwl AUC. It is 0.912 and it is the head that works. The rollup changes its TARGET,
# so this job is also the check that the change does not damage it. If AUC drops, tell me and I
# will revert — a cleaner identity is not worth a worse head.
# Sanity ceiling: rel-err should IMPROVE. It is 43.7% today, and part of that is that the head was
# scored against fragment labels for the longest nets.
#
# NOTE: NORM_VERSION 4 -> 5. The rollup changes y_net_hpwl's distribution, so old norm caches are
# invalid and will be rebuilt automatically. Do not reuse a v4 cache.
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
export WNS_HEAD=1
export HPWL_COMPOSE=sum                      # the identity under test
export ENDPT_TARGET=${ENDPT_TARGET:-slack}   # held at the SHIPPED value — do not confound with endpt_ablate

TARGETS=(own family)
I=${SLURM_ARRAY_TASK_ID}
NT=${TARGETS[$((I / 3))]}
FOLD=$((I % 3))

NG=$(ls cache/graphs/*.npz 2>/dev/null | wc -l)
if [ "$NG" -eq 0 ]; then echo "FATAL: cache/graphs is empty." >&2; exit 1; fi
for C in coords fp_arrival net_family; do
  NC=$(ls cache/$C/*.npz 2>/dev/null | wc -l)
  if [ "$NC" -ne "$NG" ]; then
    echo "FATAL: cache/$C has ${NC} flows, cache/graphs has ${NG}. All ship via git: git pull" >&2
    exit 1
  fi
done
python -c "import torch; assert torch.cuda.is_available(), 'CUDA NOT VISIBLE'; print('GPU:', torch.cuda.get_device_name(0))"

echo "############ NET_TARGET=${NT}  fold=${FOLD} ############"
FOLD=${FOLD} NET_TARGET=${NT} OUT=runs/netfam_${NT} python src/train_fplace.py
