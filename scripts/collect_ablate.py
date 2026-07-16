#!/usr/bin/env python3
"""
Read the ablation (slurm/ablate.sbatch) and print ONE comparison table.

    python3 scripts/collect_ablate.py

Reads runs/ab_{base,arriv,sum,geo,all}/results_fold*.json and averages over folds.
Every arm uses the SAME folds, so a difference is attributable to the CONFIG.

HOW TO READ IT — in this order:
  1. `base` is the CONTROL. It must reproduce the shipped model (tot_hpwl_dev ~ +0.65,
     buf_area_dev ~ +0.47, buf_cnt_dev ~ +0.60, wns ~ -1.10, AUC ~ 0.91). If it does NOT,
     something drifted and NOTHING else in the table is readable. Check this first.
  2. Compare each arm to `base`, per metric. The delta is the effect of that one change.
  3. Look at SPREAD (the +/- column = std across the 3 folds). Fold variance has been large
     (CTS folds ranged +0.008 / +0.325 / -0.247). If |delta| < spread, IT IS NOT A RESULT.

WHAT EACH ARM SHOULD MOVE:
  arriv : ENDPT_TARGET=arrival. endpt r2 should clearly improve (-0.508 -> positive).
          wns may improve less: WNS = min over ~700 predicted slacks is an EXTREME-ORDER
          statistic, so one bad endpoint corrupts it. If endpt improves but wns does not,
          the target was never the problem — the min() amplification is, and the fix is
          MasterRTL's level 3 (a model correcting the aggregate), which we do not have.
  sum   : HPWL_COMPOSE=sum. tot_hpwl med_rel should drop a lot (52% -> ~20%).
          WATCH net_hpwl auc_top10 — the sum loss forces ABSOLUTE calibration and may fight
          the ranking objective. A big AUC loss would not be worth it.
  geo   : GEO_FEATS+CRIT_KNOB. EXPECTED TO DO NOTHING (die_area is collinear with utilization
          within a design). If geo == base, DELETE both features.
  all   : arrival + sum.
"""
import json, glob, os, sys
import numpy as np

ROOT = os.environ.get("PD_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARMS = ["base", "arriv", "sum", "geo", "all"]
# (json key, sub-metric, label, higher_is_better)
ROWS = [
    ("tot_hpwl_dev", "within_r2", "tot_hpwl knob-R2", True),
    ("buf_area_dev", "within_r2", "buf_area knob-R2", True),
    ("buf_cnt_dev",  "within_r2", "buf_cnt  knob-R2", True),
    ("wns",          "within_r2", "wns within-R2",    True),
    ("tns",          "within_r2", "tns within-R2",    True),
    ("tot_hpwl",     "med_rel",   "tot_hpwl rel-err", False),
    ("endpt",        "r2",        "endpt R2",         True),
    ("net_hpwl",     "auc_top10", "net rank AUC",     True),
]

def load(arm):
    """-> {(key,sub): [per-fold values]}"""
    out = {}
    for p in sorted(glob.glob(f"{ROOT}/runs/ab_{arm}/results_fold*.json")):
        try: folds = json.load(open(p))
        except Exception: continue
        for f in folds:
            for k, sub, _, _ in ROWS:
                v = f.get("metrics", {}).get(k, {}).get(sub)
                if v is not None and np.isfinite(v): out.setdefault((k, sub), []).append(float(v))
    return out

data = {a: load(a) for a in ARMS}
have = [a for a in ARMS if data[a]]
if not have:
    sys.exit("no results yet — runs/ab_*/results_fold*.json is empty. Still training?")
nf = {a: max((len(v) for v in data[a].values()), default=0) for a in have}
print(f"\nABLATION — mean over folds (n folds: " + ", ".join(f"{a}={nf[a]}" for a in have) + ")")
print("Every arm uses the SAME folds. base = the shipped model = the CONTROL.\n")
hdr = f"  {'metric':18}" + "".join(f"{a:>16}" for a in have)
print(hdr); print("  " + "-" * (len(hdr) - 2))
for k, sub, label, hib in ROWS:
    cells = []
    for a in have:
        v = data[a].get((k, sub))
        if not v: cells.append(f"{'-':>16}"); continue
        m, s = np.mean(v), (np.std(v) if len(v) > 1 else 0.0)
        cells.append(f"{m:>9.3f}±{s:<5.2f}")
    print(f"  {label:18}" + "".join(cells))
# deltas vs base — the actual result
if "base" in have:
    print(f"\n  DELTA vs base  (|delta| must exceed the fold spread to mean anything)")
    print(f"  {'metric':18}" + "".join(f"{a:>16}" for a in have if a != "base"))
    for k, sub, label, hib in ROWS:
        b = data["base"].get((k, sub))
        if not b: continue
        cells = []
        for a in have:
            if a == "base": continue
            v = data[a].get((k, sub))
            if not v: cells.append(f"{'-':>16}"); continue
            d = np.mean(v) - np.mean(b)
            good = (d > 0) == hib
            spread = np.std(b) if len(b) > 1 else 0.0
            tag = "" if abs(d) > max(spread, 1e-9) else " ns"      # ns = within fold noise
            cells.append(f"{d:>+11.3f}{('OK' if good else 'xx') if abs(d)>spread else '  '}{tag:>3}")
        print(f"  {label:18}" + "".join(cells))
    print("\n  OK = improved beyond fold spread | xx = WORSE beyond spread | ns = inside noise")
print()
