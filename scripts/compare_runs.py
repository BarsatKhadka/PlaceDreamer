#!/usr/bin/env python3
"""
Compare N run variants that share folds. ONE table, one verdict per metric.

    python scripts/compare_runs.py runs/endpt_slack runs/endpt_arrival runs/endpt_delta
    python scripts/compare_runs.py runs/netfam_own runs/netfam_family

Every arm uses the SAME folds/SEED, so a difference IS attributable to the flag. That is the
whole point — and it is why this only compares TEST blocks, never the per-epoch VAL line.

WHY TEST-ONLY, AND WHY THIS SCRIPT EXISTS (a mistake this prevents):
the per-epoch VAL line uses a DIFFERENT held-out pair per fold, so its numbers are not comparable
across folds. MEASURED, ENDPT_TARGET=slack, same arm, two folds:
    fold 0  VAL=[des3_area, spi]      tot 38.1% rel   wns 0.40ns
    fold 2  VAL=[ac97_ctrl, sasc]     tot  4.2% rel   wns 1.05ns
A 9x swing in tot and a 2.6x swing in wns from the SAME code — that is the VAL split, not the
model. des3_area is a genuine outlier (median |arrival delta| 7.963ns vs ac97's 0.214; free-prior
wns error 10.393ns vs spi's 2.545). Reading a per-epoch line across folds will make you believe
things that are not true. Average TEST over folds, or do not compare.

ABSOLUTE vs RELATIVE: `tot 24000um` means nothing without the design's scale. This prints med_rel
(a ratio) next to med_ae (real units) for exactly that reason.
"""
import json, glob, os, sys
import numpy as np

RUNS = sys.argv[1:]
if not RUNS:
    sys.exit(__doc__)

# (metric, field, label, higher_is_better, fmt)
ROWS = [
    ("net_hpwl", "auc_top10", "net_hpwl  AUC@top10%", True,  "{:.4f}"),
    ("net_hpwl", "med_rel",   "net_hpwl  med rel-err", False, "{:.1%}"),
    ("net_hpwl", "r2",        "net_hpwl  R2",          True,  "{:+.4f}"),
    ("net_hpwl", "calib_z2",  "net_hpwl  calib z2->1", None,  "{:.3f}"),
    ("endpt",    "med_ae",    "endpt     med_ae (ns)", False, "{:.3f}"),
    ("endpt",    "r2",        "endpt     R2",          True,  "{:+.4f}"),
    ("wns",      "med_ae",    "wns       med_ae (ns)", False, "{:.3f}"),
    ("tns",      "med_ae",    "tns       med_ae (ns)", False, "{:.1f}"),
    ("tot_hpwl", "med_rel",   "tot_hpwl  med rel-err", False, "{:.1%}"),
    ("tot_hpwl_dev", "r2",    "tot_hpwl  KNOB-R2",     True,  "{:+.4f}"),
    ("buf_cnt_dev",  "r2",    "buf_cnt   KNOB-R2",     True,  "{:+.4f}"),
    ("buf_area_dev", "r2",    "buf_area  KNOB-R2",     True,  "{:+.4f}"),
]


def load(run):
    """average each metric over folds; return {(metric, field): (mean, n_folds)}"""
    out, folds = {}, []
    for f in sorted(glob.glob(f"{run}/results_fold*.json")):
        try:
            folds += json.load(open(f))
        except Exception as e:
            print(f"  !! {f}: {e}", file=sys.stderr)
    acc = {}
    for fo in folds:
        for k, v in (fo.get("metrics") or {}).items():
            if not isinstance(v, dict): continue
            for fld, val in v.items():
                if isinstance(val, (int, float)) and np.isfinite(val):
                    acc.setdefault((k, fld), []).append(float(val))
    for key, vals in acc.items():
        out[key] = (float(np.mean(vals)), len(vals))
    return out, len(folds)


data, nfold = {}, {}
for r in RUNS:
    data[r], nfold[r] = load(r)
    if nfold[r] == 0:
        print(f"  !! {r}: no results_fold*.json — did the job finish?", file=sys.stderr)

names = [os.path.basename(r.rstrip("/")) for r in RUNS]
W = max(22, max((len(n) for n in names), default=0) + 2)
print()
print("  " + "METRIC".ljust(24) + "".join(n.rjust(W) for n in names) + "   WINNER")
print("  " + "-" * (24 + W * len(names) + 10))
verdict = {}
for metric, field, label, hib, fmt in ROWS:
    vals, cells = [], []
    for r in RUNS:
        v = data[r].get((metric, field))
        vals.append(v[0] if v else None)
        cells.append(fmt.format(v[0]) if v else "-")
    got = [(i, v) for i, v in enumerate(vals) if v is not None]
    win = ""
    if len(got) > 1 and hib is not None:
        best = (max if hib else min)(got, key=lambda t: t[1])[0]
        spread = max(v for _, v in got) - min(v for _, v in got)
        # only call a winner if the arms actually differ
        win = names[best] if spread > 1e-9 else "tie"
        verdict[label] = win
    elif hib is None and len(got) > 1:
        win = "(1.0 = honest)"
    print("  " + label.ljust(24) + "".join(c.rjust(W) for c in cells) + "   " + win)
print()
for r, n in nfold.items():
    print(f"  {os.path.basename(r.rstrip('/')):22} {n} fold(s)"
          + ("   <-- INCOMPLETE, expected 3" if n != 3 else ""))
if verdict:
    tally = {}
    for w in verdict.values():
        tally[w] = tally.get(w, 0) + 1
    print("\n  tally: " + "  ".join(f"{k} {v}" for k, v in sorted(tally.items(), key=lambda kv: -kv[1])))
print("""
  HOW TO DECIDE — in this order:
   1. net_hpwl AUC is the head that works (0.912). If an arm wins everything but drops AUC,
      it does not win. Nothing here is worth regressing our only good head for.
   2. KNOB-R2 is what f_place EXISTS for — the RL agent picks knobs, so it needs the response,
      not the level. A level win with a knob-R2 loss is not a win.
   3. wns/endpt med_ae are in real ns and ARE comparable across arms here (same folds).
   4. calib z2 should be ~1.0. >1 = overconfident (residuals bigger than sigma claims),
      <1 = underconfident. This has NEVER been reported by a real run; look at it once.
   5. A 3-fold mean over 4 test designs is a SMALL sample. If two arms are within a few percent,
      that is a tie, not a winner. Do not build an architecture on it. We already retracted one
      priority (T7b) that was built on a number nobody re-derived.
""")
