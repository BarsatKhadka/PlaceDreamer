#!/usr/bin/env python3
"""Collect every runs/*/results_fold*.json into ONE pasteable RESULTS.md.

Usage (on the cluster, after the sweeps finish):
    python scripts/collect_results.py
    cat RESULTS.md          # <- paste this

Reports, per run variant, averaged over the CV folds, on UNSEEN designs:
  * KNOB-RESPONSE R2  — the deviation heads scored alone. THE number f_place exists for.
  * RANKING (per-net) — top-10% AUC / recall@10% / 20-bin r. The axis the field reports
                        (Net2 ASP-DAC'21: AUC 0.922). Absolute per-net length is
                        under-determined pre-placement; ranking is the achievable problem.
  * absolute error    — median |err| in real units.
  * pooled R2 is NOT reported: ~99% of the global targets' variance is just design size.
"""
import json, glob, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
runs = sorted(glob.glob(f"{ROOT}/runs/*/results_fold*.json"))
if not runs:
    print("no runs/*/results_fold*.json found — are the jobs finished?"); sys.exit(1)

by_variant = {}
for f in runs:
    v = os.path.basename(os.path.dirname(f))
    by_variant.setdefault(v, []).extend(json.load(open(f)))

def agg(folds, key, sub):
    xs = [f["metrics"][key][sub] for f in folds
          if key in f["metrics"] and sub in f["metrics"][key]
          and np.isfinite(f["metrics"][key][sub])]
    return (np.mean(xs), np.std(xs), len(xs)) if xs else (np.nan, np.nan, 0)

L = []
L.append("# PlaceDreamer — f_place results (UNSEEN designs, leave-designs-out CV)\n")
L.append("`KNOB-RESPONSE R2` = the deviation head scored alone — how well the model predicts the")
L.append("effect of the KNOBS with design size held constant. **This is the number f_place exists for.**")
L.append("(Bar: a 3-parameter OLS on the raw knobs scores +0.68 .. +0.76.)\n")
L.append("`RANKING` = the axis the field actually reports. Net2 (ASP-DAC'21, leave-design-out):")
L.append("AUC 0.922. Absolute per-net length is under-determined pre-placement — Net2, MacroRank")
L.append("and Huang'19 all abandoned absolute regression for ranking. f_route needs 'which nets")
L.append("are long', not 'this net is 47.3um'.\n")
L.append("Pooled R2 is deliberately NOT reported: ~99% of the global targets' variance is merely")
L.append("'how big is this design', so it flatters badly.\n")

L.append("\n## KNOB RESPONSE  (R2, deviation head, unseen designs)\n")
L.append("| variant | folds | tot_hpwl | buf_area | buf_cnt |")
L.append("|---|---|---|---|---|")
for v in sorted(by_variant):
    fs = by_variant[v]
    row = [v, str(len(fs))]
    for k in ("tot_hpwl_dev", "buf_area_dev", "buf_cnt_dev"):
        m, s, n = agg(fs, k, "r2")
        row.append(f"{m:+.3f} ± {s:.3f}" if n else "—")
    L.append("| " + " | ".join(row) + " |")

L.append("\n## RANKING — per-net HPWL  (unseen designs)\n")
L.append("| variant | top-10% AUC | recall@10% | 20-bin r | med rel-err |")
L.append("|---|---|---|---|---|")
for v in sorted(by_variant):
    fs = by_variant[v]
    row = [v]
    for k, sub in (("net_hpwl","auc_top10"), ("net_hpwl","recall_top10"),
                   ("net_hpwl","bin20_r"), ("net_hpwl","med_rel")):
        m, s, n = agg(fs, k, sub)
        if not n: row.append("—")
        elif sub == "med_rel": row.append(f"{m*100:.0f}%")
        else: row.append(f"{m:.3f}")
    L.append("| " + " | ".join(row) + " |")
L.append("\n*(Net2 reference, leave-design-out: AUC 0.922, 20-bin r 0.98 — and they EXCLUDE the")
L.append("top 5% longest nets from that correlation; we do not.)*")

L.append("\n## ABSOLUTE ERROR  (median, real units, unseen designs)\n")
L.append("| variant | net_hpwl | tot_hpwl | buf_cnt | endpt slack | wns |")
L.append("|---|---|---|---|---|---|")
for v in sorted(by_variant):
    fs = by_variant[v]
    row = [v]
    for k, u in (("net_hpwl","um"), ("tot_hpwl","um"), ("buf_cnt","cells"),
                 ("endpt","ns"), ("wns","ns")):
        m, s, n = agg(fs, k, "med_ae")
        row.append(f"{m:.2f} {u}" if n else "—")
    L.append("| " + " | ".join(row) + " |")

L.append("\n## Test designs per fold\n")
seen = set()
for v in sorted(by_variant):
    for f in by_variant[v]:
        t = (f["fold"], tuple(f["test_designs"]))
        if t not in seen:
            seen.add(t)
            L.append(f"- fold {f['fold']}: {', '.join(f['test_designs'])}")
    break

out = "\n".join(L) + "\n"
open(f"{ROOT}/RESULTS.md", "w").write(out)
print(out)
print(f"\n[wrote {ROOT}/RESULTS.md]")
