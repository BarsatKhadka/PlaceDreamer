#!/usr/bin/env python3
"""
Comprehensive baseline: predict EVERY scalar PPA target with gradient-boosted trees,
from {floorplan size + knobs}, LEAVE-ONE-DESIGN-OUT. For each target report:
  - variance split: between-design (size) vs within-design (knobs)
  - ruler (linear on size only)  vs  full (GBT on size+knobs)  →  does adding knobs help?
  - within-design r (predicted vs actual across a design's 108 configs) = the knob-effect signal
Verdict per target: RULER (size nails it, GNN won't help) vs SIGNAL (knobs/structure matter).
"""
import pyarrow.dataset as ds
import numpy as np, pandas as pd, warnings
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneGroupOut
from scipy.stats import pearsonr
warnings.filterwarnings("ignore")

DATA = "/Users/barsat/PlaceDreamer/datasets/sky130hd"
def tbl(name, stage, cols):
    return ds.dataset(f"{DATA}/{name}/table.parquet").to_table(
        filter=ds.field("stage") == stage, columns=["flow_id"] + cols).to_pandas()

# --- features: floorplan size + knobs ---
knobs = ds.dataset(f"{DATA}/constraints/table.parquet").to_table(
    columns=["flow_id","clock_period","core_utilization","aspect_ratio"]).to_pandas()
sz   = tbl("netlists","floorplan",["no_of_cells","no_of_nets","no_of_pins"])
area = tbl("area_metrics","floorplan",["cell_area"])
base = knobs.merge(sz,on="flow_id").merge(area,on="flow_id")
base["design"] = base["flow_id"].str.replace(r"-\d+$","",regex=True)
SIZE = ["no_of_cells","no_of_nets","no_of_pins","cell_area"]
FULL = SIZE + ["clock_period","core_utilization","aspect_ratio"]

TARGETS = [  # name, table, stage, column, log?
    ("hpwl(place)",   "netlists",       "global_place",   "total_hpwl",           True),
    ("routed_WL",     "netlists",       "detailed_route", "total_wirelength",     True),
    ("WNS",           "timing_metrics", "final",          "worst_slack",          False),
    ("TNS",           "timing_metrics", "final",          "total_negative_slack", False),
    ("power",         "power_metrics",  "final",          "total_power",          True),
    ("buffer_area",   "area_metrics",   "cts",            "buffer_area",          True),
]

print(f"{'target':13} {'between%':>8} {'within%':>8} | {'rulerR2':>8} {'fullR2':>8} {'Δ':>6} | {'within-r':>8}  verdict")
print("-"*88)
for name, tab, stg, col, islog in TARGETS:
    d = base.merge(tbl(tab,stg,[col]), on="flow_id").dropna(subset=[col]+FULL)
    if islog: d = d[d[col] > 0]; d["y"] = np.log(d[col])
    else:     d["y"] = d[col].astype(float)
    if len(d) < 100: print(f"{name:13}  (too few rows: {len(d)})"); continue
    y = d["y"].values; g = d["design"].values
    # variance split
    grand=y.mean(); bt=(pd.Series(y).groupby(g).transform("mean")-grand); wt=y-pd.Series(y).groupby(g).transform("mean")
    btw=(bt**2).mean(); wth=(wt**2).mean(); tot=btw+wth
    # leave-one-design-out
    def cv(cols, mk):
        p=np.zeros_like(y); X=d[cols].values
        for tr,te in LeaveOneGroupOut().split(X,y,g):
            m=mk(); m.fit(X[tr],y[tr]); p[te]=m.predict(X[te])
        return p
    GBT = lambda: HistGradientBoostingRegressor(max_iter=400,max_depth=4,learning_rate=0.05)
    pr = cv(SIZE, GBT)   # fair size-only baseline (same model class)
    pf = cv(FULL, GBT)   # + knobs → Δ is the real knob contribution
    r2=lambda p: 1-((y-p)**2).sum()/((y-y.mean())**2).sum()
    wc=[pearsonr(pf[g==dd],y[g==dd])[0] for dd in np.unique(g) if y[g==dd].std()>1e-9 and pf[g==dd].std()>1e-9]
    wr=np.nanmean(wc)
    verdict = "RULER (size)" if (btw/tot>0.9 and r2(pf)-r2(pr)<0.05) else \
              ("weak signal" if wr<0.5 else "SIGNAL (knobs)")
    print(f"{name:13} {btw/tot*100:7.1f}% {wth/tot*100:7.1f}% | {r2(pr):8.3f} {r2(pf):8.3f} {r2(pf)-r2(pr):+6.3f} | {wr:8.3f}  {verdict}")
