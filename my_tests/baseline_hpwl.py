#!/usr/bin/env python3
"""
Baseline signal test: does {size + knobs} predict placement total_hpwl, CROSS-DESIGN?
Honest protocol — not one cherry-picked R²:
  1. variance decomposition: how much hpwl variance is between-design (size, trivial)
     vs within-design (knobs, what we care about)?
  2. ruler floor (linear on log cell_area) vs full model (grad-boosted trees on size+knobs),
     LEAVE-ONE-DESIGN-OUT (no design leakage).
  3. within-design correlation (predicted vs actual across a held-out design's configs)
     = the real knob-effect signal, with size held constant.

Input features = FLOORPLAN netlist (pre-placement). Target = global_place total_hpwl.
"""
import pyarrow.dataset as ds
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneGroupOut
from scipy.stats import pearsonr

DATA = "/Users/barsat/PlaceDreamer/datasets/sky130hd"

def tbl(name, stage, cols):
    return ds.dataset(f"{DATA}/{name}/table.parquet").to_table(
        filter=ds.field("stage") == stage, columns=["flow_id"] + cols).to_pandas()

# --- assemble the dataframe ---
knobs = ds.dataset(f"{DATA}/constraints/table.parquet").to_table(
    columns=["flow_id", "clock_period", "core_utilization", "aspect_ratio"]).to_pandas()
sz   = tbl("netlists", "floorplan", ["no_of_cells", "no_of_nets", "no_of_pins"])
area = tbl("area_metrics", "floorplan", ["cell_area"])
tgt  = tbl("netlists", "global_place", ["total_hpwl"])

df = knobs.merge(sz, on="flow_id").merge(area, on="flow_id").merge(tgt, on="flow_id")
df["design"] = df["flow_id"].str.replace(r"-\d+$", "", regex=True)
df = df[(df["total_hpwl"] > 0) & (df["cell_area"] > 0)].dropna().reset_index(drop=True)
df["y"] = np.log(df["total_hpwl"])
print(f"{len(df)} flows, {df['design'].nunique()} designs, "
      f"{df.groupby('design').size().min()}-{df.groupby('design').size().max()} configs/design\n")

# --- 1. variance decomposition ---
grand = df["y"].mean()
between = df.groupby("design")["y"].mean().sub(grand).pow(2).mul(df.groupby("design").size()).sum() / len(df)
within  = df.groupby("design")["y"].transform(lambda s: s - s.mean()).pow(2).mean()
tot = between + within
print("=== 1. variance decomposition of log(hpwl) ===")
print(f"  between-design (SIZE, trivial):  {between/tot*100:5.1f}%")
print(f"  within-design  (KNOBS, real):    {within/tot*100:5.1f}%")
print(f"  → knobs move hpwl by ~{np.sqrt(within)*100:.0f}% (1 std, size-controlled)\n")

FEATS = ["no_of_cells", "no_of_nets", "no_of_pins", "cell_area",
         "clock_period", "core_utilization", "aspect_ratio"]
X = df[FEATS].values; y = df["y"].values; groups = df["design"].values
Xruler = np.log(df[["cell_area"]].values)   # size-only

def cv(model, X):
    logo = LeaveOneGroupOut(); pred = np.zeros_like(y)
    for tr, te in logo.split(X, y, groups):
        m = model(); m.fit(X[tr], y[tr]); pred[te] = m.predict(X[te])
    return pred

pred_ruler = cv(lambda: LinearRegression(), Xruler)
pred_full  = cv(lambda: HistGradientBoostingRegressor(max_iter=400, max_depth=4,
                                                      learning_rate=0.05), X)

def report(name, pred):
    ss_res = ((y - pred)**2).sum(); ss_tot = ((y - y.mean())**2).sum()
    r2 = 1 - ss_res/ss_tot
    relerr = np.median(np.abs(np.expm1(pred - y)))  # median relative error on hpwl
    # within-design corr (size held constant) — the knob-effect signal
    wc = []
    for d in df["design"].unique():
        m = groups == d
        if m.sum() > 2 and y[m].std() > 0 and pred[m].std() > 0:
            wc.append(pearsonr(pred[m], y[m])[0])
    print(f"  {name:26} R²(log)={r2:6.3f}  medRelErr={relerr*100:5.1f}%  "
          f"within-design r={np.mean(wc):.3f}")

print("=== 2/3. leave-one-DESIGN-out (the honest cross-design test) ===")
report("ruler (size only)", pred_ruler)
report("full (size + knobs, trees)", pred_full)
print(f"\n  SIGNAL = does full beat ruler cross-design, and is within-design r high?")
