#!/usr/bin/env python3
"""Add FLOORPLAN-stage WNS/TNS to cache/meta.parquet as f_place CONDITIONING INPUTS.

Rationale: cross-design timing LEVEL doesn't transfer (place_resized WNS R²=-0.77 on
unseen designs) — the model learns knob->timing response (within-r 0.72) but can't peg an
unseen design's baseline. Floorplan timing is the leakage-free anchor for that level: it's
the INPUT stage (before placement), pearson +0.444 with place_resized WNS across designs.

Fed as an input feature (NOT an additive delta — floorplan scale is wildly off: ethernet
floorplan WNS -84 vs place_resized -3). The model learns the nonlinear mapping.

Per-flow scalars -> meta only, NO graph rebuild, NO re-cache.
"""
import pandas as pd, pyarrow.dataset as ds, numpy as np
DATA  = "/Users/barsat/PlaceDreamer/datasets/sky130hd"
META  = "/Users/barsat/PlaceDreamer/cache/meta.parquet"
STAGE = "floorplan"

m = pd.read_parquet(META)
t = ds.dataset(f"{DATA}/timing_metrics/table.parquet").to_table(
    filter=(ds.field("stage") == STAGE),
    columns=["flow_id", "worst_slack", "total_negative_slack"]).to_pandas()
t = t.drop_duplicates("flow_id").set_index("flow_id")
m["fp_wns"] = m.flow_id.map(t.worst_slack)
m["fp_tns"] = m.flow_id.map(t.total_negative_slack)

m.to_parquet(META)
print(f"✓ meta updated: fp_wns/fp_tns @ {STAGE} ({m.fp_wns.notna().sum()}/{len(m)} flows)\n")
print(m[["fp_wns", "fp_tns", "wns", "tns"]].describe().round(2).to_string())
