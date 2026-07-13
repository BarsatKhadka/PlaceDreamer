#!/usr/bin/env python3
"""Add placement-stage TIMING to cache/meta.parquet: WNS + TNS + #violating endpoints
@ place_resized. These are per-flow scalars -> meta only, NO graph rebuild, NO re-cache.

WNS/TNS are placement metrics: they are what the placer+resizer hands you, and they move
with the knobs (aes_core: WNS ranges -4.22 .. -0.13 across its 108 configs). f_place has
to predict them for its picture of the placement state to be complete.
"""
import pandas as pd, pyarrow.dataset as ds, numpy as np
DATA  = "/Users/barsat/PlaceDreamer/datasets/sky130hd"
META  = "/Users/barsat/PlaceDreamer/cache/meta.parquet"
STAGE = "place_resized"          # same stage as buffer_area / buffer_count

m = pd.read_parquet(META)
t = ds.dataset(f"{DATA}/timing_metrics/table.parquet").to_table(
    filter=(ds.field("stage") == STAGE),
    columns=["flow_id", "worst_slack", "total_negative_slack",
             "no_of_violating_endpoints", "no_of_endpoints"]).to_pandas()

t = t.drop_duplicates("flow_id").set_index("flow_id")
m["wns"]      = m.flow_id.map(t.worst_slack)
m["tns"]      = m.flow_id.map(t.total_negative_slack)
m["n_viol"]   = m.flow_id.map(t.no_of_violating_endpoints)
m["n_endpts"] = m.flow_id.map(t.no_of_endpoints)

m.to_parquet(META)
print(f"✓ meta updated @ {STAGE}: wns/tns/n_viol added "
      f"({m.wns.notna().sum()}/{len(m)} flows)\n")
print(m[["wns", "tns", "n_viol"]].describe().round(3).to_string())
print("\nper-design WNS spread (knobs must MOVE it, or there's nothing to learn):")
g = m.assign(d=m.flow_id.str.replace(r"-\d+$", "", regex=True)).groupby("d").wns
print(g.agg(n="size", mean="mean", std="std", lo="min", hi="max").round(3).to_string())
