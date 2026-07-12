#!/usr/bin/env python3
"""Add buffer COUNT (# is_buffer cells @ place_resized) to cache/meta.parquet,
alongside the existing buffer_area. Per-flow scalar → meta only, no graph re-cache."""
import pandas as pd, pyarrow.dataset as ds, numpy as np
DATA = "/Users/barsat/PlaceDreamer/datasets/sky130hd"
META = "/Users/barsat/PlaceDreamer/cache/meta.parquet"

sc = ds.dataset(f"{DATA}/standard_cells/table.parquet").to_table(
    columns=["name", "is_buffer", "is_inverter"]).to_pandas()
buf_types = set(sc.loc[sc.is_buffer.fillna(False), "name"])
inv_types = set(sc.loc[sc.is_inverter.fillna(False), "name"])
print(f"library: {len(buf_types)} buffer types, {len(inv_types)} inverter types")

m = pd.read_parquet(META)
designs = sorted(m.flow_id.str.replace(r"-\d+$", "", regex=True).unique())
counts = {}
for i, d in enumerate(designs):
    flows = m.loc[m.flow_id.str.startswith(d + "-"), "flow_id"].tolist()
    g = ds.dataset(f"{DATA}/gates/table.parquet").to_table(
        filter=(ds.field("stage") == "place_resized") & (ds.field("flow_id").isin(flows)),
        columns=["flow_id", "standard_cell"]).to_pandas()
    g["is_buf"] = g.standard_cell.isin(buf_types)
    g["is_inv"] = g.standard_cell.isin(inv_types)
    for fid, sub in g.groupby("flow_id"):
        counts[fid] = (int(sub.is_buf.sum()), int(sub.is_inv.sum()), len(sub))
    print(f"  [{i+1}/{len(designs)}] {d}", flush=True)

m["buffer_count"]   = m.flow_id.map(lambda f: counts.get(f, (np.nan,)*3)[0])
m["inverter_count"] = m.flow_id.map(lambda f: counts.get(f, (np.nan,)*3)[1])
m["cells_resized"]  = m.flow_id.map(lambda f: counts.get(f, (np.nan,)*3)[2])
m.to_parquet(META)
print(f"\n✓ meta updated: buffer_count added ({m.buffer_count.notna().sum()}/{len(m)} flows)")
print(m[["flow_id", "buffer_area", "buffer_count", "cells_resized"]].head(3).to_string())
