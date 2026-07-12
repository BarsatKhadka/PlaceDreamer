#!/usr/bin/env python3
"""
Cache all flows' f_place graphs (floorplan stage) to disk, extensibly.
Batched per DESIGN (one gates-read per design's 108 flows) → minutes not hours.

Per flow → cache/graphs/<flow_id>.npz : structure (edges, part_id, PE) + identities
(cell_names, net_names, cell_type) + current features (cell_x, net_x, design_features).
Labels + knobs → cache/meta.parquet.

To add features later: read the new table, join by cell_names, append to cell_x — no rebuild.

Usage: python3 scripts/cache_graphs.py [n_designs] [n_configs_per_design]
"""
import os, sys, json, time, numpy as np, pandas as pd
import pyarrow.dataset as ds
sys.path.insert(0, os.path.dirname(__file__))
from build_graph import build, load_cell_lib, DATA

OUT = "/Users/barsat/PlaceDreamer/cache"
GDIR = f"{OUT}/graphs"; os.makedirs(GDIR, exist_ok=True)
IN_STAGE = "floorplan"        # f_place input netlist
LABEL_STAGE = "global_place"  # placement outcome

def read_stage(table, stage, cols, flows):
    return ds.dataset(f"{DATA}/{table}/table.parquet").to_table(
        filter=(ds.field("stage") == stage) & (ds.field("flow_id").isin(flows)),
        columns=["flow_id"] + cols).to_pandas()

def rudy_per_net(nets_df):
    """per-net demand (RUDY) from bbox: (dx+dy)*w / area; wire_width=156.9 dbu (met1-5)."""
    W = 156.9
    dx = (nets_df["x_max"] - nets_df["x_min"]).clip(lower=1)
    dy = (nets_df["y_max"] - nets_df["y_min"]).clip(lower=1)
    return ((dx + dy) * W / (dx * dy)).values

def main():
    cell_tid, cell_feat = load_cell_lib()
    designs = sorted(set(f.rsplit("-",1)[0] for f in
        ds.dataset(f"{DATA}/constraints/table.parquet").to_table(columns=["flow_id"]).to_pydict()["flow_id"]))
    knob_tbl = ds.dataset(f"{DATA}/constraints/table.parquet").to_table(
        columns=["flow_id","clock_period","core_utilization","aspect_ratio"]).to_pandas().set_index("flow_id")
    nd = int(sys.argv[1]) if len(sys.argv) > 1 else len(designs)
    nc = int(sys.argv[2]) if len(sys.argv) > 2 else 108
    meta = []
    for di, design in enumerate(designs[:nd]):
        flows = [f"{design}-{i:06d}" for i in range(1, nc+1)]
        t0 = time.time()
        # batched reads for this design
        gjt = ds.dataset(f"{DATA}/netlists/graph.parquet").to_table(
            filter=(ds.field("stage")==IN_STAGE)&(ds.field("flow_id").isin(flows)),
            columns=["flow_id","graph_json"]).to_pandas().set_index("flow_id")
        gates = ds.dataset(f"{DATA}/gates/table.parquet").to_table(
            filter=(ds.field("stage")==IN_STAGE)&(ds.field("flow_id").isin(flows)),
            columns=["flow_id","name","standard_cell"]).to_pandas()
        gate_map = {fid: dict(zip(g["name"], g["standard_cell"])) for fid, g in gates.groupby("flow_id")}
        nets_lab = read_stage("nets", LABEL_STAGE, ["name","hpwl","x_min","y_min","x_max","y_max"], flows)
        nl_lab = read_stage("netlists", LABEL_STAGE, ["total_hpwl"], flows).set_index("flow_id")
        buf_lab = read_stage("area_metrics", "place_resized", ["buffer_area"], flows).set_index("flow_id")
        ok = 0
        for fid in flows:
            if fid not in gjt.index or fid not in gate_map: continue
            try:
                g = build(fid, IN_STAGE, cell_tid, cell_feat,
                          graph_json=gjt.loc[fid,"graph_json"], gate_cell=gate_map[fid])
                # per-net labels (from global_place), aligned to the floorplan net order by name
                nl = nets_lab[nets_lab["flow_id"]==fid]
                n2h = dict(zip(nl["name"], nl["hpwl"]))
                n2d = dict(zip(nl["name"], rudy_per_net(nl)))
                net_hpwl = np.array([n2h.get(n, np.nan) for n in g["net_names"]], np.float32)
                net_dem  = np.array([n2d.get(n, np.nan) for n in g["net_names"]], np.float32)
                np.savez_compressed(f"{GDIR}/{fid}.npz",
                    edge_driver=g["edge_driver"], edge_sink=g["edge_sink"],
                    pe_cell=g["pe_cell"], part_cell=g["part_cell"], part_net=g["part_net"],
                    cell_x=g["cell_x"], net_x=g["net_x"], cell_type=g["cell_type"],
                    cell_names=g["cell_names"], net_names=g["net_names"],
                    net_hpwl=net_hpwl, net_demand=net_dem,   # per-net f_place labels
                    df_keys=np.array(list(g["design_features"].keys())),
                    df_vals=np.array(list(g["design_features"].values()), np.float32))
                meta.append(dict(flow_id=fid, design=design,
                    clock_period=knob_tbl.loc[fid,"clock_period"],
                    utilization=knob_tbl.loc[fid,"core_utilization"],
                    aspect_ratio=knob_tbl.loc[fid,"aspect_ratio"],
                    total_hpwl=nl_lab.loc[fid,"total_hpwl"] if fid in nl_lab.index else np.nan,
                    buffer_area=buf_lab.loc[fid,"buffer_area"] if fid in buf_lab.index else np.nan,
                    n_cells=g["n_cells"], n_nets=g["n_nets"]))
                ok += 1
            except Exception as e:
                print(f"  [skip] {fid}: {type(e).__name__}: {e}")
        print(f"[{di+1}/{nd}] {design}: cached {ok}/{len(flows)} ({time.time()-t0:.0f}s)", flush=True)
    pd.DataFrame(meta).to_parquet(f"{OUT}/meta.parquet")
    print(f"\ndone: {len(meta)} flows cached → {GDIR} | meta → {OUT}/meta.parquet")

if __name__ == "__main__":
    main()
