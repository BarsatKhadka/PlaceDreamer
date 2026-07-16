#!/usr/bin/env python3
"""Per-cell CTS features -> cache/cts/{design}.npz  (computed ONCE per design, no graph rebuild)

f_cts reuses f_place's FLOORPLAN graph (the world model anchors the whole pipeline on it).
It needs three things f_place didn't:
  is_sink   : is this cell a CLOCK SINK? Read from the clock net's fanout (the TRUE sink set —
              562 for aes_core, matches clock_trees exactly), NOT a guessed is_seq flag.
  activity  : switching activity of the cell (pins.switching_activity, aggregated to the cell).
              THE #1 clock feature in the literature: clock power ~ activity x cap. Fully
              populated in EDA-Schema.
  (positions are NOT needed — the METIS partition is a measured spatial proxy: same-partition
   cells are 2-3.4x closer in the real placement. See docs/architecture.md.)
"""
import pyarrow.dataset as ds, numpy as np, glob, os
ROOT=os.environ.get("PD_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); DATA=f"{ROOT}/datasets/sky130hd"; CACHE=f"{ROOT}/cache/graphs"
OUT=f"{ROOT}/cache/cts"; os.makedirs(OUT, exist_ok=True)

# PER FLOW — the netlist is re-synthesized per knob config, so a design's flows have DIFFERENT
# cell sets (ac97: 9994 vs 8197 cells). is_sink/activity must be per-flow, aligned to THAT
# flow's cell order. Batched read per design (one pins scan per design's flows).
designs=sorted({os.path.basename(p).rsplit("-",1)[0] for p in glob.glob(f"{CACHE}/*.npz")})
pins_ds=ds.dataset(f"{DATA}/pins/table.parquet")
tot=0
for di,dsg in enumerate(designs):
    flows=sorted(glob.glob(f"{CACHE}/{dsg}-*.npz"))
    fids=[os.path.basename(f)[:-4] for f in flows]
    # one pins read for all flows of this design
    p=pins_ds.to_table(filter=(ds.field("flow_id").isin(fids))&(ds.field("stage")=="place_resized"),
                       columns=["flow_id","name","switching_activity"]).to_pandas()
    by_flow=dict(tuple(p.groupby("flow_id")))
    for f,fid in zip(flows,fids):
        z=np.load(f,allow_pickle=True)
        names=[str(c) for c in z["cell_names"]]; nidx={n:i for i,n in enumerate(names)}; C=len(names)
        clk_nets=set(np.where(z["net_x"][:,2]>0.5)[0].tolist())
        is_sink=np.zeros(C,np.float32)
        for c,n in zip(z["edge_sink"][0], z["edge_sink"][1]):
            if int(n) in clk_nets: is_sink[int(c)]=1.0
        act=np.zeros(C,np.float64)
        sub=by_flow.get(fid)
        if sub is not None:
            for nm,a in zip(sub.name, sub.switching_activity):
                cell=nm.rsplit("/",1)[0] if "/" in nm else nm
                if cell in nidx and np.isfinite(a): act[nidx[cell]]+=a
        act=np.log1p(np.maximum(act,0)).astype(np.float32)
        np.savez(f"{OUT}/{fid}.npz", is_sink=is_sink, activity=act)
        tot+=1
    print(f"  [{di+1}/{len(designs)}] {dsg}: {len(flows)} flows", flush=True)
print(f"\n✓ wrote {tot} FLOWS to {OUT}")
