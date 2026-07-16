#!/usr/bin/env python3
"""Per-net ROUTED-LENGTH labels -> cache/route/{flow}.npz  (per flow, no graph rebuild)

f_route predicts the routing state from the placement state. Its per-net target is the ACTUAL
ROUTED length (nets.length @ detailed_route) — the genuine routing quantity, vs the HPWL estimate
f_place predicts. The ratio routed/hpwl is the DETOUR (how far the router snaked around
congestion), which is the real routability signal EDA-Schema's empty routability_metrics can't give.

Aligned to f_place's FLOORPLAN net order by name (nets that survive to detailed_route). Nets that
split during buffering/CTS won't match — masked, same handling as f_place's net_hpwl.
"""
import pyarrow.dataset as ds, numpy as np, glob, os
ROOT=os.environ.get("PD_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); DATA=f"{ROOT}/datasets/sky130hd"; CACHE=f"{ROOT}/cache/graphs"
OUT=f"{ROOT}/cache/route"; os.makedirs(OUT, exist_ok=True)
nets_ds=ds.dataset(f"{DATA}/nets/table.parquet")

designs=sorted({os.path.basename(p).rsplit("-",1)[0] for p in glob.glob(f"{CACHE}/*.npz")})
tot=cov=0
for di,dsg in enumerate(designs):
    flows=sorted(glob.glob(f"{CACHE}/{dsg}-*.npz")); fids=[os.path.basename(f)[:-4] for f in flows]
    t=nets_ds.to_table(filter=(ds.field("flow_id").isin(fids))&(ds.field("stage")=="detailed_route"),
                       columns=["flow_id","name","length"]).to_pandas()
    by_flow={k:dict(zip(v.name,v.length)) for k,v in t.groupby("flow_id")}
    for f,fid in zip(flows,fids):
        z=np.load(f,allow_pickle=True); names=[str(x) for x in z["net_names"]]; N=len(names)
        lm=by_flow.get(fid,{})
        rl=np.full(N,np.nan,np.float32)
        for i,n in enumerate(names):
            v=lm.get(n)
            if v is not None and np.isfinite(v) and v>0: rl[i]=v
        np.savez(f"{OUT}/{fid}.npz", routed_len=rl)
        tot+=N; cov+=np.isfinite(rl).sum()
    print(f"  [{di+1}/{len(designs)}] {dsg}: {len(flows)} flows", flush=True)
print(f"\n✓ {tot} nets, {cov} ({cov/tot*100:.0f}%) matched to a routed length")
