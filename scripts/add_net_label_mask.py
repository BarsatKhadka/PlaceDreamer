#!/usr/bin/env python3
"""Build cache/netmask/{flow}.npz — which per-net HPWL labels are TRUSTWORTHY.

THE BUG (docs/fplace_audit.md B2): the graph is built @floorplan but net_hpwl labels come from
@global_place. In between, OpenROAD inserts buffers that SPLIT high-fanout nets. The net NAME
survives, so name-matching finds a "hit" and nothing is masked — but the label then describes
only a FRAGMENT of the net the model is looking at.

  ethernet-000003, net _00005_:  model sees fanout 2052 (floorplan)
                                 label is for the net with 36 sinks left (global_place)

Only ~0.4% of nets, BUT they carry 4-6% of total HPWL with a median HPWL 20x the rest — i.e.
the highest-leverage nets in the per-net head were trained against wrong answers, silently.

Detection: a net is UNTRUSTWORTHY if its fanout changed floorplan -> global_place.
(place_resized is worse, not better: MORE buffer splits, less HPWL coverage. Verified.)

Per-flow scalars/arrays only — NO graph rebuild.
"""
import pyarrow.dataset as ds, numpy as np, glob, os
ROOT  = os.environ.get("PD_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA  = f"{ROOT}/datasets/sky130hd"
CACHE = f"{ROOT}/cache/graphs"
OUT   = f"{ROOT}/cache/netmask"
os.makedirs(OUT, exist_ok=True)

nets = ds.dataset(f"{DATA}/nets/table.parquet")
designs = sorted({os.path.basename(p).rsplit("-", 1)[0] for p in glob.glob(f"{CACHE}/*.npz")})
tot_nets = tot_bad = 0
for di, dsg in enumerate(designs):
    fids = [os.path.basename(p)[:-4] for p in sorted(glob.glob(f"{CACHE}/{dsg}-*.npz"))]
    for stage, key in (("floorplan", "fp"), ("global_place", "gp")):
        pass
    fp_t = nets.to_table(filter=(ds.field("stage") == "floorplan") & (ds.field("flow_id").isin(fids)),
                         columns=["flow_id", "name", "no_of_fanouts"]).to_pandas()
    gp_t = nets.to_table(filter=(ds.field("stage") == "global_place") & (ds.field("flow_id").isin(fids)),
                         columns=["flow_id", "name", "no_of_fanouts"]).to_pandas()
    fp_by = {k: v.set_index("name").no_of_fanouts for k, v in fp_t.groupby("flow_id")}
    gp_by = {k: v.set_index("name").no_of_fanouts for k, v in gp_t.groupby("flow_id")}
    for fid in fids:
        names = [str(x) for x in np.load(f"{CACHE}/{fid}.npz", allow_pickle=True)["net_names"]]
        fp, gp = fp_by.get(fid), gp_by.get(fid)
        ok = np.ones(len(names), bool)
        if fp is not None and gp is not None:
            fpd, gpd = fp.to_dict(), gp.to_dict()
            for i, n in enumerate(names):
                a, b = fpd.get(n), gpd.get(n)
                if a is None or b is None or a != b:      # missing, or SPLIT by buffering
                    ok[i] = False
        np.savez(f"{OUT}/{fid}.npz", label_ok=ok)
        tot_nets += len(ok); tot_bad += int((~ok).sum())
    print(f"  [{di+1}/{len(designs)}] {dsg}: {len(fids)} flows", flush=True)

print(f"\n✓ {OUT}: {tot_nets:,} nets, {tot_bad:,} masked ({tot_bad/tot_nets*100:.2f}%) "
      f"— fragment labels from buffer-split nets")
