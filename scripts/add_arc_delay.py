#!/usr/bin/env python3
"""
PER-NET and PER-GATE DELAY labels — step 1 of the timing rebuild, and the cheapest way to kill it.

WHY THIS EXISTS: our timing head is broken cross-design, MEASURED on the fold-0 TEST block:
    endpt  r2 -0.382        wns  pooled r2 +0.332  but  within_r2 -0.720
`within_r2 -0.720` is the one that matters: WITHIN a design, as the knobs move, the model is worse
than that design's own mean. It has ZERO knob signal on timing. The pooled +0.332 is design
identity (big design -> bad slack), which is useless to an RL agent that works inside one design.
And placement WNS is worth having: within-design corr(place_wns, FINAL wns) = +0.796 (~63% of the
knob variance; cts is +0.974). So the signal is there and we capture none of it.

*** THREE THINGS docs/model_architecture.md SECTION 6 CLAIMS THAT ARE FALSE. Measured here. ***
 1. "33,617 arcs/flow, 48x more supervision". NO. net_arcs/cell_arcs are PATH-INDEXED, not
    arc-indexed: `startpoint`/`endpoint` are the PATH's flop-to-flop endpoints, NOT the arc's two
    pins. ac97-000001 @global_place: 17,003 setup rows = 4,237 paths through 1,122 NETS.
    Real label count is ~1,122 nets + ~2,170 gates ~= 3,300/flow -> 4.7x our current ~700, not 48x.
    (The giveaway: topo-sorting those startpoint/endpoint pairs gives max level 1 over 1,633 pins.
     A real netlist has ~300 levels. It is bipartite because they are path endpoints.)
 2. "wire delay is a function of net length -- the chain closes on our strength (AUC 0.912)". NO.
    corr(log hpwl, log delay) = +0.583 / +0.362 / +0.255 / +0.404 -> median ~+0.38, i.e. length
    explains ~14%. Linear R2 from [hpwl, fanout] (all we know or predict): +0.343 / +0.184 /
    +0.080 / +0.214 -> median ~0.20. Even ADDING post-placement cap+slew LABELS only reaches ~0.38.
    A GNN sees driver type and local topology a linear probe cannot, so 0.20 is a FLOOR, not a
    ceiling -- but nobody should promise +0.90 from this. I nearly did.
 3. The prior (T7b, "+0.98 R2 free") -- RETRACTED, it fit its shift on the test design's truth.

WHAT SURVIVES, and it is why this is still worth one day:
  * net delay is WELL-DEFINED PER NET. within-net std ACROSS paths = 0.0000 / 0.0019 / 0.0000 ns
    vs across-net std 0.242 / 0.280 / 0.203. One net, one delay, whichever path traverses it.
    => it is a PER-NET target: the exact shape of our best head (net_hpwl, AUC 0.912).
  * cell delay is WELL-DEFINED PER GATE. within-gate std 0.0000 / 0.0003 vs across-gate 0.0125 /
    0.0301. => a per-gate target, 2,170-4,164 labels/flow.
  * net delay is 414x cell delay at global_place. Pre-route, WIRE delay IS the timing.

THE POINT OF SHIPPING THIS ALONE FIRST: it fails fast. Train ONE head on net delay. If it cannot
beat the ~0.20 linear floor, the whole levelized chain is dead and we spent a day, not three weeks.
Only if it lands do the `max` channel and the topological schedule become worth building.

STAGE: global_place == cache_graphs.LABEL_STAGE. Do not change without re-checking: `length` and
`resistance` in the `nets` table are 100% NaN pre-route (they are ROUTED values) -- a query joining
them silently drops every row, which cost me a debugging cycle here.

OUTPUT, aligned to cache/graphs (like cache/fp_arrival, cache/net_family):
    net_delay   float32  per-net delay (ns), aligned to net_names
    net_mask    bool     True where this net is on a setup timing path (~24% of nets)
    cell_delay  float32  per-gate delay (ns), aligned to cell_names
    cell_mask   bool     True where this gate is on a setup timing path
"""
import pyarrow.dataset as ds, numpy as np, glob, os

ROOT  = os.environ.get("PD_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA  = f"{ROOT}/datasets/sky130hd"
CACHE = f"{ROOT}/cache/graphs"
OUT   = f"{ROOT}/cache/arc_delay"
STAGE = "global_place"          # == cache_graphs.LABEL_STAGE
os.makedirs(OUT, exist_ok=True)

NA = ds.dataset(f"{DATA}/net_arcs/table.parquet")
CA = ds.dataset(f"{DATA}/cell_arcs/table.parquet")

designs = sorted({os.path.basename(p).rsplit("-", 1)[0] for p in glob.glob(f"{CACHE}/*.npz")})
tot_f = 0
cov_n, cov_c = [], []

for di, dsg in enumerate(designs):
    fids = [os.path.basename(f)[:-4] for f in sorted(glob.glob(f"{CACHE}/{dsg}-*.npz"))]
    # ONE read per design — these tables are large; never read per flow.
    n = NA.to_table(filter=(ds.field("stage") == STAGE) & (ds.field("path_type") == "setup")
                    & (ds.field("flow_id").isin(fids)),
                    columns=["flow_id", "net_name", "delay"]).to_pandas()
    c = CA.to_table(filter=(ds.field("stage") == STAGE) & (ds.field("path_type") == "setup")
                    & (ds.field("flow_id").isin(fids)),
                    columns=["flow_id", "gate_name", "delay"]).to_pandas()
    nb = dict(tuple(n.groupby("flow_id"))) if len(n) else {}
    cb = dict(tuple(c.groupby("flow_id"))) if len(c) else {}
    for fid in fids:
        z = np.load(f"{CACHE}/{fid}.npz", allow_pickle=True)
        nn = [str(x) for x in z["net_names"]]
        cn = [str(x) for x in z["cell_names"]]
        nd = np.zeros(len(nn), np.float32); nm = np.zeros(len(nn), bool)
        cd = np.zeros(len(cn), np.float32); cm = np.zeros(len(cn), bool)
        sn = nb.get(fid)
        if sn is not None and len(sn):
            # median over paths — VERIFIED identical across paths (within-net std 0.0000 ns), so
            # this is a de-duplication, not an aggregation that loses information.
            d = sn.groupby("net_name").delay.median()
            idx = {k: i for i, k in enumerate(nn)}
            for k, v in d.items():
                i = idx.get(str(k))
                if i is not None and np.isfinite(v):
                    nd[i] = float(v); nm[i] = True
        sc = cb.get(fid)
        if sc is not None and len(sc):
            d = sc.groupby("gate_name").delay.median()
            idx = {k: i for i, k in enumerate(cn)}
            for k, v in d.items():
                i = idx.get(str(k))
                if i is not None and np.isfinite(v):
                    cd[i] = float(v); cm[i] = True
        np.savez_compressed(f"{OUT}/{fid}.npz", net_delay=nd, net_mask=nm,
                            cell_delay=cd, cell_mask=cm)
        cov_n.append(nm.mean()); cov_c.append(cm.mean()); tot_f += 1
    print(f"[{di+1:2}/{len(designs)}] {dsg:14} {len(fids):4} flows", flush=True)

print(f"\nwrote {tot_f} flows -> {OUT}")
print(f"  nets  with a delay label: median {100*float(np.median(cov_n)):.1f}% of our nets")
print(f"  gates with a delay label: median {100*float(np.median(cov_c)):.1f}% of our cells")
print(f"  ^ sparse by nature: only nets/gates ON a setup timing path get one. That is the")
print(f"    supervision available — ~4.7x our current ~700 endpoints, NOT the 48x section 6 claims.")
