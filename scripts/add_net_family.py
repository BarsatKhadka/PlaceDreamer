#!/usr/bin/env python3
"""
NET-FAMILY ROLLUP — the fix for "the graph grows", without predicting a single buffer.

THE PROBLEM (docs/model_architecture.md section 5, section 10 "UNDECIDED"):
Our graph is frozen at FLOORPLAN. The resizer then inserts buffers and SPLITS nets, so by
place_resized there are nets we have no node for. Summing HPWL over OUR nets misses them:
MEASURED Sigma(ours)/true = 0.68 .. 0.82. That gap is why the composed-sum identity was only ever
"absorbable", never exact.

THE IDEA (not ours — Kahng's MLBuf/ISPD26 line avoids mutating the graph; we take the dual):
a resizer split does not CREATE wire, it REDISTRIBUTES one original net's span across a family.
    floorplan:       net N  ---------------------------------> sinks
    place_resized:   net N  --> [buf] --> net191 -------------> sinks
So roll the label up: the target for floorplan net N is Sigma HPWL over N's DESCENDANTS. The
label changes; the NODE SET does not. No graph growth to model, no buffer to predict.

MEASURED — and the stage decides everything:
                                    Sigma(families) / true total
    global_place   (LABEL_STAGE)          1.0000        <- AN EXACT IDENTITY
    place_resized                         0.9521        <- NOT an identity
WHY: gate cloning happens in the RESIZER, not in global_place. At global_place the only insertion
is BUFFERING, every buffer is single-input, so every new net traces back cleanly and no HPWL is
unattributed. At place_resized, repair_design also CLONES gates (a clone's output net is driven by
a COPY of multi-input logic, so our single-input walk stops): ac97 191 new nets, 48 untraceable
(37 with a 2-input driver, 10 with 3-input, 1 with 0) = 4.24% of HPWL. Per-design 0.33% (jpeg) to
15.67% (wb_dma).
  => at LABEL_STAGE this is a TRUE IDENTITY and HPWL_COMPOSE=sum becomes exact rather than
     approximate. If we ever move labels to place_resized, handle cloning first (match a clone to
     its original by identical input-net set). NOT DONE — we do not need it at global_place.
I built this at place_resized first and nearly wrote off a real identity as "not an identity."
Check the stage before believing the number.

*** THIS REPLACES cache/netmask, IT DOES NOT COMPLEMENT IT. ***
The netmask (scripts/add_net_label_mask.py, audit B2) exists for THIS EXACT BUG: a split net keeps
its name, so name-matching silently attached a FRAGMENT's HPWL to the full net. Its fix is to
DISCARD those nets: 0.39% of nets — but they carry 4-6% of total HPWL at a median 20x the rest,
i.e. the highest-leverage nets in our best head (AUC 0.912) were being thrown away.
MEASURED: the rollup EXPLAINS 323/323 = 100% of them. It fixes the cause; the mask deleted the
symptom. With NET_TARGET=family the mask should be OFF — masking them again would discard the
nets this was built to recover.

WHY IT SHIPS VIA GIT (same trap as fp_arrival, caught the same way):
this reads datasets/sky130hd/netlists/graph.parquet, which is not on the cluster and cannot be
rsynced (double jump host). The cache is small. Ship it. Do NOT put a "just regenerate it" note
in an sbatch guard -- that is exactly the advice that was wrong for fp_arrival.

OUTPUT, aligned to cache/graphs' net_names (like cache/cts, cache/coords, cache/fp_arrival):
    fam_hpwl  float32  rolled-up HPWL for this floorplan net (its own + its descendants')
    fam_mask  bool     False where the net has no STAGE HPWL at all
    fam_n     int16    family size (1 = never split). A FEATURE in its own right: it is the
                       per-net buffer-insertion count, which is what MLBuf/BufFormer predict --
                       except they need placement first and we do not.
"""
import pyarrow.dataset as ds, numpy as np, glob, os, json, sys

ROOT  = os.environ.get("PD_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA  = f"{ROOT}/datasets/sky130hd"
CACHE = f"{ROOT}/cache/graphs"
OUT   = f"{ROOT}/cache/net_family"
STAGE = os.environ.get("FAMILY_STAGE", "global_place")   # == cache_graphs.LABEL_STAGE
os.makedirs(OUT, exist_ok=True)

G = ds.dataset(f"{DATA}/netlists/graph.parquet")
N = ds.dataset(f"{DATA}/nets/table.parquet")


def _graphs(fid):
    """floorplan + STAGE graph_json for one flow, in ONE read."""
    t = G.to_table(filter=(ds.field("flow_id") == fid) & (ds.field("stage").isin(["floorplan", STAGE])),
                   columns=["stage", "graph_json"]).to_pandas()
    d = {r.stage: json.loads(r.graph_json) for r in t.itertuples()}
    return d.get("floorplan"), d.get(STAGE)


def families(fid):
    """map each FLOORPLAN net -> [its descendants at STAGE].  None if the flow is unusable."""
    fp, gj = _graphs(fid)
    if fp is None or gj is None: return None, None
    ty = dict(zip(gj["nodes"], gj["node_types"]))
    fpnets = {n for n, t in zip(fp["nodes"], fp["node_types"]) if t == "NET"}
    # graph is GATE -PIN- NET -PIN- GATE, edges directed. A pin is named "<gate>/<pin>".
    drv, ins = {}, {}
    for s, d in gj["edges"]:
        ts, td = ty.get(s), ty.get(d)
        if ts == "PIN" and td == "NET":      # gate's OUTPUT pin drives this net
            drv.setdefault(s.rsplit("/", 1)[0], set()).add(d)
        elif ts == "NET" and td == "PIN":    # net feeds a gate's INPUT pin
            ins.setdefault(d.rsplit("/", 1)[0], set()).add(s)
    gate_driving = {n: g for g, ns in drv.items() for n in ns}

    def ancestor(n):
        """walk a NEW net back through single-input (buffer) chains to its floorplan net."""
        seen = set()
        while n not in fpnets and n not in seen:
            seen.add(n)
            g = gate_driving.get(n)
            if g is None: return None                  # port-driven / dangling
            i = ins.get(g, set())
            if len(i) != 1: return None                # 2+ inputs => GATE CLONE, not a buffer
            n = next(iter(i))
        return n if n in fpnets else None

    fam = {}
    for n, t in zip(gj["nodes"], gj["node_types"]):
        if t != "NET": continue
        a = n if n in fpnets else ancestor(n)
        if a is not None: fam.setdefault(a, []).append(n)
    return fam, fpnets


designs = sorted({os.path.basename(p).rsplit("-", 1)[0] for p in glob.glob(f"{CACHE}/*.npz")})
tot_f = tot_split = 0
lost_acc = []

for di, dsg in enumerate(designs):
    fids = [os.path.basename(f)[:-4] for f in sorted(glob.glob(f"{CACHE}/{dsg}-*.npz"))]
    # one read of `nets` per design — it is large; never read per flow
    nt = N.to_table(filter=(ds.field("stage") == STAGE) & (ds.field("flow_id").isin(fids)),
                    columns=["flow_id", "name", "hpwl"]).to_pandas()
    by_flow = dict(tuple(nt.groupby("flow_id"))) if len(nt) else {}
    for fid in fids:
        names = [str(c) for c in np.load(f"{CACHE}/{fid}.npz", allow_pickle=True)["net_names"]]
        fam, _ = families(fid)
        sub = by_flow.get(fid)
        fh = np.zeros(len(names), np.float32)
        fm = np.zeros(len(names), bool)
        fn = np.zeros(len(names), np.int16)
        if fam is not None and sub is not None and len(sub):
            h = dict(zip(sub.name.astype(str), sub.hpwl))
            for i, n in enumerate(names):
                kids = fam.get(n)
                if not kids: continue
                v = [h[c] for c in kids if c in h and np.isfinite(h[c])]
                if not v: continue
                fh[i] = float(np.sum(v)); fm[i] = True; fn[i] = len(v)
                if len(v) > 1: tot_split += 1
            tv = float(np.nansum(list(h.values())))
            if tv > 0: lost_acc.append(1.0 - float(fh[fm].sum()) / tv)
        np.savez_compressed(f"{OUT}/{fid}.npz", fam_hpwl=fh, fam_mask=fm, fam_n=fn)
        tot_f += 1
    print(f"[{di+1:2}/{len(designs)}] {dsg:14} {len(fids):4} flows", flush=True)

print(f"\nwrote {tot_f} flows -> {OUT}")
print(f"  split families (fam_n > 1): {tot_split:,}  (= per-net buffer insertions, a free label)")
if lost_acc:
    med, mx = 100*float(np.median(lost_acc)), 100*max(lost_acc)
    print(f"  UNATTRIBUTED HPWL: median {med:.2f}%  max {mx:.2f}%")
    if mx < 1e-6:
        print(f"  ^ EXACT: Sigma(families) == true total over our FIXED floorplan nets, all flows.")
        print(f"    tot_hpwl = Sigma_i fam_hpwl(i) is now a TRUE IDENTITY (HPWL_COMPOSE=sum).")
    else:
        print(f"  ^ NOT an identity — the gate-cloning residual (expected at place_resized, ~5%).")
