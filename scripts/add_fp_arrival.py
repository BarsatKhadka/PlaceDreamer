#!/usr/bin/env python3
"""
Per-cell FLOORPLAN ARRIVAL — the free prior the whole timing model should residual against.

WHY THIS EXISTS (measured, stress_test.py T7):
  arrival_place := arrival_floorplan + one global shift    -> R2 +0.476   (ZERO parameters)
  our trained 680k-param endpt head                        -> R2 -0.508
The floorplan arrival — which is our INPUT STAGE, free — beats our trained head by +0.98 R2.
We were predicting from scratch what the input already tells us.

This is the Delta-ML pattern ("pick the prior for smoothness, not accuracy"; the prior buys a
persistent learning-curve offset ~= a fixed multiple of designs, which is the currency we lack at
n=18 designs) and RTL-Timer's (their analytic STA arrival is R=0.26 alone, R=0.86 through the
model). TimingGCN seeds level 0 with the true arrival for the same reason.

WHAT IT IS:  arrival = the worst (MAX) setup arrival_time over the timing paths ending at each cell,
             at the FLOORPLAN stage. Floorplan precedes placement, so this is LEAK-FREE.
WHY MAX:     timing is a max over converging paths (slack = min over paths <=> arrival = max).
             This mirrors add_endpoint_slack.py's np.minimum.at on slack — the same fix, dual form.

ALIGNMENT (verified before writing this): 100% of floorplan endpoints map onto cells we KEEP for
ac97/ethernet; 80.7% for sasc (the remainder are primary OUTPUTS — no cell — and are masked).

Arrays are FULL-LENGTH (aligned to cell_names) like cache/cts and cache/coords; load_graph applies
`keep`.
"""
import pyarrow.dataset as ds, numpy as np, glob, os

ROOT  = os.environ.get("PD_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA  = f"{ROOT}/datasets/sky130hd"
CACHE = f"{ROOT}/cache/graphs"
OUT   = f"{ROOT}/cache/fp_arrival"
STAGE = "floorplan"          # the PRIOR stage — strictly before placement => leak-free
os.makedirs(OUT, exist_ok=True)

def ep_to_cell(e):
    return e.rsplit("/", 1)[0] if "/" in e else None      # None = a primary output, not a cell

designs = sorted({os.path.basename(p).rsplit("-", 1)[0] for p in glob.glob(f"{CACHE}/*.npz")})
tp_ds = ds.dataset(f"{DATA}/timing_paths/table.parquet")
tot_c = tot_f = 0

for di, dsg in enumerate(designs):
    fids = [os.path.basename(f)[:-4] for f in sorted(glob.glob(f"{CACHE}/{dsg}-*.npz"))]
    # one read per design — timing_paths is huge (30GB); never read it per flow
    tp = tp_ds.to_table(
        filter=(ds.field("stage") == STAGE) & (ds.field("path_type") == "setup")
               & (ds.field("flow_id").isin(fids)),
        columns=["flow_id", "endpoint", "arrival_time"]).to_pandas()
    by_flow = dict(tuple(tp.groupby("flow_id"))) if len(tp) else {}
    for fid in fids:
        names = [str(c) for c in np.load(f"{CACHE}/{fid}.npz", allow_pickle=True)["cell_names"]]
        cidx = {c: i for i, c in enumerate(names)}
        arr = np.zeros(len(names), np.float32); msk = np.zeros(len(names), bool)
        sub = by_flow.get(fid)
        if sub is not None and len(sub):
            cells = sub.endpoint.map(ep_to_cell)
            ok = cells.notna() & cells.isin(cidx)
            if ok.any():
                idx = np.array([cidx[c] for c in cells[ok]], np.int64)
                at = sub.arrival_time[ok].to_numpy(np.float32)
                # WORST arrival per cell = MAX (dual of add_endpoint_slack's minimum.at on slack).
                # A cell owns several endpoint pins (/D, /SET_B); a plain fancy-index write would
                # keep whichever came last — that exact bug corrupted 15% of the slack labels.
                raw = np.full(len(names), -np.inf, np.float32)
                np.maximum.at(raw, idx, at)
                hit = np.unique(idx)
                arr[hit] = raw[hit]; msk[hit] = True
        np.savez_compressed(f"{OUT}/{fid}.npz", fp_arrival=arr, mask=msk)
        tot_c += int(msk.sum()); tot_f += 1
    print(f"[{di+1:2}/{len(designs)}] {dsg:14} {len(fids):4} flows", flush=True)

print(f"\nwrote {tot_f} flows -> {OUT}")
print(f"  cells with a floorplan arrival: {tot_c:,}  (avg {tot_c/max(tot_f,1):.0f}/flow)")
