#!/usr/bin/env python3
"""
Per-cell PLACEMENT GEOMETRY labels — where every cell actually landed at place_resized.

WHY THIS EXISTS. The seam (f_place -> f_cts) was forwarding only SUMMARY METRICS of placement
(total HPWL, buffer area/count, endpoint slack). Measured result: feeding f_cts the imagined
placement state vs the real one changed nothing, because none of those carry the thing CTS
actually consumes -- WHERE THE SINKS ARE. Clock-tree buffering is a function of sink geometry:
their spatial spread sets clock wirelength, which sets buffer count and clock power. A scalar
total-HPWL cannot express that.

So f_place must learn geometry, and this script builds the labels for it.

TARGET. Per cell, the NORMALIZED centroid inside the placement region:
    x = (cx - x0) / W,  y = (cy - y0) / H      in [0,1]
normalized because the die itself MOVES WITH THE KNOBS -- aspect_ratio drives sasc from a
214.8x103.4um die (AR 2.08) to 101.2x149.6um (AR 0.68), and core_utilization scales it. So the
absolute micron is a knob artifact; the normalized position is the placement DECISION. The die
W/H are saved alongside (they are knowable pre-placement from the floorplan knobs, so they are
an input, not something to predict).

COVERAGE. Verified 100% on the cells f_place keeps: the only NaN coordinates in `gates` are
TAP_TAPCELL_* rows, which live_cells() already drops (they tiled the core and leaked design
identity). Arrays here are FULL-LENGTH (aligned to cell_names); load_graph applies `keep`,
matching how cache/cts is consumed.
"""
import pyarrow.dataset as ds, numpy as np, pandas as pd, glob, os

DATA  = "/Users/barsat/PlaceDreamer/datasets/sky130hd"
CACHE = "/Users/barsat/PlaceDreamer/cache/graphs"
OUT   = "/Users/barsat/PlaceDreamer/cache/coords"
STAGE = "place_resized"          # f_place's target stage — same one add_endpoint_slack uses
os.makedirs(OUT, exist_ok=True)

designs = sorted({os.path.basename(p).rsplit("-", 1)[0] for p in glob.glob(f"{CACHE}/*.npz")})
g_ds = ds.dataset(f"{DATA}/gates/table.parquet")
tot_cells = tot_missing = tot_flows = 0

for di, d in enumerate(designs):
    fids = [os.path.basename(f)[:-4] for f in sorted(glob.glob(f"{CACHE}/{d}-*.npz"))]
    # one read per design (gates is ~20M rows at this stage — never read it per flow)
    t = g_ds.to_table(filter=(ds.field("stage") == STAGE) & (ds.field("flow_id").isin(fids)),
                      columns=["flow_id", "name", "x_min", "y_min", "x_max", "y_max"]).to_pandas()
    by_flow = dict(tuple(t.groupby("flow_id")))
    for fid in fids:
        names = [str(c) for c in np.load(f"{CACHE}/{fid}.npz", allow_pickle=True)["cell_names"]]
        sub = by_flow.get(fid)
        if sub is None:
            print(f"  !! {fid}: no gates rows at {STAGE} — skipped"); continue
        sub = sub.drop_duplicates("name").set_index("name").reindex(names)
        cx = ((sub.x_min + sub.x_max) / 2).values
        cy = ((sub.y_min + sub.y_max) / 2).values
        ok = np.isfinite(cx) & np.isfinite(cy)          # False for tapcells (dropped downstream)
        # placement region = bbox of the cells that HAVE coordinates
        x0, x1 = np.nanmin(sub.x_min.values), np.nanmax(sub.x_max.values)
        y0, y1 = np.nanmin(sub.y_min.values), np.nanmax(sub.y_max.values)
        W, H = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
        xn = np.where(ok, (cx - x0) / W, 0.5)           # fill non-coord cells with die centre
        yn = np.where(ok, (cy - y0) / H, 0.5)
        np.savez_compressed(f"{OUT}/{fid}.npz",
                            x=xn.astype(np.float32), y=yn.astype(np.float32),
                            mask=ok, die_w=np.float32(W), die_h=np.float32(H))
        tot_cells += ok.sum(); tot_missing += (~ok).sum(); tot_flows += 1
    print(f"[{di+1:2}/{len(designs)}] {d:14} {len(fids):4} flows", flush=True)

print(f"\nwrote {tot_flows} flows -> {OUT}")
print(f"  cells with coords: {tot_cells:,}   without (tapcells etc): {tot_missing:,}")
