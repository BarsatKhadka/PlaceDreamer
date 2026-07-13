#!/usr/bin/env python3
"""Extract per-ENDPOINT setup slack @ place_resized as per-cell labels for f_place.

Timing lives on endpoints (register D-pins / output ports), not nets. We predict a slack
at each register endpoint (node-level), then READ OUT WNS = min and TNS = sum-of-negatives
from those predictions — replacing the two global scalar heads with one richer per-endpoint
head that's physically where slack actually lives.

Per flow @ place_resized we store (aligned to the cached FLOORPLAN graph's cell order):
  ep_idx    int   indices into cell_names of the endpoint (register) cells we have a label for
  ep_slack  f32   worst (min) setup slack at that endpoint
Register endpoints map 100% to floorplan cells (verified all 18 designs). Primary-output
endpoints (0 in most designs, ~50% in wb_dma) are NOT cell nodes → not stored here; they are
covered by the recorded-total aggregate constraint in the loss (see train_fplace.wloss).

timing_paths is a TOP-N truncated report (WNS reconstructs exactly everywhere; TNS coverage
ranges 9–100%). So these direct labels teach the SHAPE; the complete recorded WNS/TNS
(cache/meta.parquet) constrain the aggregate. Labels join by cell_names — NO graph rebuild.
"""
import pyarrow.dataset as ds, numpy as np, glob, os, sys
DATA  = "/Users/barsat/PlaceDreamer/datasets/sky130hd"
CACHE = "/Users/barsat/PlaceDreamer/cache/graphs"
OUT   = "/Users/barsat/PlaceDreamer/cache/endpt"
STAGE = "place_resized"
os.makedirs(OUT, exist_ok=True)

def ep_to_cell(e): return e.rsplit("/", 1)[0] if "/" in e else None   # None = primary output

designs = sorted({os.path.basename(p).rsplit("-", 1)[0] for p in glob.glob(f"{CACHE}/*.npz")})
tp_ds = ds.dataset(f"{DATA}/timing_paths/table.parquet")
tot_ep = tot_flows = 0
for di, d in enumerate(designs):
    flows = sorted(glob.glob(f"{CACHE}/{d}-*.npz"))
    fids  = [os.path.basename(f)[:-4] for f in flows]
    # one read per design: all setup paths for its flows at this stage
    tp = tp_ds.to_table(
        filter=(ds.field("stage") == STAGE) & (ds.field("path_type") == "setup")
               & (ds.field("flow_id").isin(fids)),
        columns=["flow_id", "endpoint", "slack"]).to_pandas()
    by_flow = dict(tuple(tp.groupby("flow_id")))
    for fid in fids:
        cell_names = list(str(c) for c in np.load(f"{CACHE}/{fid}.npz", allow_pickle=True)["cell_names"])
        cidx = {c: i for i, c in enumerate(cell_names)}
        sub = by_flow.get(fid)
        idx, slk = [], []
        if sub is not None:
            worst = sub.groupby("endpoint").slack.min()      # worst setup slack per endpoint
            for ep, s in worst.items():
                c = ep_to_cell(ep)
                if c is not None and c in cidx:
                    idx.append(cidx[c]); slk.append(float(s))
        np.savez(f"{OUT}/{fid}.npz",
                 ep_idx=np.array(idx, np.int64), ep_slack=np.array(slk, np.float32))
        tot_ep += len(idx); tot_flows += 1
    print(f"  [{di+1}/{len(designs)}] {d}: {len(fids)} flows", flush=True)

print(f"\n✓ wrote {tot_flows} flows to {OUT}  ({tot_ep} endpoint labels, "
      f"avg {tot_ep/max(tot_flows,1):.0f}/flow)")
