#!/usr/bin/env python3
"""
The SEAM — how one stage's state feeds the next.

The whole world model is f_place -> f_cts -> f_route, and the research problem is that each stage
must consume the PREVIOUS stage's IMAGINED output, not a real one, so errors compound.

This module defines the STATE INTERFACE and the ONE SWITCH that makes the seam measurable:

    mode = "real"      -> the downstream stage consumes the REAL upstream state (from EDA-Schema).
                          This is TEACHER FORCING — the baseline. It is also what f_place's targets
                          ARE, so "real placement state" == f_place's labels.
    mode = "imagined"  -> the downstream stage consumes the UPSTREAM MODEL's PREDICTION.
                          This is the actual pipeline. The gap between imagined and real final
                          error IS the compounding — the thing we measure and then fix.

The symmetry that keeps this clean: real state = the upstream targets; imagined state = the
upstream outputs. Same shape, same injection code. Flip one flag.

STATE VECTORS (what crosses each boundary):
  placement state (f_place -> f_cts, f_route):
      per-net   : log HPWL                       (net feature)
      per-cell  : endpoint slack                 (cell feature; 0 where not an endpoint)
      global    : tot_hpwl, buf_area, buf_cnt, wns, tns   (design context)
  cts state (f_cts -> f_route):
      global    : cts_buffers, cts_power, cts_wns, cts_tns

Injection appends these to the graph BEFORE the encoder, so the same encoder ingests them.
"""
import numpy as np, torch
import fplace
from fplace import norm, ROOT, CACHE, live_cells, _z, _slog

# ---- what flows, as ordered lists (single source of truth) ----
# f_place has LEVEL heads for these three (tot_hpwl/buf_area/buf_cnt). WNS/TNS in f_place are
# READ OUT from the endpoint head, not a level head — so the placement GLOBAL state we forward is
# the three level heads; timing is already carried per-cell via the endpoint slack feature.
PLACE_GLOBAL = ("tot_hpwl", "buf_area", "buf_cnt")                 # f_place level-head outputs
CTS_GLOBAL   = ("cts_buffers", "cts_power", "cts_wns", "cts_tns")  # f_cts global outputs
STATE_MODE   = None   # set per-run: "real" | "imagined". None during pure-standalone training.

def _keep(flow_id):
    return live_cells(np.load(f"{CACHE}/{flow_id}.npz", allow_pickle=True))

# ---------------------------------------------------------------------------
# REAL placement state — straight from EDA-Schema (== f_place's targets).
# These are the SAME transforms f_place uses, so real-state injection is consistent with what
# f_place would predict.
# ---------------------------------------------------------------------------
def real_place_state(flow_id, g, device="cpu"):
    d = np.load(f"{CACHE}/{flow_id}.npz", allow_pickle=True)
    keep = _keep(flow_id); nm = norm()
    # per-net log HPWL (standardized like f_place's y_net_hpwl)
    net_hpwl = _z("net_hpwl", torch.log(torch.tensor(np.asarray(d["net_hpwl"]), dtype=torch.float).clamp(min=1e-6)))
    net_hpwl = torch.nan_to_num(net_hpwl)
    # per-cell endpoint slack (already loaded into g by f_place's load path as y_endpt)
    cell_slack = g["y_endpt"] if "y_endpt" in g else torch.zeros(g["n_cells"])
    m = fplace.meta().loc[flow_id]
    raws = dict(tot_hpwl=np.log(max(m.total_hpwl,1e-6)), buf_area=np.log(max(m.buffer_area,0)+1),
                buf_cnt=np.log(max(getattr(m,'buffer_count',np.nan),0)+1))
    glob = torch.tensor(_glob_lvl_dev(nm, flow_id, raws), dtype=torch.float)
    return dict(net=net_hpwl.to(device), cell=cell_slack.to(device), glob=torch.nan_to_num(glob).to(device))

def _glob_lvl_dev(nm, flow_id, raws):
    """Forward each global as TWO channels: level and deviation, each at UNIT scale.

    THE BUG THIS FIXES: a single channel (raw_log - L_m)/L_s standardizes by the CROSS-DESIGN
    spread L_s. But the knob response only spans W (the WITHIN-design spread), and W/L_s is
    0.055 / 0.026 / 0.019 for tot_hpwl / buf_area / buf_cnt -- so the knob signal arrived
    compressed 18x / 38x / 53x under design identity and f_cts could not see it. This is the
    same level-vs-deviation normalization trap that cost f_place -14.7 R2 before it was split.
    Splitting restores the knob response to scale ~1, where it is actually visible:
        level     = (mu_design - L_m) / L_s     -> WHICH design (design identity)
        deviation = (raw_log - mu_design) / W_d -> WHAT THE KNOBS DID  <- the agent needs this
    f_place already predicts exactly these two quantities, so imagined mode forwards its
    {k}_lvl and {k}_dev heads directly -- same spaces, no rescaling.
    """
    dsg = flow_id.rsplit("-", 1)[0]; out = []
    for k in PLACE_GLOBAL:
        L_m, L_s = float(nm[f"L_{k}_m"]), float(nm[f"L_{k}_s"])
        i = int(np.where(nm[f"MU_{k}_keys"] == dsg)[0][0])
        mu_d = float(nm[f"MU_{k}_vals"][i]); w_d = float(nm[f"W_{k}_vals"][i])
        out.append((mu_d - L_m) / L_s)              # level
        out.append((raws[k] - mu_d) / max(w_d, 1e-3))  # deviation -- unit scale, knob response
    return out

# ---------------------------------------------------------------------------
# IMAGINED placement state — f_place's OWN outputs on this flow.
# ---------------------------------------------------------------------------
@torch.no_grad()
def imagined_place_state(flow_id, g, place_model, device="cpu"):
    place_model.eval()
    o = place_model(g)
    net  = torch.nan_to_num(o["net_hpwl"][:, 0])     # per-net predicted HPWL
    cell = torch.nan_to_num(o["endpt"][:, 0])        # per-cell predicted endpoint slack
    # Level AND deviation, forwarded as separate unit-scale channels -- see _glob_lvl_dev.
    # f_place's heads already emit these in exactly the spaces real mode builds, so the two
    # modes are directly comparable and the knob response is NOT crushed by cross-design scale.
    glob = torch.stack([o[f"{k}_{p}"][0] for k in PLACE_GLOBAL for p in ("lvl", "dev")])
    return dict(net=net.to(device), cell=cell.to(device),
                glob=torch.nan_to_num(glob).to(device))

def place_state(flow_id, g, mode, place_model=None, device="cpu"):
    if mode == "real":     return real_place_state(flow_id, g, device)
    if mode == "imagined": return imagined_place_state(flow_id, g, place_model, device)
    raise ValueError(f"bad mode {mode}")
