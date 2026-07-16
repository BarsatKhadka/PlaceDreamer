#!/usr/bin/env python3
"""
f_cts — the Clock-Tree-Synthesis stage of the world model.

Input:  the SAME floorplan netlist graph f_place uses (the whole pipeline anchors on it) +
        the CLOCK structure (which cells are sinks, their switching activity) + knobs.
        (Standalone training uses the real graph; the SEAM later swaps in f_place's imagined
         placement state as extra node features.)
Output: the CTS state, WITHOUT running CTS —
        cts_buffers : # clock buffers CTS inserts       (EASY: LODO R2 0.89 from #sinks alone)
        cts_power   : post-CTS total power              (EASY: LODO R2 0.95 from sinks+activity)
        cts_wns/tns : post-CTS timing                   (HARD: level doesn't transfer, knob does)

Architecture: REUSES f_place's exact winning encoder (FPlace.encode: DE-HNN + VN-structure +
multi-aggr + concat-fuse) — so a stage boundary is just a change of heads on a shared
representation, which is what makes the seam possible. The ONE CTS-specific addition is a
SINK-SPECIFIC READOUT: clock metrics are functions of the SINKS, not all cells, so we pool
(mean+max) over the clock-sink nodes and read the clock heads from that. mean -> count/power
(sum-like); MAX -> the worst-case sinks that drive skew/WNS.
"""
import os, glob, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear, LeakyReLU
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fplace, seam
from fplace import (FPlace, meta, norm, set_norm as _set_norm_place, load_graph as _load_place,
                    ROOT, _slog, gnll)

# THE SEAM SWITCH. mode=None -> standalone f_cts (no placement state).
#   "real"     -> teacher forcing: consume the REAL placement state from EDA-Schema.
#   "imagined" -> consume f_place's PREDICTION. The gap between the two final errors is the
#                 compounding we measure. The model is IDENTICAL in both; only the input differs.
# When the seam is on, the model gains: +1 cell feature (placement slack), +1 net feature
# (placement HPWL), +len(PLACE_GLOBAL) design features (placement globals).
_SEAM = {"mode": None, "place_model": None}
def set_seam(mode, place_model=None):
    _SEAM["mode"] = mode; _SEAM["place_model"] = place_model
def seam_dims():
    on = _SEAM["mode"] is not None
    # each placement global crosses as TWO channels (level, deviation) -- see seam._glob_lvl_dev
    return dict(cell=(1 if on else 0), net=(1 if on else 0),
                df=(2 * len(seam.PLACE_GLOBAL) if on else 0))

KNOB = 5                                      # raw knob vector width (matches fplace)
# A/B FLAG, not an assumption: do the RAW knobs get a direct path into the deviation heads?
# f_place has exactly this (fplace.DIRECT_KNOB, default on, A/B-validated). f_cts never did —
# its knobs only reached the dev heads through ctx = MLP([knobs, dfeat]), mixed with 16 design
# features and diffused through K message-passing layers. Measured OLS ceilings from the raw
# knobs alone (within-design R2): cts_power 0.905 (clock_period; P = a*C*V^2*f),
# cts_buffers 0.271 (utilization -> die area -> sink spread -> clock WL), cts_wns 0.538.
# f_cts was scoring ~0 on all three, so the signal was present but unreachable. Flag it so the
# claim is MEASURED by an A/B rather than asserted.
CTS_DIRECT_KNOB = bool(int(os.environ.get("CTS_DIRECT_KNOB") or "1"))
CTS_GLOBAL = ("cts_buffers", "cts_power")     # level + knob-deviation (like tot_hpwl/buf_area)
CTS_TIMING = ("cts_wns", "cts_tns")           # signed; level + deviation

# ---- CTS cell features appended to cell_x: [is_sink, switching_activity] ----
# is_sink from the clock net's fanout (the TRUE sink set, matches clock_trees exactly);
# activity is the literature's #1 clock feature (clock power ~ activity x cap). Both in cache/cts.
def load_graph(flow_id, device="cpu"):
    g = _load_place(flow_id, device)          # reuse f_place's graph EXACTLY (incl. tapcell drop)
    dsg = flow_id.rsplit("-", 1)[0]
    c = np.load(f"{ROOT}/cache/cts/{flow_id}.npz")   # PER-FLOW (netlist differs per config)
    keep = fplace.live_cells(np.load(f"{fplace.CACHE}/{flow_id}.npz", allow_pickle=True))
    is_sink = c["is_sink"][keep]
    act = c["activity"][keep]
    nm = norm()
    act_z = (act - nm["cts_act_m"]) / nm["cts_act_s"]
    # CTS cell features: is_sink, activity. Plus, IF the SEAM is active, f_place's per-cell
    # placement state (endpoint slack) — real (teacher-forced) or imagined (f_place's prediction).
    cell_extra = [is_sink, act_z]
    if _SEAM["mode"] is not None:
        ps = seam.place_state(flow_id, g, _SEAM["mode"], _SEAM["place_model"], device)
        g["_place_state"] = ps                                   # net + glob injected below
        cell_extra.append(np.asarray(ps["cell"].cpu()))          # per-cell placement slack
    extra = torch.tensor(np.stack(cell_extra, 1), dtype=torch.float, device=device)
    # INSERT into the LIBRARY block (before PE) — PE must stay the last 10 dims for SignNet.
    cx = g["cell_x"]
    lib, pe = cx[:, :fplace.PE_SLICE.start], cx[:, fplace.PE_SLICE.start:]
    g["cell_x"] = torch.cat([lib, extra, pe], 1)
    if _SEAM["mode"] is not None:
        # inject per-net placement HPWL into net_x, and the placement globals into dfeat
        ps = g["_place_state"]
        g["net_x"] = torch.cat([g["net_x"], ps["net"].unsqueeze(1)], 1)
        g["dfeat"] = torch.cat([g["dfeat"], ps["glob"]], 0)
    g["sink_mask"] = torch.tensor(is_sink > 0.5, dtype=torch.bool, device=device)
    m = meta().loc[flow_id]
    # targets, decomposed level+deviation exactly like f_place's global targets
    for k, col, signed in (("cts_buffers","cts_buffers",False), ("cts_power","cts_power",False),
                           ("cts_wns","cts_wns",True), ("cts_tns","cts_tns",True)):
        raw = float(m[col])
        yv = _slog(raw) if signed else np.log(max(raw, 1e-6))
        i = int(np.where(nm[f"MU_{k}_keys"] == dsg)[0][0])
        mu_d, w_d = float(nm[f"MU_{k}_vals"][i]), float(nm[f"W_{k}_vals"][i])
        g[f"y_{k}_lvl"] = torch.tensor((mu_d - float(nm[f"L_{k}_m"])) / float(nm[f"L_{k}_s"]), dtype=torch.float, device=device)
        g[f"y_{k}_dev"] = torch.tensor((yv - mu_d) / w_d, dtype=torch.float, device=device)
        g[f"w_{k}"] = w_d; g[f"deg_{k}"] = bool(nm[f"DEG_{k}_vals"][i]); g[f"y_{k}"] = yv
    return g

def set_norm(train_designs, force=False):
    """f_place norm (features + its targets) PLUS the CTS additions (activity stats, and the
    level/deviation decomposition for the 4 CTS targets). Reuses f_place's set_norm wholesale."""
    _set_norm_place(train_designs, force=force)
    nm = fplace._NORM
    # activity normalization (train designs only; sample a few flows each — per-flow files now)
    acts = []
    for d in sorted(train_designs):
        for p in sorted(glob.glob(f"{ROOT}/cache/cts/{d}-*.npz"))[::11][:6]:
            acts.append(np.load(p)["activity"])
    a = np.concatenate(acts)
    nm["cts_act_m"] = np.float32(a.mean()); nm["cts_act_s"] = np.float32(a.std() + 1e-6)
    # level+deviation stats for the CTS targets (mirrors fplace.set_norm's global-target block)
    m = meta(); dcol = m.index.str.replace(r"-\d+$", "", regex=True)
    for k, col, signed in (("cts_buffers","cts_buffers",False), ("cts_power","cts_power",False),
                           ("cts_wns","cts_wns",True), ("cts_tns","cts_tns",True)):
        v = m[col].values.astype(np.float64)
        y = np.sign(v)*np.log1p(np.abs(v)) if signed else np.log(np.maximum(v, 1e-6))
        s = pd.Series(y, index=dcol)
        mu_d, w_d = s.groupby(level=0).mean(), s.groupby(level=0).std()
        tr = [d for d in mu_d.index if d in set(train_designs)]
        nm[f"L_{k}_m"] = np.float32(mu_d[tr].mean()); nm[f"L_{k}_s"] = np.float32(mu_d[tr].std()+1e-6)
        nm[f"W_{k}_keys"] = np.array(w_d.index); nm[f"W_{k}_vals"] = np.maximum(w_d.values,1e-3).astype(np.float32)
        nm[f"W_{k}"] = np.float32(w_d[tr].mean()+1e-6)
        nm[f"DEG_{k}_keys"] = np.array(w_d.index); nm[f"DEG_{k}_vals"] = (w_d < 0.03).values
        nm[f"MU_{k}_keys"] = np.array(mu_d.index); nm[f"MU_{k}_vals"] = mu_d.values.astype(np.float32)
    return nm

def recon(k, lvl, dev, nm, w=None):
    if w is None: w = float(nm[f"W_{k}"])
    return (lvl * float(nm[f"L_{k}_s"]) + float(nm[f"L_{k}_m"])) + dev * float(w)


class FCTS(nn.Module):
    """f_place's encoder (reused, +2 CTS cell features) + a SINK-SPECIFIC readout + CTS heads."""
    def __init__(self, d=64, K=4, encoder="dehnn"):
        super().__init__()
        sd = seam_dims()                        # extra dims when the seam is on (else 0)
        n_cell_extra = 2 + sd["cell"]           # is_sink, activity (+ placement slack if seam)
        # cell_x = [lib(12) | CTS-extra | PE(10)]. Build the encoder for that width and point
        # its PE split AFTER the extra features so SignNet still sees the last 10 dims as PE.
        cell_in = fplace.CELL_IN + n_cell_extra
        net_in  = fplace.NET_IN + sd["net"]
        dfeat   = fplace.DF_IN + sd["df"]
        self.enc = FPlace(d=d, K=K, cell_in=cell_in, net_in=net_in, dfeat=dfeat, encoder=encoder)
        self.enc.pe_start = fplace.PE_SLICE.start + n_cell_extra
        self.d = d
        # TWO readouts, because CTS targets have different SCOPE:
        #   sink-pool  (clock sinks only, mean+max)  -> cts_buffers (a clock-tree quantity)
        #   whole-pool (all cells + all nets)        -> cts_power, cts_wns, cts_tns
        # Bug found on the first run: pooling total_power over only the ~5% clock-sink cells
        # threw away the design-wide switching info -> power knob-R2 went NEGATIVE while its OLS
        # ceiling is +0.96. total_power is a WHOLE-design quantity; only buffers are sink-scoped.
        self.fc_sink  = Seq(Linear(2*d + d, 256), LeakyReLU(), Linear(256, 256))   # sinks + ctx
        self.fc_whole = Seq(Linear(4*d + d, 256), LeakyReLU(), Linear(256, 256))   # cells,nets + ctx
        # SEPARATE DEVIATION PATHWAY WITH A DIRECT KNOB PATH — mirrors f_place's DIRECT_KNOB.
        # THE BUG THIS FIXES: every CTS knob response is driven by a knob f_cts ALREADY has, and
        # it was scoring ~0 on all of them. Measured within-design R2 from the raw knobs alone:
        #     cts_power   0.905 from clock_period   (P = alpha*C*V^2*f -> power ~ 1/period)
        #     cts_buffers 0.271 from utilization    (die area -> sink spread -> clock WL)
        #     cts_wns     0.538 from clock_period
        # NONE of it needs placement geometry. But the knobs only reached the dev heads through
        # ctx = MLP([knobs, dfeat]) -- mixed with 16 design features and diffused through K layers
        # of message passing. f_place hit the same wall and solved it by feeding the RAW knobs
        # straight into the deviation head; f_cts never got that path. A 0.9-R2 signal was sitting
        # in the input unreachable.
        self.direct_knob = CTS_DIRECT_KNOB
        kd = KNOB if self.direct_knob else 0
        self.fc_sink_dev  = Seq(Linear(2*d + d + kd, 256), LeakyReLU(), Linear(256, 256))
        self.fc_whole_dev = Seq(Linear(4*d + d + kd, 256), LeakyReLU(), Linear(256, 256))
        self.scope = {"cts_buffers": "sink", "cts_power": "whole",
                      "cts_wns": "whole", "cts_tns": "whole"}
        self.h_lvl = nn.ModuleDict({k: Linear(256, 2) for k in CTS_GLOBAL + CTS_TIMING})
        self.h_dev = nn.ModuleDict({k: Linear(256, 2) for k in CTS_GLOBAL + CTS_TIMING})

    def forward(self, g):
        h, h_net, ctx = self.enc.encode(g)          # SAME winning encoder as f_place
        sm = g["sink_mask"]
        hs = h[sm] if sm.sum() >= 1 else h
        pool_sink  = [hs.mean(0), hs.max(0).values, ctx]
        pool_whole = [h.mean(0), h.max(0).values, h_net.mean(0), h_net.max(0).values, ctx]
        # LEVEL: which design is this — pooled graph is what carries design size.
        zL = {"sink":  F.leaky_relu(self.fc_sink(torch.cat(pool_sink))),
              "whole": F.leaky_relu(self.fc_whole(torch.cat(pool_whole)))}
        # DEVIATION: what the knobs did — the RAW knobs go in DIRECTLY (see __init__).
        kn = [g["knobs"]] if self.direct_knob else []
        zD = {"sink":  F.leaky_relu(self.fc_sink_dev(torch.cat(pool_sink + kn))),
              "whole": F.leaky_relu(self.fc_whole_dev(torch.cat(pool_whole + kn)))}
        o = {}
        for k in CTS_GLOBAL + CTS_TIMING:
            s = self.scope[k]
            o[f"{k}_lvl"] = self.h_lvl[k](zL[s])
            o[f"{k}_dev"] = self.h_dev[k](zD[s])       # knob response, fed the knobs directly
        return o
