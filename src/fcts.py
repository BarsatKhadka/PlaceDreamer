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
import fplace
from fplace import (FPlace, meta, norm, set_norm as _set_norm_place, load_graph as _load_place,
                    ROOT, _pre, _slog, gnll)

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
    # INSERT the two CTS features into the LIBRARY block (before the PE slice), not at the end —
    # the encoder splits cell_x at PE_SLICE.start into [library | PE], and PE must stay the last
    # 10 dims for SignNet. So cell_x goes [lib(12) | CTS(2) | PE(10)] and PE_SLICE shifts by 2.
    extra = torch.tensor(np.stack([is_sink, act_z], 1), dtype=torch.float, device=device)
    cx = g["cell_x"]
    lib, pe = cx[:, :fplace.PE_SLICE.start], cx[:, fplace.PE_SLICE.start:]
    g["cell_x"] = torch.cat([lib, extra, pe], 1)
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

CELL_IN_CTS = None   # set after first set_norm (fplace CELL_IN + 2)

def set_norm(train_designs, force=False):
    """f_place norm (features + its targets) PLUS the CTS additions (activity stats, and the
    level/deviation decomposition for the 4 CTS targets). Reuses f_place's set_norm wholesale."""
    global CELL_IN_CTS
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
    CELL_IN_CTS = fplace.CELL_IN + 2
    return nm

def recon(k, lvl, dev, nm, w=None):
    if w is None: w = float(nm[f"W_{k}"])
    return (lvl * float(nm[f"L_{k}_s"]) + float(nm[f"L_{k}_m"])) + dev * float(w)


class FCTS(nn.Module):
    """f_place's encoder (reused, +2 CTS cell features) + a SINK-SPECIFIC readout + CTS heads."""
    def __init__(self, d=64, K=4, encoder="dehnn"):
        super().__init__()
        # the encoder is FPlace's — but cell_in is +2 for the CTS features. We build a full
        # FPlace to reuse encode(); its own heads exist but are unused (cheap, and keeps ONE
        # tested code path). cell_in override via fplace.CELL_IN is set in set_norm.
        self.enc = FPlace(d=d, K=K, cell_in=CELL_IN_CTS, encoder=encoder)
        # cell_x is [lib(12) | CTS(2) | PE(10)]. The encoder splits at self.enc.pe_start into
        # [library-block | PE]. Point it AFTER the 2 CTS features so the library MLP sees
        # lib+CTS (14 dims) and SignNet sees the last 10 (PE). node_encoder was built for
        # cell_in - n_pe = CELL_IN_CTS - 10 = 14 inputs, which matches.
        self.enc.pe_start = fplace.PE_SLICE.start + 2
        self.d = d
        # SINK readout: pool cell embeddings over CLOCK SINKS only (mean + max) + ctx.
        # mean -> buffer count / power (sum-like); max -> worst-case sink for skew/WNS.
        self.fc_sink = Seq(Linear(2*d + d, 256), LeakyReLU(), Linear(256, 256))
        self.h_lvl = nn.ModuleDict({k: Linear(256, 2) for k in CTS_GLOBAL + CTS_TIMING})
        self.h_dev = nn.ModuleDict({k: Linear(256, 2) for k in CTS_GLOBAL + CTS_TIMING})

    def forward(self, g):
        h, h_net, ctx = self.enc.encode(g)          # SAME winning encoder as f_place
        sm = g["sink_mask"]
        if sm.sum() < 1:                            # no sinks (shouldn't happen) -> whole-graph pool
            sink = torch.cat([h.mean(0), h.max(0).values, ctx])
        else:
            hs = h[sm]
            sink = torch.cat([hs.mean(0), hs.max(0).values, ctx])
        z = F.leaky_relu(self.fc_sink(sink))
        o = {}
        for k in CTS_GLOBAL + CTS_TIMING:
            o[f"{k}_lvl"] = self.h_lvl[k](z)
            o[f"{k}_dev"] = self.h_dev[k](z)
        return o
