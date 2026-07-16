#!/usr/bin/env python3
"""
f_route — the routing stage of the world model.

Input:  the SAME floorplan netlist graph f_place/f_cts use, + the placement state.
        (Standalone: real graph. SEAM later: f_place's imagined placement + f_cts's imagined
         clock state as node features.)
Output: the routing state, WITHOUT running the router —
        per-net ROUTED LENGTH (nets.length @ detailed_route) — the DETOUR signal, the genuine
                              routing quantity. Measured knob-transfer LODO +0.66 — the BEST of
                              any stage (routed WL tracks HPWL, which we predict well).
        rt_wl   : total routed wirelength       (level LODO +0.97)
        rt_power/rt_wns/rt_tns : post-route PPA  (level transfers, knob response is grounding-gated)

Architecture: REUSES f_place's encoder (FPlace.encode). A per-NET head for routed length (dense,
like f_place's net_hpwl) + level/deviation global heads for the aggregates. No new machinery —
a stage boundary is a change of heads on the shared representation.
"""
import os, glob, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear, LeakyReLU
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fplace
from fplace import (FPlace, meta, norm, set_norm as _set_norm_place, load_graph as _load_place,
                    ROOT, _slog, gnll, _z)

RT_GLOBAL = ("rt_wl", "rt_power")            # level + knob-deviation
RT_TIMING = ("rt_wns", "rt_tns")             # signed; level + deviation

def load_graph(flow_id, device="cpu"):
    g = _load_place(flow_id, device)          # reuse f_place's graph EXACTLY
    dsg = flow_id.rsplit("-", 1)[0]; nm = norm()
    # per-NET routed length (aligned to the floorplan net order, masked where unmatched)
    rl = np.load(f"{ROOT}/cache/route/{flow_id}.npz")["routed_len"]
    ok = np.isfinite(rl) & (rl > 0)
    y = np.zeros(len(rl), np.float32); y[ok] = (np.log(rl[ok]) - float(nm["rt_len_m"])) / float(nm["rt_len_s"])
    g["y_rt_len"] = torch.tensor(y, dtype=torch.float, device=device)
    g["m_rt_len"] = torch.tensor(ok, dtype=torch.bool, device=device)
    m = meta().loc[flow_id]
    for k, col, signed in (("rt_wl","rt_wl",False), ("rt_power","rt_power",False),
                           ("rt_wns","rt_wns",True), ("rt_tns","rt_tns",True)):
        raw = float(m[col]); yv = _slog(raw) if signed else np.log(max(raw, 1e-6))
        if not np.isfinite(raw):                        # 17 flows have no route (DRV/timeout)
            g[f"y_{k}_lvl"] = torch.tensor(0., device=device); g[f"y_{k}_dev"] = torch.tensor(0., device=device)
            g[f"w_{k}"] = 1.0; g[f"deg_{k}"] = True; g[f"y_{k}"] = np.nan; continue
        i = int(np.where(nm[f"MU_{k}_keys"] == dsg)[0][0])
        mu_d, w_d = float(nm[f"MU_{k}_vals"][i]), float(nm[f"W_{k}_vals"][i])
        g[f"y_{k}_lvl"] = torch.tensor((mu_d - float(nm[f"L_{k}_m"]))/float(nm[f"L_{k}_s"]), dtype=torch.float, device=device)
        g[f"y_{k}_dev"] = torch.tensor((yv - mu_d)/w_d, dtype=torch.float, device=device)
        g[f"w_{k}"] = w_d; g[f"deg_{k}"] = bool(nm[f"DEG_{k}_vals"][i]); g[f"y_{k}"] = yv
    return g

def set_norm(train_designs, force=False):
    _set_norm_place(train_designs, force=force); nm = fplace._NORM
    # per-net routed-length normalization (train designs, sample flows)
    rls = []
    for d in sorted(train_designs):
        for p in sorted(glob.glob(f"{ROOT}/cache/route/{d}-*.npz"))[::11][:10]:
            a = np.load(p)["routed_len"]; rls.append(np.log(a[np.isfinite(a) & (a > 0)]))
    r = np.concatenate(rls); nm["rt_len_m"] = np.float32(r.mean()); nm["rt_len_s"] = np.float32(r.std()+1e-6)
    # level+deviation for the global route targets
    m = meta(); dcol = m.index.str.replace(r"-\d+$", "", regex=True)
    for k, col, signed in (("rt_wl","rt_wl",False), ("rt_power","rt_power",False),
                           ("rt_wns","rt_wns",True), ("rt_tns","rt_tns",True)):
        v = m[col].values.astype(np.float64)
        y = np.sign(v)*np.log1p(np.abs(v)) if signed else np.log(np.maximum(v, 1e-6))
        s = pd.Series(y, index=dcol); mu_d, w_d = s.groupby(level=0).mean(), s.groupby(level=0).std()
        tr = [d for d in mu_d.index if d in set(train_designs)]
        nm[f"L_{k}_m"] = np.float32(np.nanmean(mu_d[tr])); nm[f"L_{k}_s"] = np.float32(np.nanstd(mu_d[tr])+1e-6)
        nm[f"W_{k}_keys"] = np.array(w_d.index); nm[f"W_{k}_vals"] = np.nan_to_num(np.maximum(w_d.values,1e-3),nan=1.0).astype(np.float32)
        nm[f"W_{k}"] = np.float32(np.nanmean(w_d[tr])+1e-6)
        nm[f"DEG_{k}_keys"] = np.array(w_d.index); nm[f"DEG_{k}_vals"] = (np.nan_to_num(w_d.values,nan=0.0) < 0.03)
        nm[f"MU_{k}_keys"] = np.array(mu_d.index); nm[f"MU_{k}_vals"] = np.nan_to_num(mu_d.values).astype(np.float32)
    return nm

def recon(k, lvl, dev, nm, w=None):
    if w is None: w = float(nm[f"W_{k}"])
    return (lvl*float(nm[f"L_{k}_s"]) + float(nm[f"L_{k}_m"])) + dev*float(w)


class FRoute(nn.Module):
    def __init__(self, d=64, K=4, encoder="dehnn"):
        super().__init__()
        self.enc = FPlace(d=d, K=K, encoder=encoder)     # reuse the winning encoder (no CTS feats)
        self.fc_net  = Linear(d + d, 256)                # per-net readout: [net emb, ctx]
        self.h_rt_len = Linear(256, 2)                   # per-net ROUTED LENGTH (the win)
        self.fc_glob = Linear(4*d + d, 256)
        self.h_lvl = nn.ModuleDict({k: Linear(256, 2) for k in RT_GLOBAL + RT_TIMING})
        self.h_dev = nn.ModuleDict({k: Linear(256, 2) for k in RT_GLOBAL + RT_TIMING})

    def forward(self, g):
        h, h_net, ctx = self.enc.encode(g)
        hn = F.leaky_relu(self.fc_net(torch.cat([h_net, ctx.expand(h_net.size(0), -1)], 1)))
        hg = F.leaky_relu(self.fc_glob(torch.cat([
            h.mean(0), h.max(0).values, h_net.mean(0), h_net.max(0).values, ctx])))
        o = {"rt_len": self.h_rt_len(hn)}
        for k in RT_GLOBAL + RT_TIMING:
            o[f"{k}_lvl"] = self.h_lvl[k](hg); o[f"{k}_dev"] = self.h_dev[k](hg)
        return o
