#!/usr/bin/env python3
"""
f_place — knob-conditioned DE-HNN.

STRICT FIDELITY: the encoder is DE-HNN's `model_att.py` / `HyperConvLayer`, reproduced exactly
(see paperCodes/DEHNN/de_hnn/models/). Our ONLY additions, all specified in docs/architecture.md:
  (a) knob+design context INJECTED INTO THE VIRTUAL NODE (architecture.md §2 knob-injection),
      so it is broadcast to nodes every layer rather than fading from a one-shot input add;
  (b) multi-task uncertainty heads (mean+logvar → Gaussian NLL): per-net hpwl, per-net demand,
      total hpwl, buffer area, buffer count.

DE-HNN details reproduced verbatim:
  - HyperConvLayer: lin_node/lin_net + residual; driver(source)/sink split via SimpleConv;
    psi() for the net update, mlp() for the cell update; residual to the layer INPUT.
  - VN acts on CELLS only; init = virtualnode_encoder(concat[mean_pool, max_pool] of input feats);
    per layer: broadcast = back_mlp(concat[h, vn[part]]) + h  (BEFORE conv);
    then conv → norm → LeakyReLU; then (except last layer) vn = mlp(concat[mean,max] of h) + vn.
  - edge dropout p=0.2 (train only); heads fc1(d→256) → LeakyReLU → fc2.
"""
import os, glob, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear, ReLU, LeakyReLU
from torch_geometric.nn import SimpleConv
from torch_geometric.utils import scatter, dropout_edge

# repo-relative so the same code runs on the laptop and on the cluster.
# override with PD_ROOT=/path/to/repo if the cache lives elsewhere (e.g. scratch).
ROOT  = os.environ.get("PD_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = f"{ROOT}/cache/graphs"
N_TYPES = 5293
_META = _NORM = None

def meta():
    global _META
    if _META is None: _META = pd.read_parquet(f"{ROOT}/cache/meta.parquet").set_index("flow_id")
    return _META

# Targets, in the log space they're trained in. Standardized to ~N(0,1) so the five
# heads are commensurate: raw log-means run 2.4 (net_hpwl) to 11.2 (tot_hpwl), which
# would let tot_hpwl dominate the gradient purely because of its units.
TARGETS = ("net_hpwl", "net_dem", "tot_hpwl", "buf_area", "buf_cnt")

def set_norm(train_designs, force=False):
    """Build feature + target normalization from TRAINING designs ONLY.

    Two things this fixes vs. the old norm():
      * old code used sorted(glob)[:40], which is 40 flows of ONE design (ac97_ctrl) —
        every design got z-scored by one small design's statistics.
      * stats computed over all designs leak test/OOD statistics into training.
    Call this ONCE per fold, before any load_graph().
    """
    global _NORM
    key = hash(tuple(sorted(train_designs))) & 0xffffffff
    f = f"{ROOT}/cache/norm_{key:08x}.npz"
    if not force and os.path.exists(f):
        _NORM = dict(np.load(f)); return _NORM

    cx, nx, df, ynh, ynd = [], [], [], [], []
    for dsg in sorted(train_designs):                       # stratified: every train design
        fl = sorted(glob.glob(f"{CACHE}/{dsg}-*.npz"))
        for p in fl[:2]:                                    # graph feats barely move with knobs
            d = np.load(p, allow_pickle=True)
            cx.append(np.nan_to_num(np.concatenate([d["cell_x"], d["pe_cell"]], 1)))
            nx.append(d["net_x"]); df.append(d["df_vals"])
        for p in fl[::11][:10]:                             # per-net targets DO move with knobs
            d = np.load(p, allow_pickle=True)
            a = d["net_hpwl"];   ynh.append(np.log(a[np.isfinite(a) & (a > 0)]))
            b = d["net_demand"]; ynd.append(np.log(b[np.isfinite(b) & (b > 0)]))
    cx, nx, df = np.concatenate(cx), np.concatenate(nx), np.stack(df)
    _NORM = dict(cx_m=cx.mean(0), cx_s=cx.std(0)+1e-6, nx_m=nx.mean(0), nx_s=nx.std(0)+1e-6,
                 df_m=df.mean(0), df_s=df.std(0)+1e-6)

    # global targets: read straight off meta (free), train designs only
    m = meta(); tr = m.index.str.replace(r"-\d+$", "", regex=True).isin(set(train_designs))
    mt = m[tr]
    gl = dict(tot_hpwl=np.log(np.maximum(mt.total_hpwl.values, 1e-6)),
              buf_area=np.log(np.maximum(mt.buffer_area.values, 0) + 1.0),
              buf_cnt =np.log(np.maximum(mt.buffer_count.values, 0) + 1.0))
    ynh, ynd = np.concatenate(ynh), np.concatenate(ynd)
    ys = dict(net_hpwl=ynh, net_dem=ynd, **gl)
    for k in TARGETS:
        v = ys[k]; v = v[np.isfinite(v)]
        _NORM[f"y_{k}_m"] = np.float32(v.mean())
        _NORM[f"y_{k}_s"] = np.float32(v.std() + 1e-6)
    np.savez(f, **_NORM)
    print(f"[norm] built from {len(train_designs)} TRAIN designs → {os.path.basename(f)}")
    for k in TARGETS:
        print(f"       {k:9} log-mean {float(_NORM[f'y_{k}_m']):7.3f}  std {float(_NORM[f'y_{k}_s']):6.3f}")
    return _NORM

def norm():
    if _NORM is None:
        raise RuntimeError("call set_norm(train_designs) before load_graph() — "
                           "normalization must come from TRAIN designs only (no leakage).")
    return _NORM

def denorm(k, v):
    """standardized log-space -> log-space (for reporting in real units)."""
    n = norm(); return v * float(n[f"y_{k}_s"]) + float(n[f"y_{k}_m"])

def _z(k, v):
    n = norm(); return (v - float(n[f"y_{k}_m"])) / float(n[f"y_{k}_s"])

def load_graph(flow_id, device="cpu"):
    d = np.load(f"{CACHE}/{flow_id}.npz", allow_pickle=True)
    m = meta().loc[flow_id]; nm = norm()
    ct = d["cell_type"].astype(np.int64); ct[ct < 0] = N_TYPES - 1
    ed = torch.tensor(d["edge_driver"], dtype=torch.long)
    es = torch.tensor(d["edge_sink"],   dtype=torch.long)
    t  = lambda a, dt=torch.float: torch.tensor(np.asarray(a), dtype=dt)
    cell_x = (np.nan_to_num(np.concatenate([d["cell_x"], d["pe_cell"]], 1)) - nm["cx_m"]) / nm["cx_s"]
    net_x  = (d["net_x"]  - nm["nx_m"]) / nm["nx_s"]
    dfeat  = (d["df_vals"] - nm["df_m"]) / nm["df_s"]
    knb    = np.array([m.clock_period/10.0, m.utilization/30.0, m.aspect_ratio], np.float32)
    part_c = torch.tensor(d["part_cell"], dtype=torch.long)      # DE-HNN: VN over CELLS
    bufcnt = float(getattr(m, "buffer_count", np.nan))
    g = dict(
        cell_x=t(cell_x), cell_type=torch.tensor(ct), net_x=t(net_x),
        part_cell=part_c, num_vn=int(part_c.max()) + 1,
        n_cells=int(len(ct)), n_nets=int(len(d["net_x"])),
        dfeat=t(dfeat), knobs=t(knb),
        ntn=torch.cat([ed, es], 1),                                    # cell→net [cell; net]
        ntn_type=torch.cat([torch.ones(ed.size(1)), torch.zeros(es.size(1))]),
        ntc=torch.cat([ed.flip(0), es.flip(0)], 1),                    # net→cell [net; cell]
        # all targets: log space, then STANDARDIZED with train-fold stats -> ~N(0,1).
        # (raw log-means span 2.4..11.2; unstandardized, tot_hpwl swamps the gradient.)
        y_net_hpwl=_z("net_hpwl", torch.log(t(d["net_hpwl"]).clamp(min=1e-6))),
        m_net_hpwl=t(np.isfinite(d["net_hpwl"]) & (d["net_hpwl"] > 0), torch.bool),
        y_net_dem =_z("net_dem",  torch.log(t(d["net_demand"]).clamp(min=1e-6))),
        m_net_dem =t(np.isfinite(d["net_demand"]) & (d["net_demand"] > 0), torch.bool),
        y_tot_hpwl=_z("tot_hpwl", t(np.log(max(m.total_hpwl, 1e-6)))),
        y_buf_area=_z("buf_area", t(np.log(max(m.buffer_area, 0) + 1.0))),
        y_buf_cnt =_z("buf_cnt",  t(np.log(max(bufcnt, 0) + 1.0))) if np.isfinite(bufcnt) else t(0.0),
        has_bufcnt=bool(np.isfinite(bufcnt)),
    )
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in g.items()}

# ---------- DE-HNN HyperConvLayer (verbatim) ----------
class HyperConv(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin_node, self.lin_net = Seq(Linear(d, d)), Seq(Linear(d, d))
        self.psi = Seq(Linear(d*3, d), ReLU(), Linear(d, d))   # net update
        self.mlp = Seq(Linear(d*3, d), ReLU(), Linear(d, d))   # cell update
        self.forward_conv, self.back_conv = SimpleConv(), SimpleConv()

    def forward(self, x, x_net, ntn, ttype, ntc):
        h_net = self.lin_net(x_net) + x_net
        h     = self.lin_node(x)    + x
        sm, km = ttype == 1, ttype == 0                        # source(driver) / sink
        h_net_source = self.forward_conv((h, h_net), ntn[:, sm]) + h_net
        h_net_sink   = self.forward_conv((h, h_net), ntn[:, km]) + h_net
        h_net = self.psi(torch.cat([h_net, h_net_sink, h_net_source], 1)) + x_net
        h_source = self.back_conv((h_net, h), ntc[:, sm]) + h
        h_sink   = self.back_conv((h_net, h), ntc[:, km]) + h
        h = self.mlp(torch.cat([h, h_sink, h_source], 1)) + x
        return h, h_net

# ---------- ABLATION ENCODERS (so ENCODER= actually does what it says) ----------
class HyperConvUndirected(nn.Module):
    """DE-HNN minus the driver/sink split — all cell↔net edges treated the same.
    Tests whether the directional (driver≠sink) asymmetry matters."""
    def __init__(self, d):
        super().__init__()
        self.lin_node, self.lin_net = Seq(Linear(d, d)), Seq(Linear(d, d))
        self.psi = Seq(Linear(d*2, d), ReLU(), Linear(d, d))
        self.mlp = Seq(Linear(d*2, d), ReLU(), Linear(d, d))
        self.forward_conv, self.back_conv = SimpleConv(), SimpleConv()
    def forward(self, x, x_net, ntn, ttype, ntc):
        h_net = self.lin_net(x_net) + x_net
        h     = self.lin_node(x)    + x
        h_net_all = self.forward_conv((h, h_net), ntn) + h_net       # NO source/sink split
        h_net = self.psi(torch.cat([h_net, h_net_all], 1)) + x_net
        h_all = self.back_conv((h_net, h), ntc) + h
        h = self.mlp(torch.cat([h, h_all], 1)) + x
        return h, h_net

class BipartiteConv(nn.Module):
    """Standard GNN baseline (GraphSAGE / GATv2) on the same bipartite graph.
    No driver/sink typing, no DE-HNN psi/mlp structure."""
    def __init__(self, d, kind="sage"):
        super().__init__()
        from torch_geometric.nn import SAGEConv, GATv2Conv
        C = (lambda: SAGEConv((d, d), d)) if kind == "sage" else \
            (lambda: GATv2Conv((d, d), d, heads=2, concat=False, add_self_loops=False))
        self.c2n, self.n2c = C(), C()
    def forward(self, x, x_net, ntn, ttype, ntc):
        h_net = self.c2n((x, x_net), ntn) + x_net      # cells → nets
        h     = self.n2c((h_net, x),  ntc) + x         # nets  → cells
        return h, h_net

ENCODERS = {"dehnn", "dehnn_novn", "dehnn_undirected", "sage", "gat"}

class FPlace(nn.Module):
    def __init__(self, d=64, K=4, cell_in=25, net_in=4, knob=3, dfeat=18, encoder="dehnn"):
        super().__init__()
        assert encoder in ENCODERS, f"unknown encoder {encoder}; pick from {ENCODERS}"
        self.encoder = encoder
        self.use_vn = encoder != "dehnn_novn"     # the no-VN ablation
        self.K = K
        self.node_encoder = Seq(Linear(cell_in, d), LeakyReLU(), Linear(d, d))
        self.net_encoder  = Seq(Linear(net_in, d),  LeakyReLU(), Linear(d, d))
        self.type_emb = nn.Embedding(N_TYPES, d)
        # --- VN (DE-HNN exact): cells only, mean+max pooling, concat-MLP broadcast ---
        self.virtualnode_encoder = Seq(Linear(cell_in*2, d*2), LeakyReLU(), Linear(d*2, d))
        self.mlp_vn  = nn.ModuleList([Seq(Linear(d*2, d), LeakyReLU(), Linear(d, d)) for _ in range(K)])
        self.back_vn = nn.ModuleList([Seq(Linear(d*2, d), LeakyReLU(), Linear(d, d)) for _ in range(K)])
        # --- OUR addition (architecture.md §2): knob+design context injected INTO the VN ---
        # (if VN is ablated away, ctx falls back to a one-shot input add — the only place it can go)
        self.ctx = Seq(Linear(knob + dfeat, d), LeakyReLU(), Linear(d, d))
        mk = {"dehnn": lambda: HyperConv(d), "dehnn_novn": lambda: HyperConv(d),
              "dehnn_undirected": lambda: HyperConvUndirected(d),
              "sage": lambda: BipartiteConv(d, "sage"), "gat": lambda: BipartiteConv(d, "gat")}[encoder]
        self.convs = nn.ModuleList([mk() for _ in range(K)])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(K)])
        # --- heads: DE-HNN fc1(d→256)→LeakyReLU→fc2 ; ours output (mu, logvar) ---
        self.fc1_net, self.fc1_glob = Linear(d, 256), Linear(2*d, 256)
        self.h_net_hpwl, self.h_net_dem = Linear(256, 2), Linear(256, 2)
        self.h_tot, self.h_buf_area, self.h_buf_cnt = Linear(256, 2), Linear(256, 2), Linear(256, 2)

    def forward(self, g):
        h     = self.node_encoder(g["cell_x"]) + self.type_emb(g["cell_type"])
        h_net = self.net_encoder(g["net_x"])
        part, nvn = g["part_cell"], g["num_vn"]
        ctx = self.ctx(torch.cat([g["knobs"], g["dfeat"]]))
        if self.use_vn:
            # VN init: DE-HNN pooled input feats + OUR knob/design ctx (→ broadcast every layer)
            vn_in = torch.cat([scatter(g["cell_x"], part, 0, dim_size=nvn, reduce="mean"),
                               scatter(g["cell_x"], part, 0, dim_size=nvn, reduce="max")], 1)
            vn = self.virtualnode_encoder(vn_in) + ctx
        else:
            h, h_net = h + ctx, h_net + ctx        # no VN → ctx can only be added at input
        ntn, ttype, ntc = g["ntn"], g["ntn_type"], g["ntc"]
        if self.training:                                     # DE-HNN edge dropout p=0.2
            ntn, mask = dropout_edge(ntn, p=0.2)
            ttype, ntc = ttype[mask], ntc[:, mask]
        for l in range(self.K):
            if self.use_vn:
                h = self.back_vn[l](torch.cat([h, vn[part]], 1)) + h   # VN → cells (carries knobs)
            h, h_net = self.convs[l](h, h_net, ntn, ttype, ntc)
            h     = F.leaky_relu(self.norms[l](h))
            h_net = F.leaky_relu(self.norms[l](h_net))
            if self.use_vn and l < self.K - 1:
                vn_t = torch.cat([scatter(h, part, 0, dim_size=nvn, reduce="mean"),
                                  scatter(h, part, 0, dim_size=nvn, reduce="max")], 1)
                vn = self.mlp_vn[l](vn_t) + vn
        hn = F.leaky_relu(self.fc1_net(h_net))
        hg = F.leaky_relu(self.fc1_glob(torch.cat([h.mean(0), h_net.mean(0)])))
        return dict(net_hpwl=self.h_net_hpwl(hn), net_dem=self.h_net_dem(hn),
                    tot_hpwl=self.h_tot(hg), buf_area=self.h_buf_area(hg), buf_cnt=self.h_buf_cnt(hg))

def gnll(pred, y, mask=None, nll=True):
    """Gaussian NLL on standardized targets.

    nll=False -> plain MSE on the mean (the variance head is ignored).
    WHY: with NLL from step 0 the model shrinks logvar to cut the loss on points it
    already fits (train loss goes NEGATIVE), then any miss on val is divided by
    exp(logvar) and explodes. Seen here: train -0.13 / val 27.6 at epoch 1.
    So we warm up the mean with MSE, then switch NLL on once mu is sane.
    Clamp is (-5,5): targets are unit-scale now, so logvar has no business outside that.
    """
    mu, lv = pred[..., 0], pred[..., 1].clamp(-5, 5)
    if mask is not None:
        if mask.sum() == 0: return torch.zeros((), device=mu.device)
        mu, lv, y = mu[mask], lv[mask], y[mask]
    se = (y - mu).pow(2)
    if not nll: return 0.5 * se.mean()
    return 0.5 * (lv + se / lv.exp()).mean()

def loss_fn(out, g):
    L = (gnll(out["net_hpwl"], g["y_net_hpwl"], g["m_net_hpwl"])
       + gnll(out["net_dem"],  g["y_net_dem"],  g["m_net_dem"])
       + gnll(out["tot_hpwl"], g["y_tot_hpwl"])
       + gnll(out["buf_area"], g["y_buf_area"]))
    if g["has_bufcnt"]: L = L + gnll(out["buf_cnt"], g["y_buf_cnt"])
    return L

if __name__ == "__main__":
    import time
    paths = sorted(glob.glob(f"{CACHE}/*.npz"), key=os.path.getsize)[:8]
    graphs = [load_graph(os.path.basename(p)[:-4]) for p in paths]
    print(f"smoke test: {len(graphs)} graphs, {graphs[0]['n_cells']} cells, "
          f"{graphs[0]['num_vn']} VNs (DE-HNN-exact model)")
    model = FPlace(); model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3); t0 = time.time()
    for step in range(15):
        opt.zero_grad(); tot = 0.0
        for g in graphs:
            l = loss_fn(model(g), g); l.backward(); tot += l.item()
        opt.step()
        if step % 3 == 0 or step == 14:
            print(f"  step {step:3d}  loss/graph = {tot/len(graphs):.4f}  ({time.time()-t0:.0f}s)")
