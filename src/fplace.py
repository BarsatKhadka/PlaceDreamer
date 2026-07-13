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

# f_place placement-state targets. All standardized to ~N(0,1) before training so the
# heads are commensurate (raw scales span log-means 2.4..11.2 and slacks 0.7..-35446).
#   LOG targets:    strictly positive magnitudes -> log() then z-score.
#   SIGNED targets: slack is <=0 and heavy-tailed -> signed-log compress, then z-score.
# net_dem (per-net RUDY-from-bbox) was DROPPED: it's -0.94 correlated with net_hpwl (same
# bounding box, two views) — a circular target that proves nothing. Real congestion is a
# per-TILE field / router quantity, not per-net, and lives on the data_gen path, not here.
LOG_TARGETS    = ("net_hpwl", "tot_hpwl", "buf_area", "buf_cnt")
TARGETS        = LOG_TARGETS
# Timing (WNS/TNS) is NO LONGER a direct global head. Slack lives on ENDPOINTS (register
# D-pins), so we predict PER-ENDPOINT slack on the cell nodes and READ OUT WNS = min,
# TNS = sum-of-negatives from those predictions (train_fplace). Per-endpoint slack is
# intensive (a register's slack is a bounded number regardless of chip size) → transfers
# across sizes, unlike the extensive global TNS that blew up on extrapolation.
# v1: register endpoints only (100% node-mapped, covers all high-TNS designs). Primary-
# output endpoints are dropped here (see scripts/add_endpoint_slack.py) — add via the
# output-net node in v2 if PO-heavy designs (wb_dma) read out badly.

def _slog(x):
    """signed log1p: sign(x)*log1p(|x|). Handles both signs and crushes the fp_wns tail
    (floorplan slacks run to -85). Used for the floorplan-timing ANCHOR inputs."""
    return np.sign(x) * np.log1p(np.abs(x))

# ---------- feature pre-transform (BEFORE z-scoring) ----------
# Unbounded non-negative magnitudes get log1p first: they have brutal right tails
# (net fanout on ethernet: mean 3.2, MAX 10,016 — a clock net; out_cap_max: mean 72,
# max 2,050). Z-scoring those raw leaves a few nodes at +50 sigma and crushes the rest.
# Bounded/indicator dims (flags, fractions, drive strength, w/h, PE) are left alone.
#   cell_x 15: 0 w, 1 h, 2 n_in, 3 n_out, 4 seq, 5 inv, 6 buf, 7 fill, 8 diode,
#              9 drive, 10 in_cap, 11 out_cap, 12 leak, 13 area, 14 degree   (+10 PE = 25)
LOG_CELL = [9, 10, 11, 12, 13, 14]              # drive, caps, leakage, area, degree
LOG_NET  = [0]                                  # fanout

# Indicator dims are left as raw 0/1 — NEVER z-scored. z-scoring a rare binary divides
# by a tiny std and detonates: is_buf (0.004% of cells) hit +41 sigma, is_reset +80 sigma.
IDENT_CELL = [4, 5, 6, 7, 8]                    # is_seq/inv/buf/filler/diode
IDENT_NET  = [1, 2, 3]                          # is_io/is_clock/is_reset
IDENT_DF   = [4, 5, 6, 7, 8, 12, 13, 14]        # frac_* (already in [0,1])
PE_SLICE   = slice(15, 25)                      # the 10 Laplacian PE dims of cell_x
#   design_features 18 (insertion order in build_graph.py):
#     0 n_cells 1 n_nets 2 n_pins 3 total_cell_area 4-8 frac_* 9 fanout_mean
#     10 fanout_max 11 fanout_p90 12-14 frac_*pin 15 clock_fanout 16 n_clock 17 n_reset
LOG_DF   = [0, 1, 2, 3, 9, 10, 11, 15, 16, 17]  # counts / areas / fanout magnitudes

def _pre(a, idx):
    """log1p the heavy-tailed magnitude dims; leave flags/fractions/PE alone."""
    a = np.array(a, np.float32, copy=True)
    if a.ndim == 1:
        a[idx] = np.log1p(np.maximum(a[idx], 0))
    else:
        a[:, idx] = np.log1p(np.maximum(a[:, idx], 0))
    return a

def _stats(a, ident=()):
    """mean/std for z-scoring, with two guards.

    dead dims: a CONSTANT feature must be neutralized, not divided by ~1e-6.
      `height` is the same for every sky130 std cell (it IS the std-cell row height),
      is_filler/is_diode are all-zero at floorplan. The threshold is RELATIVE
      (s < 1e-6*(|m|+1)) — height's std is 2.4e-07 against a mean of 2.72, which an
      absolute 1e-6 threshold misses, leaving it to emit a constant 1.0 from float dust.
    ident dims: indicators pass through untouched (m=0, s=1) — see IDENT_* above.
    """
    m, s = a.mean(0), a.std(0)
    dead = s < 1e-6 * (np.abs(m) + 1.0)
    m = np.where(dead, 0.0, m); s = np.where(dead, 1.0, s + 1e-6)
    if len(ident):
        m[list(ident)] = 0.0; s[list(ident)] = 1.0
    return m.astype(np.float32), s.astype(np.float32), dead

def _cellfeat(d):
    """cell_x (15) ++ per-design-standardized Laplacian PE (10) -> (C,25).
    Single source of truth: set_norm() and load_graph() BOTH go through this, so the
    stats can never drift from what the model is actually fed."""
    return np.nan_to_num(np.concatenate([d["cell_x"], _pe_norm(d["pe_cell"])], 1))

def _pe_norm(pe):
    """Laplacian PE: passed through RAW, exactly as DE-HNN does. Do not "normalize" it.

    eigsh returns unit-L2 eigenvectors, so every entry is already bounded in [-1,1]
    (observed max 0.596) — they need no scaling. Two things we tried and reverted:
      * global z-score across designs  -> +61 sigma outliers (PE scale ~1/sqrt(n_cells),
        so a 48k-cell design and a 575-cell one are on different scales; pooling their
        stats is meaningless).
      * per-design std standardization -> +138 sigma. WORSE: eigenvectors on large sparse
        graphs localize on a few nodes, so dividing by std inflates exactly those spikes.
    Raw is bounded, cross-design-consistent, and what the paper feeds. Leave it alone.
    """
    return np.nan_to_num(np.asarray(pe, np.float32))

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

    cx, nx, df, ynh = [], [], [], []
    for dsg in sorted(train_designs):                       # stratified: every train design
        fl = sorted(glob.glob(f"{CACHE}/{dsg}-*.npz"))
        for p in fl[:2]:                                    # graph feats barely move with knobs
            d = np.load(p, allow_pickle=True)
            cx.append(_pre(_cellfeat(d), LOG_CELL))
            nx.append(_pre(d["net_x"], LOG_NET)); df.append(_pre(d["df_vals"], LOG_DF))
        for p in fl[::11][:10]:                             # per-net targets DO move with knobs
            d = np.load(p, allow_pickle=True)
            a = d["net_hpwl"];   ynh.append(np.log(a[np.isfinite(a) & (a > 0)]))
    cx, nx, df = np.concatenate(cx), np.concatenate(nx), np.stack(df)
    cx_m, cx_s, cx_d = _stats(cx, IDENT_CELL)
    nx_m, nx_s, nx_d = _stats(nx, IDENT_NET)
    df_m, df_s, df_d = _stats(df, IDENT_DF)
    cx_m[PE_SLICE] = 0.0; cx_s[PE_SLICE] = 1.0     # PE passes through RAW (see _pe_norm)
    _NORM = dict(cx_m=cx_m, cx_s=cx_s, nx_m=nx_m, nx_s=nx_s, df_m=df_m, df_s=df_s)
    n_dead = int(cx_d.sum() + nx_d.sum() + df_d.sum())
    if n_dead:
        print(f"[norm] {n_dead} constant/dead feature dims zeroed "
              f"(cell {list(np.where(cx_d)[0])}, net {list(np.where(nx_d)[0])}, "
              f"design {list(np.where(df_d)[0])})")

    # global targets: read straight off meta (free), train designs only
    m = meta(); tr = m.index.str.replace(r"-\d+$", "", regex=True).isin(set(train_designs))
    mt = m[tr]
    gl = dict(tot_hpwl=np.log(np.maximum(mt.total_hpwl.values, 1e-6)),
              buf_area=np.log(np.maximum(mt.buffer_area.values, 0) + 1.0),
              buf_cnt =np.log(np.maximum(mt.buffer_count.values, 0) + 1.0))
    ys = dict(net_hpwl=np.concatenate(ynh), **gl)
    for k in TARGETS:
        v = ys[k]; v = v[np.isfinite(v)]
        _NORM[f"y_{k}_m"] = np.float32(v.mean())
        _NORM[f"y_{k}_s"] = np.float32(v.std() + 1e-6)
    # per-ENDPOINT slack target: standardize raw slack over train designs' endpoint labels
    es = []
    for dsg in sorted(train_designs):
        for p in sorted(glob.glob(f"{ROOT}/cache/endpt/{dsg}-*.npz"))[::11][:10]:
            es.append(np.load(p)["ep_slack"])
    es = np.concatenate(es) if es else np.zeros(1, np.float32)
    _NORM["y_endpt_m"] = np.float32(es.mean()); _NORM["y_endpt_s"] = np.float32(es.std() + 1e-6)
    # floorplan-timing ANCHOR inputs (conditioning, not targets): signed-log + standardize
    for k in ("fp_wns", "fp_tns"):
        v = _slog(mt[k].values.astype(np.float32)); v = v[np.isfinite(v)]
        _NORM[f"a_{k}_m"] = np.float32(v.mean()); _NORM[f"a_{k}_s"] = np.float32(v.std() + 1e-6)
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
    # log1p the heavy-tailed magnitude dims, THEN z-score (train-fold stats). Must match set_norm().
    cell_x = (_pre(_cellfeat(d), LOG_CELL) - nm["cx_m"]) / nm["cx_s"]
    net_x  = (_pre(d["net_x"],  LOG_NET) - nm["nx_m"]) / nm["nx_s"]
    dfeat  = (_pre(d["df_vals"], LOG_DF) - nm["df_m"]) / nm["df_s"]
    # knobs + FLOORPLAN-TIMING ANCHOR (fp_wns/fp_tns). The anchor gives the model the
    # design's baseline timing LEVEL — the part that doesn't transfer cross-design and
    # tanked WNS to R²=-0.77 without it. Leakage-free: floorplan is BEFORE placement.
    a_wns = (_slog(float(m.fp_wns)) - nm["a_fp_wns_m"]) / nm["a_fp_wns_s"]
    a_tns = (_slog(float(m.fp_tns)) - nm["a_fp_tns_m"]) / nm["a_fp_tns_s"]
    knb    = np.array([m.clock_period/10.0, m.utilization/30.0, m.aspect_ratio,
                       a_wns, a_tns], np.float32)
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
        y_tot_hpwl=_z("tot_hpwl", t(np.log(max(m.total_hpwl, 1e-6)))),
        y_buf_area=_z("buf_area", t(np.log(max(m.buffer_area, 0) + 1.0))),
        y_buf_cnt =_z("buf_cnt",  t(np.log(max(bufcnt, 0) + 1.0))) if np.isfinite(bufcnt) else t(0.0),
        has_bufcnt=bool(np.isfinite(bufcnt)),
    )
    # per-ENDPOINT slack labels (register endpoints @ place_resized), standardized.
    # y_endpt is per-CELL (0 where no label); m_endpt masks the labeled endpoint cells.
    ep = np.load(f"{ROOT}/cache/endpt/{flow_id}.npz")
    y_ep = np.zeros(g["n_cells"], np.float32); mask = np.zeros(g["n_cells"], bool)
    if len(ep["ep_idx"]):
        idx = ep["ep_idx"]
        y_ep[idx] = (ep["ep_slack"] - nm["y_endpt_m"]) / nm["y_endpt_s"]
        mask[idx] = True
    g["y_endpt"] = t(y_ep); g["m_endpt"] = t(mask, torch.bool)
    g["ep_idx"] = torch.tensor(ep["ep_idx"], dtype=torch.long)   # labeled endpoint cells
    # raw recorded WNS/TNS — for eval readout comparison (complete, untruncated)
    g["wns_true"] = float(m.wns); g["tns_true"] = float(m.tns)
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
    def __init__(self, d=64, K=4, cell_in=25, net_in=4, knob=5, dfeat=18, encoder="dehnn"):
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
        # per-net head: DE-HNN-exact (reads net embedding directly).
        # GLOBAL head readout (OURS, not DE-HNN): mean+MAX pool of cells & nets, PLUS a direct
        # conditioning skip. Rationale — WNS is the WORST slack (an extreme, dominated by ONE
        # critical path) and TNS is a SUM; mean-pool alone washes the critical cell out among
        # ~48k others. max-pool exposes it. The ctx skip lets clock_period + the floorplan
        # anchor reach WNS/TNS directly instead of surviving VN->graph->pool dilution.
        self.fc1_net  = Linear(d, 256)
        self.fc1_cell = Linear(d + d, 256)            # per-cell readout: [cell embedding, ctx skip]
        self.fc1_glob = Linear(4*d + d, 256)          # [h.mean,h.max,hn.mean,hn.max, ctx]
        self.h_net_hpwl = Linear(256, 2)
        self.h_tot, self.h_buf_area, self.h_buf_cnt = Linear(256, 2), Linear(256, 2), Linear(256, 2)
        # per-ENDPOINT slack head (per-cell). WNS=min, TNS=sum-neg are READOUTS, not heads.
        # ctx skip so clock_period + the floorplan anchor reach each endpoint's slack directly.
        self.h_endpt = Linear(256, 2)

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
        hc = F.leaky_relu(self.fc1_cell(torch.cat([h, ctx.expand(h.size(0), -1)], 1)))  # per-cell
        hg = F.leaky_relu(self.fc1_glob(torch.cat([
            h.mean(0), h.max(0).values, h_net.mean(0), h_net.max(0).values, ctx])))
        return dict(net_hpwl=self.h_net_hpwl(hn),
                    tot_hpwl=self.h_tot(hg), buf_area=self.h_buf_area(hg), buf_cnt=self.h_buf_cnt(hg),
                    endpt=self.h_endpt(hc))          # per-cell slack; WNS/TNS read out in train

LOSS   = os.environ.get("LOSS", "decoupled")     # decoupled | beta | nll | mse
BETA   = float(os.environ.get("BETA", 0.5))      # for LOSS=beta
LAM_V  = float(os.environ.get("LAM_VAR", 1.0))   # weight on the variance term (decoupled)

def gnll(pred, y, mask=None, nll=True, mode=None):
    """Mean + variance, trained TOGETHER but DECOUPLED (the default).

        L = MSE(mu, y)  +  lam * NLL(stopgrad(mu), sigma^2, y)

    The whole pathology of Gaussian NLL is one term: d(NLL)/d(mu) = -(y-mu)/sigma^2.
    The mean's gradient is DIVIDED by the variance, so the model shrinks sigma on points
    it already fits, which amplifies exactly those points' gradient, which shrinks sigma
    further. Easy points drown out hard ones; the mean stops learning; then one bad val
    point gets divided by exp(-5) and the loss detonates. Observed twice:
        plain NLL:       ep1  train  -0.13  val  27.6
        + MSE warm-up:   ep20 train  -5.98  val  62.4  -> 161.9 / 228.1
    (the warm-up did NOT save it — mu was healthy at mse 0.24 when NLL engaged, proving
     the fault is the objective's shape, not a bad starting mean.)

    Decoupling removes that term outright:
      * mu   trains on pure MSE, every epoch, forever. No 1/sigma^2 factor exists in its
             gradient, so there is no runaway to fall into. Stable by construction.
      * sigma trains on NLL against a DETACHED mu, i.e. it learns to predict the residual
             magnitude (y-mu)^2 — which IS the calibrated uncertainty we want for the
             grounding loop. It cannot move mu, so it cannot sabotage it.
    No warm-up, no phase switch, no LR drop, no flatline. Both heads learn from step 0.

    Other modes kept for ablation: beta (Seitzer 2022), nll (broken, for the record), mse.
    """
    mode = (LOSS if mode is None else mode)
    if not nll: mode = "mse"                       # warm-up override, if anyone sets WARMUP
    mu, lv = pred[..., 0], pred[..., 1].clamp(-5, 5)
    if mask is not None:
        if mask.sum() == 0: return torch.zeros((), device=mu.device)
        mu, lv, y = mu[mask], lv[mask], y[mask]

    if mode == "mse":
        return 0.5 * (y - mu).pow(2).mean()

    if mode == "decoupled":
        mean_loss = 0.5 * (y - mu).pow(2).mean()               # trains mu ONLY
        se_d      = (y - mu.detach()).pow(2)                   # mu detached -> trains lv ONLY
        var_loss  = 0.5 * (lv + se_d / lv.exp()).mean()
        return mean_loss + LAM_V * var_loss

    se = (y - mu).pow(2)
    l  = 0.5 * (lv + se / lv.exp())
    if mode == "beta":
        l = l * lv.mul(BETA).exp().detach()        # stopgrad(sigma^(2*beta))
    return l.mean()

def loss_fn(out, g):
    L = (gnll(out["net_hpwl"], g["y_net_hpwl"], g["m_net_hpwl"])
       + gnll(out["tot_hpwl"], g["y_tot_hpwl"])
       + gnll(out["buf_area"], g["y_buf_area"])
       + gnll(out["endpt"], g["y_endpt"], g["m_endpt"]))
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
