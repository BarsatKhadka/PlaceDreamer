#!/usr/bin/env python3
"""
f_place — knob-conditioned DE-HNN (netlist + knobs -> placement state).

REPRODUCED FROM DE-HNN (`paperCodes/DEHNN/de_hnn/models/`), verified line-by-line:
  - HyperConvLayer: lin_node/lin_net + residual; driver(source)/sink split via SimpleConv;
    psi() for the net update, mlp() for the cell update; residual to the layer INPUT.
  - VN over CELLS only; init = virtualnode_encoder(concat[mean_pool, max_pool] of input feats);
    per layer: broadcast = back_mlp(concat[h, vn[part]]) + h  (BEFORE conv);
    then conv -> norm -> LeakyReLU; then (except the last layer) vn = mlp(concat[mean,max]) + vn.
  - edge dropout p=0.2 (train only, unlike DE-HNN which drops at eval too — theirs is the bug).

OUR DELIBERATE DEVIATIONS (all measured; see docs/fplace_audit.md):
  (a) knob+design context injected INTO THE VIRTUAL NODE, so it is broadcast every layer.
  (b) multi-task heads emitting (mu, logvar): per-net hpwl, per-cell endpoint slack, total hpwl,
      buffer area, buffer count. WNS/TNS are READ OUT from the endpoint head (min / sum-neg).
  (c) `type_emb` — a cell-type embedding (441 types) added to the node encoder. DE-HNN has none.
  (d) SIGN-INVARIANT PE (SignNet): eigenvectors have an arbitrary sign, so raw PE was NOISE
      across designs. DE-HNN does not fix this — they can afford not to (their targets are
      z-scored PER DESIGN and predicted per-node). We pool into ABSOLUTE cross-design scalars,
      so we cannot. This is a place where fidelity to DE-HNN would be WRONG for our task.
  (e) tapcells (34-58% of cells, zero signal pins, zero PE) are DROPPED. DE-HNN's designs have
      none; keeping them leaked design identity into the pooled readout.
  (f) global readout is mean+MAX pool + a direct ctx skip (WNS is a worst-case, not a mean).

NOT reproduced (and it matters): DE-HNN passes gcn_norm edge weights and drops nets with
fanout >= 3000. We do neither -> the clock net (fanout 10k) sends a message of norm ~68,672 vs
8.3 for a 2-pin net. See docs/fplace_audit.md A2. STILL OPEN.
"""
import os, glob, hashlib, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear, ReLU, LeakyReLU
from torch_geometric.nn import SimpleConv
from torch_geometric.utils import scatter, dropout_edge

# repo-relative so the same code runs on the laptop and on the cluster.
# override with PD_ROOT=/path/to/repo if the cache lives elsewhere (e.g. scratch).
ROOT  = os.environ.get("PD_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = f"{ROOT}/cache/graphs"
_META = _NORM = _T2N = None

# ---------- cell-type vocabulary ----------
# The sky130 standard_cells parquet has 5292 rows = the SAME 441 cells repeated 12x.
# scripts/build_graph.py's load_cell_lib() enumerated the ROWS, so cached cell_type ids are
# last-occurrence row indices (range 4858..5290) and the embedding was sized 5293 — of which
# 92% (310,528 params) never received a gradient and only 174 rows are ever used.
# Fix: a canonical 441-name vocabulary + a remap of the cached ids. Done at LOAD time so no
# graph re-cache (and no 970MB re-upload) is needed.
DIRECT_KNOB = bool(int(os.environ.get("DIRECT_KNOB") or "1"))  # A/B: raw knobs -> dev head
AGGR = os.environ.get("AGGR", "mean")   # A/B: sum | mean | multi | gcn  — see HyperConv
FUSE = os.environ.get("FUSE", "concat") # A/B: how the cell encoder fuses its 3 sources
VN_KNOBS = bool(int(os.environ.get("VN_KNOBS") or "1"))
# A/B FLAG — the SUPER-VN (DE-HNN's "two-level VN hierarchy"). AUDITED from their source:
# it is METIS at ONE granularity + ONE GLOBAL ROOT (pyg_dataset.py:109-111: top_part_id =
# zeros(num_vn), num_top_vn = 1) — NOT two METIS granularities, which is what we assumed.
# Their ablation is CUMULATIVE on single-design demand RMSE (Supp C.3 Table 7):
#     +PD 8.765 -> +single-VN 8.687 (0.9%) -> +two-level 8.381 (3.5%);  vs NO VN at all: 4.4%.
# We currently run METIS clusters with NO root — a config THEY NEVER TEST (their "single VN"
# is one GLOBAL VN, graph_conv_hetero.py:473 batch = zeros_like(batch)). So we are neither
# ablation row. Mechanism copied from graph_conv_hetero.py:386-391 (both levels init to
# CONSTANT ZERO), :497 (down: h += (vn + top[top_batch])[batch]), :538-543 (up: top = top +
# top_mlp(mean_pool(vn) + top)). Mean-pool only, no max. Updated only while l < K-1.
SUPER_VN = bool(int(os.environ.get("SUPER_VN") or "0"))
# The dehnn_novn sweep win likely means knobs-through-VN DILUTES them (LOSTIN warns of
# exactly this: a knob supernode -> 40.7%% MAPE, late-concat -> 3.11%%). VN_KNOBS=0 keeps
# the VN doing its STRUCTURAL job (long-range pooling) but stops routing knobs through it
# — the knobs still reach every head via the direct ctx skip. Tests: is it the VN that
# hurts, or just the knob DELIVERY through it?
# NOTE: DE-HNN uses NEITHER sum NOR mean. They use gcn_norm-weighted sum
# (train_all_cross.py:85-87): each edge weighted 1/sqrt(deg_i*deg_j). AGGR=gcn is that.
# bump when set_norm's CONTENT changes (new keys/stats) so stale caches invalidate
NORM_VERSION = 4   # +geometry feats, +wns_g/tns_g targets, +crit_path knob
N_TYPES = 442            # 441 real cell types + 1 UNK slot (index 441)
UNK_TYPE = 441

def type_remap():
    """old cached cell_type id (row index, 0..5291) -> dense id (0..440).
    Precomputed into cache/type_remap.npz by scripts/make_type_remap.py so this works on the
    cluster, which has cache/ but NOT the 71GB datasets/ dir."""
    global _T2N
    if _T2N is None:
        _T2N = np.load(f"{ROOT}/cache/type_remap.npz")["old2new"]
    return _T2N

def meta():
    global _META
    if _META is None: _META = pd.read_parquet(f"{ROOT}/cache/meta.parquet").set_index("flow_id")
    return _META

# f_place TARGETS (the placement state). Log-space then z-scored to ~N(0,1) so the heads are
# commensurate (raw log-means span 2.4 .. 11.2; unstandardized, tot_hpwl swamps the gradient).
#   per-net  : net_hpwl            (dense, ~10k/flow)
#   per-cell : endpt  (slack)      (dense, ~700/flow — WNS = min, TNS = sum-neg are READOUTS)
#   global   : tot_hpwl, buf_area, buf_cnt
# DROPPED: net_dem (per-net RUDY-from-bbox) — it was -0.94 correlated with net_hpwl (the same
# bounding box, twice). Real congestion is a per-TILE router quantity, not per-net, and lives
# on the data_gen path (EDA-Schema's routability_metrics table is EMPTY).
#
# KNOWN LABEL LIMITS (docs/fplace_audit.md B):
#   - net_hpwl labels are @global_place but the graph is @floorplan. Buffer insertion SPLITS
#     high-fanout nets in between; the name survives so nothing is masked, but the label then
#     describes a FRAGMENT. 0.41% of nets, carrying 4-6% of HPWL, median 20x the rest. OPEN.
#   - ~7% of total HPWL is on nets that have no floorplan node at all. So net_hpwl and tot_hpwl
#     cannot be made consistent; any sum-readout is biased low 0-20%, design-dependent. OPEN.
#   - endpoint labels come from timing_paths, a TOP-N truncated report: TNS coverage 9-100%
#     (ethernet 32%). WNS reconstructs EXACTLY (the worst path is always reported). So the
#     TNS readout undercounts by construction — a PERFECT endpoint predictor scores tns R2=0.34.
#   - primary-output endpoints are not cell nodes and are dropped (0% on the high-TNS designs;
#     50% on wb_dma, where the min-readout is consequently wrong). OPEN.
LOG_TARGETS    = ("net_hpwl", "tot_hpwl", "buf_area", "buf_cnt")
TARGETS        = LOG_TARGETS
# A/B FLAG. WNS_HEAD=0 reproduces the SHIPPED model (WNS/TNS exist only as READOUTS off the
# per-cell endpt head). WNS_HEAD=1 gives them real level+deviation heads.
# Fair-split evidence (cluster fold 0, raw-knob baseline, no OOD): the shipped readout scores
# wns -1.102 / tns -0.126 while a 3-knob OLS gets 0.091 / 0.118 — the model is WORSE THAN
# TRIVIAL, and endpt (its source) is our worst head (pooled R2 -0.508). This flag is the fix.
# A/B FLAG — what does the per-cell timing head predict?
#   "slack"   : the shipped target. Carries the CLOCK KNOB, so the head must learn structure
#               AND the knob dependence at once. Pooled R2 -0.508 (our worst head).
#   "arrival" : MasterRTL's decomposition (graph_stat.cal_timing):
#                   slack = require_time - arrival,   require_time = clock_period
#               arrival is STRUCTURAL — measured up to 18x more stable across knob configs
#               (per-endpoint CV: arrival 0.076 vs slack 1.364 on ac97). The head then has a
#               purely structural job and slack is recovered by EXACT ARITHMETIC.
ENDPT_TARGET = os.environ.get("ENDPT_TARGET", "slack")   # slack | arrival
WNS_HEAD = bool(int(os.environ.get("WNS_HEAD") or "1"))
# A/B FLAG. crit_path = clock_period - fp_wns = the design TIMING SCALE theta(G_D).
# Fair-split gain is SMALL: wns 0.091 -> 0.118, tns 0.118 -> 0.183 (not the 0.143 ->
# 0.419 first claimed, which came from a contaminated split). A probe showed NO model
# gain (+0.001). Flagged so the cluster decides rather than me.
CRIT_KNOB = bool(int(os.environ.get("CRIT_KNOB") or "1"))
KNOB_DIM  = 5 + (1 if CRIT_KNOB else 0)   # clk, util, AR, fp_wns, fp_tns [, crit_path]
GLOBAL_TARGETS = (("tot_hpwl", "buf_area", "buf_cnt") + (("wns_g", "tns_g") if WNS_HEAD else ()))
# each -> a LEVEL head + a DEVIATION head.
#
# wns_g/tns_g ADDED — and this is the biggest measured miss in f_place. WNS/TNS used to exist
# ONLY as READOUTS off the per-cell endpt head (WNS=min, TNS=sum-neg), i.e. we derived timing
# from our single worst prediction: endpt scores pooled R2 -0.508, ~100% rel err, calib z^2 9.63.
# Result: f_place's wns knob-response was -1.102 and tns -0.126 -- WORSE THAN A CONSTANT.
# But the signal is right there and trivially available:
#     OLS on the 3 raw knobs alone -> wns within-design R2 0.649, tns 0.657  (clock_period)
# So a 3-parameter linear model beat our GNN by 1.75 R2 on wns. That was never "timing does not
# transfer cross-design" (a claim we repeated and which is false) -- it was an ARCHITECTURE gap:
# no head, no direct knob path, derived from the broken endpt readout. Give them the same
# level+deviation heads every other global target has, and the dev head already receives the raw
# knobs directly (DIRECT_KNOB), which is exactly where the 0.649 lives.
SIGNED_TARGETS = ("wns_g", "tns_g")     # signed -> signed-log transform, not log

# A/B FLAG: how is tot_hpwl produced?
#   "pool" — the pooled mean/max readout -> MLP -> level+dev heads   (what we have)
#   "sum"  — tot_hpwl = SUM_net HPWL_net, an IDENTITY, not a model. The GNN predicts the per-net
#            unknown (where it has ~1e6 labels and scores AUC 0.912) and an exact formula
#            aggregates. This is GRANNITE's verified pattern (DAC'20): keep P = sum(a*C*V^2*f)
#            exact, have the net predict only the per-gate alpha. <5.5% err, 18.7x speedup.
#
# WHY: the pooled head has ~1e2 effective labels (and since the input graph is IDENTICAL across a
# design's 108 knob configs, the LEVEL task has just 18 — one per distinct graph). Measured on a
# 182-flow probe, the pooled head DIVERGES (rel-err 29%->170%, knob-R2 -1->-9.6) while the sum
# stays flat at ~15-20% rel-err and beats it at every checkpoint. The full-data pooled head gets
# 52% rel-err; this sum gets 17.4% on a toy set.
# Theory agrees it should work: tot_hpwl is a smooth aggregate of LOCAL quantities, squarely
# inside the 1-WL-decomposable class (Xu GIN ICLR'19; Chen NeurIPS'20), so the failure was never
# the representation — it was aggregation + sample size. And MasterRTL (ICCAD'23) ran exactly our
# pooled-GCN experiment on WNS/TNS/power/area and shipped XGBoost instead.
HPWL_COMPOSE = os.environ.get("HPWL_COMPOSE", "pool")   # pool | sum
# (endpt stays: it is the per-cell timing state the seam forwards, and it is dense supervision.
#  It is simply no longer the ONLY path to WNS/TNS.)

def recon(k, lvl, dev, nm, w=None):
    """level + deviation -> log(target). Inverse of the decomposition in set_norm.
    `w` = that design's OWN within-std (g[f"w_{k}"]); falls back to the pooled mean when the
    design is unknown (e.g. a truly novel design at deploy time)."""
    if w is None: w = float(nm[f"W_{k}"])
    return (lvl * float(nm[f"L_{k}_s"]) + float(nm[f"L_{k}_m"])) + dev * float(w)

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
# Indices below are in the RAW cache layout; they are remapped to the post-drop layout at the
# bottom of this block (see KEEP_CELL/KEEP_DF in _cellfeat).
_LOG_CELL_RAW = [9, 10, 11, 12, 13, 14]          # drive, caps, leakage, area, degree

# Indicator dims are left as raw 0/1 — NEVER z-scored. z-scoring a rare binary divides
# by a tiny std and detonates: is_buf (0.004% of cells) hit +41 sigma, is_reset +80 sigma.
_IDENT_CELL_RAW = [4, 5, 6]                      # is_seq/inv/buf   (filler/diode are DEAD)
IDENT_NET       = [1, 2, 3]                      # is_io/is_clock/is_reset

# NET features = [fanout, is_io, is_clock, is_reset] ++ 14 Net2 PARTITION features.
# Net2 (ASP-DAC'21): 725 nets with IDENTICAL local features had lengths spanning 1um..100um —
# a net model with only local features "cannot distinguish them at all". Cells in DIFFERENT
# clusters get placed FAR APART, so cluster disagreement is a pre-placement proxy for physical
# distance = exactly what HPWL is, and what fanout can never express.
# 7 METIS granularities x {span, disagreement-fraction}. Verified on our data: top-10%-longest-net
# AUC 0.864 -> 0.937 (ethernet), mean 0.918 vs Net2's reported 0.922.
N_PART_NET  = 14
# ELECTRICAL net features (3): drive_strength, load/drive ratio, max sink cap.
# Buffers are inserted for CAP/SLEW violations, NOT fanout (Kahng ISPD'26 §2.2). A driver of
# strength D driving load C has slew ~ C/D; exceed the limit and the resizer INSERTS A BUFFER.
# So load/drive is literally the trigger condition, and we fed the model nothing about it.
# MEASURED first: driven_cap alone is 96% correlated with fanout (redundant, NOT added);
# load/drive is only 62% correlated with fanout => ~38% new information. That is the one.
N_ELEC_NET  = 3
NET_IN      = 4 + N_PART_NET + N_ELEC_NET         # 21
_PART0, _ELEC0 = 4, 4 + N_PART_NET
LOG_NET     = ([0]                                   # fanout
               + list(range(_PART0, _PART0 + 14, 2)) # the 7 partition SPAN dims (frac is [0,1])
               + [_ELEC0, _ELEC0 + 1, _ELEC0 + 2])   # drive, load/drive, max_cap (heavy tails)
#   design_features 18 (insertion order in build_graph.py):
#     0 n_cells 1 n_nets 2 n_pins 3 total_cell_area 4-8 frac_* 9 fanout_mean
#     10 fanout_max 11 fanout_p90 12-14 frac_*pin 15 clock_fanout 16 n_clock 17 n_reset
_IDENT_DF_RAW = [4, 5, 6, 12, 13, 14]            # frac_* in [0,1]  (frac_filler/diode DEAD)
_LOG_DF_RAW   = [0, 1, 2, 3, 9, 10, 11, 15, 16, 17]   # counts / areas / fanout magnitudes

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
    # float64: a float32 mean(0) over ~343k rows accumulates enough error to inflate a truly
    # constant column's std (height: true 4.8e-07 -> 0.0125), so the guard below never fired
    # and height was z-scored by float dust. Verified.
    a = np.asarray(a, np.float64)
    m, s = a.mean(0), a.std(0)
    dead = s < 1e-6 * (np.abs(m) + 1.0)
    m = np.where(dead, 0.0, m); s = np.where(dead, 1.0, s + 1e-6)
    if len(ident):
        m[list(ident)] = 0.0; s[list(ident)] = 1.0
    return m.astype(np.float32), s.astype(np.float32), dead

# DEAD input dims — verified identically constant across ALL 1944 flows, so they carry zero
# information and are dropped (not merely zeroed):
#   cell_x[1]  height     — every sky130 std cell has the same height (it IS the row height)
#   cell_x[7]  is_filler  — fillers are not inserted until after placement
#   cell_x[8]  is_diode   — likewise
#   df_vals[7] frac_filler, df_vals[8] frac_diode — same two, at design level
DEAD_CELL = [1, 7, 8]
DEAD_DF   = [7, 8]
KEEP_CELL = [i for i in range(15) if i not in DEAD_CELL]     # 12 real library dims
KEEP_DF   = [i for i in range(18) if i not in DEAD_DF]       # 16 real design dims

# remap the RAW-layout index lists into the post-drop layout (single source of truth)
_c = {raw: new for new, raw in enumerate(KEEP_CELL)}
_f = {raw: new for new, raw in enumerate(KEEP_DF)}
LOG_CELL   = [_c[i] for i in _LOG_CELL_RAW]
IDENT_CELL = [_c[i] for i in _IDENT_CELL_RAW]
LOG_DF     = [_f[i] for i in _LOG_DF_RAW]
IDENT_DF   = [_f[i] for i in _IDENT_DF_RAW]
CELL_IN    = len(KEEP_CELL) + 10          # 12 library + 10 PE = 22
# A/B FLAG. GEO_FEATS=0 reproduces the shipped 16 netlist-only design features.
# MEASURED (fair fold-0 split, no OOD): die_area adds NOTHING to knob response
# (0.702 -> 0.702) and never could — within a design log(die)=log(cell_area)-log(util)
# and cell_area is design-constant, so it is COLLINEAR with the utilization knob.
# Kept flagged only because it may still help the cross-design LEVEL (untested).
GEO_FEATS  = bool(int(os.environ.get("GEO_FEATS") or "1"))
# A/B FLAG — do the GEOMETRY HEADS (h_pos per-cell x/y, h_vnbox per-METIS-cluster bbox) exist?
# They were UNGATED and therefore training in every arm, and they are MEASURED DEAD: "GEO skill"
# (1 - err/trivial-baseline) is NEGATIVE every epoch at FULL scale (d=64/K=4, 756 flows) —
# box -0.001..-0.062, pos -0.064..-0.172. Yet at W_POS=5 (x2 terms) and W_VNBOX=5 (x4 terms)
# they were consuming 68.2% OF THE ENTIRE LOSS while net_hpwl — the head that works (AUC
# 0.912) — got 11.4%. And they share the encoder, so they shape the representation toward a
# task that does not learn. Default OFF: dead heads are not free.
GEO_HEADS  = bool(int(os.environ.get("GEO_HEADS") or "0"))
DF_IN      = len(KEEP_DF) + (2 if GEO_FEATS else 0)   # 16 netlist + 2 floorplan geometry
                                          # (log die_area, log sqrt(n*A) physics prior)
PE_SLICE   = slice(len(KEEP_CELL), CELL_IN)   # the 10 PE dims, post-drop

def live_cells(d):
    """Boolean mask of cells with at least one signal pin.

    34-58% of cells are TAP_TAPCELL_ROW_* — physical-only tap cells with ZERO signal pins.
    Verified: 100% of degree-0 cells are tapcells, they carry 0 endpoint labels, and their
    Laplacian PE is EXACTLY zero (max |PE| ~1e-17). They are pure dead weight, and worse:
    the FRACTION of them varies 34%->58% across designs, so mean-pooling the PE block
    encodes "what fraction of this design is tapcells" — a design-identity leak straight
    into the global readout. Dropped at load time (no graph re-cache needed).
    """
    C = len(d["cell_type"])
    deg = np.zeros(C, np.int64)
    for arr in (d["edge_driver"], d["edge_sink"]):
        np.add.at(deg, np.asarray(arr)[0], 1)
    return deg > 0

def _cellfeat(d, keep=None):
    """cell_x (15 -> 12 live dims) ++ Laplacian PE (10) -> (C,22), tapcells dropped.
    Single source of truth: set_norm() and load_graph() BOTH go through this, so the
    stats can never drift from what the model is actually fed."""
    if keep is None: keep = live_cells(d)
    cx = np.asarray(d["cell_x"])[keep][:, KEEP_CELL]
    return np.nan_to_num(np.concatenate([cx, _pe_norm(np.asarray(d["pe_cell"])[keep])], 1))

def _dffeat(d):
    """design_features (18 -> 16 live dims)."""
    return np.nan_to_num(np.asarray(d["df_vals"])[KEEP_DF])

def _netfeat(d, flow_id):
    """net_x (4) ++ Net2 partition features (14) -> (N, 18).
    The graph is identical across a design's 108 flows, so the partition is computed ONCE per
    design (scripts/add_partition_features.py) and shared."""
    dsg = flow_id.rsplit("-", 1)[0]
    pf  = np.load(f"{ROOT}/cache/part/{dsg}.npz")["part_net"]
    ef  = np.load(f"{ROOT}/cache/elec/{dsg}.npz")["elec_net"]
    nx  = np.asarray(d["net_x"])
    assert len(pf) == len(nx) == len(ef), f"{flow_id}: part {len(pf)} elec {len(ef)} nets {len(nx)}"
    return np.nan_to_num(np.concatenate([nx, pf, ef], 1))

def _pe_norm(pe):
    """Laplacian PE, RMS-normalized per design. Sign is handled by the model (SignNet), NOT here.

    Two separate pathologies, both measured:
      (1) SIGN AMBIGUITY. eigsh returns eigenvectors with an ARBITRARY sign (and arbitrary
          rotation inside degenerate eigenspaces). Across reruns, 4-8 of our 10 dims flip.
          So the "same" structural position encodes differently in different designs.
          => the model must be sign-INVARIANT. Handled in FPlace.pe_encoder (SignNet:
          phi(v) + phi(-v), which is identical for v and -v by construction).
          NOTE: DE-HNN does NOT fix this anywhere in their codebase — but their targets are
          z-scored PER DESIGN and predicted per-node, so a per-design random sign costs them
          little. We pool into ABSOLUTE cross-design scalars, so it costs us everything.
          We are outside what DE-HNN validated; copying them faithfully would not save us.
      (2) SCALE. Entries scale ~1/sqrt(C), so RMS spans 0.0045 (ethernet, 48k cells) to
          0.042 (sasc, 575) — a 10x spread. DE-HNN's designs are all ~1M cells so this never
          bit them. RMS-normalize per design (NOT std-normalize: an earlier attempt divided by
          std and inflated the localized spikes to +138 sigma).
    """
    pe = np.nan_to_num(np.asarray(pe, np.float32))
    rms = np.sqrt((pe ** 2).mean(0, keepdims=True))
    return pe / (rms + 1e-8)

def set_norm(train_designs, force=False):
    """Build feature + target normalization from TRAINING designs ONLY.

    Two things this fixes vs. the old norm():
      * old code used sorted(glob)[:40], which is 40 flows of ONE design (ac97_ctrl) —
        every design got z-scored by one small design's statistics.
      * stats computed over all designs leak test/OOD statistics into training.
    Call this ONCE per fold, before any load_graph().
    """
    global _NORM
    # STABLE key. This used to be hash(tuple(...)), but Python randomizes string hashing per
    # process (PYTHONHASHSEED), so the same train split produced a DIFFERENT filename every run:
    # the cache never hit across processes, every run silently rebuilt, and cache/ filled with
    # orphan norm_*.npz. It also hid the allow_pickle bug below, which only fires when the cache
    # actually loads. NORM_VERSION invalidates stale caches whenever the norm CONTENT changes
    # (bump it when you add/alter a key here) — otherwise a working cache would serve a norm
    # missing newly-added entries and fail far away with a confusing KeyError.
    key = hashlib.md5(("|".join(sorted(train_designs)) + f"|v{NORM_VERSION}").encode()).hexdigest()[:8]
    f = f"{ROOT}/cache/norm_{key}.npz"
    if not force and os.path.exists(f):
        # allow_pickle=True is REQUIRED, not cosmetic: set_norm saves per-design key arrays
        # (MU_{k}_keys / W_{k}_keys come from a pandas index of design names -> object dtype).
        # Loading them without it raises "Object arrays cannot be loaded when allow_pickle=False",
        # so ANY run that hits an existing norm cache dies — e.g. re-running a fold, or f_cts
        # reusing the norm f_place just built for the same train split.
        _NORM = dict(np.load(f, allow_pickle=True)); return _NORM

    cx, nx, df, ynh = [], [], [], []
    for dsg in sorted(train_designs):                       # stratified: every train design
        fl = sorted(glob.glob(f"{CACHE}/{dsg}-*.npz"))
        for p in fl[:2]:                                    # graph feats barely move with knobs
            d = np.load(p, allow_pickle=True)
            cx.append(_pre(_cellfeat(d), LOG_CELL))
            nx.append(_pre(_netfeat(d, os.path.basename(p)[:-4]), LOG_NET)); df.append(_pre(_dffeat(d), LOG_DF))
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
    _mt = meta()
    for dsg in sorted(train_designs):
        for p in sorted(glob.glob(f"{ROOT}/cache/endpt/{dsg}-*.npz"))[::11][:10]:
            _v = np.load(p)["ep_slack"]
            if ENDPT_TARGET == "arrival":        # stats must match the TARGET (see ENDPT_TARGET)
                _fid = os.path.basename(p)[:-4]
                if _fid in _mt.index: _v = float(_mt.loc[_fid].clock_period) - _v
            es.append(_v)
    es = np.concatenate(es) if es else np.zeros(1, np.float32)
    _NORM["y_endpt_m"] = np.float32(es.mean()); _NORM["y_endpt_s"] = np.float32(es.std() + 1e-6)
    # FLOORPLAN GEOMETRY feature stats (train designs only) — see load_graph. die = cell_area/util
    # and sqrt(n*A) are both pre-placement, so these are inputs, not leakage.
    _m = meta()                       # NOTE: m_all is only bound further down; use meta() here
    gd, gs = [], []
    for dsg in sorted(train_designs):
        for p in sorted(glob.glob(f"{CACHE}/{dsg}-*.npz"))[::7][:16]:
            fid = os.path.basename(p)[:-4]
            if fid not in _m.index: continue
            dv = np.asarray(np.load(p, allow_pickle=True)["df_vals"])
            die = float(dv[3]) / max(float(_m.loc[fid, "utilization"]) / 100.0, 1e-6)
            gd.append(np.log(die)); gs.append(0.5 * np.log(max(float(dv[0]), 1.0) * die))
    gd, gs = np.array(gd), np.array(gs)
    _NORM["g_die_m"] = np.float32(gd.mean()); _NORM["g_die_s"] = np.float32(gd.std() + 1e-6)
    _NORM["g_snA_m"] = np.float32(gs.mean()); _NORM["g_snA_s"] = np.float32(gs.std() + 1e-6)

    # GEOMETRY BASELINE (train designs only): the mean VN box. This is the "learned nothing"
    # predictor the geometry head must BEAT — boxes are normalized by the die, so every design
    # is already on the same [0,1] scale and a single mean box is a fair, honest baseline.
    bxs = []
    for dsg in sorted(train_designs):
        for p in sorted(glob.glob(f"{ROOT}/cache/coords/{dsg}-*.npz"))[::11][:6]:
            c = np.load(p); gp = f"{CACHE}/{os.path.basename(p)}"
            if not os.path.exists(gp): continue
            d_ = np.load(gp, allow_pickle=True); kp = live_cells(d_)
            pt = np.asarray(d_["part_cell"])[kp]; _, pt = np.unique(pt, return_inverse=True)
            x_, y_, mk_ = c["x"][kp], c["y"][kp], c["mask"][kp]
            for q in range(int(pt.max()) + 1):
                s = (pt == q) & mk_
                if s.sum() >= 5: bxs.append((x_[s].min(), y_[s].min(), x_[s].max(), y_[s].max()))
    _NORM["vnbox_m"] = (np.mean(bxs, 0) if bxs else np.array([.25, .25, .75, .75])).astype(np.float32)

    # ---- GLOBAL TARGET DECOMPOSITION:  log(y) = DESIGN LEVEL + KNOB DEVIATION ----
    # THE bug that made within-R2 = -14. We z-scored by the CROSS-design std, but the knob
    # effect lives inside the WITHIN-design std, and those differ by 12-33x:
    #     tot_hpwl  cross 1.764  within 0.152  -> knob signal arrives as +-0.086 std units
    #     buf_area  cross 1.514  within 0.074  -> +-0.049
    #     buf_cnt   cross 1.190  within 0.036  -> +-0.030
    # i.e. the ONLY thing f_place exists to predict was ~1% of the target's variance, and the
    # optimizer (correctly) ignored it and nailed the design level instead.
    # Fix: predict the two parts SEPARATELY, each standardized to O(1):
    #     level_k = (mean_of_design - M_k) / S_k          <- easy; ~n_cells
    #     dev_k   = (y - mean_of_design) / W_k            <- THE KNOB RESPONSE, now O(1)
    # reconstruct: log(y) = (level*S_k + M_k) + dev*W_k
    # mu_design is a LABEL statistic used only to FORM the targets — the model never sees it,
    # so there is no leakage. (Test-design means are used only inside evaluate(), which is what
    # "within-design error" means by definition.)
    m_all = meta()
    dcol  = m_all.index.str.replace(r"-\d+$", "", regex=True)
    for k, col, off in (("tot_hpwl","total_hpwl",0.0), ("buf_area","buffer_area",1.0),
                        ("buf_cnt","buffer_count",1.0), ("wns_g","wns",0.0), ("tns_g","tns",0.0)):
        v_ = m_all[col].values.astype(np.float64)
        # wns/tns are SIGNED (slack is negative when violating) -> signed-log, same transform the
        # endpt target and f_cts's cts_wns/cts_tns already use. log() would silently clamp them.
        y = (np.sign(v_) * np.log1p(np.abs(v_)) if k in SIGNED_TARGETS
             else np.log(np.maximum(v_, 0) + off + 1e-12))
        s = pd.Series(y, index=dcol)
        mu_d = s.groupby(level=0).mean()                       # per-design LEVEL (all designs)
        w_d  = s.groupby(level=0).std()                        # per-design WITHIN std
        tr_d = [d for d in mu_d.index if d in set(train_designs)]
        _NORM[f"L_{k}_m"] = np.float32(mu_d[tr_d].mean())      # M_k  (train designs only)
        _NORM[f"L_{k}_s"] = np.float32(mu_d[tr_d].std() + 1e-6)  # S_k

        # PER-DESIGN within-std, not a pooled one. Dividing every design's deviation by ONE
        # pooled W silently weighted the loss by design: per-design within-std varies 6-10x, so
        # ethernet's buf_cnt deviation target came out at std 1.89 while mem_ctrl's was 0.18 —
        # and MSE squares that, so ethernet contributed ~100x the deviation gradient. Nobody
        # chose that. Per-design W => every design's deviation target has std ~1 and every
        # design contributes equally to the knob-response.
        _NORM[f"W_{k}_keys"] = np.array(w_d.index)
        _NORM[f"W_{k}_vals"] = np.maximum(w_d.values, 1e-3).astype(np.float32)
        _NORM[f"W_{k}"]      = np.float32(w_d[tr_d].mean() + 1e-6)   # kept: used to reconstruct

        # DEGENERATE designs: some chips are so small the resizer inserts the same handful of
        # buffers no matter what the knobs say. usb_phy has TWO distinct buffer_area values
        # across all 108 flows; ss_pcm has 3; simple_spi sits at one value in 91 of 108. There
        # is no knob response to learn OR to score — including them made R2 a division by
        # ~zero, which is the whole reason buf_area_dev read exactly +0.000.
        degen = (w_d < 0.03).values                            # <3% log-spread across 108 knobs
        _NORM[f"DEG_{k}_keys"] = np.array(w_d.index)
        _NORM[f"DEG_{k}_vals"] = degen
        _NORM[f"MU_{k}_keys"]  = np.array(mu_d.index)          # per-design level lookup
        _NORM[f"MU_{k}_vals"]  = mu_d.values.astype(np.float32)
    # floorplan-timing ANCHOR inputs (conditioning, not targets): signed-log + standardize
    for k in ("fp_wns", "fp_tns"):
        v = _slog(mt[k].values.astype(np.float32)); v = v[np.isfinite(v)]
        _NORM[f"a_{k}_m"] = np.float32(v.mean()); _NORM[f"a_{k}_s"] = np.float32(v.std() + 1e-6)
    # KNOBS: properly z-scored (train designs only). They used to be divided by ad-hoc constants
    # I invented (clock/10, util/30, aspect raw) -> utilization arrived with std ~0.27 and a mean
    # of ~1.0, sitting next to unit-variance z-scored anchors, so the ctx MLP was dominated by
    # the anchors and the knobs were a small offset. Utilization is the knob that drives the
    # target hardest (corr -0.75..-0.86 with log tot_hpwl). It must not be the quietest input.
    for k in ("clock_period", "utilization", "aspect_ratio"):
        v = mt[k].values.astype(np.float32)
        _NORM[f"k_{k}_m"] = np.float32(v.mean()); _NORM[f"k_{k}_s"] = np.float32(v.std() + 1e-6)
    # design TIMING SCALE stats (train designs only) — crit_path = clock_period - fp_wns.
    cv = np.log(np.maximum(mt["clock_period"].values.astype(np.float64)
                           - mt["fp_wns"].values.astype(np.float64), 1e-3))
    cv = cv[np.isfinite(cv)]
    _NORM["k_crit_m"] = np.float32(cv.mean()); _NORM["k_crit_s"] = np.float32(cv.std() + 1e-6)
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
    # cached ids are row indices into the 12x-repeated parquet -> remap to the dense 441-type
    # vocabulary (see type_remap). Unknown (-1) -> UNK slot.
    # DROP TAPCELLS (34-58% of cells, zero signal pins, zero PE, zero labels — see live_cells).
    # Every cell-indexed array must be remapped into the compacted space.
    keep = live_cells(d)
    old2new = np.full(len(keep), -1, np.int64)
    old2new[keep] = np.arange(int(keep.sum()))
    ct = d["cell_type"].astype(np.int64)[keep]
    ct = np.where(ct < 0, UNK_TYPE, type_remap()[np.clip(ct, 0, None)])
    ed_np = np.asarray(d["edge_driver"]).copy(); ed_np[0] = old2new[ed_np[0]]
    es_np = np.asarray(d["edge_sink"]).copy();   es_np[0] = old2new[es_np[0]]
    assert ed_np[0].min() >= 0 and es_np[0].min() >= 0, "an edge referenced a dropped cell"
    ed = torch.tensor(ed_np, dtype=torch.long)
    es = torch.tensor(es_np, dtype=torch.long)
    t  = lambda a, dt=torch.float: torch.tensor(np.asarray(a), dtype=dt)
    # log1p the heavy-tailed magnitude dims, THEN z-score (train-fold stats). Must match set_norm().
    cell_x = (_pre(_cellfeat(d, keep), LOG_CELL) - nm["cx_m"]) / nm["cx_s"]
    net_x  = (_pre(_netfeat(d, flow_id), LOG_NET) - nm["nx_m"]) / nm["nx_s"]
    dfeat  = (_pre(_dffeat(d), LOG_DF) - nm["df_m"]) / nm["df_s"]
    # ---- FLOORPLAN GEOMETRY + PHYSICS PRIOR (leakage-free: both known BEFORE placement) ----
    # The 18 design features are all netlist-derived (n_cells, total_cell_area, fanouts...) —
    # NOTHING told the model how big the die is. But the floorplan sets die = cell_area/util, so
    # the model had to learn a division to recover it. Measured knob-response ceilings (OLS,
    # within-design): tot_hpwl 0.719 from knobs alone -> 0.857 once die_area is available.
    # f_place actually scores 0.654, so this feature is worth ~0.2 R2 and costs nothing.
    # Leak check: log(cell_area/util) vs the die area measured from the PLACED cell bbox
    # correlates R2 0.9961 — it genuinely is a pre-placement quantity, not placement info.
    _cell_area = float(np.asarray(d["df_vals"])[3])                  # total_cell_area (synthesis)
    _die = _cell_area / max(float(m["utilization"]) / 100.0, 1e-6)   # floorplan identity
    # sqrt(n*A): the Beardwood-Halton-Hammersley / Rent scaling for total edge length over n
    # points in area A. VERIFIED in the CTS literature as the clock-WL law (Charikar et al.,
    # SIAM J. Discrete Math 2004: 1.5*sqrt(n) on a unit grid; Han/Kahng/Li TCAD'18 rederive
    # sqrt(N*A) from H-tree algebra). Handing the model the physics form instead of making it
    # discover sqrt of a product from log features.
    _geo = np.array([(np.log(_die)          - nm["g_die_m"])  / nm["g_die_s"],
                     (0.5*np.log(max(float(np.asarray(d["df_vals"])[0]),1.0) * _die)
                                            - nm["g_snA_m"])  / nm["g_snA_s"]], np.float32)
    if GEO_FEATS: dfeat = np.concatenate([dfeat, _geo])
    # knobs + FLOORPLAN-TIMING ANCHOR (fp_wns/fp_tns). The anchor gives the model the
    # design's baseline timing LEVEL — the part that doesn't transfer cross-design and
    # tanked WNS to R²=-0.77 without it. Leakage-free: floorplan is BEFORE placement.
    a_wns = (_slog(float(m.fp_wns)) - nm["a_fp_wns_m"]) / nm["a_fp_wns_s"]
    a_tns = (_slog(float(m.fp_tns)) - nm["a_fp_tns_m"]) / nm["a_fp_tns_s"]
    kz = lambda k: (float(m[k]) - float(nm[f"k_{k}_m"])) / float(nm[f"k_{k}_s"])
    # DESIGN TIMING SCALE — theta(G_D) made explicit. THE point: a knob is only meaningful
    # RELATIVE to the design's own scale. clock_period is design-specific (each design's clock is
    # set from its critical path: ac97 1.8-3.0ns, des3_area 6.0-9.3ns), so "3ns" is tight for one
    # and loose for another. The knob response is an INTERACTION y_dev = f(k ; theta(G_D)), and
    # without theta the model cannot decode the knob at all.
    # crit_path ~= clock_period - fp_wns  (floorplan timing => PRE-placement => leak-free).
    # MEASURED (raw knobs -> +crit_path, held-out designs, within-design knob response):
    #     cts_power  0.143 -> 0.419      cts_wns  0.121 -> 0.372      cts_buffers 0.043 -> 0.043
    # i.e. ~3x on the timing targets, and correctly NOTHING on buffers (not timing-driven).
    # It recovers ~43% of the gap to an oracle that is handed mean_design(clock_period) (0.970).
    # Goes in the KNOB vector, NOT dfeat: the deviation head reads the raw knobs DIRECTLY
    # (DIRECT_KNOB), while dfeat only reaches it smeared through ctx and K message-passing layers.
    # Fed explicitly rather than left to be derived: it is a subtraction of two differently
    # transformed, z-scored quantities (Delta-ML — give the model the prior's derivation inputs,
    # not just the prior: R2 81.88% -> 99.15%).
    crit = max(float(m.clock_period) - float(m.fp_wns), 1e-3)
    a_crit = (np.log(crit) - nm["k_crit_m"]) / nm["k_crit_s"]
    knb = np.array(([kz("clock_period"), kz("utilization"), kz("aspect_ratio"), a_wns, a_tns]
                    + ([a_crit] if CRIT_KNOB else [])), np.float32)   # unit-variance, zero-mean
    # VN partition, compacted to the surviving cells. Re-densify the partition ids: dropping
    # tapcells can empty a partition, and scatter(dim_size=max+1) would leave a hole.
    pc = np.asarray(d["part_cell"])[keep]
    _, pc = np.unique(pc, return_inverse=True)
    part_c = torch.tensor(pc, dtype=torch.long)                  # DE-HNN: VN over CELLS
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
        # mask = finite AND positive AND the label is TRUSTWORTHY. The graph is @floorplan but
        # labels are @global_place; buffer insertion SPLITS high-fanout nets in between, and the
        # name survives so name-matching silently attached a FRAGMENT's HPWL to the full net.
        # 0.41% of nets — but they carry 4-6% of total HPWL, median 20x the rest, i.e. exactly
        # the highest-leverage nets in the per-net head. cache/netmask marks them. (audit B2)
        m_net_hpwl=t(np.isfinite(d["net_hpwl"]) & (d["net_hpwl"] > 0)
                     & np.load(f"{ROOT}/cache/netmask/{flow_id}.npz")["label_ok"], torch.bool),
        has_bufcnt=bool(np.isfinite(bufcnt)),
    )
    # GLOBAL targets, DECOMPOSED (see set_norm): log(y) = design_level + knob_deviation.
    # Both parts standardized to O(1) — the knob deviation used to arrive as a +-0.03..0.09
    # wiggle on a unit-variance target (1% of the variance) and the optimizer ignored it.
    dsg  = flow_id.rsplit("-", 1)[0]
    raws = dict(tot_hpwl=np.log(max(m.total_hpwl, 1e-6)),
                buf_area=np.log(max(m.buffer_area, 0) + 1.0),
                buf_cnt =np.log(max(bufcnt, 0) + 1.0) if np.isfinite(bufcnt) else np.nan,
                wns_g   =_slog(float(m.wns)),      # SIGNED -> signed-log (see GLOBAL_TARGETS)
                tns_g   =_slog(float(m.tns)))
    for k in GLOBAL_TARGETS:
        i    = int(np.where(nm[f"MU_{k}_keys"] == dsg)[0][0])
        mu_d = float(nm[f"MU_{k}_vals"][i])                      # this design's LEVEL
        w_d  = float(nm[f"W_{k}_vals"][i])                       # this design's OWN within-std
        deg  = bool(nm[f"DEG_{k}_vals"][i])                      # knobs don't move it at all
        yv   = float(raws[k])
        g[f"y_{k}_lvl"] = t((mu_d - float(nm[f"L_{k}_m"])) / float(nm[f"L_{k}_s"]))
        g[f"y_{k}_dev"] = t((yv - mu_d) / w_d) if np.isfinite(yv) else t(0.0)
        g[f"w_{k}"]     = w_d                                    # to reconstruct absolute
        g[f"deg_{k}"]   = deg                                    # skip in loss AND metric
        g[f"y_{k}"]     = t(yv)                                  # raw log, for real-unit error
    # per-ENDPOINT slack labels (register endpoints @ place_resized), standardized.
    # y_endpt is per-CELL (0 where no label); m_endpt masks the labeled endpoint cells.
    # TWO fixes here:
    #  (a) remap ep_idx into the tapcell-compacted cell space.
    #  (b) a DFF owns SEVERAL endpoint pins (/D, /SET_B). The old fancy-index write kept the
    #      LAST one, and groupby sorts alphabetically, so /SET_B overwrote /D and the LESS
    #      critical slack systematically won: 15.2% of labels corrupted, and in 45% of flows
    #      the cell holding the WORST endpoint stored a non-worst slack. Take the MIN per cell.
    ep = np.load(f"{ROOT}/cache/endpt/{flow_id}.npz")
    y_ep = np.zeros(g["n_cells"], np.float32); mask = np.zeros(g["n_cells"], bool)
    ep_idx = np.empty(0, np.int64)
    if len(ep["ep_idx"]):
        idx_new = old2new[ep["ep_idx"]]
        ok = idx_new >= 0                                    # (endpoints are never tapcells)
        idx_new, slk = idx_new[ok], ep["ep_slack"][ok]
        raw = np.full(g["n_cells"], np.inf, np.float32)
        np.minimum.at(raw, idx_new, slk)                     # WORST slack per cell — fix (b)
        ep_idx = np.unique(idx_new)
        # ENDPT_TARGET — slack | arrival.  MasterRTL's cal_timing is literally
        #     slack = require_time - arrival,   require_time = THE CLOCK PERIOD
        # so slack carries the knob and arrival carries the STRUCTURE. Predicting slack forces one
        # head to learn both at once; predicting ARRIVAL leaves it a purely structural job (which
        # is what a graph is for) and recovers slack by exact arithmetic.
        # MEASURED — per-endpoint CV across a design's knob configs (std/|mean|, lower = more
        # structural): slack 1.364 / 2.028 / 0.492 / 0.181  vs  arrival 0.076 / 0.329 / 0.103 /
        # 0.191 (ac97 / usb_funct / sasc / systemcdes) — arrival is up to 18x more stable.
        # Our slack head scores pooled R2 -0.508; this is the likeliest reason.
        if ENDPT_TARGET == "arrival":
            raw = float(m.clock_period) - raw                 # arrival = require_time - slack
        y_ep[ep_idx] = (raw[ep_idx] - nm[f"y_endpt_m"]) / nm[f"y_endpt_s"]
        mask[ep_idx] = True
    g["y_endpt"] = t(y_ep); g["m_endpt"] = t(mask, torch.bool)
    g["ep_idx"] = torch.tensor(ep_idx, dtype=torch.long)         # labeled endpoint cells
    # per-cell PLACEMENT GEOMETRY targets: normalized (x, y) in [0,1] at place_resized.
    # Stored full-length (aligned to cell_names) like cache/cts, so [keep] compacts them the
    # same way every other per-cell array is compacted. NOT standardized: the target is already
    # a bounded fraction of the die, and the die itself moves with the knobs, so [0,1] IS the
    # canonical frame. mask is False only for cells with no coordinate (tapcells -> dropped by
    # keep anyway), so coverage on the kept cells is ~100%.
    pc_ = np.load(f"{ROOT}/cache/coords/{flow_id}.npz")
    cxn, cyn, cmk = pc_["x"][keep], pc_["y"][keep], pc_["mask"][keep]
    g["y_pos_x"] = t(cxn.astype(np.float32))
    g["y_pos_y"] = t(cyn.astype(np.float32))
    g["m_pos"]   = t(cmk, torch.bool)
    # per-VIRTUAL-NODE BOUNDING BOX — the geometry target that is actually well-posed.
    # Predicting all ~20k cell positions is under-determined cross-design (a short run converged
    # straight to the predict-die-centre baseline). But a VN is a METIS CONNECTIVITY cluster, and
    # the placer keeps connected cells together — measured, a cluster occupies a median 1.4%
    # (jpeg) to 21% (ac97) of the die, NOT the whole die. So a cluster's box is a real region:
    #   ~110 cells averaged per target -> per-cell placement noise cancels
    #   ~40-350 targets/flow instead of 20,000 -> vastly better conditioned
    #   the box carries LOCATION (where the cluster sits) AND SPREAD (its extent) — and spread
    #   is what sets clock wirelength, which is what f_cts needs from the seam.
    nvn = int(g["num_vn"])
    box = np.zeros((nvn, 4), np.float32); mvn = np.zeros(nvn, bool)
    for p in range(nvn):
        s = (pc == p) & cmk
        if s.sum() < 5: continue                      # too few cells -> box is noise, mask it out
        box[p] = (cxn[s].min(), cyn[s].min(), cxn[s].max(), cyn[s].max())
        mvn[p] = True
    g["y_vnbox"] = t(box); g["m_vnbox"] = t(mvn, torch.bool)   # [nvn,4] = xmin,ymin,xmax,ymax
    # raw recorded WNS/TNS — for eval readout comparison (complete, untruncated)
    # TRUE SUM-IDENTITY TARGET for HPWL_COMPOSE=sum.
    # STRESS TEST (scripts/stress_test.py T1): meta.total_hpwl is the sum at GLOBAL_PLACE over
    # 21,517 nets, but our graph is FLOORPLAN with 20,806 — the 711 extra nets are created DURING
    # placement by the resizer's buffer insertion, and they carry up to 23% of the total (ethernet
    # ratio 0.772; ac97 0.850; sasc 1.000 — small designs get no buffers).
    # So `tot_hpwl = SUM_net HPWL_net` is NOT an identity over OUR nets, and supervising the
    # composed sum against meta.total_hpwl would inflate every per-net prediction by 13-30% to
    # close a gap the net head did not cause — DAMAGING our best head (AUC 0.912).
    # The gap is a DESIGN-LEVEL CONSTANT (within-design std 0.0103 vs across-design 0.0567 =>
    # ratio 0.18), and the knob response survives it: corr(dev log SUM(ours), dev log meta) =
    # +0.9936 (R2 0.9872, 648 flows / 18 designs). So: supervise the sum against the sum over OUR
    # OWN nets — a TRUE identity — and let the level head carry the buffer-net offset.
    _nh = np.asarray(d["net_hpwl"], np.float64)
    _ok = np.isfinite(_nh) & (_nh > 0)
    g["y_hpwl_sum"] = float(np.log(_nh[_ok].sum())) if _ok.any() else float("nan")
    g["wns_true"] = float(m.wns); g["tns_true"] = float(m.tns)
    g["clock_period_raw"] = float(m.clock_period)   # for the arrival->slack identity
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in g.items()}

# ---------- DE-HNN HyperConvLayer ----------
class HyperConv(nn.Module):
    """DE-HNN's HyperConvLayer, with ONE deliberate change: MEAN aggregation, not SUM.

    SimpleConv defaults to aggr='sum'. Unnormalized, that made the message into a net scale with
    its fanout. Measured on ethernet:
        2-pin net           ->  |msg|      8.3
        8-32 fanout         ->  |msg|     65.4
        clock (fanout 10k)  ->  |msg| 68,671.9      <- 4 ORDERS of magnitude
    and this feeds psi's first Linear, then propagates back to cells BEFORE any LayerNorm — so
    every flip-flop's embedding was dominated by the clock net.

    DE-HNN defends against this TWO ways we didn't: gcn_norm degree weights on every edge
    (dehnn_layers.py:66-72, train_all_cross.py:85-87) AND dropping nets with fanout >= 3000
    (train_all_cross.py:70-78). Mean-aggregation achieves the same normalization in one step
    and keeps every net (we can't afford to drop the clock net — it drives timing).
    """
    def __init__(self, d, aggr=None):
        super().__init__()
        # AGGR is an A/B FLAG, because neither sum NOR mean is obviously right and we should
        # MEASURE rather than assume:
        #   sum  -> a net's message scales with fanout. Measured: clock net |msg| 97x a 2-pin net.
        #   mean -> normalized, BUT: HPWL is a half-perimeter = a MAX-MIN SPAN. A mean provably
        #           CANNOT compute an extremal statistic of its neighbours. So mean-aggregating
        #           cells into a net is structurally mismatched to the target we ask it to predict.
        #   multi -> [mean || max || std] concatenated: the net sees CENTER, EXTREME and SPREAD.
        #           This is the one that can actually express a span. Costs 3x the net-update width.
        # DE-HNN uses sum + gcn_norm edge weights + a fanout>=3000 drop. We have neither of the
        # latter two, so raw sum is not even their configuration.
        self.aggr = aggr or AGGR
        super_wide = 2 if self.aggr == "multi" else 1   # multi = [mean || max]
        self.lin_node, self.lin_net = Seq(Linear(d, d)), Seq(Linear(d, d))
        self.psi = Seq(Linear(d*(1 + 2*super_wide), d), ReLU(), Linear(d, d))   # net update
        self.mlp = Seq(Linear(d*3, d), ReLU(), Linear(d, d))                    # cell update
        if self.aggr == "multi":
            self.convs_f = nn.ModuleList([SimpleConv(aggr=a) for a in ("mean", "max")])
        elif self.aggr == "gcn":
            self.forward_conv = SimpleConv(aggr="sum")     # weights supplied per-edge (gcn_norm)
        else:
            self.forward_conv = SimpleConv(aggr=self.aggr)
        self.back_conv = SimpleConv(aggr="mean")               # nets -> cells: always mean

    def _agg(self, h, h_net, ei):
        """cells -> nets.
        multi: concat mean AND max so the net can see a SPAN, not just a centre — HPWL is a
               max-min extent, which a mean provably cannot represent.
        gcn:   DE-HNN's ACTUAL aggregator — sum weighted by 1/sqrt(deg_i*deg_j) (gcn_norm).
               They use neither raw sum nor mean."""
        if self.aggr == "multi":
            return torch.cat([c((h, h_net), ei) for c in self.convs_f], 1)
        if self.aggr == "gcn":
            from torch_geometric.nn.conv.gcn_conv import gcn_norm
            ei2, w = gcn_norm(ei, add_self_loops=False,
                              num_nodes=max(h.size(0), h_net.size(0)))
            return self.forward_conv((h, h_net), ei2, w)
        return self.forward_conv((h, h_net), ei)

    def forward(self, x, x_net, ntn, ttype, ntc):
        h_net = self.lin_net(x_net) + x_net
        h     = self.lin_node(x)    + x
        sm, km = ttype == 1, ttype == 0                        # source(driver) / sink
        rep = 2 if self.aggr == "multi" else 1
        h_net_source = self._agg(h, h_net, ntn[:, sm]) + h_net.repeat(1, rep)
        h_net_sink   = self._agg(h, h_net, ntn[:, km]) + h_net.repeat(1, rep)
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
        self.forward_conv = SimpleConv(aggr="mean")     # mean, same as HyperConv (see its docstring)
        self.back_conv    = SimpleConv(aggr="mean")
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
    def __init__(self, d=64, K=4, cell_in=CELL_IN, net_in=NET_IN, knob=None, dfeat=DF_IN, encoder="dehnn"):
        knob = knob if knob is not None else KNOB_DIM   # follows CRIT_KNOB
        super().__init__()
        assert encoder in ENCODERS, f"unknown encoder {encoder}; pick from {ENCODERS}"
        self.encoder = encoder
        self.use_vn = encoder != "dehnn_novn"     # the no-VN ablation
        self.vn_knobs = VN_KNOBS                  # do knobs ride into the VN? (A/B)
        self.K = K
        # SIGN-INVARIANT PE (SignNet, Lim et al. 2023). Laplacian eigenvectors have an
        # ARBITRARY SIGN: across eigsh reruns 4-8 of our 10 dims flip, so the same structural
        # position encodes differently in different designs — the PE was NOISE cross-design
        # (measured: +0.06..0.16 R2 within-design, only +0.03 pooled). A plain Linear on raw
        # v cannot be sign-invariant. phi(v) + phi(-v) is, BY CONSTRUCTION: swapping v -> -v
        # permutes the two terms and leaves the sum identical.
        # Applied per-eigenvector (each of the 10 dims is independently sign-ambiguous).
        self.n_pe = PE_SLICE.stop - PE_SLICE.start                       # 10
        self.pe_start = PE_SLICE.start   # where the PE block begins in cell_x (f_cts shifts this)
        self.pe_phi = Seq(Linear(1, 32), LeakyReLU(), Linear(32, 32))    # phi: per-eigvec
        self.pe_rho = Seq(Linear(32 * self.n_pe, d), LeakyReLU(), Linear(d, d))   # rho: combine
        self.node_encoder = Seq(Linear(cell_in - self.n_pe, d), LeakyReLU(), Linear(d, d))
        self.net_encoder  = Seq(Linear(net_in, d),  LeakyReLU(), Linear(d, d))
        self.type_emb = nn.Embedding(N_TYPES, d)

        # HOW THE CELL ENCODER FUSES ITS THREE SOURCES — library / cell-type / structure(PE).
        # It used to ADD them:  h = MLP(library) + type_emb + SignNet(PE).
        # Addition ENTANGLES: all three land in the same 64 dims superimposed, so the model
        # cannot cleanly separate "I am a NAND2" from "I sit HERE in the graph" — it has to
        # learn to keep them separable and may never manage it. And I never chose this: DE-HNN's
        # node_encoder is a plain MLP and I bolted type_emb + SignNet on with `+` because it was
        # the path of least resistance.
        # CONCAT keeps them distinct and lets a fusion MLP decide how to combine them.
        self.fuse = FUSE
        if self.fuse == "concat":
            self.fuse_mlp = Seq(Linear(3 * d, d), LeakyReLU(), Linear(d, d))

        # --- VN (DE-HNN exact): cells only, mean+max pooling, concat-MLP broadcast ---
        # VN pools the SIGN-INVARIANT features (library dims + SignNet PE embedding), NOT raw
        # cell_x — pooling raw PE would leak the arbitrary sign straight back into the VN.
        self.vn_in_dim = (cell_in - self.n_pe) + d
        self.virtualnode_encoder = Seq(Linear(self.vn_in_dim*2, d*2), LeakyReLU(), Linear(d*2, d))
        self.mlp_vn  = nn.ModuleList([Seq(Linear(d*2, d), LeakyReLU(), Linear(d, d)) for _ in range(K)])
        self.back_vn = nn.ModuleList([Seq(Linear(d*2, d), LeakyReLU(), Linear(d, d)) for _ in range(K)])
        # SUPER-VN: one global root over the cluster-VNs. DE-HNN inits BOTH levels to constant
        # zero (graph_conv_hetero.py:386-391) — not feature-pooled; VNs differentiate only
        # through pooling. top_mlp mirrors their per-layer root update (:538-543).
        self.super_vn = SUPER_VN
        if self.super_vn:
            self.top_emb = nn.Parameter(torch.zeros(d))
            self.top_mlp = nn.ModuleList([Seq(Linear(d, d), LeakyReLU(), Linear(d, d)) for _ in range(K)])
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
        # per-net head NOW HAS A CTX SKIP. It didn't, and knobs were effectively DEAD there:
        # measured |d(mu)/d(knobs)| = 0.0083 vs 0.19-0.37 for every ctx-skip head, and raising
        # utilization by 50% moved the per-net HPWL prediction by 0.11% of its own output std.
        # Utilization physically rescales the die and therefore EVERY net's HPWL. For a
        # knob-conditioned world model whose whole point is counterfactuals, that is fatal.
        self.fc1_net  = Linear(d + d, 256)            # [net embedding, ctx skip]
        self.fc1_cell = Linear(d + d, 256)            # per-cell readout: [cell embedding, ctx skip]
        self.fc1_glob = Linear(4*d + d, 256)          # [h.mean,h.max,hn.mean,hn.max, ctx]
        self.h_net_hpwl = Linear(256, 2)
        # Each global target gets TWO heads: the design LEVEL (easy, ~n_cells) and the knob
        # DEVIATION (the thing f_place exists for). Both targets are O(1), so the knob response
        # finally gets real gradient instead of a +-0.03 wiggle on a unit-variance target.
        self.h_lvl = nn.ModuleDict({k: Linear(256, 2) for k in GLOBAL_TARGETS})

        # --- KNOB-DEVIATION HEAD: the knobs go in DIRECTLY, not just via ctx. ---
        # For ONE design the graph is byte-identical across all 108 flows — only the knobs
        # change. So of the 320 dims fc1_glob sees, ~256 (h.mean/h.max/hn.mean/hn.max) are
        # CONSTANT, and the real signal (3 knobs) was buried inside the 64-dim ctx, itself
        # entangled with 16 design features that are ALSO constant. The head had to learn to
        # ignore 256 frozen dims and recover a tiny varying subspace — an absurdly indirect way
        # to express what is nearly a straight line: hpwl ~ a - b*utilization.
        # Measured: a 3-parameter OLS on the raw knobs scores R2 = +0.68..0.76 within-design.
        # Our 500k-param model scored -0.18. The task is not hard; we buried the input.
        # Fix: feed the raw knobs straight to the deviation head. The model can now trivially
        # represent the linear law, and use the GRAPH to MODULATE it ("this design is more
        # utilization-sensitive than that one") — which is the part only a GNN can do.
        # A/B FLAG, not an assumption: DIRECT_KNOB=1 feeds the raw knobs straight to the
        # deviation head; =0 keeps them only inside ctx. The decomposition fix ALONE already
        # took the knob response from -14.7 to +0.66, so whether this extra path earns its
        # place is an open question the RUN should answer, not me.
        self.direct_knob = DIRECT_KNOB
        dev_in = 4*d + d + (knob if self.direct_knob else 0)
        self.fc1_dev = Seq(Linear(dev_in, 256), LeakyReLU(), Linear(256, 256))
        self.h_dev   = nn.ModuleDict({k: Linear(256, 2) for k in GLOBAL_TARGETS})
        # per-ENDPOINT slack head (per-cell). WNS=min, TNS=sum-neg are READOUTS, not heads.
        # ctx skip so clock_period + the floorplan anchor reach each endpoint's slack directly.
        self.h_endpt = Linear(256, 2)
        # per-cell PLACEMENT GEOMETRY head -> normalized (x, y) in [0,1], each (mu, logvar).
        # WHY: the seam measured that forwarding placement SUMMARY metrics changed nothing for
        # f_cts (imagined == real), because CTS is driven by WHERE THE SINKS LANDED -- sink
        # spread sets clock wirelength, which sets buffer count and clock power. A scalar
        # total-HPWL cannot express that; coordinates can. Verified learnable: adjacent knob
        # configs place the same cell at corr x/y 0.97/0.92 (the placer is deterministic, so
        # this is not the PE sign-ambiguity trap), while the aspect_ratio knob reorganizes the
        # layout (sasc corr y 0.92 -> 0.13 as the die goes AR 2.08 -> 0.68) -- so geometry
        # CARRIES the knob response the summary globals were crushing 18-53x.
        self.geo_heads = GEO_HEADS
        if self.geo_heads:
            self.h_pos = Linear(256, 4)             # [mu_x, logvar_x, mu_y, logvar_y]
        # per-VN BOUNDING BOX head: pool the VN's cells (mean+max) -> xmin,ymin,xmax,ymax,
        # each (mu, logvar). Coarser and far better-posed than per-cell position — see load_graph.
            self.h_vnbox = Seq(Linear(2 * 256, 256), LeakyReLU(), Linear(256, 8))

    def encode_pe(self, pe):
        """SignNet: rho( concat_i [ phi(v_i) + phi(-v_i) ] ). Invariant to v_i -> -v_i by
        construction — swapping the sign permutes the two phi terms and the sum is unchanged."""
        v = pe.unsqueeze(-1)                                  # (C, n_pe, 1)
        s = self.pe_phi(v) + self.pe_phi(-v)                  # (C, n_pe, 32)  <- sign-invariant
        return self.pe_rho(s.flatten(1))                      # (C, d)

    def encode(self, g):
        """The shared graph encoder: raw features -> (cell embeddings h, net embeddings h_net,
        design/knob context ctx). Everything up to the readout heads. f_cts reuses this EXACT
        encoder (winning config: DE-HNN + VN-structure + multi-aggr + concat-fuse), so the seam
        can feed one stage's imagined state into the next through the same representation."""
        lib = g["cell_x"][:, :self.pe_start]
        pe  = g["cell_x"][:, self.pe_start:self.pe_start + self.n_pe]
        e_lib, e_typ, e_pe = self.node_encoder(lib), self.type_emb(g["cell_type"]), self.encode_pe(pe)
        feat  = torch.cat([lib, e_pe], 1)                     # sign-INVARIANT cell features (for VN)
        h = (self.fuse_mlp(torch.cat([e_lib, e_typ, e_pe], 1)) if self.fuse == "concat"
             else e_lib + e_typ + e_pe)
        h_net = self.net_encoder(g["net_x"])
        part, nvn = g["part_cell"], g["num_vn"]
        ctx = self.ctx(torch.cat([g["knobs"], g["dfeat"]]))
        if self.use_vn:
            vn_in = torch.cat([scatter(feat, part, 0, dim_size=nvn, reduce="mean"),
                               scatter(feat, part, 0, dim_size=nvn, reduce="max")], 1)
            vn = self.virtualnode_encoder(vn_in) + (ctx if self.vn_knobs else 0)
            top = self.top_emb if self.super_vn else None
        else:
            h, h_net = h + ctx, h_net + ctx
        ntn, ttype, ntc = g["ntn"], g["ntn_type"], g["ntc"]
        if self.training:
            ntn, mask = dropout_edge(ntn, p=0.2)
            ttype, ntc = ttype[mask], ntc[:, mask]
        for l in range(self.K):
            if self.use_vn:
                # DOWN: the root is SUMMED INTO the cluster VN, then broadcast to cells
                # (graph_conv_hetero.py:497 — additive, every layer).
                vn_eff = (vn + top) if self.super_vn else vn
                h = self.back_vn[l](torch.cat([h, vn_eff[part]], 1)) + h
            h, h_net = self.convs[l](h, h_net, ntn, ttype, ntc)
            h     = F.leaky_relu(self.norms[l](h))
            h_net = F.leaky_relu(self.norms[l](h_net))
            if self.use_vn and l < self.K - 1:
                vn_t = torch.cat([scatter(h, part, 0, dim_size=nvn, reduce="mean"),
                                  scatter(h, part, 0, dim_size=nvn, reduce="max")], 1)
                vn = self.mlp_vn[l](vn_t) + vn
                if self.super_vn:      # UP: VN -> root (mean-pool only), :538-543
                    top = top + self.top_mlp[l](vn.mean(0) + top)
        return h, h_net, ctx

    def forward(self, g):
        h, h_net, ctx = self.encode(g)
        hn = F.leaky_relu(self.fc1_net(torch.cat([h_net, ctx.expand(h_net.size(0), -1)], 1)))
        hc = F.leaky_relu(self.fc1_cell(torch.cat([h, ctx.expand(h.size(0), -1)], 1)))  # per-cell
        hg = F.leaky_relu(self.fc1_glob(torch.cat([
            h.mean(0), h.max(0).values, h_net.mean(0), h_net.max(0).values, ctx])))
        # deviation head: pooled graph + ctx + THE RAW KNOBS (a direct, unmediated path)
        dev_in = [h.mean(0), h.max(0).values, h_net.mean(0), h_net.max(0).values, ctx]
        if self.direct_knob: dev_in.append(g["knobs"])
        hd = self.fc1_dev(torch.cat(dev_in))
        p = self.h_pos(hc) if self.geo_heads else None
        # ANALYTIC COMPOSITION (HPWL_COMPOSE="sum"): tot_hpwl = SUM_net HPWL_net is an IDENTITY.
        # Undo net_hpwl's standardization to get log-um, exp -> um, sum -> log. Kept inside the
        # model so the aggregate is DIFFERENTIABLE and can be supervised: that is what forces the
        # per-net predictions to be calibrated in absolute terms rather than merely well-ranked
        # (our per-net AUC is 0.912 but its absolute rel-err is 43.7% — ranking alone will not
        # sum to the right total). See GRANNITE (DAC'20) for the pattern.
        hpwl_sum = None
        if HPWL_COMPOSE == "sum":
            nm_ = norm()
            lg = out_net_log = (self.h_net_hpwl(hn)[:, 0] * float(nm_["y_net_hpwl_s"])
                                + float(nm_["y_net_hpwl_m"]))            # standardized -> log um
            m_ = g.get("m_net_hpwl")
            lg = lg[m_] if m_ is not None and m_.any() else lg
            hpwl_sum = torch.logsumexp(lg, 0)        # log(SUM exp(log_len)) — stable
        # VN boxes: pool each METIS cluster's cell embeddings, then read its bbox off that.
        vb = torch.cat([scatter(hc, g["part_cell"], 0, dim_size=g["num_vn"], reduce="mean"),
                        scatter(hc, g["part_cell"], 0, dim_size=g["num_vn"], reduce="max")], 1)
        o = dict(net_hpwl=self.h_net_hpwl(hn),
                 endpt=self.h_endpt(hc),             # per-cell slack; WNS/TNS read out in train
                 )
        if self.geo_heads:
            o["pos_x"], o["pos_y"] = p[:, 0:2], p[:, 2:4]    # each (mu, logvar)
            o["vn_box"] = self.h_vnbox(vb).view(-1, 4, 2)    # [nvn, 4 coords, (mu, logvar)]
        if hpwl_sum is not None:
            o["tot_hpwl_sum"] = hpwl_sum             # log um, composed by identity — supervised
                                                     # in train_fplace against the true log total
        for k in GLOBAL_TARGETS:                     # level + knob-deviation, both O(1)
            o[f"{k}_lvl"] = self.h_lvl[k](hg)        # design level: pooled graph (~n_cells)
            o[f"{k}_dev"] = self.h_dev[k](hd)        # knob response: knobs go in DIRECTLY
        return o

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

# NOTE: the training objective lives in train_fplace.wloss (it carries the loss WEIGHTS).
# A duplicate loss_fn() used to live here, unweighted and out of sync with what training
# actually optimized — a trap for anyone reusing it. Removed.

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
        for g in graphs:   # the real objective (with weights) is train_fplace.wloss
            l = (gnll(model(g)["net_hpwl"], g["y_net_hpwl"], g["m_net_hpwl"])
                 + gnll(model(g)["endpt"], g["y_endpt"], g["m_endpt"]))
            l.backward(); tot += l.item()
        opt.step()
        if step % 3 == 0 or step == 14:
            print(f"  step {step:3d}  loss/graph = {tot/len(graphs):.4f}  ({time.time()-t0:.0f}s)")
