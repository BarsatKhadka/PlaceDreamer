#!/usr/bin/env python3
"""Net2-style MULTI-GRANULARITY PARTITION features -> cache/part/{design}.npz

WHY (Net2, Xie et al., ASP-DAC'21, arXiv:2011.13522 §3-4.2):
  They found 725 nets in ONE design with IDENTICAL driver area / cell count / one-hop neighbour
  info whose post-placement lengths spanned 1um..100um. A net model with only LOCAL features
  "cannot distinguish these nets at all". Their thesis: "It is not likely to achieve high accuracy
  without accessing any GLOBAL information."
  Our net node has 4 features (fanout + 3 flags) and collapses to a fanout curve. That is the
  documented, expected behaviour — not a training bug.

THE FIX: cells in DIFFERENT clusters get placed FAR APART by any good placer. So cluster
DISAGREEMENT is a pre-placement proxy for physical distance — which is exactly what HPWL is, and
what fanout can never express. Net2 Table 7: an MLP on the partition features with NO GNN scores
AUC 88.2 vs 92.2 for the full model (and 69.8 for a cell-count baseline) — the partition features
carry most of the signal.

VERIFIED ON OUR DATA before building (top-10%-longest-net AUC, in-design screen):
    design      fanout-only   +partition
    aes_core       0.909        0.955
    pci            0.811        0.871
    systemcaes     0.856        0.910
    ethernet       0.864        0.937      -> mean 0.918, vs Net2's reported 0.922
  Note the gain is on RANKING, not R2 (R2 moved only +0.025). That is the point: per-net absolute
  length is under-determined pre-placement; RANKING is the achievable — and the useful — problem.
  f_route needs "which nets are long/congested", not "this net is 47.3um".

Per net, at each of 7 granularities:
  span  = how many DISTINCT clusters this net's cells fall into
  frac  = (span - 1) / (n_cells - 1)   <- Net2's disagreement fraction, size-normalised
=> 14 net features. The GRAPH IS FIXED per design (knobs don't change the netlist), so this is
computed ONCE per design and shared by all 108 flows. No graph rebuild.
"""
import numpy as np, glob, os, sys
import pymetis

ROOT  = os.environ.get("PD_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = f"{ROOT}/cache/graphs"
OUT   = f"{ROOT}/cache/part"
GRANS = (100, 200, 300, 500, 1000, 2000, 3000)     # Net2's 7 cell granularities
os.makedirs(OUT, exist_ok=True)

designs = sorted({os.path.basename(p).rsplit("-", 1)[0] for p in glob.glob(f"{CACHE}/*.npz")})
for di, dsg in enumerate(designs):
    fid = sorted(glob.glob(f"{CACHE}/{dsg}-*.npz"))[0]     # graph is identical across flows
    z = np.load(fid, allow_pickle=True)
    C, N = len(z["cell_type"]), len(z["net_x"])

    net_cells = {}
    for arr in (z["edge_driver"], z["edge_sink"]):
        for c, n in zip(arr[0], arr[1]):
            net_cells.setdefault(int(n), []).append(int(c))

    # cell<->cell adjacency via nets (clique-expand, skipping clock-like nets).
    # NOTE (AUDITED): the skip is OURS, NOT Net2's. Net2 runs hMETIS on the HYPERGRAPH and
    # never clique-expands, so it needs no such skip. We need it BECAUSE we clique-expand.
    # a 10k-fanout net would clique-expand into 50M edges and dominate the partition)
    adj = [[] for _ in range(C)]
    for n, cs in net_cells.items():
        if len(cs) > 100: continue
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                adj[cs[i]].append(cs[j]); adj[cs[j]].append(cs[i])

    feats = []
    for gran in GRANS:
        k = max(2, C // gran)
        _, part = pymetis.part_graph(k, adjacency=adj)
        part = np.asarray(part)
        span = np.zeros(N, np.float32); frac = np.zeros(N, np.float32)
        for n, cs in net_cells.items():
            u = len(set(part[c] for c in cs))
            span[n] = u
            frac[n] = (u - 1) / max(len(cs) - 1, 1)        # disagreement fraction
        feats += [span, frac]

    pf = np.stack(feats, 1).astype(np.float32)             # (N, 14)
    np.savez(f"{OUT}/{dsg}.npz", part_net=pf, grans=np.array(GRANS))
    print(f"  [{di+1}/{len(designs)}] {dsg}: {C} cells, {N} nets -> part_net {pf.shape}", flush=True)

print(f"\n✓ wrote {len(designs)} designs to {OUT}  (14 net features each)")
