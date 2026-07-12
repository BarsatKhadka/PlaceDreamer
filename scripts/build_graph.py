#!/usr/bin/env python3
"""
Build the DE-HNN bipartite representation (cell<->net, driver/sink edges) from
EDA-Schema-V2 sky130hd, for one (flow_id, stage).

The raw graph_json is pin-level (PORT/NET/GATE/PIN). Edge DIRECTION encodes driver/sink:
    GATE -> PIN -> NET   = gate's output pin DRIVES the net   -> driver edge (cell->net)
    NET  -> PIN -> GATE  = net feeds gate's input pin         -> sink   edge (cell->net)
We contract the PIN nodes to get the bipartite cell<->net graph the encoder wants.

Node features:
  cell : standard_cell physical/functional (w,h,#pins,is_seq/inv/buf,drive,caps) + type_id + degree
  net  : fanout(#sinks) + is_io   (NOT hpwl/length — those are f_place LABELS)
Positional: top-k Laplacian eigenvectors — DE-HNN construction (cell↔cell driver→sink
graph, cells only; nets carry zero PE).

Usage: python3 scripts/build_graph.py <flow_id> <stage>
"""
import sys, json, numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import pymetis

DATA = "/Users/barsat/PlaceDreamer/datasets/sky130hd"

# --- standard-cell library: name -> numeric features + type id (loaded once) ---
def load_cell_lib():
    t = pq.read_table(f"{DATA}/standard_cells/table.parquet").to_pydict()
    names = t["name"]
    tid = {n: i for i, n in enumerate(names)}
    feat = {}
    for i, n in enumerate(names):
        feat[n] = np.array([
            t["width"][i], t["height"][i],
            t["no_of_input_pins"][i], t["no_of_output_pins"][i],
            float(t["is_sequential"][i]), float(t["is_inverter"][i]),
            float(t["is_buffer"][i]), float(t["is_filler"][i]), float(t["is_diode"][i]),
            t["drive_strength"][i],
            t["input_capacitance_max"][i], t["output_capacitance_max"][i],
            t["leakage_power_max"][i],
        ], dtype=np.float32)
        feat[n] = np.nan_to_num(feat[n])   # some library cells have NULL caps/leakage
    return tid, feat

def build(flow_id, stage, cell_tid=None, cell_feat=None, part_size=250,
          graph_json=None, gate_cell=None):
    """graph_json + gate_cell can be pre-read (batched caching) to skip per-flow table scans."""
    if cell_tid is None:
        cell_tid, cell_feat = load_cell_lib()
    # 1) the pin-level graph
    if graph_json is None:
        graph_json = ds.dataset(f"{DATA}/netlists/graph.parquet").to_table(
            filter=(ds.field("flow_id") == flow_id) & (ds.field("stage") == stage)
        ).to_pydict()["graph_json"][0]
    gj = json.loads(graph_json)
    nodes, types, edges = gj["nodes"], gj["node_types"], gj["edges"]
    tmap = dict(zip(nodes, types))
    # 2) gate -> standard_cell type (this flow/stage)
    if gate_cell is None:
        gt = ds.dataset(f"{DATA}/gates/table.parquet").to_table(
            filter=(ds.field("flow_id") == flow_id) & (ds.field("stage") == stage),
            columns=["name", "standard_cell"]).to_pydict()
        gate_cell = dict(zip(gt["name"], gt["standard_cell"]))

    # 3) contract pins -> driver/sink edges via edge direction
    pin_gate, pin_dir, pin_net = {}, {}, {}
    net_is_io = set()
    for s, d in edges:
        ts, td = tmap.get(s), tmap.get(d)
        if   ts == "GATE" and td == "PIN": pin_gate[d] = s; pin_dir[d] = "out"
        elif ts == "PIN"  and td == "GATE": pin_gate[s] = d; pin_dir[s] = "in"
        elif ts == "PIN"  and td == "NET":  pin_net[s]  = d
        elif ts == "NET"  and td == "PIN":  pin_net[d]  = s
        elif "PORT" in (ts, td):
            net_is_io.add(s if ts == "NET" else d if td == "NET" else None)

    cells = [n for n, t in zip(nodes, types) if t == "GATE"]
    nets  = [n for n, t in zip(nodes, types) if t == "NET"]
    cidx = {c: i for i, c in enumerate(cells)}
    nidx = {n: i for i, n in enumerate(nets)}

    CLK_FN = {"CLK", "CLK_N", "CLKN", "GCLK", "GATE"}
    RST_FN = {"RESET", "RESET_B", "RST", "RST_B", "SET", "CLEAR", "CLR", "SET_B"}
    net_sink_fn = {}   # net -> set of sink-pin functions (to classify clock/reset)

    drv, snk = [], []   # (cell_i, net_i)
    for pin, gate in pin_gate.items():
        net = pin_net.get(pin)
        if net is None or gate not in cidx or net not in nidx: continue
        if pin_dir[pin] == "out":
            drv.append((cidx[gate], nidx[net]))
        else:
            snk.append((cidx[gate], nidx[net]))
            net_sink_fn.setdefault(net, set()).add(pin.rsplit("/", 1)[-1])
    drv = np.array(drv, dtype=np.int64).T if drv else np.zeros((2, 0), np.int64)
    snk = np.array(snk, dtype=np.int64).T if snk else np.zeros((2, 0), np.int64)

    # 4) node features
    UNK = np.zeros(13, np.float32)
    cell_lib = np.stack([cell_feat.get(gate_cell.get(c, ""), UNK) for c in cells])  # (C,13)
    cell_type = np.array([cell_tid.get(gate_cell.get(c, ""), -1) for c in cells], np.int64)
    cell_area = (cell_lib[:, 0] * cell_lib[:, 1])[:, None]          # explicit area = w*h
    cell_deg = np.zeros((len(cells), 1), np.float32)
    for arr in (drv, snk):
        for ci in arr[0]: cell_deg[ci, 0] += 1
    cell_x = np.concatenate([cell_lib, cell_area, cell_deg], axis=1)  # (C,15)

    net_fanout = np.zeros(len(nets), np.float32)
    for ni in snk[1]: net_fanout[ni] += 1
    net_io = np.array([1.0 if n in net_is_io else 0.0 for n in nets], np.float32)
    net_clk = np.array([1.0 if net_sink_fn.get(n, set()) & CLK_FN else 0.0 for n in nets], np.float32)
    net_rst = np.array([1.0 if net_sink_fn.get(n, set()) & RST_FN else 0.0 for n in nets], np.float32)
    net_x = np.stack([net_fanout, net_io, net_clk, net_rst], axis=1)   # (N,4)

    # 5) Laplacian PE — DE-HNN construction: on the cell↔cell graph (driver→sink edges),
    #    cells only (nets get zero PE, as in DE-HNN). Sym-normalized, 10 lowest eigenvectors.
    C, N = len(cells), len(nets)
    net_driver = {int(ni): int(ci) for ci, ni in zip(drv[0], drv[1])}
    net_sinks = {}
    for ci, ni in zip(snk[0], snk[1]): net_sinks.setdefault(int(ni), []).append(int(ci))
    cc_r, cc_c = [], []
    for ni, dci in net_driver.items():
        for sci in net_sinks.get(ni, []):
            cc_r += [dci, sci]; cc_c += [sci, dci]          # source↔sink, symmetric
    A = sp.coo_matrix((np.ones(len(cc_r)), (cc_r, cc_c)), shape=(C, C)).tocsr()
    deg = np.asarray(A.sum(1)).ravel(); deg[deg == 0] = 1
    Dinv = sp.diags(1.0 / np.sqrt(deg))
    L = sp.eye(C) - Dinv @ A @ Dinv
    k = min(10, C - 2)
    try:
        _, vecs = eigsh(L, k=k + 1, which="SM"); pe_cell = vecs[:, 1:k + 1]
    except Exception:
        pe_cell = np.zeros((C, 10), np.float32)
    pe_cell = pe_cell.astype(np.float32)
    pe_net = np.zeros((N, 10), np.float32)                  # DE-HNN: nets carry no PE

    # 5b) METIS partition of the BIPARTITE graph → part_id per node (DE-HNN VN hierarchy).
    #     One cluster-VN per partition; all cluster-VNs roll up to 1 top VN (2 levels).
    ei = np.concatenate([drv, snk], axis=1)                 # cell→net edges
    src = np.concatenate([ei[0], ei[1] + C]); dst = np.concatenate([ei[1] + C, ei[0]])  # undirected bipartite
    adj = [[] for _ in range(C + N)]
    for s, d in zip(src.tolist(), dst.tolist()): adj[s].append(d)
    k = max(1, round((C + N) / part_size))                  # #partitions scales with graph size
    if k > 1:
        _, membership = pymetis.part_graph(k, adjacency=adj)
        part_id = np.array(membership, dtype=np.int64)
    else:
        part_id = np.zeros(C + N, dtype=np.int64)
    num_vn = int(part_id.max()) + 1
    top_part_id = np.zeros(num_vn, dtype=np.int64)           # all cluster-VNs → top VN 0
    num_top_vn = 1

    # 6) design-level features (features.md Group B — the cross-design signal)
    design_features = dict(
        n_cells=float(C), n_nets=float(N), n_pins=float(drv.shape[1] + snk.shape[1]),
        total_cell_area=float(cell_area.sum()),
        frac_seq=float(cell_lib[:, 4].mean()), frac_inv=float(cell_lib[:, 5].mean()),
        frac_buf=float(cell_lib[:, 6].mean()), frac_filler=float(cell_lib[:, 7].mean()),
        frac_diode=float(cell_lib[:, 8].mean()),
        fanout_mean=float(net_fanout.mean()), fanout_max=float(net_fanout.max()),
        fanout_p90=float(np.percentile(net_fanout, 90)),
        frac_2pin=float((net_fanout == 1).mean()), frac_3pin=float((net_fanout == 2).mean()),
        frac_ge4pin=float((net_fanout >= 3).mean()),
        clock_fanout=float(net_fanout[net_clk == 1].max()) if net_clk.any() else 0.0,
        n_clock_nets=float(net_clk.sum()), n_reset_nets=float(net_rst.sum()),
    )

    return dict(flow_id=flow_id, stage=stage,
                n_cells=C, n_nets=N, cell_x=cell_x, net_x=net_x,
                edge_driver=drv, edge_sink=snk, cell_type=cell_type,
                cell_names=np.array(cells), net_names=np.array(nets),   # identities → future joins
                pe_cell=pe_cell, pe_net=pe_net, design_features=design_features,
                part_cell=part_id[:C], part_net=part_id[C:],
                num_vn=num_vn, top_part_id=top_part_id, num_top_vn=num_top_vn)

if __name__ == "__main__":
    flow, stage = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("aes_core-000001", "floorplan")
    g = build(flow, stage)
    print(f"=== {flow} @ {stage} ===")
    print(f"  cells={g['n_cells']}  nets={g['n_nets']}")
    print(f"  driver edges={g['edge_driver'].shape[1]}  sink edges={g['edge_sink'].shape[1]}")
    print(f"  cell_x {g['cell_x'].shape}  net_x {g['net_x'].shape}  PE {g['pe_cell'].shape}")
    print(f"  net types: clock={int(g['net_x'][:,2].sum())} reset={int(g['net_x'][:,3].sum())} io={int(g['net_x'][:,1].sum())}")
    print(f"  cell types seen={len(set(g['cell_type']))}  (-1=unknown: {int((g['cell_type']==-1).sum())})")
    import numpy as _np
    sizes = _np.bincount(_np.concatenate([g['part_cell'], g['part_net']]))
    print(f"  VNs (METIS partitions)={g['num_vn']}  top_vn={g['num_top_vn']}  "
          f"partition sizes: min={sizes.min()} mean={sizes.mean():.0f} max={sizes.max()}")
    print("  === design_features ===")
    for k, v in g['design_features'].items(): print(f"     {k:16} {v:.4g}")
