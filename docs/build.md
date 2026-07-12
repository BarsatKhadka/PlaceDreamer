# PlaceDreamer — Build Log

*The implementation log. `architecture.md` holds the **design** (what & why); this holds the
**build** (the code, and whether it actually does what the design says).*

**Ground rule (same as architecture):** every build step is **verified against the dataset's own
recorded truth**, not trusted. When we build a graph, we check its counts against the numbers the
dataset independently records; we write down the matches *and* the mismatches.

**Data source:** EDA-Schema-V2 `sky130hd` → `datasets/sky130hd/` — 18 designs × 108 configs × 8
stages = 1,944 flows; columnar Parquet (`gates` 219M rows, `nets` 115M rows, `netlists/graph`
= per-flow-stage `graph_json`, `standard_cells` = the library).

---

## 1. Representation — netlist encoding  (implements architecture §1.1)

**Builder:** `scripts/build_graph.py` — `build(flow_id, stage)` → the DE-HNN bipartite graph.

### The key mechanism (how the raw graph maps to ours)
The dataset's `graph_json` is **pin-level** — 4 node types: `PORT / NET / GATE / PIN`. **Edge
direction encodes driver/sink**, which is exactly what we need:
```
GATE → PIN → NET   = gate's OUTPUT pin drives the net  → DRIVER edge (cell→net)
NET  → PIN → GATE  = net feeds gate's INPUT pin         → SINK   edge (cell→net)
```
We **contract the PIN nodes** → bipartite `cell(GATE) ↔ net(NET)` with driver/sink edge types.

### Checklist against §1.1

| §1.1 item | status | notes |
|---|---|---|
| bipartite cell↔net, driver/sink edges | ✅ **done + verified** | driver from `GATE→PIN→NET`, sink from `NET→PIN→GATE` |
| Laplacian PE (top-10) | ✅ done | **DECISION: match DE-HNN** — computed on the cell↔cell graph (driver→sink edges), sym-normalized, **cells only** (nets carry zero PE). Chose fidelity to the validated paper over our bipartite variant. |
| cell features: type, w/h, #pins, degree | ✅ done | joined from `standard_cells`; **0 unknown types** |
| net features: fanout, is_io | ✅ done | `is_io` from PORT-connected nets |
| **net type (clock/reset/signal)** | ✅ **done** | via sink-pin function (`/CLK` → clock); found 1 clock net in aes_core |
| explicit `area` feature | ✅ done | w×h, added to cell_x |
| **design-level features (features.md Group B)** | ✅ **done** | 18 scalars: gate/net/pin count, total area, cell-type fractions, fanout mean/max/p90, net-degree 2/3/≥4-pin fractions, seq ratio, clock fanout |
| **virtual nodes (METIS hierarchy)** | ✅ **done** | `pymetis` partition of bipartite graph, part_size≈250 → `num_vn` scales with size (sasc 3, aes_core 115, jpeg 436), balanced; 2-level (cluster-VNs → 1 top VN). part_cell/part_net returned. |
| persistent homology | ⏸ deferred | later ablation (per §1.1) |

**§1.1 is COMPLETE** — the netlist representation is fully DE-HNN-faithful (bipartite cell↔net +
driver/sink + DE-HNN Laplacian PE + VN hierarchy) plus our added features (net type, design-level
vector). Only *caching* (a preprocessing optimization, not a representation item) and *persistent
homology* (deferred ablation) remain outside the core.

### Verification (the trust check)
Cross-checked the build against `netlists/table.parquet` (which records `no_of_cells/nets/pins`
independently — never used in building):

| | dataset | built | |
|---|---|---|---|
| cells | 17,145 | 17,145 | ✅ exact |
| nets | 11,683 | 11,683 | ✅ exact |
| pins | 45,376 | 45,140 | ⚠️ −236 (~0.5%) = IO/port pins (we count only gate-pin↔net; ports handled via `is_io`) |

(aes_core-000001 @ floorplan. Also validated on jpeg: 63,907 cells / 45,024 nets.)

### Feature vectors as built
- **cell_x** (14): width, height, #in_pins, #out_pins, is_seq, is_inv, is_buf, is_filler, is_diode,
  drive_strength, in_cap_max, out_cap_max, leakage_max, **degree**.  + `cell_type` id (for embedding).  + `pe_cell` (10).
- **net_x** (2): fanout, is_io.  + `pe_net` (10).
- **edges**: `edge_driver` (2×E), `edge_sink` (2×E) — [cell_idx, net_idx].

### To complete §1.1 (next build steps, in order)
1. **net type** (clock/reset/signal) — classify nets (clock net = high fanout / driver is clock source; reset similarly).
2. **virtual nodes** — METIS-partition the bipartite graph → add one VN per partition (+ top VN), with VN↔node edges. Toggle.
3. **explicit area** feature (w×h) — trivial.
4. **caching** — the ~8.6s/flow `gates`-scan is fine once; preprocess all 1,944 flows → tensors on disk so training doesn't rebuild.

*(§1.2 — placement state / f_place outputs — is a separate build section, started once §1.1 is complete.)*

## Caching (§1.1 → training-ready)
- **`scripts/cache_graphs.py`** — caches all flows' f_place graphs (floorplan stage) to
  `cache/graphs/<flow_id>.npz`, batched per-design (one gates-read per 108 flows → ~36s/design,
  ~11min total). **Extensible by design**: each npz stores the expensive+fixed structure
  (edges, part_id, PE) + **identities (cell_names, net_names, cell_type)** + current features
  (cell_x, net_x, design_features) + **per-net labels** (net_hpwl, net_demand=RUDY-from-bbox,
  99.4% name-matched to global_place). `cache/meta.parquet` = per-flow design/knobs/total_hpwl/buffer_area.
- **Adding features later** = read new table → join by `cell_names` → append column. No graph rebuild.
- Builder refactored: `build(..., graph_json=, gate_cell=)` accepts pre-read data for batched caching.

## Baseline signal test (my_tests/) — the honest reality check
Gradient-boosted trees on {size + knobs}, leave-one-DESIGN-out, all scalar PPA targets:
- **hpwl / routed_WL / power / buffer_area = RULERS** (97–99% design-size; size-only GBT R²=0.87–0.99;
  knobs add ~0). A GNN is **not justified** for aggregate PPA.
- **WNS / TNS = knob-driven (80–90% within-design) but cross-design prediction FAILS** (negative R²) —
  idiosyncratic per design. The one place structure (a timing GNN) or congestion might earn the GNN.
- **Congestion untested** (RUDY maps were trimmed) — recover via RUDY-from-bbox (now a per-net label).
→ The GNN lives or dies on **congestion + timing**, not wirelength/power.
