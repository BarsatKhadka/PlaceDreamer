# PlaceDreamer — Features & Targets

The canonical list of **what we predict (targets = PPA)** and **what we predict from
(input features)**. Everything here is extractable from a single archived run:

```
data_gen/archive/<design>_<cfg>/
  <design>.synth.v   ← netlist (pure connectivity graph, pre-placement)
  placement.odb      ← placement DB (cells + positions + nets)
  routed.def         ← routed layout
  metrics.json       ← all PPA numbers (LibreLane cumulative metrics)
  config.json        ← the knobs (full resolved config)
  features/          ← RUDY/density/pin maps (64×64 CSV) + summary.json
```

We extract **all** features up front (not tiered) so nothing needs re-running; models
may then use any subset.

---

## 1. Targets — PPA (what the world model predicts)

The deliverable: predict the routed **P**ower / **P**erformance / **A**rea of a design,
from a pre-route state, without running the router. Metric keys are LibreLane's
`metrics.json` names.

### Performance — timing (this is where WNS lives)
| target | metric key | meaning |
|---|---|---|
| **WNS (setup)** | `timing__setup__ws` | worst negative slack, setup — the headline timing number (<0 = violation) |
| WNS (hold) | `timing__hold__ws` | worst slack, hold |
| **TNS (setup)** | `timing__setup__tns` | total negative slack (sum over violating paths) |
| TNS (hold) | `timing__hold__tns` | total negative slack, hold |
| setup/hold vio count | `timing__setup_vio__count`, `timing__hold_vio__count` | # violating endpoints |
| per-corner WNS | `timing__setup__ws__corner:<corner>` | slack at each corner (tt/ss/ff); the worst corner is what signs off |
| clock skew | `clock__skew__worst_setup` | CTS-induced skew (post-CTS target) |

### Power
| target | metric key | meaning |
|---|---|---|
| **total power** | `power__total` | total power |
| breakdown | `power__internal`, `power__switching`, `power__leakage` | dynamic vs leakage split (if present) |

### Area — the *emergent* part (what the flow ADDS, not what we set)
> Die area / core area / target utilization are **inputs, not targets** — we set them
> (`FP_CORE_UTIL`, `DIE_AREA`; die = cell_area ÷ utilization). They belong under §2
> Config, not here. The predictable "area" is the **extra cells the flow inserts** that
> you can't know without running it — and these cost **area *and* power**.

| target | metric key | meaning |
|---|---|---|
| **clock buffers** | `design__instance__count__class:clock_buffer` | CTS buffering cost — unknown until CTS runs |
| **timing-repair buffers** | `design__instance__count__class:timing_repair_buffer` | resizer buffering to fix slack |
| clock inverters | `design__instance__count__class:clock_inverter` | clock-tree inverters |
| setup / hold buffers | `design__instance__count__setup_buffer`, `..._hold_buffer` | added to fix setup/hold |
| antenna diodes | `design__instance__count__class:antenna_cell` | added during routing |
| final cell count | `design__instance__count__stdcell` | total after all insertions (vs. synth count → *added area*) |

**Added area** = final cell count − synthesized cell count = the flow's overhead. This is
the interesting quantity: given a floorplan, imagine how much CTS+routing will bloat it.

### Routability / physical quality (also targets)
| target | metric key | meaning |
|---|---|---|
| **routed wirelength** | `route__wirelength` | true routed WL (primary "does it learn" target) |
| **DRC violations** | `route__drc_errors` | routing rule violations (0 in the safe band; needs edge-of-routability configs to vary) |
| GRT wirelength | `global_route__wirelength` | global-route WL (intermediate truth; `routed ≈ 0.727 × GRWL`) |

> **Note (macro designs):** timing on hard-macro designs (chameleon, microwatt) is not
> meaningful — the macros have no `.lib`, so paths through them aren't timed. Use their
> PPA (esp. WNS) only for std-cell designs.

---

## 2. Input features (what we predict from)

Three groups by source. All extracted per run.

### Group A — Config knobs (the "action"; from `config.json`)
The choices that generate each data point.
- `FP_CORE_UTIL`, `PL_TARGET_DENSITY_PCT`, `FP_ASPECT_RATIO`
- `SYNTH_STRATEGY` (AREA/DELAY level → area-vs-speed tradeoff)
- `PL_ROUTABILITY_DRIVEN`, `PL_TIME_DRIVEN`
- `CLOCK_PERIOD` (target frequency)
- `FP_CORE_UTIL` / `DIE_AREA` **set** the die & core area (so *area-as-a-size* is an input
  here, not a target — the emergent buffering is the target; see §1 Area).

### Group B — Design-intrinsic (from `synth.v`; **constant across a design's configs**)
This is what enables **cross-design generalization** — without it the model can't tell
designs apart. All derivable from the synthesized netlist / odb.
- **scalars:** gate count, net count, pin count, total cell area
- **cell-type histogram:** counts by class — `design__instance__count__class:*`
  (sequential_cell, multi_input_combinational_cell, inverter, buffer, …)
- **fanout distribution:** avg / max / p90 net fanout
- **net-degree distribution:** fraction of 2-pin / 3-pin / ≥4-pin nets (drives the √p
  HPWL→routed bias)
- **sequential ratio:** #flops / #cells (clock-tree / timing proxy)
- **clock fanout:** #sinks on the clock net (CTS difficulty)
- **the netlist graph itself:** nodes = cells (feature: type, area, #pins), edges = nets
  (hyperedges → bipartite cell↔net or clique) → for a GNN

### Group C — Placement-state (from `placement.odb`; **changes per config**)
The "imagined placement" the staged world model produces at end-of-GP.
- **scalars:** HPWL, utilization, cell-area density, RUDY max / mean
- **spatial maps (64×64 grids):**
  - **RUDY** — routing-demand / congestion proxy (OpenROAD-faithful; see below)
  - **cell density** — where cells pack
  - **pin density** — routing demand at pins
- (later) GRT congestion map — intermediate truth via `global_route`

**RUDY** (our congestion proxy) is a faithful port of OpenROAD `Rudy.cpp`:
`wire_width = 1.25/Σ(1/pitch)` over signal layers; per net (via `getTermBBox()`)
`net_congestion = (dx+dy)·wire_width / bbox_area`; per tile `+= net_congestion ·
(intersect/tile_area) · 100`. Computed by `scripts/extract_features.py`.

---

## 3. Extraction sources (where each feature comes from)

| feature group | source file | tool |
|---|---|---|
| Config knobs | `config.json` | direct |
| Design scalars + cell-type histogram | `metrics.json` (`design__*__class:*`) + `synth.v` / `placement.odb` | odb Python |
| Netlist graph, fanout, net-degree | `synth.v` or `placement.odb` (nets/iterms) | odb Python |
| Placement maps + HPWL + RUDY | `placement.odb` | `extract_features.py` |
| All PPA targets | `metrics.json` | direct |
| Routing detail (if needed) | `routed.def` | odb Python |

---

## 4. How features feed the model

- **Everything is extracted up front** and stored; models pick subsets.
- **Design + config features** are the cross-design backbone (Group A + B). **Placement
  features** (Group C) are the per-config signal and the staged-imagination target.
- Evaluation is always **leave-one-design-out** (train on N−1 designs, predict an unseen
  design) — the honest generalization test, not config-interpolation on one design.
- Model progression (subsets, each justified by the data): scalars → +placement scalars
  → +spatial maps (CNN) → +netlist graph (GNN). We add richness only when the simpler
  set leaves residual it can capture.

---

## 5. Open items

- **DRC as a target needs signal:** it's 0 in the current safe routable band. To make it
  vary we need an edge-of-routability sweep (higher density / tighter configs).
- **GRT congestion map** not yet archived — add a `global_route` pass if we want the
  intermediate-truth spatial target.
- **Per-corner timing:** decide whether to predict the worst corner or all three
  (tt/ss/ff) for WNS/TNS.
