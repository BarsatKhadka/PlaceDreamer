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

---

## f_place — first trained result (3-fold leave-designs-out CV, DE-HNN + knob/anchor conditioning)

**Target set (final).** Dropped `net_dem` (per-net RUDY-from-bbox was −0.94 correlated with
`net_hpwl` — the same bounding box twice, a circular target). f_place now predicts the placement
state: **per-net HPWL · total HPWL · buffer area · buffer count · WNS · TNS**, each as (μ, logvar).

**3-fold CV on unseen designs (`runs/cv_readout`):**
| target | R² (mean±std) | within-design r |
|---|---|---|
| tot_hpwl | 0.944 ± 0.028 | 0.63 |
| buf_area | 0.884 ± 0.104 | 0.34 |
| buf_cnt  | 0.803 ± 0.085 | 0.31 |
| net_hpwl | 0.518 ± 0.019 | — |
| wns      | **−1.93 ± 2.31** | **0.83** |
| tns      | **−4.43 ± 3.77** | **0.87** |

**The finding (measured, not assumed): absolute cross-design timing level is COVERAGE-limited,
not model-limited.** Per fold, WNS R² = −1.26 / −5.03 / **+0.49**. It is POSITIVE (fold 2, small/mid
test designs whose level is inside the training span) and catastrophic (fold 1, contains ethernet,
48k cells — a size EXTRAPOLATION). Same model, same readout; the only variable is interpolation vs
extrapolation. The ±2–4 std IS the result: timing is learnable, we lack coverage. Confirmed 5 ways:
GBT baseline (−0.42), the floorplan anchor only half-helped, the slog-residual made it WORSE
(between-design variance 75%→102%), the variance decomposition (~70% of timing variance is
between-design), and the fold-2-positive / fold-1-negative split.

**What works:** geometry (tot 0.94), buffers (0.80–0.88), and the timing **knob-response**
(within-r 0.83–0.87 every fold — the model learned the physics; the RL agent ranks knobs per
design, so this is most of what it needs). Per-net HPWL 0.52 is a hard, consistent floor.

**Loss / training lessons (all cost a run to learn):**
- Normalization stats must come from TRAIN designs only, stratified over all of them. Original bug:
  `sorted(glob)[:40]` = 40 flows of ONE design (ac97) → every design z-scored by ac97's stats.
- Targets must be standardized (log-means spanned 2.4→11.2; tot_hpwl swamped the gradient).
- Feature transforms: log1p heavy tails (fanout max 10k), leave indicators raw 0/1 (z-scoring a
  0.004%-ones binary → +41σ), Laplacian PE raw (unit-L2, bounded), kill dead dims (height const).
- Plain Gaussian NLL collapses the variance head (d(NLL)/dμ ∝ 1/σ² runaway). Fix = DECOUPLED loss:
  MSE trains μ, NLL(stopgrad(μ)) trains σ. Both from step 0, no warm-up, no flatline.
- Global readout for WNS must include MAX-pool (WNS is the WORST path, not a mean) + a direct
  conditioning skip. This jumped tot 0.75→0.94 and recovered timing within-r 0.39→0.83. But raw
  max-pool is size/OOD-sensitive → amplifies the extrapolation blowup (candidate fix: attention pool).

**Calibration (σ/rmse) is worst exactly where the model extrapolates** (0.08 on the ethernet fold,
0.60 when interpolating) — a single model doesn't know it's out-of-distribution. This is the case
for the ENSEMBLE (epistemic uncertainty = member disagreement), which the grounding loop keys on.

**Next levers (earned, not guessed):** (1) MORE training designs spanning the size/timing range —
the only thing that moves absolute cross-design timing; (2) ensemble — fixes calibration + gives
honest OOD uncertainty; (3) attention pooling — tame the max-pool OOD blowup. Congestion stays a
`data_gen` problem (real target = per-tile RUDY / router demand; EDA-Schema `routability_metrics`
is empty).

---

## f_place v2 — timing reformulated as PER-ENDPOINT slack (WNS/TNS become readouts)

**Why.** v1 predicted WNS/TNS as two global scalar heads. TNS is *extensive* (a sum over the
whole chip) → on OOD-large designs the magnitude is unseen → catastrophic extrapolation
(tns R²=−8.98 on the ethernet fold). Root cause: timing LEVEL is between-design idiosyncratic
and the head regresses an unbounded total.

**Reformulation.** Slack physically lives on ENDPOINTS (register D-pins), not the whole chip.
So predict **per-endpoint slack** on the cell nodes (a per-cell masked head, like per-net HPWL),
and **read out WNS = min, TNS = sum-of-negatives** from those predictions. Per-endpoint slack is
*intensive* — a register's slack is a bounded number regardless of chip size → transfers across
sizes; the global heads are gone.

**Data (validated before building).**
- `scripts/add_endpoint_slack.py` → `cache/endpt/{flow}.npz` = `ep_idx` (endpoint cell indices),
  `ep_slack` (worst setup slack per endpoint) @ place_resized. 1944 flows, ~745 endpoints/flow.
- Source = `timing_paths` (setup, worst-per-endpoint). Register endpoints map to floorplan
  cells **100%** across all 18 designs (join by cell name, no graph rebuild).
- **WNS reconstructs EXACTLY** from these on all 18 (worst path always in the report).
- **TNS reconstruction is truncated** (`timing_paths` is a top-N report): 9–100% coverage
  (ethernet 32%, mem_ctrl 9%, jpeg 99%). So the per-endpoint sum UNDERCOUNTS TNS on some
  designs — a known v1 limitation; v2b can add the recorded-total (complete) as an aggregate
  sum-constraint in the loss to close it.
- **Primary-output endpoints** (0 in most designs; ~50% in wb_dma) are NOT cell nodes → dropped
  in v1. They can set WNS on I/O-heavy designs (wb_dma: worst is a PO). Add via the output-net
  node in v2 if PO-heavy designs read out badly. The high-TNS designs are 0% PO, so v1 covers them.

**Code.** fplace: dropped `h_wns`/`h_tns`; added `h_endpt` (per-cell, ctx-skip readout) + slack
norm stats; `load_graph` loads endpoint labels (`y_endpt`,`m_endpt`,`ep_idx`) + raw recorded
`wns_true`/`tns_true` for eval. train: `evaluate()` computes per-endpoint R²/calib AND reads out
per-flow WNS(min)/TNS(sum-neg) vs recorded. Loss weight `W_ENDPT`. NOT yet run on the cluster.

**Open question for the run:** does WNS(min-readout) now transfer better than the old global
head, and does TNS undercount as the coverage table predicts? The numbers decide v2b (aggregate
constraint) and v2c (PO endpoints).

### v2b — supervise the READOUTS directly (soft-min WNS + sum-constraint TNS)

**v2 result (fold 0, `runs/probe_endpt`):** per-endpoint slack LEARNS (val ep R²=0.83) but the
readouts collapsed — val WNS R² = **−2.57** (was +0.92 with the old global head), and on unseen
designs `endpt` itself was −0.42.

**Diagnosis: `min` is a fragile readout.** Endpoints trained to be right ON AVERAGE don't protect
an EXTREME statistic — one endpoint predicted too-negative becomes a spurious minimum and tanks
WNS. The old global head regressed WNS *directly* so it never had this failure mode.

**v2b fix — train the readout, not just the average:**
- `L += (softmin_TAU(pred endpoint slacks) − recorded_WNS)²`. Hard `min` gives gradient to ONE
  endpoint; **soft-min** (temperature-weighted, TAU=0.1) spreads it over every near-worst endpoint.
  Verified soft-min tracks hard min to 0.007–0.03. Eval still reports the TRUE hard min.
- `L += (slog(Σ neg pred slacks) − slog(recorded_TNS))²`. The per-endpoint LABELS are truncated
  (ethernet: 32% of violations), but the RECORDED TNS is complete → this term supervises the
  endpoints that have no individual label. That is its entire purpose.
- Weights `W_WNS_RO` / `W_TNS_RO` (default 1), `TAU` env-tunable.

Verified locally: soft-min faithful, readout loss backprops to the head (grad-norm 81.6).
**If v2b also fails → stop patching the target and revisit the architecture properly.**

**v2b RESULT: FAILED (converged, fold 0).** val WNS **−0.59**, val per-endpoint **−0.29**.
The readout supervision stopped v2's catastrophic collapse (−2.57) but never reached the OLD
GLOBAL HEAD's +0.92 — and it *destroyed* per-endpoint accuracy (v2 had +0.83). The model games
the min/sum constraints while getting individual endpoints wrong: worst of both.

**One real finding, though: the SUM constraint works, the MIN readout doesn't.** TNS reached
**+0.44** (best we've seen) — sums are robust to per-element error. WNS-via-min stays fragile no
matter how it's supervised, because a minimum is decided by ONE endpoint and nothing forces the
model to get *that* one right.

### Verdict on target reformulation — STOP.
Four formulations tried and measured on the same fold:
| formulation | val WNS | cross-design |
|---|---|---|
| global head (v1)               | **+0.92** | −1.9 avg (coverage-limited) |
| + floorplan anchor             | +0.92 | −0.49 (helped, not enough) |
| per-endpoint, min-readout (v2) | −2.57 | endpt −0.42 |
| + readout supervision (v2b)    | −0.59 | — |
None beat the global head. The target is not the problem — **the architecture is next.**
Keep from this line of work: (a) the global WNS/TNS heads, (b) the TNS sum-constraint idea,
(c) per-endpoint slack as an AUXILIARY signal (it learns fine at +0.83 when not fighting the
readout loss, and f_route will want it).

---

## Literature check — what the field actually does (and what it refuses to report)

Surveyed the papers that predict placement metrics from a PRE-placement state. Findings that
directly change what we build and how we measure.

### 1. Our per-net "failure" is a documented, EXPECTED result — not a bug
**Net² (Xie et al., ASP-DAC'21, arXiv:2011.13522, §3):** in one design they found **725 nets with
identical driver area, cell count and one-hop neighbour info** whose post-placement lengths ranged
**1 um to >100 um**. A model with only local features *"cannot distinguish these nets at all."*
Their thesis: *"It is not likely to achieve high accuracy without accessing any GLOBAL information."*

Our net node has 4 features (fanout + 3 flags). Collapsing to a fanout curve is exactly what the
paper predicts. **No loss/architecture tuning fixes this** — it needs global structure.

### 2. THE fix — partition-disagreement features (Net² §4.2, Alg. 2)
Run hMETIS at **7 cell granularities** (#cells/100,200,300,500,1000,2000,3000) + 3 net
granularities. For each edge, compute **cluster-ID DISAGREEMENT** between the two endpoints'
neighbourhoods. Cells in different clusters get placed far apart in any good placement -> cluster
disagreement is a **pre-placement proxy for physical distance**, which is what HPWL IS and what
fanout can never express.
**Their ablation (Table 7) is the kicker:** an "Edge ANN" with the partition features and NO GNN
scores **AUC 88.2**, vs 92.2 for full Net² and 69.8 for a cell-count baseline. **The partition
features carry most of the signal, not the message passing.**
We ALREADY run METIS for the virtual nodes and throw the partition away. Cheapest, highest-leverage
change available.

### 3. Net features: they use 12, we use 4
Net² Alg.1: fan-in size, fan-out size, **driver cell area**, **sum of ALL cell areas on the net**,
plus **sum AND std of neighbouring nets' fan-in/fan-out sizes, kept SEPARATE for in- vs
out-neighbours**. We have none of the areas, no direction split, no second-order stats.

### 4. Buffering needs ELECTRICAL features (we feed none)
GraPhSyM (arXiv:2308.03944, Table I): input cap, **driven cap**, slew, per-pin delay. Buffers are
inserted for **slew and max-cap violations, NOT fanout** (Kahng ISPD'26 §2.2). A fanout-only model
*structurally cannot* predict buffering. Also: GraPhSyM labels each output pin with the **summed
area of the buffer tree attached to it** and totals ANALYTICALLY — no global pool. They name our
exact trap: predicting an absolute post-value when the input is close to it makes the model copy
the input.

### 5. Timing: MasterRTL does NOT pool (ICCAD'23, arXiv:2311.08441, §II.B)
Predict **per-path delay**, then compute `WNS = min(clk - delay)` and `TNS = sum(clk - delay)`
**analytically** — so `clock_period` enters as the literal constant in the slack equation, not as a
soft conditioning input. Then calibrate with a small tree on design scale + slack percentiles.

### 6. Knob injection — LOSTIN ablated this on an UNSEEN-DESIGN split (arXiv:2201.08455 §III-A)
Late concat of [graph emb || knob emb] WINS (3.11% area MAPE). A **knob SUPERNODE collapses to
40.7%** — the knob signal *"is gradually diluted"* by message passing. Per-node knob broadcast is
also called out as a failure mode. Our direct ctx-skip is on the right side of this; our VN
injection is the thing they say fails. **Worth an ablation.**
HARP (ICCAD'23 §IV-B2) goes further for PER-NET conditioning: a per-knob MLP that TRANSFORMS node
embeddings (FiLM-like), then ONE more message-passing layer to propagate the knob effect. Beats
plain concat by 12-19% RMSE. (Caveat: their eval is within-design across knobs.)

### 7. HONEST CALIBRATION of our numbers
**Nobody publishes cross-design absolute per-net HPWL error.** Net²'s full text has ZERO
MAE/RMSE/MAPE. They report top-10%-longest-net **ROC-AUC (92.2)** and a **20-bin correlation
(0.98, with the top 5% of nets EXCLUDED)** — binning cancels per-net error.
~~**Net², MacroRank and Huang (DATE'19) ALL independently abandoned absolute regression for
ranking.**~~ **RETRACTED 2026-07-16 — verified FALSE from the Net2 PDF.** Net2 *regresses* net
length: *"The net length is the label for training and prediction. Each net's length is the HPWL of
the bounding box of the net after placement."* There is no ranking loss. What is TRUE: Net2 never
publishes a um-level absolute error — it reports only binned correlation (>0.98) and top-10% AUC
(**92.2** for nets; their 92.5 is for PATHS), and calls the binned protocol "a classical criterion
used in many net length estimation works". So the FIELD evaluates this ORDINALLY — a convention,
not an impossibility proof. MacroRank's stated reason is decision-theoretic, not
under-determination: *"the relative relationship between them is noteworthy instead of the absolute
value ... a ranking model, rather than a regression model, is needed"*, and they show the
dissociation (EHNN has the best MRE but ~0 Kendall tau until the loss is swapped).

Cross-design reference points:
  - DE-HNN cross-design per-net log2 RMSE 1.677 (Pearson 0.754) => ~3.2x multiplicative tail error
  - MacroRank cross-design TOTAL-WL MRE: **24-49%**
  - MasterRTL cross-design WNS: R 0.92, **MAPE 27%** — nobody has shown <15%
  - Ghose cross-design node-level congestion Pearson **0.23-0.31** <- the real SOTA for hard
    cross-design node-level regression. Anyone reporting 0.9 is reporting WITHIN-design.
=> **our ~40% median per-net rel err (cross-design) is in the plausible band, and is MORE than
   anyone else is willing to report.** We are not an outlier.

### 8. The ceiling (stated, not guessed)
~~Per-net absolute HPWL from a pre-placement netlist is **fundamentally under-determined** — Net²'s
725-net example is a proof by construction.~~ **PARTIALLY RETRACTED 2026-07-16.** The 725 nets
(identical local features, lengths 1um..100um, design B20) are REAL — but **Net2 uses them to argue
for a GLOBAL RECEPTIVE FIELD, not to drop regression** (Net2a's fix is an edge-convolution reaching
the whole netlist). The honest statement: the field evaluates net length ordinally by convention
and does not publish absolute error; that is not the same as proving regression impossible.
NOTE the contrast with PLACEMENT GEOMETRY, where under-determination IS established:
absolute per-cell POSITION has a genuine gauge freedom (placement is a global optimisation), our
NLL head provably converged to the die-centre baseline (0.397 vs 0.395), and TransPlace's own
Table 11 shows their geometry head alone is 25,753x worse on overflow without a gradient
fine-tuner. Position: under-determined. Net length: merely evaluated ordinally.
=> ADDED top-10% AUC / recall@10% / 20-bin correlation to evaluate(). f_route needs to know WHICH
nets are long and congested, not their exact length — that is a ranking problem, and it is the
achievable one.

### 9. Independent confirmation of our "coverage-limited" finding
MasterRTL Fig.11: path-model accuracy collapses R 0.93 -> 0.46 as training designs shrink; their
fix was **generating synthetic RTL designs**, not a better model. SwiftCTS: clock power varies 8.9x
across design families but only 10-30% across knobs within a design — the cross-design scale term
dominates the knob term ~30x. Exactly our decomposition finding, arrived at independently.

### 10. Our buffer target appears to be UNPUBLISHED
Extensive search found no pre-placement buffer count/area predictor. Kahng's ISPD'26 contest states
tools deliberately defer buffering to post-detailed-placement because interconnect delay can't be
estimated earlier. Novel — but there is a headwind and no baseline. Closest reference: GraPhSyM's
**15.6% area MAE on unrelated designs** (vs 3.86% within-family — a 4x degradation on leaving the
training family).
