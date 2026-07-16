# PlaceDreamer — Architecture

**What this is:** a world model of the physical-design flow. Given the netlist at floorplan plus
the knobs, predict what the flow *does* — and let PPA fall out of the predicted state.

**Companion:** `learning.md` (the evidence log — every claim MEASURED / CITED / HYPOTHESIS,
including 7 retractions) and `scripts/stress_test.py` (the claims below, executable).

**Ground rule** (from `docs/architecture.md`, and today proved why): *no claim is DECIDED until
it's confirmed against real runs. No assuming.* Every number here is traceable to a stress-test ID
or a paper read **in source**.

---

## 1. The thesis

> **The flow is one physical STATE, propagated. PPA is a READOUT of that state, not a prediction.**

Everyone else predicts PPA directly and hits a wall: **MasterRTL (ICCAD'23) built exactly a pooled-
GCN graph-level regressor for WNS/TNS/power/area, measured XGBoost beating it, and shipped trees.**
DE-HNN predicts **zero** global scalars. CircuitGNN's readout taxonomy doesn't list graph-level.

They wall out because a global scalar has **n = #designs** (18 for us) while the per-node state has
**n ≈ 10⁶**. So we don't predict PPA. We predict the state; PPA is arithmetic.

### MEASURED — the PPA identities CLOSE (T3, T8)

| PPA | identity | verified |
|---|---|---|
| **Area** | `cell_area = Σ(comb + seq)` | ratio **1.0000** — exact |
| **Power** | `total_power = Σ_gates total_power` | 61,854 vs 61,800 → **0.09%** |
| **Timing** | `slack = T − arrival` ; `wns = min(slack)` | exact to **1 ps**; `meta.wns == min(slack)` to **0.00e+00** |
| **WL** | `Σ_nets length` | 0.77–1.00 — the gap is buffer nets (§5) |

And `gates` carries **per-gate `internal/switching/leakage_power`**; `pins` carries per-pin slack;
`net_arcs`/`cell_arcs` carry per-arc delay+arrival. **The labels for the state are all there.**

### MEASURED — the state exists at EVERY stage, with the SAME types

| stage | gates | per-gate pwr | nets | hpwl | pins(slack) | arcs |
|---|---|---|---|---|---|---|
| floorplan | 9,994 | yes | 4,701 | — | yes | 35,585 |
| global_place | 10,060 | yes | 4,767 | yes | yes | 34,624 |
| place_resized | 10,184 | yes | 4,891 | yes | yes | 37,641 |
| cts | 10,623 | yes | 5,154 | yes | yes | 74,566 |
| global_route | 10,630 | yes | 5,159 | yes | yes | 74,989 |
| detailed_route | 10,630 | yes | 5,159 | len | yes | 74,989 |

**The flow is a state sequence.** Each stage transforms the same state types.

---

## 2. The state, and why arrival is the spine

### MEASURED (T7) — `arrival` chains, and the floorplan gives it to us FREE
`slack = required − arrival` is **exact at all six stages**. Δarrival per stage (ac97, T=3.0ns):

| stage | max arrival | min slack | Δ |
|---|---|---|---|
| **floorplan** | 3.022 | −0.177 | — ← **our INPUT** |
| global_place | 3.031 | −0.215 | +0.009 |
| **place_resized** | 3.711 | −0.917 | **+0.680** ← the resizer does the work |
| cts | 3.800 | +0.007 | +0.089 |
| global_route | 3.672 | +0.100 | **−0.128** ← routing *improves* timing |
| detailed_route | 3.672 | +0.100 | **+0.000** ← settled at global_route |

Endpoints **chain**: 91.2% overlap floorplan→place_resized; **97.8–100% between consecutive stages**.

### 🛑 MEASURED (T7b) — the free prior beats our trained head by **+0.98 R²**
`arrival_place := arrival_floorplan + one global shift`, fold-0 test designs:

| | R² |
|---|---|
| copy + a single global shift, **zero parameters** | **+0.476** |
| our trained 680k-param `endpt` head | **−0.508** |

**We spend 680k parameters predicting from scratch what we already know — and lose to copying it.**
⇒ **every stage predicts Δ on a known prior.** Δ-ML's pattern (a prior buys a persistent
learning-curve offset ≈ a fixed multiple of designs — the currency we lack at n=18). RTL-Timer does
exactly this: their analytic STA arrival is **R=0.26 alone, R=0.86 through the model**.

---

## 3. THE ARCHITECTURE

```
                netlist @ floorplan  +  knobs  +  arrival_floorplan (FREE prior)
                                   │
                    ┌──────────────┴──────────────┐
                    │   DE-HNN ENCODER (shared)   │   ← §4. Barsat's call, and our conv
                    │   directed hypergraph,      │      MATCHES their source line-for-line
                    │   driver/sink split,        │
                    │   METIS VNs + SUPER-VN,     │
                    │   SignNet PE                │
                    └──────────────┬──────────────┘
                                   │  h_cell, h_net, θ = pooled(G)   [θ is CONSTANT in k — F1]
                                   │
   ┌───────────────────────────────┼───────────────────────────────┐
   │  PER-NODE STATE HEADS — this is the model. n ≈ 10⁶ labels.    │
   │    per-net   : Δlength        (HPWL @ place → routed @ route) │
   │    per-arc   : net delay, cell delay                          │
   │    per-pin   : Δarrival       (residual on the prior)         │
   │    per-gate  : power (internal / switching / leakage)         │
   │    per-cell  : area                                           │
   └───────────────────────────────┬───────────────────────────────┘
                                   │
              LEVELIZED PROPAGATION (§6) — topological, max over fanin
                                   │
   ┌───────────────────────────────┴───────────────────────────────┐
   │  PPA = IDENTITIES. Zero parameters. Verified exact.            │
   │    slack   = T − arrival        wns = min(slack)               │
   │    tns     = Σ min(slack, 0)                                   │
   │    power   = Σ_gates power                                     │
   │    area    = Σ_cells area                                      │
   │    WL      = Σ_nets length      (+ §5 buffer correction)       │
   └───────────────────────────────┬───────────────────────────────┘
                                   │
                    + LEVEL-3 CORRECTION (§5) on the aggregate
                      [wns_hat, tns_hat, θ, knobs] → small head
                      (RTL-Timer §3.4.3: "another tree-based model")
```

**The stage boundary is a change of prior, not a change of model.** f_place, f_cts, f_route are the
*same* architecture applied at three points in the sequence, each residualing on the previous
state. **That is what makes the seam coherent** — it carries `arrival`, `length`, `power`: physical,
per-node, chaining at 97.8–100% overlap. Not our old bag of (broken slack + 1 HPWL column + 6
crushed globals).

**The knob never enters the network for timing.** It enters at `slack = T − arrival`. What the
network learns is structure; the knob is applied by arithmetic.

---

## 4. The encoder — DE-HNN, AUDITED

**Keep it.** Our `HyperConv.forward` matches `dehnn_layers.py:49-75` line-for-line (concat order,
shared forward/back conv, residual-to-input). It is the part that works: **net_hpwl AUC 0.912** vs
Net2's 0.922.

**What the audit corrected (all now fixed in code):**
- ❌ "DE-HNN: nets carry no PE" → **their nets DO get PE** (`pyg_dataset.py:145`).
- ❌ "DE-HNN drops nets with fanout ≥3000" → **their filter is a NO-OP BUG** (thresholds the first
  Laplacian eigenvector against 3000 ⇒ zero nets dropped). We are not behind them.
- ❌ "Net2 partition features" → **we do not implement Net2.** Ours is a per-net scalar; theirs is
  per-edge pairwise disagreement (Alg. 2) via **hMETIS on the hypergraph** (we clique-expand with
  pymetis). We have 0 of their 3 net-partitions and lack `f1`, their strongest feature.
  **Our AUC 0.918 is real; the attribution was not.**
- **SUPER-VN** (`SUPER_VN=1`, implemented): DE-HNN's "two-level hierarchy" is **METIS at one
  granularity + ONE GLOBAL ROOT** (`top_part_id = zeros`, `num_top_vn = 1`) — not two granularities.
  Their ablation is **cumulative**, single-design: `+PD 8.765 → +single-VN 8.687 (0.9%) → +two-level
  8.381 (3.5%)`; vs no VN, 4.4%. **We are neither row** — their "single VN" is one *global* VN;
  we run METIS clusters with **no root**, which they never test. Verified wired (+33k params, root
  gradient 2342). **Low priority: a +3.5% single-design encoder tweak is noise next to a −0.98 R²
  architecture error.**
- **Do NOT calibrate to their numbers**: their cross-design script has **val == test**
  (`load_data_indices[10:]` twice) and selects checkpoints on **training** loss, with per-design
  z-scored targets so level transfer is never tested. Our protocol is strictly better.

### Would a TRANSFORMER help? — No. Scientifically:
1. **Cai et al. ICML'23** (the usual justification for VNs/GTs): MPNN+VN approximates a **linear**
   transformer at O(1) width; **full self-attention needs O(n^d) width** — vacuous at our n
   (20k–50k cells, 10⁶ pins).
2. **"Distinguished in Uniform" (2405.11951)**: GT and MPNN+VN are **both non-uniform-universal and
   INCOMPARABLE**. We generalise across designs of very different sizes — that **is** the uniform
   setting, where neither is universal.
3. **Our bottleneck is not global reach.** It is (a) **levelized depth** — ~300 logic levels, which
   TimingGCN solves with a topological **schedule**, not attention or depth; and (b) **n=18 for
   anything global**, which no architecture fixes.
4. **Southern ICLR'25**: a VN's sensitivity to distinct node features is "often uniform" ⇒ a global
   readout **degenerates toward a learned mean**. More global capacity is not the missing thing.
**Verdict: structure (schedule + identities + priors), not attention.** Revisit only if a measured
long-range failure appears that the schedule cannot reach.

---

## 5. The one real complication: THE GRAPH GROWS

MEASURED: gates 9,994 → 10,630 across the flow (**+6.4%**) — the resizer and CTS **insert buffers**.
Our graph is fixed at floorplan, so sums over *our* nodes are incomplete:

| | our share | verdict |
|---|---|---|
| area | buffers ≈ 2.2% of cell_area | small |
| **HPWL** | **0.77–1.00** (ethernet 0.772) | **big — buffers land on LONG nets** |

**MEASURED (T1) — and this caught a real bug:** `meta.total_hpwl` sums **21,517 global_place** nets;
our floorplan graph has **20,806**. Supervising the composed sum against `meta.total_hpwl` would
have **inflated every per-net prediction 13–30%** to close a gap the net head did not cause —
damaging our best head. **Fixed**: supervise against `Σ` over **our own** nets (a true identity).

**The gap is absorbable where it matters (T1b/T1c):** within-design std **0.0103** vs across-design
**0.0567** (ratio **0.18** ⇒ a design-level constant), and
`corr(dev log Σ(ours), dev log meta) = +0.9971` → **R² 0.9942**. So the **knob response** — the part
the agent needs — survives the identity; the level head carries the buffer offset.

**⛔ MEASURED (T2) — the same trick FAILS for routing.** `rt_wl = Σ routed_len`: ratio **0.664**,
gap **NOT** design-constant (0.50), and it tracks only **R² 0.62** of the knob response (vs 0.9942
for HPWL) — **worse than f_route's existing +0.66.** `RT_COMPOSE` is **disabled**.
**Lesson: the two identities look identical and are not. Reasoning by analogy is how I generated 7
retractions in a day.**

---

## 6. Timing — the biggest measured gap

**Current:** one head predicts per-cell **slack** from a synchronous GNN with ~700 labels; WNS/TNS
are read off it. **−0.508 / −1.102.**

**Why it cannot work — four independent reasons:**
1. **Wrong prior.** We ignore `arrival_floorplan`, which alone scores **+0.476** (T7b).
2. **Wrong target.** `slack` carries the knob; `arrival` carries the structure. Per-endpoint CV
   across knob configs (T4): **arrival wins 6/8** — ss_pcm 11.19→0.16, ac97 1.36→0.08.
   RTL-Timer footnote 3: *"fixed clock frequency, implying slack is solely determined by arrival."*
   **Neither TimingGCN nor RTL-Timer predicts slack.**
3. **Wrong architecture.** TimingGCN Table 5, cross-design: vanilla GCNII **−0.84 / −0.78 / −1.51**
   at 4/8/16 layers (deeper is *worse*) vs **+0.8957** levelized. **Our −0.508 is inside that band.**
   Max logic depth ~300 ⇒ depth must come from the **schedule**, not stacked layers.
4. **Wrong aggregation, and starved.** `min` is an ORDER STATISTIC — it reads the error *tail*, not
   the average. And we use ~700 labels when **33,617 arcs/flow** are on disk (T6a — 48×).

**The rebuild (TimingGCN + RTL-Timer, both read in source):**

| stage | what | supervision | verified |
|---|---|---|---|
| 0 | **prior**: `arrival_floorplan` | — free, our input | T7: corr 0.669, R² +0.476 alone |
| 1 | GNN → **net delay** (local) | `net_arcs.delay` | 16,825/flow |
| 2 | GNN → **cell delay** (per-arc) | `cell_arcs.delay` | 16,793/flow |
| 3 | **levelized propagation**, net→cell→net, seeded at PIs, **`max` over fanin** | arc `arrival_time` | T6 |
| 4 | `slack = T − arrival` | **IDENTITY** | exact to 1 ps (T3a) |
| 5 | `wns = min(slack)`, `tns = Σ min(slack,0)` | **IDENTITY** | 0.00e+00 (T3b) |
| 6 | **learned CORRECTION** on `[wns_hat, tns_hat, θ, knobs]` | design WNS/TNS | RTL-Timer §3.4.3 |

**MEASURED (T6b): net delay median 0.297 ns vs cell delay 0.0001 ns — 2970×.** Pre-route, **wire
delay IS the timing**, and wire delay is a function of net length — **the head we already score
AUC 0.912 on.** The chain closes on our strength.

**Details that matter (VERIFIED from TimingGCN source):**
- **`sum` AND `max` reduction channels in parallel**, sigmoid-gated, concat+MLP (`model.py:60-61`).
  A mean/sum-only aggregator **cannot represent `arrival = max over fanin`** — the core STA operator.
  Hard max (subgradient), not softmax.
- **Teacher forcing** (`model.py:85-88`): `groundtruth=True` swaps the predecessor's *predicted*
  embedding for the true one and **skips the topological loop entirely** — training is O(1) levels
  instead of ~300. **This is what makes it tractable**, and it is **our seam at the timing-graph
  level**. Track free-running vs teacher-forced loss to watch the exposure gap (`train_gnn.py:130`).
- Loss: three **unweighted** MSE terms (all 1.0). No NLL. Don't over-think weighting.
- Transforms: arrival **raw**; slew `log(1e-4+x)+3`; net delay `log(1e-4+x)+7.6`. ⚠️ They keep
  arrival raw because 21 designs share one PDK/clock; **we are cross-design** → normalise `AT/T_clk`.
- Aux ablation: net-delay alone **0.8513**, cell-delay alone **0.8150**, full **0.8957** ⇒
  **net delay is the more valuable auxiliary.**

---

## 7. Geometry — the question was wrong

CTS needs it (SwiftCTS: placement adds **+0.124** to clock_buffers once CTS knobs vary). But:

**MEASURED — our two "failures" were CORRECT.** Placement is a global optimisation with a **gauge
freedom**, so the conditional mean of `p(position | local topology)` **genuinely IS the die centre**.
Our NLL landing there (0.397 vs a 0.395 baseline) is the right answer to the question we asked. The
VN box (0.253 vs 0.242) is the same fact coarser — `(xmin,ymin,xmax,ymax)` is **four absolute
coordinates**, same gauge.

**CITED — the literature agrees by omission and by ablation:**
- **No paper reports per-cell cross-design position accuracy.** TransPlace had every incentive; it
  reports none.
- **TransPlace Table 11**: the GNN head **alone** is **25,753× worse OVFL / 2.47× worse RWL** without
  a gradient fine-tuner. It is a **warm start for a placer**, not a geometry predictor.
- **MacroRank DISCARDS standard-cell positions** (`fake_pos[macro_index] = node_pos[macro_index]` —
  every non-macro zeroed) and builds cluster boxes from **areas + a density target (Eq 5) — no
  position**.
- **Translation invariance ONLY.** MacroRank **measured** it: translation → WL std **<0.5%** (holds);
  rotation/flip → *"very significant impact"* (**broken**). TransPlace assumed SE(2) and pays with a
  hand-tuned per-netlist `Theta ∈ {0,72,…,300}` + `Δx,Δy`.

**⇒ THE REFRAME:** f_cts does not need *which sink is where*. Clock WL/skew/buffers are functionals
of the **sink POINT SET** — permutation-invariant over sinks. We were solving identity-resolved
localisation: harder than needed **and** unlearnable.
- **A.** Sink **density map** (K=16/32, die normalised to 1×1). Reuse MacroRank's
  `util.py:41 get_ensity_map()` (differentiable bilinear splat). **Loss: Sinkhorn/OT or soft-histogram
  CE — NOT per-pixel L2**, which collapses to the marginal exactly as our NLL collapsed to the centre.
  Beat the **train-set-mean heatmap**.
- **B.** **Translation-invariant scalars**: sink count, bbox **extent (W,H — not corners)**, radius of
  gyration, mean pairwise distance. Retarget `h_vnbox` from corners → `(w,h,area,aspect,R_g)`.
- **C.** Anchored relative coords (TransPlace) — **skip**; it needs the fine-tuner, i.e. a placer.
- **Evaluate on DOWNSTREAM f_cts error, not position error** — which is what both papers do.

---

## 8. What is NOT the fix (retracted — see `learning.md`)

- **die_area as a feature** — 0.702 → 0.702. Within a design `log(die) = log(cell_area) − log(util)`,
  cell_area design-constant ⇒ **collinear with the utilization knob**.
- **crit_path in f_place** — fair gain 0.091→0.118; probe showed **+0.001** model gain. (Different in
  f_route: power 0.158→0.354.)
- **√(n·A) as built** — the law is `√A·N^p`, p∈[0.5,0.75] (Donath); ours is the p=0.5 case. Salvaged
  only because `dfeat` carries `log A` and `log N` separately, so the head can **learn p**.
- **Replacing the WNS readout with a direct head** — BACKWARDS. The readout **is** MasterRTL's
  architecture. The defects were the prior, the target, level-1, and the missing level-3.
- **RT_COMPOSE=sum** — T2. Disabled.
- **"DE-HNN demand Pearson 0.372 vs 0.723"** — that was **their** number (0.372/0.683). We dropped
  `net_dem` as a target and never measured it.

---

## 9. Priority

1. **The Δarrival prior** (§2, §6 step 0). **+0.98 R² measured**, zero parameters, one cache.
   Nothing else on this list is within an order of magnitude.
2. **Timing rebuild** (§6): per-arc delay heads + levelized propagation + identities + correction.
   −0.508 → a demonstrated +0.90, with 48× more supervision already on disk.
3. **PPA as readout** (§1): power = Σ per-gate, area = Σ per-cell. Identities verified exact; the
   per-gate labels exist. This is the contribution nobody has.
4. **Geometry as point-set** (§7) — gated on the f_cts downstream metric, not position error.
5. **Super-VN** (§4) — implemented, untested, low priority.
6. **Data** — CTS knobs (SwiftCTS: 5,400 runs, 540 placements × 10 configs) and more designs.
   **n=18 is the ceiling on everything global; 1–5 only make the model correct.**

---

## 10. Open / needs data or a decision

- **The graph grows** (§5). We model a fixed floorplan graph. Buffers are ~2% of area but up to 23%
  of HPWL. Absorbable as a design-level constant for HPWL (T1b); **unresolved for routing** (T2).
  Do we predict *which* cells get added, or stay with a level correction? **UNDECIDED.**
- **f_cts's clock representation.** We encode `is_sink` + activity. GAN-CTS feeds sink-distribution
  **images**; Kahng SLIP'13 uses an explicit **clock entry point** (`M_CEP`) and shows **fall delay
  varies 43%** across entry points at fixed aspect ratio. **What must the graph EXPRESS about the
  tree?** A real clock-source node changes what is expressible. **NOT YET STUDIED.**
- **Papers that would help**: anything on (a) predicting *cell insertion* (buffering) as a graph
  operation, (b) cross-stage/flow-level state propagation in EDA, (c) OT/Sinkhorn losses on point
  sets for layout.
