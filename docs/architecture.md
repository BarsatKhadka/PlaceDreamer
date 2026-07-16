# PlaceDreamer — Architecture

*Living doc. We decide step by step, one choice at a time.*

> **THIS FILE = the SYSTEM** (pipeline, agent, MBPO training, grounding loop, decision log).
> **[`docs/model_architecture.md`](model_architecture.md) = the MODEL** (encoder, heads, state,
> identities, the seam) — rewritten 2026-07-16 from papers read IN SOURCE (DE-HNN, Net2, TimingGCN,
> MasterRTL, RTL-Timer, TransPlace, MacroRank) and stress-tested against the real data.
> **[`learning.md`](../learning.md)** = the evidence log (MEASURED / CITED / HYPOTHESIS + 7
> retractions). **[`scripts/stress_test.py`](../scripts/stress_test.py)** = this file's ground rule,
> executable.
>
> **The headline that changes this document:** the flow is **ONE STATE (arrival, length, power,
> area) propagated**, and **PPA is a READOUT of that state, not a prediction**. The PPA identities
> are verified EXACT (area ratio 1.0000; power 0.09%; `wns == min(T − arrival)` to 0.00e+00). A
> **zero-parameter copy of the floorplan arrival beats our trained timing head by +0.98 R²** — we
> predict from scratch what the input already tells us. See `model_architecture.md` §1–§3.

**Ground rule: no claim is DECIDED until it's confirmed against real runs.** No assuming.
When we reach a decision, we open the actual archived data, check the claim, and only then
write it down. Every number and shape in this doc must be traceable to something we looked at.

---

## 0. The pipeline — what feeds what

Two model stages, with the agent picking knobs before each. CTS added later.

```
          design / netlist
                │
   agent ──────►│  (picks floorplan / placement knobs)
                ▼
           ┌─────────┐
           │ f_place │
           └─────────┘
                │
                ▼
         placement state
                │
   agent ──────►│  (picks routing knobs)
                ▼
           ┌─────────┐
           │ f_route │
           └─────────┘
                │
                ▼
          routing + PPA
```

*(Stage details — synth in/out, floorplan split, etc. — filled in slowly later.)*

---

## 1. Representations
*Two things to settle, in order:*
1. **How we encode the netlist** ← in progress
2. **What we get as the placement state**

### 1.1 How we encode the netlist

**Graph structure — reasoned (from DE-HNN + our scale):**
- **Directed hypergraph**: a net = one driver cell + its set of sink cells (DE-HNN's core).
  Keeps driver/sink roles separate (matters for timing). This is our hyperedge representation.
- **Virtual nodes: INCLUDE (default on), via DE-HNN's METIS-partition hierarchy.**
  Grounded in *measured* diameter of our real netlists (bipartite cell↔net, layers-to-cross,
  from `placement.odb`): gcd ~1, usb ~1, picorv32 ~8, sha256 ~7, aes ~10. Mid designs need
  **7–10 layers** to span — *beyond* the ~3–6 where plain GNNs over-smooth → a normal-depth
  GNN can't reach across, so VNs are justified. Diameter grows with size, and big designs
  (microwatt ~10×) are coming, so we bake VNs in for all designs. The hierarchy (VNs per
  METIS partition) **scales with graph size** — few regional VNs on gcd, many on microwatt —
  so no single global-average over-smoothing. Built as a **toggle (default on)** so we can
  still confirm it's net-positive at training. Caveat: high-fanout nets (clock/reset,
  excluded from the measurement) already act as *natural* hubs — their help-vs-over-smooth
  role is a separate thing to watch.
- **Persistent homology: still a later ablation** (compute-heavy; contribution unproven for us).
- **Laplacian PE: INCLUDE.** Top-k Laplacian eigenvectors as node features = a "graph GPS"
  (global position each node otherwise can't know from local message passing). Cheap (one
  eigendecomp/design), and congestion depends on global span → likely helps. Random sign-flip
  during training (eigenvector sign ambiguity).

**Feature direction set (exact list refined in `docs/features.md`; extract from `placement.odb`):**
- **Cell node:** cell type (synth.v) · width/height/area + #pins (LEF/odb) · degree · Laplacian PE
  · *(later: timing/power from `.lib`, FastTuner-style)*
- **Net node:** degree/fanout · net type (clock/reset/signal) · *(Laplacian PE)*
- Driver/sink flag comes free from pin direction (`.Y`=driver, inputs=sinks). All the above is
  in `placement.odb` (netlist+LEF merged). *Confirm exact extraction against a real `synth.v`
  when we implement — deferred, not blocking.*

## 2. f_place  (netlist + knobs → placement state)
*In progress. Designing this = answering §1.2 (the placement state is exactly f_place's output).*

**What it does:** takes the (fixed, post-synth) netlist + the placement knobs (the agent's
action) → predicts the **placement state** (the intermediate we imagine after placement).

**Placement state = graph-native OUTCOME quantities (Option A), NOT positions.** *(provisional)*
- **Per-cell** local congestion, **per-net** predicted length/HPWL — values on the graph nodes.
- **Why not positions:** predicting per-cell (x,y) = reimplementing the placer — huge output,
  overlap/legalization constraints, and it encodes the *algorithm* when our aim is to **learn
  knob effects from data**. We don't need positions anyway: f_route and the final metrics need
  the congestion picture / outcomes, not coordinates. So predict the *result* of placement.
- **Consequence:** no positions → the graph⊗grid / MAGNet / CNN spatial machinery does **not**
  drive the imagined pipeline (it needs positions to scatter onto a grid) → f_route is
  graph-based too. A spatial heatmap stays a "maybe later, via a coarse spectral anchor," not core.
- Laplacian PE gives *graph* position (topological, knob-independent) — a cheap head-start
  toward placement, not physical position itself.

**Architecture:** DE-HNN backbone (directed bipartite conv + VN hierarchy + Laplacian PE) +
knob conditioning + per-element heads. Not "just a GNN" — a *knob-conditioned* one (that
intersection — netlist-GNN predicting per-cell/net outcomes *as a function of P&R knobs* — is
unoccupied in the literature; it's also the right lever against the ceiling, see below).

**Knob injection (how the action conditions f_place):**
- Encode the knobs in a **separate small MLP branch** (~6 *unordered scalars*: util, density,
  aspect, routability, time — NOT a sequence, so no LSTM). Evidence: LOSTIN — circuit-graph and
  flow-knobs are "separate concepts," encode separately.
- Because our outputs are **per-node**, inject the knob embedding **globally via the
  virtual-node / supernode** (knobs broadcast to all nodes during message passing). LOSTIN's
  supernode is the evidenced per-node-reaching option, and **we already have VNs** → free.
- **Do NOT** concat knobs onto every node feature (over-redundant — LOSTIN). **FiLM is an
  untested alternative, not a recommendation** (LOSTIN itself fuses by concat, not FiLM; and its
  concat is graph-level, which doesn't serve our per-node outputs).

**Outputs (placement state) — driven by what the downstream needs:**
- **Per-net length / HPWL** → total wirelength (sum), *and* the placement-part of **timing**
  (wire delay) and **power** (wire capacitance). Most predictable target (Net² ~0.98 single-design).
- **Congestion** (per-cell and/or per-net routing demand, RUDY-like) → **routability / DRC**, and
  the **gap between HPWL and routed WL**. Harder/noisier (cross-design ceiling ~0.37).
- These two = **DE-HNN's proven targets** → evidence-backed, not invented. Aggregate HPWL is
  derivable (sum). Timing paths + power are computed downstream from these + netlist/library.
- **Buffer count (added cells)** → *both* a target (emergent area/power — buffers cost both) *and*
  the **composition bridge** (see below). f_place predicts placement + **resize-buffer count**;
  f_cts predicts **clock-buffer count**. Labels from `area_metrics.buffer_area` / buffer-cell
  counts, stage-resolved (fillers excluded — die-determined). Learnable: buffers ∝ clock fanout +
  timing tightness (driven by the `clock_period` knob).

**The buffer / stage-mismatch problem, and the fix (the composition seam, §5c):** the netlist
*grows* across stages — resize adds ~timing buffers, CTS adds ~clock buffers, each *splitting*
nets (e.g. aes_core 11,683 nets @ floorplan → 12,403 @ route). So **per-net predictions from an
early stage don't align with a later stage** — feeding f_place's per-net vector into f_route is a
mess. Fix:
1. **Anchor the whole imagined pipeline on ONE netlist** (floorplan, pre-buffer). Both f_place and
   f_route operate on it → no mid-pipeline net mismatch. Buffering happens in reality (labels
   include it) but is *predicted/absorbed*, not passed as per-net structure.
2. **Compose via buffer-robust state**: aggregate HPWL + per-cell congestion field + **predicted
   buffer count** — NOT the exact per-net vector. Per-net HPWL stays an f_place *auxiliary/diagnostic*
   head (for its own accuracy + inspectability), not fed forward net-by-net.
3. **Predicting buffer count makes buffering explicit** → f_route conditions on it instead of
   blindly absorbing it → explains variance otherwise dumped into the noise floor.

**The ceiling (design around it, don't fight it):** netlist-only prediction is fundamentally
limited — cross-design congestion tops out ~0.37 Pearson for *all* models (DE-HNN), and
logically-null netlist/floorplan changes swing routed WL ±7–11% (Chan/Kahng noise floor). Our
setting is cross-design (leave-one-design-out), so expect this. Two responses: (1) **knob
conditioning** attacks the source (it fixes *which* placement is realized → less
under-determined); (2) **predict uncertainty**
(distribution/quantiles or seed-ensemble), don't output a false-precision point estimate.

> **VERIFIED 2026-07-16 (this was the flagged "verify before quoting" — now done, from the PDFs):**
> Net2's "~0.98" is its **20-BIN CORRELATION** (bin by true length, top 5% EXCLUDED, correlate bin
> means) — **not** an absolute-accuracy number. Its net AUC is **92.2** (92.5 is for PATHS). Net2
> **does regress** net length; it simply never publishes a um-level error.
> Our own numbers on the same axes: **AUC 0.912**, 20-bin r **0.956**, absolute rel-err **43.7%**.
> Also verified: **DE-HNN's own cross-design script has val == test** (`load_data_indices[10:]`
> twice) and selects checkpoints on **training** loss, with per-design z-scored targets — so its
> "cross-design" numbers never test level transfer. **Do not calibrate against them.**

## 2.5 f_cts  (placement state + CTS knobs → post-CTS state)  — LATER
*Deferred. The clock-tree stage; slots between place and route once the 2-stage core works.*

## 3. f_route  (placement state → routing/PPA)

**Key survey finding — position-free viability splits by metric** (our input is graph-native, no
positions): **WL and WNS predict well position-free** (routed WL ~7% error from HPWL+congestion;
SOTA timing predictors are *already* graph-native, no grid). **DRC/congestion genuinely needs the
2D map** (aggregate R² collapses 0.55→0.035 with macros) → handle DRC by grounding, not pretending.

**Input:** netlist graph (bipartite cell↔net) enriched with the placement state as node features:
net nodes ← per-net length; cell nodes ← per-cell congestion (+ the usual type/size/degree). Route
knobs via VN/supernode slot (fixed in our data → v1 off).

**Not one uniform GNN — a shared light encoder (2–3 DE-HNN layers) + metric-matched heads:**
- **WL head** *(v1, solid ~7%)* — per-net routed length = `per_net_length × (1 + detour(local
  congestion))` (bake in the `RSMT × ratio(congestion)` physics), then **sum-pool → total routed
  WL**. Light (each net sees its own cells' congestion, ~1 hop).
- **power head** *(v1, untested)* — per-net `length × activity` → sum-pool. Rides on WL.
- **WNS head** *(v2)* — **STA-style *levelized* propagation** through the timing DAG (arrival =
  cell delay + wire delay∝length), read out min-slack. NOT the bipartite conv (generic GCN →
  negative R²). TimingPredict/PreRoutGNN pattern. TNS noisier than WNS.
- **DRC head** *(coarse/grounded)* — position-free can't give a calibrated count; aggregate
  congestion (max/overflow pool) → **risk score (low/med/high)**, a ranking not a count. Or defer
  and ground with a real route when DRC matters. *(DRC is doubly weak: needs positions AND is ~0
  in our current data.)*

**Cross-cutting:** ensemble + probabilistic outputs (mean+var) per head — uncertainty as before;
**train on real + imagined placement states** (the compounding fix — where the seam is built).

**Why matched heads:** WL/power are *sums* (pooling), WNS is *path-accumulation* (levelized MP),
DRC is *local-spatial* (position-free can't). One pooled GNN for all four would tank timing and
over-promise DRC.

**v1 build order:** WL → WNS → power → DRC(coarse/grounded).

**Novel niche:** graph-level *pooled* readout of total routed WL / WNS from positionless per-net/
per-cell features — DE-HNN/CongestionNet are per-node only; whole-chip position-free pooling is
unoccupied. *(Numbers from survey — verify before quoting.)*

> **North star:** our only aim is to predict **both the intermediate states and the final
> metrics** accurately. So every representation choice (esp. the placement state) is judged by
> one thing: **does it carry enough for the downstream stage to predict its output well?**

## 4. Agent — amortized knob policy  (crucial, not an afterthought)

**Goal: a policy that picks *optimal* knobs — trained ONCE** (against the world model, across many
designs), then deployed **zero-shot** on a new design (good knobs instantly, no per-design search).
Then **invoke real OpenROAD only on the chosen config(s)** — few real runs, the right ones.

**MDP (stage-by-stage):**
- **State:** design (netlist GNN embedding) + knobs chosen so far + the **imagined intermediate
  state** (f_place's placement output). The intermediate visibility is what makes it stage-wise —
  the agent sees the imagined placement *before* choosing routing knobs.
- **Action:** knobs for the current stage — mixed continuous (util, aspect, density) + discrete
  (synth strategy, routability).
- **Reward:** PPA objective from f_route (weighted WL/WNS/power − DRC-risk), at the end.

**Algorithm:** PPO (FastTuner pattern), **model-based** — rollouts imagined against f_place→f_route
(cheap), so the policy trains on millions of imagined trajectories.

**Training loop (amortized, over designs):**
1. sample a design → policy picks knobs stage-by-stage;
2. world model imagines → predicts PPA → reward;
3. PPO update; repeat over ALL training designs → policy generalizes (zero-shot on unseen);
4. periodically **ground** (real runs) where the ensemble is uncertain — improves the model *and*
   the reward signal the policy learns from.

**Deployment:** new design → policy picks optimal knobs **zero-shot** → invoke real OpenROAD only
on the chosen config(s) to validate/commit.

**Policy + grounding are coupled:** the policy proposes *which* configs to consider; ensemble
uncertainty decides *which of those* to ground. Explore → ground the uncertain ones → model
improves → truer rewards → better policy.

**Stage-by-stage earns its keep IF routing knobs should adapt to the imagined placement** — test
this; else it collapses to all-knobs-at-once (= FastTuner).

**Honest note:** the policy *mechanism* is FastTuner-like (PPO over knobs, GNN transfer). The
contribution is the **staged world model it plans in** (visible intermediates + grounding), not the
RL itself. Build **after** f_place/f_route are validated — MBPO's warning: don't optimize a policy
against an untrusted model (it will exploit the model's errors).

## 5. Training

**North star:** predict intermediate states + final metrics accurately, cross-design, using as
few real OpenROAD runs as possible. Structure follows MBPO (read first-hand).

### 5a. Initial supervised training (the seed)
- Train each stage on the **600 real runs** (`data_gen/archive/*`). Labels are aligned per run:
  `placement.odb` → f_place targets (per-net length, per-cell congestion); `routed.def`+`metrics.json`
  → f_route targets (routed WL, DRC, PPA).
- **Eval = leave-one-design-out** (train on N−1 designs, predict the held-out one) — the honest
  cross-design test, always. Expect the ceiling: per-net length good, congestion ~0.37-ish.

### 5b. Uncertainty — ensemble of probabilistic models (MBPO)
- Each stage = an **ensemble of probabilistic nets**, each head outputting **mean + variance**
  (train with Gaussian NLL, optionally quantile heads). Captures **aleatoric** (the physical
  ceiling noise — the netlist under-determines the outcome) *and* **epistemic** (ensemble
  disagreement = where the model hasn't seen data). Not optional — this uncertainty is the
  **steering signal for the grounding loop.**

### 5c. The compounding fix (the seam — MBPO's core lesson)
- Chaining `f_place → f_route` feeds f_route an **imagined, error-laden** placement state
  (out-of-distribution vs. the real states it trained on) → errors compound. MBPO's fix = keep
  rollouts short and branch from real states; ours is short already (2–3 stages), but:
- **Train f_route on BOTH real placement states AND f_place's imagined states**, so it learns to
  tolerate the noise it will actually receive at inference. This is *the* research problem, not a
  GNN detail — measure how much the composition degrades vs. real-placement input.

### 5d. The grounding loop (MBPO / Dyna — "use the model a lot, real data sparingly")
1. Train on the data we have.
2. **Imagine cheaply** — agent/model evaluates thousands of configs against the world model (near-free).
3. **Ground where uncertain** — pick the config+design with **highest ensemble uncertainty**
   (where a real run cuts model error most). **Staged grounding:** if the unsure stage is f_place,
   a **placement-only** real run suffices (cheap); full route only when f_route is unsure.
4. Add the real run(s) → retrain → **validate** (did held-out accuracy improve?) → repeat until plateau.
- Never train on the model's own imagined outputs as if real (circular); imagination only *proposes*
  where to spend real runs.

---

## Decision log
*(we append every locked decision here with a one-line why)*

- **Netlist graph = bipartite cell↔net, edges typed driver/sink** (DE-HNN `HyperConvLayer`) —
  clean, handles any fanout, preserves the physically-real driver≠sink asymmetry.
- **Include virtual nodes** (DE-HNN METIS hierarchy, toggle default-on) — measured netlist
  diameter is 7–10 layers on mid designs (past plain-GNN reach), grows with size, big designs coming.
- **Defer persistent homology** to a later ablation — compute-heavy, contribution unproven at our scale.
- **Include Laplacian PE** (top-k eigenvectors as node features) — cheap "graph GPS"; congestion depends on global span.
- **Knob conditioning = separate MLP branch injected via VN/supernode** (LOSTIN evidence; FiLM untested) — not concat-on-nodes.
- **Uncertainty = ensemble of probabilistic nets** (mean+variance) — MBPO; it's the steering signal for grounding.
- **Train f_route on real + imagined placement states** — MBPO compounding fix; the f_place→f_route seam is the core research problem.
- **Grounding loop: imagine cheaply, ground where uncertain, staged (placement-only when cheap)** — MBPO/Dyna, real runs spent sparingly.
- **f_route = shared light encoder + metric-matched heads** (WL/power = sum-pool; WNS = STA-style levelized MP; DRC = coarse risk/grounded) — survey: position-free works for WL/WNS, not DRC.
- **DRC: defer / coarse risk, handle by grounding** — needs positions we don't have AND is ~0 in our current data.
- **Agent = amortized PPO policy** (train once over designs against the world model → zero-shot knob-picking, invoke real runs only on chosen configs). Build after the world model is validated (MBPO).

## Build log
- **Representation builder done** (`scripts/build_graph.py`) — turns an EDA-Schema flow into the
  DE-HNN bipartite graph: GATE→cells, NET→nets, driver/sink edges contracted from pin-edge
  direction (GATE→PIN→NET=driver, NET→PIN→GATE=sink); cell feats (14, from standard_cells+degree),
  net feats (fanout+io), Laplacian PE (10). Validated: aes_core (17k cells, 0 unknown types),
  jpeg (64k cells). ~8.6s/flow (gates-table scan) → cache for training. VNs (METIS) still TODO.
