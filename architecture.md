# architecture.md — what PlaceDreamer's models SHOULD be, and why

**Companion to `learning.md`.** That file is the evidence log (every claim MEASURED / CITED /
HYPOTHESIS, including six retractions). This file is the *design*, derived from that evidence.

**Rule: nothing here is here because it seemed reasonable.** Every choice cites either a
measurement on our data or a paper we read *in source*. Where we don't know, it says so.

---

## −1. THE WORLD MODEL IS ONE STATE PROPAGATED — not three metric predictors

*This section supersedes the per-stage thinking below. It is the answer to "look at everything as
one thing."*

### MEASURED — `arrival` exists at EVERY stage, and the identity is EXACT at every one
`ac97_ctrl-000001`, clock_period 3.0 ns, `slack = required − arrival` **EXACT (≤1 ps) at all six**:

| stage | max arrival | min slack | **Δarrival** |
|---|---|---|---|
| **floorplan** | 3.022 | −0.177 | — ← **this is our INPUT. We already have it.** |
| global_place | 3.031 | −0.215 | +0.009 |
| **place_resized** | 3.711 | −0.917 | **+0.680** ← the resizer does the work |
| cts | 3.800 | +0.007 | +0.089 |
| global_route | 3.672 | +0.100 | **−0.128** ← routing *improves* timing |
| detailed_route | 3.672 | +0.100 | **+0.000** ← timing is settled at global_route |

### MEASURED — it CHAINS: endpoints are stable
79.5% of endpoints exist at **every** stage; **consecutive stages share 97.8–100%**.
Per-endpoint Δarrival (floorplan→place_resized): median **+0.224 ns**, IQR [+0.150, +0.439],
**corr(fp, placed) = +0.669**.

### 🛑 MEASURED — A ZERO-PARAMETER COPY OF OUR INPUT BEATS OUR TRAINED HEAD BY +0.98 R²
Fold-0 test designs, `arrival_place := arrival_floorplan + one global shift`:

| | R² |
|---|---|
| copy, raw | −2.465 (a systematic offset — placement ADDS delay) |
| **copy + single global shift** | **+0.476** (ac97 .460 / sasc .500 / systemcdes .834 / usb_funct .112) |
| **our trained 680k-param `endpt` head** | **−0.508** |

**We predict from scratch what we already know, and do it worse than not predicting.**

### ⇒ THE ARCHITECTURE
```
arrival_floorplan  (KNOWN — the input stage. FREE. corr 0.669 with placed.)
      │  f_place : predict Δarrival  ⇒ arrival_place = arrival_fp + Δ
      │  f_cts   : predict Δarrival  ⇒ arrival_cts   = arrival_place + Δ
      │  f_route : predict Δarrival  ⇒ arrival_route = arrival_cts + Δ
      └─ at EVERY stage:  slack = T − arrival ;  wns = min(slack) ;  tns = Σ min(slack,0)
                          (identities — verified exact; meta.wns == min(path slack) to 0.00e+00)
```
- **The seam carries `arrival`** — per-endpoint, physical, chains at 97.8–100% endpoint overlap.
  NOT our current bag of (broken slack + one HPWL column + 6 globals).
- **Every stage has the same target type** (Δarrival), so one head design serves all three.
- **The knob never enters the network for timing** — it enters via `slack = T − arrival`.
- **Each stage is a RESIDUAL on a known prior** — Δ-ML's pattern ("pick the prior for smoothness,
  not accuracy"; the prior buys a persistent offset in the learning curve ≈ a fixed multiple of
  designs, which is the currency we lack at n=18). RTL-Timer does exactly this: their analytic STA
  arrival is R=0.26 alone and R=0.86 through the model.
- **f_route's Δ is ~0 from global_route → detailed_route**, so routing timing is settled earlier
  than we assumed.

**This is what the seam SHOULD be. Everything in §2.3 below is the mechanism for predicting Δ.**

---

## 0. The three facts that force everything

**F1 — The input graph is IDENTICAL across a design's 108 knob configs.** (MEASURED: `cell_x`
drift **exactly 0.0000**; only tapcells vary and we drop them. `cache_graphs.py: IN_STAGE="floorplan"`.)
⇒ any pooling of the graph is a **design fingerprint, constant in k**. It cannot express knob
response. ⇒ the LEVEL task has **18 training examples** (one per distinct graph), not 1944.

**F2 — Knob response is an INTERACTION, not an addition.** `clock_period = 3ns` is *tight* for
des3_area (range 6.0–9.3) and *loose* for ac97 (1.8–3.0). (MEASURED: wns knob response is **0.092**
from raw knobs alone but **0.693** once design scale is known.)
⇒ `y_dev = f(k ; θ(G_D))` — the graph supplies θ (the design's scale), the knobs supply the
excitation. **Both necessary.** Neither alone works.

**F3 — A synchronous GNN scores NEGATIVE R² on levelized (timing) problems, cross-design.**
(CITED: TimingGCN DAC'22 Table 5 — vanilla GCNII **−0.84 / −0.78 / −1.51** at 4/8/16 layers vs
**+0.8957** for their levelized model. Deeper is WORSE.) Our `endpt` scores **−0.508** — inside
that failure band.

---

## 1. What each model class is FOR

| task shape | model | why | our evidence |
|---|---|---|---|
| per-net / per-cell, 1-hop-decomposable | **GNN** | MPNN counts exactly 1-hop stars, provably nothing with 3+ nodes (Chen NeurIPS'20 Cor 3.4). Distance-decay is a *feature* here (Di Giovanni ICML'23 §5.5(i)). N ≈ 10⁶ labels. | net_hpwl **AUC 0.912** (Net2: 0.922) |
| **levelized** (timing) | **GNN + topological propagation** | max logic depth ~300 ⇒ a synchronous GNN needs 300 layers. Depth must come from the **schedule**, not stacked layers. | our endpt **−0.508** = the vanilla band |
| global = exact function of per-node | **an IDENTITY. No model.** | zero params, inherits the per-node head's 10⁶ labels instead of 18 | in-probe A/B: composed sum **17.4%** rel-err vs pooled **112.7%** |
| global, no identity | level (pooled graph) + dev (**raw knobs** × θ) | pooling IS design identity — correct for the level; the knobs must arrive **unsmeared** | f_cts direct-knob: **−0.204 → +0.301** |
| global aggregate of a noisy per-node head | **+ a learned CORRECTION model** | `min`/`sum` are ORDER STATISTICS — they read the error *tail*, not the average | our WNS readout: **−1.102** |

**Nobody in EDA produces global scalars from a pooled graph readout.** DE-HNN predicts **zero**
global scalars. CircuitGNN's readout taxonomy omits graph-level entirely. MasterRTL *built* our
pooled-GCN for WNS/TNS/power/area, measured XGBoost beating it, and **shipped trees**.

---

## 2. f_place

### 2.1 Encoder — AUDITED against DE-HNN's source. We are NOT "DE-HNN-exact".
**The conv layer MATCHES** their current `dehnn_layers.py:49-75` line-for-line (concat order,
shared forward/back conv, residual-to-input). This part is fine, and it works: **AUC 0.912**.
(Note: their *paper's* conv differs from their *repo's* — the paper has no source/sink split on
cell→net. We match the repo.)

**But everything else is ours, and two of our comments were factually FALSE:**
- ❌ *"DE-HNN: nets carry no PE"* — **their nets DO get PE** (`pyg_dataset.py:145`). FIXED.
- ❌ *"DE-HNN drops nets with fanout ≥3000"* — **their filter is a NO-OP BUG**: they threshold
  `net_features[:,1]`, which is the **first Laplacian eigenvector**, against 3000. Entries ≪3000 ⇒
  zero nets dropped. We are not behind them; they never did it. FIXED.
- **PE also differs**: ours is cell↔cell clique-expanded + `which="SM"`; theirs is **bipartite
  cell+net** with scipy's default **`'LM'` (LARGEST eigenvalues)**. Ours is arguably better —
  but it is OURS.
- **DE-HNN feeds NO knobs.** Their entire design-context mechanism is `global_info =
  Tensor([num_nodes])` — one scalar, the node count — **off by default**. Our ctx MLP has no
  counterpart there.
- **Their targets are z-scored PER DESIGN** (`train_all_cross.py:102-104`), so the model
  structurally never predicts a cross-design level. **Our level/deviation split has no DE-HNN
  counterpart and needs none** — it is the price of asking a question they never ask.
- **Their cross-design script is BROKEN**: `all_valid_indices, all_test_indices =
  load_data_indices[10:], load_data_indices[10:]` — **val IS test** — and the checkpoint is
  selected on the **training** loss. **Our protocol is strictly better. Do NOT calibrate to their
  numbers.**

### 2.1b THE SUPER-VN — the one real DE-HNN gap (`SUPER_VN=1`)
**We had the concept wrong.** "Two-level VN hierarchy" is **METIS at ONE granularity + ONE GLOBAL
ROOT** (`pyg_dataset.py:109-111`: `top_part_id = zeros(num_vn)`, `num_top_vn = 1`) — *not* two
METIS granularities.
**Their ablation is CUMULATIVE**, single-design demand RMSE (Supp C.3 Table 7):
`+PD 8.765 → +single-VN 8.687 (0.9%) → +two-level 8.381 (3.5%)`; vs **no VN at all: 4.4%**.
(There is **no cross-design VN ablation** in the paper.)
**We are NEITHER ablation row**: their "single VN" is one *global* VN
(`graph_conv_hetero.py:473 batch = zeros_like(batch)`); we run METIS clusters with **no root** —
a config they never test.
Implemented (mechanism copied from `graph_conv_hetero.py:386-391, 497, 538-543`): both levels init
to **constant zero**; DOWN `h += (vn + top)[part]` every layer; UP `top = top + top_mlp(mean(vn) +
top)` while `l < K-1`; **mean-pool only, no max**. VERIFIED: +33k params, root gradient 2342 (it
is in the computation, not dangling).
> `VN_KNOBS=0` (production) means we currently have **NO global path through the VN at all** —
> so our VN_KNOBS result is evidence that *knob delivery via the VN* hurts, **not** that a pooled
> structural root hurts. Different intervention. The super-VN is untested, not refuted.

### 2.2 Per-net head — keep, it is our best asset
`net_hpwl`. AUC 0.912. **Everything downstream should be composed from this**, not re-derived.

### 2.3 Timing — REBUILD (this is the big one)

**Current:** one head predicts per-cell **slack** from a synchronous GNN with ~700 labels; WNS/TNS
are read off it. Scores **−0.508 / −1.102**.

**Why it cannot work:**
- **Wrong target.** `slack = require_time − arrival`, `require_time = clock_period` — so slack
  carries the **knob**, arrival carries the **structure**. (MEASURED: per-endpoint CV across knob
  configs — arrival **0.076** vs slack **1.364** on ac97 → 18× more stable.) RTL-Timer footnote 3:
  *"We assume a fixed clock frequency, implying slack is solely determined by arrival time."*
  **Neither TimingGCN nor RTL-Timer predicts slack. Both predict arrival.**
- **Wrong architecture.** Synchronous MP on a levelized problem = the −0.84/−1.51 band.
- **Wrong aggregation.** `min` over ~700 noisy predictions reads the error tail.
- **Starved.** ~700 labels when the dataset holds ~75k/flow.

**The rebuild — and we already have every label:**

| stage | what | supervision | source |
|---|---|---|---|
| 1 | GNN predicts **net delay** (local) | `net_arcs.delay` — **37,641/flow** | TimingGCN `nc1..nc3`, `x[:, :4]` |
| 2 | GNN predicts **cell delay** (per-arc) | `cell_arcs.delay` — **37,609/flow** | TimingGCN `e_cell_delays` |
| 3 | **propagate arrival topologically**, net→cell→net, seeded at PIs, **`max` over fanin** | `pins`/arc `arrival_time` — **5,041/flow** | TimingGCN `SignalProp` |
| 4 | `slack = clock_period − arrival` | — **IDENTITY** | verified to 1 ps in our data |
| 5 | `wns = min(slack)`, `tns = Σ min(slack,0)` | — **IDENTITY** | |
| 6 | **learned CORRECTION** on [wns_hat, tns_hat, design feats] | design WNS/TNS | RTL-Timer §3.4.3 |

**≈75k supervision points per flow vs ~700 today — ~100× more, already downloaded.**

And the chain closes on our strength: **MEASURED at place_resized, net delay median 0.32 ns vs cell
delay median 0.0001 ns — pre-routing, wire delay IS the timing**, and wire delay is a function of
net length, which we predict at AUC 0.912.
```
net HPWL -> net delay -> arrival (max over fanin) -> slack = T - arrival -> WNS = min -> correction
 ✅0.912     new head       STA structure               arithmetic          identity      RTL-Timer
```
**The GNN never learns timing globally.** Every step after the first is a supervised local quantity
or an identity.

**Details that matter (VERIFIED from TimingGCN source):**
- `sum` AND `max` reduction channels **in parallel**, each sigmoid-gated, concat + MLP
  (`model.py:60-61`). A mean/sum-only aggregator **structurally cannot represent
  `arrival = max over fanin`** — the core STA operator. Hard max (subgradient), not softmax.
- **Teacher forcing** (`model.py:85-88`): `groundtruth=True` swaps the predecessor's *predicted*
  embedding for the true one and **skips the topological loop entirely** — training is O(1) levels
  instead of ~300. This is what makes it tractable. Track free-running vs teacher-forced test loss
  to watch the exposure gap (`train_gnn.py:130-131`). **This is our seam, at the timing-graph level.**
- Loss: three **unweighted** MSE terms, all 1.0. No NLL. Don't over-think weighting.
- Transforms: arrival **raw**; slew `log(1e-4+x)+3`; net delay `log(1e-4+x)+7.6`.
  ⚠️ They keep arrival raw because 21 designs share one PDK and one clock. **We are cross-design
  with varying knobs** → normalize `AT / T_clk` (dimensionless). This is ours to get right.
- Aux ablation: net-delay aux alone **0.8513**, cell-delay alone **0.8150**, full **0.8957**.
  **Net delay is the more valuable auxiliary.**
- Optional (+0.08 R for RTL-Timer): **max over sampled paths** — backtrack each endpoint's cone,
  take the slowest path + K random paths, `max()` before the loss.

### 2.4 Globals — compose, don't pool
- `tot_hpwl = Σ_net HPWL_net` — **identity** (`HPWL_COMPOSE=sum`), supervised via `logsumexp` so
  per-net predictions are calibrated in ABSOLUTE terms (ranking alone won't sum right: AUC 0.912
  but 43.7% absolute rel-err).
- `buf_area` / `buf_cnt` — **no identity exists.** Keep level (pooled) + dev (**raw knobs** × θ).
  These already **beat** the fair baseline (+0.474 vs 0.386; +0.595 vs 0.447). **Not broken.**

### 2.5 Geometry — SOLVED (we were asking the wrong question)

CTS **needs** geometry: in SwiftCTS (CTS knobs varied) placement adds **+0.124** to clock_buffers.
So this must work. But **our two failures were CORRECT MEASUREMENTS, not bugs.**

#### ⇒ Absolute per-cell position is UNLEARNABLE, and the die-centre collapse PROVED it
Placement is a **global optimization with a gauge freedom**. So the conditional mean of
`p(position | local netlist topology)` **genuinely IS the die centre**. Our NLL head converging to
exactly that (0.397 vs a 0.395 die-centre baseline) is the *right answer to the question we asked*.
**No architecture fixes this.** Our VN box (0.253 vs 0.242) is the same fact one scale coarser —
because `(xmin,ymin,xmax,ymax)` is **four absolute coordinates**, carrying the same gauge.

#### ⇒ The literature confirms it by OMISSION and by ABLATION
- **No paper reports per-cell cross-design position accuracy.** TransPlace had every incentive and
  reported none. The absence is the finding.
- **TransPlace Table 11 — the decisive number:**

  | | OVFL ratio | RWL ratio |
  |---|---|---|
  | TransPlace (full) | 1.00 | 1.00 |
  | DREAMPlace | 2.03 | 0.97 |
  | **TransPlace w/o fine-tuning** | **25,753.73** | **2.47** |

  **The GNN geometry head ALONE — exactly what we built — is 25,753× worse on overflow.** It is a
  *warm start* for 150–2000 iterations of gradient descent on wirelength + electrostatic density.
  **TransPlace is not evidence a GNN can predict placement; it is evidence a GNN can initialize a
  placer.** If we add that optimizer, we have written a placer — at which point, just run ours.
- **MacroRank does not predict geometry — it CONSUMES it, and only for macros:**
  ```python
  fake_pos = torch.zeros_like(node_pos)
  fake_pos[macro_index] = node_pos[macro_index]   # every standard-cell position ZEROED
  ```
  *"since the position information of cells is unknown, only macro nodes are included in the input
  graph."* Their cluster box is **Eq (5): shape from AREAS + a density target — position never
  consulted.** That is our VN box built correctly: **extent, no location.**

#### ⇒ Use TRANSLATION invariance only. NOT rotation.
- **MacroRank MEASURED it**: translation → WL/via/short std **<0.5% / 0.3% / 3.7%** (holds);
  rotation/flip → *"very significant impact"* (**broken**). They built translation invariance only,
  and said rotation should wait "if the placement and routing algorithms maintain [it] as well."
- **TransPlace assumed full SE(2)** — and paid for it: their appendix carries a **per-netlist
  `Theta` ∈ {0,72,90,120,144,180,240,270,300}** plus `Δx, Δy` under *"circuit-adaptive
  fine-tuning"*. A rotation-invariant representation cannot recover orientation, so they
  **hand-tune the global rigid transform on every design**. A 3-parameter manual patch for
  over-invariance. (INFERENCE, but their own tables.)

#### ⇒ THE REFRAME: predict the sink POINT-SET, not per-cell positions
**f_cts does not need to know WHICH sink is WHERE.** Clock wirelength / skew / buffer count are
functionals of the **sink point set** — essentially **permutation-invariant over sinks**. We were
solving identity-resolved localization: strictly harder than the downstream consumer needs, *and*
unlearnable. This is the mistake.

**PLAN — A + B. Skip per-cell coordinates entirely.**

**A. Sink density map + spread statistics.**
- `K×K` heatmap (K=16 or 32 — NOT 224; we have 18 designs) of clock-sink density over the
  die normalized to 1×1.
- **Reusable code**: `paperCodes/MacroRank/src/util.py:41 get_ensity_map()` — a differentiable
  bilinear-splat rasterizer, 3-channel, max-normalized. Swap the macro loop for a sink loop.
- **Loss: NOT per-pixel L2** — that collapses to the marginal heatmap exactly as our NLL collapsed
  to the die centre. Use **soft-histogram cross-entropy or Sinkhorn/entropic-OT** against the true
  sink point set. OT penalizes *spatial* misplacement of mass; per-pixel losses do not.
- **Baseline it must beat: the train-set-mean heatmap** (the analogue of our die-centre baseline).
  Report skill against it. If it cannot beat that, that is the answer — and given the omission
  above, it is publishable as a negative result.

**B. Translation-invariant scalars (most likely to carry signal, no gauge ⇒ no die-centre trap):**
sink count, sink-set bbox **extent (W, H — NOT corners)**, radius of gyration, mean pairwise
distance, RSMT of the sink set. **This is what our VN-box head should have predicted.**
Retarget `h_vnbox`: `(xmin,ymin,xmax,ymax)` → `(w, h, area, aspect, radius_of_gyration)`.

**C. TransPlace-style anchored relative coords — SKIP** unless placement generation becomes a goal.
Feasible (our floorplan gives I/O pads + die dims) but Table 11 says it needs the fine-tuner, i.e.
a placer.

#### ⇒ CHANGE THE METRIC
Both papers evaluate a geometry representation by **downstream task error**, never position error.
Do the same: measure **f_cts's error with vs without the geometry features**, against the same
trivial baseline. That is the literature-consistent framing, it is the question we actually care
about, and it is where our signal is most likely to survive (placement IS deterministic — corr
0.97/0.92 — and clusters ARE compact, 1.4–9% of die).

> NOTE: TimingGCN uses **distance-to-die-boundary as a PIN FEATURE** — geometry as *input*.
> Different problem from predicting it. Worth remembering when the seam carries geometry forward.

---

## 3. f_cts

**Keep:** the shared encoder; sink-scoped readout for buffers, whole-graph for power/timing (a real
bug we fixed: pooling power over ~5% sink cells sent it negative).
**Keep:** `CTS_DIRECT_KNOB` — **measured −0.204 → +0.301 on buffers**, beating the 0.043 baseline.

**The honest state:** f_cts is **at or above its honest ceiling on every target** (buffers +0.301 vs
0.043; power +0.138 vs 0.143; wns +0.145 vs 0.121). **It is not broken.** Its ceiling is *data*:
EDA-Schema pins CTS knobs, so `n_sinks` is EXACTLY constant within a design (std 0.0) and geometry
adds **−0.003**.

**OPEN — the clock-source node.** We encode the clock as an `is_sink` flag + activity. GAN-CTS feeds
**sink-distribution images**; Kahng SLIP'13 uses an explicit **clock entry point** (`M_CEP`) and
shows **fall delay varies 43%** across entry points at fixed aspect ratio. **What must the graph
express about the tree?** A real clock-source node changes what is *expressible*, not just what is
fed. Not yet studied.

---

## 4. f_route

Same two defects as f_place, one worse. (MEASURED fair baseline, fold-0: rt_wl **0.738** — and
f_route's rt_wl scores **+0.66**, i.e. it *loses to 3 knobs*.)
- `h_lvl` and `h_dev` read the **same pooled `hg`** → the deviation head had **no knob path at all**.
  Fixed (`RT_DIRECT_KNOB`).
- `rt_wl = Σ_net routed_len` — the identity was available and unused; f_route already predicts
  per-net routed length (LODO **+0.66**, its best head). Fixed (`RT_COMPOSE=sum`).
- `crit_path` matters **more** here (power 0.158→0.354, wns 0.135→0.320) than in f_place — routing
  timing is dominated by clock scale.

---

## 5. The seam

**What crosses should be the PER-NODE structure** (the GNN's actual product), not global scalars —
the downstream stage can compute those from the knobs itself.

**Fixed:** the seam standardized globals by *cross-design* spread, crushing knob response
**18× / 38× / 53×**; and the imagined path forwarded only the **level** head, which is
design-constant (std **exactly 0.0000**) — i.e. it carried **zero knob information**. Now level and
deviation cross as separate unit-scale channels.

**TimingGCN's `groundtruth=True/False` is our seam** — teacher-forced vs propagated, evaluated both
ways, *inside one model*. They need it for tractability. Our dual-eval is the same instrument.

**Honest:** in EDA-Schema the seam has almost nothing to carry (placement adds **−0.003**). In
SwiftCTS (CTS knobs varied) it has **+0.124**. **The seam is only real in the knob-varied regime.**

---

## 6. What is NOT the fix

- **die_area** — 0.702 → 0.702. Within a design `log(die) = log(cell_area) − log(util)` and
  cell_area is design-constant ⇒ **collinear with the utilization knob**. RETRACTED.
- **crit_path in f_place** — fair gain 0.091 → 0.118, and a probe showed **+0.001** model gain.
  (Different story in f_route.)
- **√(n·A) as we built it** — the defensible law is `√A · N^p`, p∈[0.5,0.75] (Donath). Ours is the
  p=0.5 case. Salvaged only because `dfeat` carries `log A` and `log N` separately, so the head can
  **learn p**.
- **Replacing the WNS readout with a direct head** — BACKWARDS. The readout *is* MasterRTL's
  architecture. The defects are the target (slack→arrival), the level-1 head, and the missing
  level-3 correction.

---

## 7. Priority

1. **Timing rebuild** (§2.3). Biggest measured gap (−0.508 vs +0.8957 achievable), ~100× more
   supervision already on disk, and the literature is unambiguous. **Sequence: target (arrival) →
   correction model → sum+max channels → aux delay supervision → levelized propagation last** (it's
   the largest refactor; it may be unnecessary once the target is right).
2. **Compose the globals** (`HPWL_COMPOSE=sum`, `RT_COMPOSE=sum`). Free, principled, +0.05–0.08.
3. **Geometry — the REFRAME** (§2.5). Not "predict positions better": predict the **sink
   point-set** (density map + translation-invariant extent stats), and evaluate on **downstream
   f_cts error**, not position error. Retarget `h_vnbox` from `(xmin,ymin,xmax,ymax)` (four
   ABSOLUTE coords = the gauge trap) to `(w, h, area, aspect, radius_of_gyration)`. Reuse
   MacroRank's `get_ensity_map()`. **Per-cell absolute position is unlearnable and we have proof —
   our own die-centre collapse plus TransPlace's 25,753× ablation. Do not try again.**
4. **SUPER-VN** (`SUPER_VN=1`) — the one real DE-HNN gap; implemented, untested. Their gain is
   +3.5% cumulative (single-design). Cheap: +33k params, no re-cache.
5. **Data** — CTS knobs (SwiftCTS: 5400 runs, 540 placements × 10 configs) and more designs (18 is
   the level bottleneck). **This is what raises the ceiling; 1–4 only make the model principled.**
