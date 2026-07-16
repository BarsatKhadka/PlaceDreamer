# learning.md — what we actually know, measured or cited

Running record of findings for the PlaceDreamer world model (f_place → f_cts → f_route).
Rule for this file: every claim is either **MEASURED** (with the number and how) or **CITED**
(with venue/year), or it is explicitly marked **HYPOTHESIS**. No vibes.

---

## 1. The single most important finding: our dataset only varies FLOORPLAN knobs

EDA-Schema-V2 sky130hd = 18 designs × 108 configs = 1944 flows. The three knobs that vary are
`clock_period`, `utilization`, `aspect_ratio`. **CTS and routing knobs are OpenROAD DEFAULTS —
they never vary.** Everything below follows from this, and it is the thing that separates our
setup from the published work.

### MEASURED — what actually drives each CTS metric's knob response

Within-design R² (design mean removed from both sides, so this is pure knob response),
single feature, verified as corr² independently of any matrix solve:

| target | clock_period | utilization | aspect_ratio |
|---|---|---|---|
| cts_buffers | 0.000 | **0.271** | 0.025 |
| cts_power | **0.905** | 0.034 | 0.000 |
| cts_wns | **0.538** | 0.002 | 0.001 |

Every CTS knob response is driven by **a knob f_cts already receives**. `cts_power` is just
P = αCV²f → power ∝ 1/clock_period. `cts_buffers` follows utilization → die area → sink spread
→ clock wirelength.

### MEASURED — placement geometry adds ~NOTHING for CTS in our data

Using *ground-truth* placement (the ceiling any seam could deliver), predicting the knob
response of cts_buffers:

| feature | within-design R² |
|---|---|
| `die_area` — **free, knob-determined** | **0.336** |
| `sink_bbox_area` — needs predicted placement | 0.332 |
| **what predicting placement adds** | **−0.003** |

Because: **within-design corr(die_area, sink_bbox_area) = +0.981 (R² 0.963)** — the placer just
spreads cells to fill the core. And **die area is knob-determined**: R² 0.938 from the 3 knobs,
`corr(utilization, log die_area) = −0.965`, exactly die = cell_area/utilization.

### MEASURED — n_sinks is EXACTLY constant within a design

std = 0.0 across all 108 configs (ac97 always 2212 sinks, ethernet always 10544). Floorplan
knobs do not change the flip-flop count. So `n_sinks` gives a great **level** (cross-design R²
0.900 for buffers) and **literally zero knob response**.

**This explains the entire f_cts result.** f_cts leans on `is_sink` → sink count → strong level,
zero knob signal. Its knob-R² ≈ 0 was inevitable from the features, not a model failure.

---

## 2. What the literature says (CITED)

### GAN-CTS (Lu et al., **ICCAD'19**; extended **TCAD vol.41 no.9, 2022**, pp.3104–3116)
- Input = three **placement images** at 700×717×3: flip-flop (sink) distribution, clock-net
  distribution, trial routing → frozen ResNet-50 → 3072-dim, + one scalar (# flip-flops).
  Sink locations enter as a **rasterized density map**, never as explicit geometry.
- **Table IV ablation — spatial images vs counts-only (4 unseen designs):**

  | target | spatial (MAPE) | counts only (MAPE) |
  |---|---|---|
  | clock power | 2.14–3.88% | 6.81–8.75% |
  | clock WL | 1.65–2.66% | 5.12–6.77% |
  | achieved skew | 3.87–4.63% | 11.78–16.86% |

  Skew MAXE 27.8–34.8% (spatial) vs **91.6–129.5%** (counts). **Counts are ~3× worse.**
- **They sweep CTS knobs** (max skew, fanout, cap, slew, latency, buffer density) — 35
  placements × 300 clock trees per design. **We do not.** This is why spatial features are
  load-bearing for them.
- DeepLIFT attribution (Fig.10): **leaf transition (slew) is the #1 driver** of clock power and
  clock WL (~0.95–1.0 normalized); max fanout and cell density rank near-lowest.
- Table III (NOVA): trunk/leaf slew 0.1/0.1 → **2556 buffers**; 0.1/**0.05** → **600**;
  **0.05**/0.05 → **1124**. Buffer count is **non-monotonic in slew** and slew moves it 4.3×,
  far harder than sink count does.

### Kahng, Lin, Nath — SLIP'13 (high-dimensional metamodeling for CTS outcomes)
- Explicit geometric parameters: `M_sink`, `M_core` (block area), `M_AR` (aspect ratio),
  `M_CEP` (clock entry point), `M_block` (blockage), `M_DCT` (**sink nonuniformity**: grid the
  block, per-grid χ² of pairwise sink Manhattan distances vs uniform, DCT across grids, sum
  |coeffs|; zero iff uniform).
- **Fall delay varies up to 43%** across clock entry points at fixed aspect ratio; **power and
  fall delay vary up to 45%** when aspect ratio is swept — *with sink count held constant*.
- Note `M_core` and `M_AR` are exactly our `utilization`→die_area and `aspect_ratio` knobs.

### The physics of clock wirelength — VERIFIED, two independent derivations
- Charikar et al., *SIAM J. Discrete Math* 17(4), 2004, §2.1: divide-and-conquer clock router
  gives total WL **1.5√n** for n terminals on a uniform grid; RSMT ≈ √n+1. Rescaled to area A:
  **L_clk ≈ 1.5·√(n·A)** (Beardwood–Halton–Hammersley √(nA) law).
- Han, Kahng, Li, TCAD 2018: exact generalized-H-tree WL; all b_p=2, N=2^P sinks collapses to
  **L ≈ (√N/2)(W+H)** — √(N·A) again, from H-tree algebra. Also `P_max = ⌈log₂N⌉`; a sink region
  "typically contains <40 sinks."
- Charikar §3 lower bound: cost ≥ ∫n(R)dR where n(R) = min balls of radius R covering the sink
  set → clock WL is lower-bounded by a **covering-number functional of the sink point set**.
  Sink count alone cannot determine it.

**So our `sqrt_nA` feature is the literature's calibrated WL estimator, not a guess.**

### Buffer count — two regimes
- *Fanout-limited*: n_buf ≈ n_sinks/(F_eff−1), independent of spread.
- *WL/slew-limited*: Harris, "Buffer Repeaters" (1998): Elmore-optimal segment count
  S = L·√(C_W R_W / (2(2p+k+1/k)·C_T R_T)) — **linear in L**.
- Cross-referencing GAN-CTS Table I vs Table V (their numbers, ratio computed by us):
  **buffer_count ≈ 0.029 × n_flip_flops, ±5% across a 7.8× span of design size** (ECG 14018 FF
  → 417 buf; JPEG 37642 → 1093; LEON3MP 108724 → 2962; VGA_LCD 17054 → 505). Implies
  F_eff ≈ 35, matching Han/Kahng's "<40 sinks per sink region".
  **Under tool auto-settings, buffer count is FANOUT-limited** — nearly linear in sink count.

**This matches our data exactly**: n_sinks → cross-design R² 0.900 for buffers, knob R² 0.000.
We are in the fanout-limited regime because our CTS knobs are pinned to defaults.

### White space (CITED)
- **Nobody predicts post-CTS WNS/TNS cross-design.** Targets in the literature are clock power /
  clock WL / skew / insertion delay. GAN-CTS §VI-C: "timing in general is a hard-to-predict
  metric… commercial tools will often **override the skew target**… which results in the
  uncorrelation between the target closure and the final achieved outcome."
- **Buffer count is not a published cross-design prediction target** — GAN-CTS reports it only
  as an achieved metric (Table V); Koh/Kwon/Shin ISLPED'20 uses it as an intermediate.

### Achievable accuracy, with the caveat that matters
GAN-CTS TCAD'22 held-out (7 train / 4 unseen, TSMC28, Innovus): clock power MAPE 2.14–3.88%,
CC 0.962–0.972; clock WL MAPE 1.65–2.66%; skew MAPE 3.87–4.63%.
**Do not use this as our target.** Their within-test-design variance is CTS-knob-driven, so the
placement image only has to supply design identity. Our variance is design-to-design at fixed
CTS knobs — the strictly harder direction. Their **counts-only ablation (7–9% power, 12–17%
skew) is the closer analogue to our regime.**


---

## 3. The seam — what we found and fixed

### MEASURED — BUG (fixed): the seam crushed knob signal 18–53×
The seam forwarded each placement global as one channel `(raw_log − L_m)/L_s`, standardized by
the **cross-design** spread. Knob response only spans the **within-design** spread W:

| metric | W | L_s | knob signal crushed |
|---|---|---|---|
| tot_hpwl | 0.146 | 2.659 | 18× |
| buf_area | 0.061 | 2.308 | 38× |
| buf_cnt | 0.032 | 1.710 | **53×** |

Worse, the **imagined** path forwarded only `o["{k}_lvl"]` — the level head — which predicts the
design *mean*: **std EXACTLY 0.0000 across a design's 108 configs.** The imagined pipeline was
feeding f_cts **zero knob information**, while real mode forwarded the true knob-varying value.
That fully explains why imagined ≈ real told us nothing.
**Fix**: forward level and deviation as separate unit-scale channels (f_place's heads already
emit both). dfeat 3 → 6. Deviation now arrives at std 0.64/2.08/1.79 — 16× more visible.

### MEASURED — the seam result (3 folds, dual-eval)
Feeding f_cts imagined vs real placement state changed **almost nothing** (buffers knob-R²
fold0 +0.151/+0.129, fold1 +0.339/+0.325, fold2 −0.368/−0.113). Given §1, we now know why:
**there was almost nothing to carry.** Not "the seam is clean" — the seam is barely connected,
and for CTS it doesn't need to be.

### MEASURED — enormous fold variance
buffers knob-R²: fold0 +0.008, fold1 +0.325, fold2 −0.247. power: +0.012 / +0.199 / −2.262.
With 7 train / 4 test designs, *which* designs are held out swings results more than anything
we test. **We cannot make confident CTS claims from 3 folds.** Error bars > effect.

---

## 4. Geometry — built, and then measured to be unnecessary (for CTS)

Built `cache/coords` (1944 flows, 12.5M cells, ~100% coverage on kept cells; the only NaN
coords are tapcells, already dropped by `live_cells`). Added `h_pos` (per-cell x,y) and
`h_vnbox` (per-METIS-cluster bbox) heads.

- **MEASURED — METIS clusters land compactly** (validating the VN-box idea): median bbox area as
  a fraction of the die — jpeg 0.014, aes_core 0.041, ethernet 0.091, ac97 0.215. The placer
  honors connectivity; a cluster occupies a real, small region.
- **MEASURED — placement is deterministic, NOT the PE sign-ambiguity trap**: adjacent knob
  configs place the same cell at corr x/y 0.97/0.92. No mirror flips (would show corr ≈ −1).
- **MEASURED — geometry IS knob-responsive**: sasc corr y drops 0.92 → 0.13 as aspect_ratio
  takes the die from AR 2.08 (214.8×103.4µm) to AR 0.68 (101.2×149.6µm).
- **MEASURED — neither head beats its trivial baseline yet.** Per-cell pos 0.397 vs 0.395
  (die-centre); VN box 0.253 vs 0.242 (train-mean box). All probes were d=48/K=3, ≤8 epochs,
  4 designs — too underpowered to conclude. `skill = 1 − err/baseline` is now reported per
  epoch so a real run answers it without interpretation.
- **CONCLUSION for CTS: predicting geometry buys ~nothing** (§1: +(−0.003) over free die_area).
  Geometry may still matter for **f_route** (congestion, routed length) — untested.

---

## 4b. CORRECTIONS — two things we believed that are FALSE (verified from the PDFs)

**These were asserted by Claude as fact, propagated into code comments, and are wrong.**

### ❌ "Net2 abandoned absolute regression for ranking because length is under-determined"
**FALSE.** Net2 (ASP-DAC'21) **regresses** net length; verbatim: *"The net length is the label for
training and prediction. Each net's length is the half perimeter wirelength (HPWL) of the
bounding box of the net after placement."* No ranking loss. The paper never says absolute length
is under-determined.
What IS true: Net2 **never reports an absolute error number** — only binned correlation (Net2a
>0.98) and top-10% AUC (Net2a 92.5), and calls the binned protocol *"a classical criterion used
in many net length estimation works"*. So the field evaluates net length **ordinally**; that's a
convention, not an impossibility proof.
The 725-nets fact IS real (identical driver area / cell count / one-hop neighborhood, lengths
1µm → >100µm, design B20) — **but Net2 uses it to argue for a GLOBAL RECEPTIVE FIELD**, not to
drop regression. Fixed in `train_fplace.py`.

### ❌ "Absolute cell position is ill-posed cross-design because placement is symmetric"
**FALSE, and MacroRank (ASP-DAC'23) MEASURED it.** On DREAMPlace + CU.GR:

| transform | effect on WL / vias / shorts |
|---|---|
| **translation** | std **<0.5% / 0.3% / 3.7%** → symmetry holds |
| **rotation 180° / flips** | *"a very significant impact"* → **symmetry BROKEN** |

Fixed I/O pads, per-layer preferred routing directions, row structure and aspect ratio all break
rotation/reflection. **Only translation is a real gauge freedom.** MacroRank therefore builds
**translation-equivariance only**. Nobody in this literature calls coordinate prediction
"ill-posed"; TransPlace justifies invariance on *generalization* grounds and claims full SE(2),
which is **stronger than MacroRank's measurement supports** — a live tension between two papers.

### ✅ Why our plain (x,y) regression head was doomed anyway — the real reason
**No paper does plain supervised L2 regression of per-cell (x,y) evaluated cross-design.** Every
group that generates coordinates uses a **distributional** objective: GraphPlanner (VAE, TODAES'22)
→ Chip Placement with Diffusion Models (ICML'25) → MacroDiff+/FlowPlace (2026). **And nobody
reports per-cell coordinate accuracy as a metric** — all are scored on downstream
HPWL/congestion/legality. A cross-design per-cell coordinate-accuracy number does not exist in the
literature.
- **TransPlace (arXiv 2501.05667) is the one worked answer for coordinates**: predict *relative*
  polar encodings `(ρ_ij, Δθ_ij)` on a Cell-flow DAG, then **decode to absolute by accumulating
  along cell-paths from FIXED CELLS (I/O pads) with known positions.** The gauge is broken by the
  design's own fixed geometry. **Anchor, don't canonicalize.**
- ICML'25 diffusion: legality **0.9970 only WITH** test-time guidance; **unguided 0.8213**. It is
  "predict then guide", not "predict".
- *(Plausible but NOT citable: the mean-collapse story — that MSE on a multi-modal placement
  conditional lands in a low-density average. The agent checked the ICML'25 text; its
  "multimodal" refs are to synthetic edge/object-size distributions, not placement modes.)*

### Realistic cross-design bar (CITED)
MacroRank's group-1/group-2 split **is** our generalization setting: best cross-design **Kendall τ
0.30–0.38** (WL), vs CNN 0.109 / GNN 0.235. Our f_place ranking AUC 0.912 vs Net2's 0.922 is
therefore genuinely competitive, and 0.3-ish rank correlation cross-design is **a realistic
target, not a low bar**.

### Two notes that got independent support
- `proxy-is-reference-not-precondition` now has a **mechanism**: LHNN (DAC'22) — *"many effective
  features in CNN models can be recovered by a one-step message passing on the G-net→G-cell
  relation"*. **A RUDY map is a degenerate LH-graph.** The proxy is something a graph model
  reconstructs for free, never a required input. Exactly our note, proven.
- `fplace-timing-is-coverage-limited` has company: DATE'19 and MacroRank hit the same wall and
  took the same exit — drop cross-design absolute scale, keep within-design order.

---

## 4c. CORRECTION: "geometry is useless" was an ARTIFACT of EDA-Schema's pinned CTS knobs

Barsat pushed back on the claim that we don't need f_place data. **He was right, and SwiftCTS's
own data proves it.**

### MEASURED — SwiftCTS `data/unified_manifest.csv`: 5400 runs, 540 placements × 10 CTS runs
This is the GAN-CTS structure we lacked. CTS knobs actually swept:
`cts_max_wire` 130–280, `cts_buf_dist` 70–150, `cts_cluster_size` 12–30, `cts_cluster_dia` 35–70.
`clock_buffers` spans **84 → 2730 (32.5×)** vs EDA-Schema's knob-invariant behaviour.

**Within-placement (geometry FIXED), CTS knobs move the outcome** — R² per knob:

| target | max_wire | buf_dist | cluster_size | cluster_dia |
|---|---|---|---|---|
| clock_buffers | 0.002 | 0.002 | 0.225 | **0.577** |
| power_total | 0.000 | 0.000 | 0.003 | **0.775** |
| skew_setup | **0.277** | 0.002 | 0.008 | 0.001 |
| wirelength | 0.001 | 0.000 | 0.037 | **0.364** |

**And placement is then LOAD-BEARING** (within-design, design mean removed):

| target | CTS knobs alone | + placement knobs | + interaction | **placement adds** |
|---|---|---|---|---|
| clock_buffers | 0.599 | 0.723 | 0.755 | **+0.124** |
| power_total | 0.141 | 0.216 | 0.224 | +0.076 |
| wirelength | 0.004 | 0.086 | 0.088 | +0.082 |
| skew_setup | 0.235 | 0.272 | 0.284 | +0.037 |

…and that is with only crude placement KNOBS as a proxy; the real sink geometry lives in
`def_path`. **Mechanism**: `cts_cluster_dia` is a *max cluster diameter* — a geometric constraint
that only bites relative to how spread the sinks actually are. `cts_cluster_size` is F_eff and
`cts_buf_dist` is d_max in `n_buf ≈ max(n_sinks/(F_eff−1), 1.5·√(n·A)/d_max)`. Sweeping them
moves CTS between the fanout-limited and WL-limited regimes — exactly where geometry matters.

**So: §1's "geometry adds −0.003" is TRUE ONLY for EDA-Schema (CTS knobs pinned). It does not
generalise.** DECISION TAKEN: sweep CTS knobs (SwiftCTS `hpc/scripts/5-run-cts.py` already does
it); keep geometry for f_route, don't run it for f_cts-on-EDA-Schema.

---

## 4d. f_place — the real problems, MEASURED (this is where the wins are)

### ❌ ANOTHER FALSE CLAIM CORRECTED: "timing knob response doesn't transfer"
Said repeatedly. **False.** OLS knob-response ceilings (within-design, 3 raw knobs):

| target | knobs only | +die_area | +√(n·A) | **f_place ACTUAL** |
|---|---|---|---|---|
| total_hpwl | 0.719 | **0.857** | 0.858 | +0.654 |
| buffer_area | 0.429 | 0.530 | 0.540 | +0.474 |
| buffer_count | 0.387 | 0.472 | 0.474 | +0.595 |
| **wns** | **0.649** | 0.650 | 0.650 | **−1.102** |
| **tns** | **0.657** | 0.658 | 0.659 | **−0.126** |

**A 3-knob linear model gets wns 0.649; our GNN gets −1.102 — 1.75 R² WORSE.** The signal is in
`clock_period` and sitting in the input. Not a data limit — an architecture gap.

**Root cause**: f_place had **no wns/tns head**. They were READOUTS off the per-cell `endpt` head
— our single worst prediction (pooled R² −0.508, ~100% rel err, calib z² 9.63). We derived timing
from the broken thing. **FIX**: `wns_g`/`tns_g` are now first-class global targets with
level+deviation heads and signed-log transform; the dev head already gets raw knobs (DIRECT_KNOB),
which is where the 0.649 lives.

### MEASURED — `die_area` was missing and is worth ~0.2 R² on tot_hpwl, for free
All 18 design features were netlist-derived (n_cells, total_cell_area, fanouts…). **Nothing told
the model how big the die is**, so it had to learn a division. Adding it lifts the tot_hpwl
ceiling 0.719 → **0.857** while f_place sits at 0.654.
**Leak-free**: `die = total_cell_area / utilization` is the floorplan identity — cell area from
synthesis, utilization is a knob, both known BEFORE placement. Verified against the die measured
from the placed-cell bbox: **R² 0.9961**. Added, with `√(n·A)` (the BHH/Rent law, literature-
verified as the clock-WL estimator) as an explicit physics prior. `DF_IN` 16 → 18.

### ✅ MEASURED — the f_cts direct-knob A/B (controlled, same data/seed, flag toggled)

| | power | buffers | wns |
|---|---|---|---|
| WITHOUT (old f_cts) | +0.108 | **−0.204** | +0.036 |
| WITH direct knobs | +0.138 | **+0.301** | +0.145 |
| **delta** | +0.030 | **+0.505** | +0.109 |

Buffers −0.204 → +0.301, **beating the 0.271 OLS ceiling**. wns ~4×. Power moved only +0.030
against a 0.905 ceiling — **still an open gap worth chasing.**

---

## 4e. WHY A GNN — the inductive bias, and the one fact that forces the architecture

This section exists because "we are just putting things" is a fair charge. Everything below is
derived from one MEASURED fact, not from taste.

### MEASURED — THE FACT: the input graph is IDENTICAL across a design's 108 knob configs

`cache_graphs.py: IN_STAGE = "floorplan"`. For every design checked (sasc, ac97_ctrl, aes_core,
ethernet), across configs 1/40/80:

| design | raw cells (incl tapcells) | KEPT cells | cell_names | cell_x |
|---|---|---|---|---|
| sasc | 575 → 560 → 474 | **275 → 275 → 275** | **IDENTICAL** | **IDENTICAL** |
| ac97_ctrl | 9994 → 9995 → 8107 | **4567 → 4567 → 4567** | **IDENTICAL** | **IDENTICAL** |

`cell_x` drift across configs = **0.0000**. Only the tapcell count moves (they tile the core, so
they scale with die area) — and `live_cells()` drops tapcells. Floorplan knobs do not
re-synthesize the netlist.

So: **G_D does not depend on k.**

### The consequences are not opinions, they are implications

Write the model as `y(D,k) = F(G_D, k)`. Since `G_D ⟂ k`:

1. **Any pooling of the graph is a design fingerprint — CONSTANT in k.** A permutation-invariant
   readout over `h(G_D)` returns the same vector for all 108 configs. It is *structurally
   incapable* of expressing knob response. Our ~680k-parameter encoder computes, with respect to
   the knobs, **a constant**.
2. **Knob response can only flow through the 5-dim knob vector.** Hence OLS-on-3-knobs beating the
   GNN (wns 0.649 vs −1.102) is **not** a tuning failure — the GNN has *no additional information*.
   The linear model already sees 100% of the varying input.
3. **Hence DIRECT_KNOB is not a trick, it is the only signal path.** Measured: +0.505 on f_cts
   buffers, +0.109 on wns. Diffusing knobs through `ctx` and K message-passing layers over a FIXED
   graph smears the only informative input into a design-constant representation.
4. **"pooled R² is ~99% design identity"** — we wrote that as a warning. It is literally true:
   the pooled readout *is* design identity, by construction.
5. **"coverage-limited" restated**: for the LEVEL task the GNN has **18 training examples** — one
   per distinct graph — not 1944. That is the real n.
6. **Where the GNN genuinely earns its keep**: *within* a fixed graph, which net is long / which
   cell is critical is a structural question. Measured: per-net HPWL ranking **AUC 0.912** vs
   Net2's 0.922. That is the job message passing is built for.

### The forced architecture

    y(D, k)  =  Level(G_D)                 <- needs the graph        -> GNN
              + KnobResponse(k, scalars)   <- needs only the knobs   -> direct MLP + physics prior

**The level/deviation split (Barsat's "is the z-score within the design?") is exactly this
factorization.** It was never a normalization trick — it is the correct decomposition of the
problem, and it works because it matches the information structure of the data.

**Physics supplies the FORM of the knob term** (so the net learns a residual, not the law):
- HPWL ∝ √A, and A = cell_area/utilization  ⇒  `log HPWL ≈ −0.5·log(util) + c`
- clock power = C·V²·f (α=1 for a clock) ⇒  `log P ≈ −log(clock_period) + c`.
  MEASURED: cts_power knob-response R² **0.905** from clock_period alone.
- buffers: `n_buf ≈ max(n_sinks/(F_eff−1), 1.5·√(n·A)/d_max)` — the two regimes (§2).
This is Barsat's old task #7 ("does the residual beat HPWL?") returning as the right architecture.

### What this predicts (falsifiable)
- A small MLP on (knobs, die_area, √(n·A), design scalars) should MATCH the full GNN on every
  global knob-response target. **Untested — this is the honest ablation we owe.**
- The GNN should only beat it on per-net/per-cell tasks (ranking) and on cross-design LEVEL.
- **Corollary for the seam**: what should cross is the PER-NODE/PER-NET structure (the GNN's
  actual product), NOT global scalars — the downstream stage can compute those from the knobs
  itself. That is a testable seam redesign.
- **Corollary for the CTS sweep**: in SwiftCTS the CTS knobs vary *per placement*, so the graph is
  still fixed but the placement (and hence real geometry) varies — which is exactly why placement
  becomes load-bearing there (+0.124, §4c) and is worthless here.

---

## 5. Open questions / decisions needed

1. **HYPOTHESIS (under test)**: f_cts's dev heads had **no direct knob path** — knobs reached
   them only via `ctx = MLP([knobs, dfeat])`, mixed with 16 design features and diffused through
   K message-passing layers. f_place hit this and solved it with `DIRECT_KNOB` (raw knobs → dev
   head, A/B-validated, default on). f_cts never got it. A 0.905-R² signal (clock_period →
   power) was sitting in the input, unreachable. **Fix implemented; test running.**
2. **MEASURED — sink nonuniformity (`M_DCT`) adds nothing either.** Kahng's is the one geometric
   quantity shown to matter beyond area/AR. 8×8 grid χ² of sink counts, 648 flows. It DOES vary
   with knobs (within-design std of log = 0.445). But **after removing die_area**: buffers R²
   **0.006**, power R² **0.000**. So *every* geometric route to CTS in our data collapses to the
   free, knob-determined die_area. Geometry for f_cts is closed.
3. **cts_power knob response ceiling is 0.905 from clock_period alone** — if the direct-knob fix
   lands, f_cts should approach it. **cts_tns has no measurable knob signal** (degenerate).
4. **The dataset's fixed CTS knobs are the core limitation.** GAN-CTS's spatial features earn
   their keep by sweeping slew/skew/fanout, which move buffer count 4.3× non-monotonically. Ours
   are pinned. Generating flows with varied CTS knobs would (a) make geometry matter, (b) make
   the seam load-bearing, (c) match the regime the literature shows is learnable.
   **DECISION FOR BARSAT.**
