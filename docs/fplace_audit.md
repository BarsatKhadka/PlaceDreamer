# f_place — full audit (code + architecture), top to bottom

Three independent audits (data pipeline / model architecture / loss+eval), each required to
VERIFY against real code and data rather than reason from memory. Every number below was
printed from the actual cache or a real forward pass. Nothing here is inferred.

**Headline: the graph network is contributing almost nothing.** Cross-design, on the exact
standardized target the model trains on:

| predictor | pooled R² (unseen designs) |
|---|---|
| 2-param ridge on `log1p(fanout)` — no graph at all | **0.467** |
| our DE-HNN (769k params, 4 layers, VN, PE, type-emb) | **0.518** |
| oracle ceiling (ANOVA: what IS learnable) | **0.837** |

The encoder buys **+0.05** over a two-parameter curve, and leaves **0.37 R²** of real,
learnable structure on the floor. Not a data problem. Not a capacity problem (we are *larger*
than DE-HNN's shipped config). A **transfer** problem — see A1.

---

## A. ROOT CAUSE — the encoder's only structural signal does not transfer

### A1. Laplacian PE is sign- and rotation-ambiguous across designs  [CONFIRMED, severity: CRITICAL]
`fplace.py:113-125 _pe_norm`. The docstring asserts raw PE is *"bounded, **cross-design-
consistent**, and what the paper feeds."* **The middle claim is false and it is the core bug.**

- `eigsh` eigenvectors have **arbitrary sign** and **arbitrary rotation within degenerate
  eigenspaces**. The same structural position in two designs encodes to different vectors.
- The cell↔cell graph has **2-3 connected components**, so eigvec 0 (and eigvec 1 on i2c) is an
  arbitrary basis vector of the **null space** — component membership, not position.
  Re-running `eigsh` gives overlap |<v,v'>| of only 0.53-0.74 on the kept dim 0.
- Scale goes as 1/sqrt(C): a 575-cell and a 48k-cell design live on different scales.
- 27-39% of cells (tap cells) get **all-zero PE**.

Measured consequence — PE-neighbourhood features added to a probe:
```
within-design gain:  +0.06 .. +0.16 R2   (jpeg 0.240 -> 0.405, ethernet 0.473 -> 0.613)
pooled  (cross-design) gain: +0.03 R2    (0.425 -> 0.456)
```
Textbook non-transferable feature. In cross-design CV — the only setting we care about — the
model's one structural signal is noise, so it collapses onto the fanout law.

**Fix direction:** sign-invariant structural encodings (RWSE / random-walk), or |PE| /
pairwise-PE-distance features, or size-normalized PE. Fix the ENCODING, not the head.

### A2. Message passing is unnormalized  [CONFIRMED, severity: HIGH]
`fplace.py:269-273`. `SimpleConv()` with no `edge_weight`; default aggr = **sum**.
DE-HNN passes `gcn_norm` degree weights (`dehnn_layers.py:66-72`, `train_all_cross.py:85-87`)
AND drops nets with out-degree >= 3000 (`train_all_cross.py:70-78`). **We do neither.**
Measured on ethernet-000001:
```
fanout bucket      || aggregated message ||
[1,2)                        8.3
[8,32)                      65.4
[128,...)                4,085.4
MAX (clock, fanout 10,016)  68,671.9
```
Straight into psi's first Linear, and back to cells BEFORE any LayerNorm. Every flip-flop's
embedding is dominated by the clock net.

### A3. Knob conditioning is DEAD in the per-net head  [CONFIRMED, severity: HIGH]
`fplace.py:341` — `fc1_net = Linear(d, 256)` reads `h_net` alone, **no ctx skip** (every other
head has one). Knobs reach nets only via VN -> cells -> conv, heavily attenuated.
Measured on a trained model, |d mu / d knobs| and finite-difference response:
```
head        |grad|     output shift for util x1.5  (as % of that head's output std)
net_hpwl    0.0083              0.11%      <- effectively knob-INVARIANT
tot_hpwl    0.1935              1.89%
buf_area    0.1849              7.87%
endpt       0.3666              3.36%
```
Utilization physically rescales the die and therefore every net's HPWL. The model does not
respond. For a world model whose purpose is counterfactuals, per-net HPWL is non-responsive.
One-line fix (`Linear(d+d,256)`, feed `cat([h_net, ctx])`) — but note the ANOVA says the knob
effect on this target is only ~0.01 R2, so **this fixes counterfactuals, NOT the 0.518.**
Do not conflate the two.

### A4. Sum-only aggregation vs an extremal target  [SUSPICIOUS]
HPWL is a half-perimeter = a **max-min span** of the net's cells. A sum/mean aggregator
provably cannot compute an extremal statistic of its neighbours in one hop. DE-HNN's target was
net *demand* (sum-like), so this never surfaced for them. Tested: a max-min span feature adds
within-design but not pooled -> a real limitation, but **not the binding one** (A1 is).

---

## B. LABELS ARE WRONG — the model trains against corrupted targets

### B1. Endpoint slack: the WRONG PIN wins  [CONFIRMED, severity: HIGH]
`scripts/add_endpoint_slack.py:47-53` stores one (cell, slack) per **endpoint**, but a DFF owns
several endpoint pins (`/D`, `/SET_B`). No dedup. `fplace.py:246-249` fancy-index writes, so
NumPy keeps the **LAST** write — and `groupby` sorts alphabetically, so **`/SET_B` lands after
`/D` and the LESS-critical slack systematically wins.**
```
flows with duplicate ep_idx:                         1444 / 1944
corrupted labels:               219,869 / 1,448,368  = 15.2%
flows where the WORST endpoint's cell stores a NON-worst slack:  877/1944 = 45%
max label error:                                     8.93 ns
```
This is the array `WNS = min` reads from. Fix: `groupby(cell).min()` before storing.

### B2. Per-net HPWL labels are FRAGMENTS on the biggest nets  [CONFIRMED, severity: HIGH]
Graph is built @floorplan; labels come from @global_place. In between OpenROAD **splits
high-fanout nets with buffers**. The NAME survives, so name-matching finds a "hit" and nothing
is masked — but the label now describes a fragment.
```
ethernet-000003, net _00005_:
  model sees:  fanout 2052   (floorplan topology)
  label from:  the net with   36 sinks left (global_place)
```
Only 0.41% of nets, but they carry **4-6% of total HPWL**, median HPWL **20x** the rest.
The highest-leverage nets in the per-net head are trained on a label for a fragment of the net
the model is looking at. Silent. Never masked.

### B3. ~7% of total HPWL is on nets the graph cannot see  [CONFIRMED, severity: MEDIUM]
New nets at global_place have no floorplan node, but `meta.total_hpwl` counts them.
`sum(cached per-net hpwl) / meta.total_hpwl`: ethernet **0.803**, jpeg 0.883, aes 0.934,
mean over 18 sampled flows = **0.931**. So `net_hpwl` and `tot_hpwl` are trained on quantities
that CANNOT be reconciled; any sum-consistency readout is biased low by 0-20%, design-dependent.

### B4. des3_area has almost no endpoint labels — and it's in a TEST fold  [CONFIRMED]
58 of its 108 flows have **ZERO** endpoints; 3% TNS coverage when it does. It lands in fold 0's
test set. A large part of the +-2.31/+-3.77 wns/tns fold variance is a **label-coverage
artifact**, not model variance. (I previously reported that variance as evidence of
"coverage-limited timing." That reading is not supported.)

### B5. wb_dma: 160/368 endpoints are primary outputs (dropped)  [CONFIRMED]
`min(ep_slack) = -0.075` vs recorded `wns = -0.378`. TNS-from-labels covers **1.2%** of recorded.

---

## C. THE INSTRUMENTS LIE — we cannot currently tell whether a fix worked

### C1. Pooled R² measures DESIGN IDENTITY, not knob response  [CONFIRMED, severity: CRITICAL]
`train_fplace.py:155` pools all test flows across 4-5 designs. Share of the scored variance
that is merely between-design:
```
tot_hpwl  98.7-99.4%      knob effect left: 0.6-1.3%
buf_area  99.7%                             0.3%
buf_cnt   99.9%                             0.1%
wns       58-71%                            29-42%
tns       23-58%                            42-77%
```
99% of what the headline R² scores is "how big is this chip." The knob effect — the ONLY thing
f_place exists to predict — is invisible in the number.
**=> report ABSOLUTE ERROR |target - predicted| in real units, and WITHIN-DESIGN error
separately. Never a bare pooled R².**

### C2. readout_loss TNS term regresses onto an unreachable target  [CONFIRMED — MY BUG]
`train_fplace.py:105-108`. `tns_p` sums over `ep_idx` = the **labeled** endpoints only (median
coverage **46%**), but is regressed onto `g["tns_true"]` = the **complete** recorded TNS. The
docstring claims it "supervises the endpoints we have no individual label for." **It cannot** —
those endpoints have no node in the sum and receive zero gradient. What it actually does is
force the labeled 46% to inflate their slack ~2.2x (des3_area: 33x), fighting the endpt MSE.
Simulating exactly that inflation reproduces the failure:
```
fold0 wns R2 = -57.5   fold1 = -164.4   fold2 = -18.1
```
**This is the direct cause of the WNS damage.** DELETE THIS TERM.

### C3. TNS R² is scored against a ceiling of ~0.34, not 1.0  [CONFIRMED]
Feeding the ORACLE (predicted slack == label exactly) through evaluate()'s readout:
```
          oracle wns R2    oracle tns R2
fold0        +0.863           +0.594
fold1        +1.000           +0.168
fold2        +0.982           +0.261
```
A PERFECT endpoint predictor scores tns R2 = 0.34 mean. Reporting -4.43 against an implicit
ceiling of 1.0 is meaningless. WNS, by contrast, IS reachable (oracle 0.86-1.0) — which
confirms wns=-1.93 is a genuine failure and C2 is its cause.

### C4. The calibration metric is mathematically wrong (Jensen)  [CONFIRMED — MY BUG]
`train_fplace.py:161`: `mean(sigma) / sqrt(mean(r^2))`. Divides an ARITHMETIC mean of sigma by
an RMS of residuals. A PERFECTLY calibrated heteroscedastic model scores:
```
log-sigma spread 0.5 -> 0.88     1.0 -> 0.61     1.5 -> 0.32     2.0 -> 0.14
```
The reported 0.08-0.6 is substantially or entirely this artifact. **Every claim I made about
the model being "badly overconfident" (and the ensemble argument built on it) is unsupported.**
Correct form: `sqrt(mean(sigma^2))/rmse`, or report `mean(r^2/sigma^2)` (should be 1.0).

### C5. Validation cannot see cross-design overfitting  [CONFIRMED, severity: HIGH]
`train_fplace.py:176-178`: val = a random 10% of flows **from the TRAINING designs**. Per C1,
~99% of the R2 is design identity — which val shares with train by construction. Yet val drives
BOTH `sched.step()` (LR schedule) and `best_state` (checkpoint selection). There is **no
held-out-design signal anywhere in the training loop.**
=> validation must be a held-out-DESIGN split.

### C6. Loss weighting is accidental  [CONFIRMED]
All weights nominally 1.0. But `net_hpwl` is a `.mean()` over ~10k nets (so 1/N per net) while
each global scalar contributes an undivided error. Measured gradient into the shared trunk:
```
term         loss     |grad| into trunk   share
net_hpwl     1.01           1.58          2.2%   <- ~10,000 supervision points/flow
tot_hpwl     2.03          12.68         17.8%   <- 1 scalar
buf_area     2.56          11.72         16.5%   <- 1 scalar
buf_cnt      1.52          10.63         14.9%   <- 1 scalar
endpt        1.92          18.09         25.4%
readout     30.23          16.43         23.1%   <- the buggy term from C2
```
The densest target gets **2.2%** of the encoder's gradient. Nobody designed that.

### C7. "decoupled" loss is a weaker claim than its docstring  [CONFIRMED]
Stopgrad is correct AT THE HEAD (verified d(var_loss)/d(mu_row) = 0.000e+00). But `lv` is a
function of the shared trunk, so var_loss backprops through the whole encoder:
```
                |grad| mean_loss   |grad| var_loss   var share of TRUNK grad
MEAN over flows      21.15             25.14              54.3%
```
At LAM_V=1, **54% of the encoder's gradient budget** goes to learning residual magnitude, not
the mean. The docstring's "it cannot sabotage mu" is false. Worth an ablation at 0.1 / 0.01.

---

## D. WHAT IS CORRECT (verified, not assumed)

- **Graph structure is right.** n_cells/n_nets match `netlists/table` EXACTLY on every design
  (aes 17145/11683, ethernet 34480/20806, jpeg 63907/45024...). Driver/sink directions verified
  against the raw `graph_json` edge census. 0 multi-driver nets. 0 unknown cell types across all
  1944 flows. Isolated cells = tap cells (legit). net_hpwl NaN rate 0.043%, correctly masked.
- **NO LEAKAGE.** Every input traces to floorplan-or-earlier. Placement coordinates and net bbox
  columns are never read. (`net_dem`, the one outcome-derived feature, is gone.)
- **HyperConv structure** matches `dehnn_layers.py` exactly (residual targets, concat order,
  driver/sink split).
- **Virtual nodes** match `model_att.py` step for step: cells only, mean+max init from raw input
  feats, per-layer broadcast BEFORE conv, update AFTER conv, gated on l < K-1. 0 empty partitions.
- **LayerNorm/activation ordering, edge-dropout masking** — correct. (Ours is gated on
  self.training; DE-HNN's is not — theirs is the bug. Keep ours.)
- **set_norm sampling is adequate** (fl[:2] vs a 10x sample: max shift 0.11 sigma) and the
  _pre/_stats path is IDENTICAL in set_norm and load_graph — no drift.
- **Capacity is not the bottleneck.** d=64/K=4 is LARGER than DE-HNN's shipped d=32/K=3.

---

## E. UNDECLARED DEVIATIONS FROM THE STATED "STRICT FIDELITY"

The module docstring (`fplace.py:5-18`) claims the encoder is "reproduced exactly" with only two
additions. Not true:
1. **`type_emb`** (`fplace.py:321`) — a 5,293-row cell-type embedding, **44% of all params**,
   of which only **172 rows** are ever used. DE-HNN's node_encoder has no such thing. It is
   neither knob-conditioning nor an uncertainty head. Sanction it in architecture.md or drop it.
2. **No `gcn_norm` edge weights** (A2).
3. **No high-fanout net filter** (A2).
4. Docstring still lists "per-net demand" as a head. Deleted long ago.
5. `N_TYPES=5293` but the library has 441 unique cells (parquet repeats it 12x).

---

## F. SMALLER, REAL

- `_stats` (`fplace.py:100-105`) — the dead-dim guard **does not fire**. float32 `mean(0)` over
  343k rows inflates `height`'s true std 4.8e-07 -> **0.0125**, sailing past the threshold. So
  height is z-scored by garbage and fed as a constant -1.00. The docstring claims this exact
  case is handled. Fix: `.astype(np.float64)`.
- `cell_x[7]` (is_filler), `cell_x[8]` (is_diode), `df_vals[7]`, `df_vals[8]` — identically
  zero across all 1944 flows. 4 dead inputs.
- **Stage mixing**: net_hpwl/tot_hpwl @global_place; buffers/wns/tns/endpt @place_resized.
  HPWL ratio pr/gp = 1.158 mean (log-corr 0.998) — defensible but the placement state is a
  chimera. `place_resized` HPWL exists; unifying would make all six targets one stage.
- **Cache is stale vs code**: `build_graph.py:44`'s nan_to_num post-dates the cache; cached
  cell_x still holds 2 NaNs/flow. Harmless (`_cellfeat` nan_to_num's them) but proves cache != code.
- **Stale norm files**: norm cache keyed on `hash(train_designs)` only, not on cache content.
  Re-caching graphs will silently reuse stale stats.
- `train_fplace.py:210` — the trailing partial-accumulation `opt.step()` skips `clip_grad_norm_`.
- `train_fplace.py:245` — prints `rel_err*100` as a **%** for endpt/wns/tns, but those are raw
  absolute errors in ns. Nonsense units.
- `within_r` is a Pearson **r**, not R² — a design can have r=0.9 with negative within-design R².
  Given C1, **within-design R² is THE number this project needs and it is not computed anywhere.**
- `fplace.py:435 loss_fn` is dead code that omits readout_loss — doesn't match what training
  optimizes. A trap for reuse.
