# PlaceDreamer

*Living plan — simple and honest. We keep editing this.*

---

## 1. The whole thing in one picture

An **agent picks flow knobs stage by stage**. Between its choices, a **learned world
model imagines what the flow would produce** — imagine placement, then imagine
routing — and predicts **PPA**, in milliseconds, without running OpenROAD. The agent
tries thousands of knob choices this way and keeps the good ones. During training it
**runs real OpenROAD sometimes** (where the model is unsure) to keep the model honest.

```
agent ── picks knobs ──►  floorplan ─[f_place]─► placement ─[f_route]─► routing → PPA
  ▲         (action)                 (imagine)             (imagine)        │
  │                                                                        │
  └──── plans over 1000s of knob choices in seconds (imagination) ─────────┘
  └──── runs real OpenROAD when unsure → adds data → retrains (grounding)
```

CTS is a later stage. Start with **floorplan → placement → routing**.

---

## 2. The three pieces

1. **World model** = two stacked nets (the learned environment):
   - `f_place`: floorplan + placement knobs → **placement state** (RUDY / density maps, HPWL)
   - `f_route`: placement state + routing knobs → **routing outcome + PPA** (WNS, power, added buffers, routed WL, DRC)
   - later: `f_cts` for the clock tree.
2. **Agent (policy)** = picks knobs **stage by stage**, planning against the world model.
3. **Grounding loop** = run real OpenROAD where the model is uncertain → add rows → retrain (MBPO / Dyna style).

---

## 3. Why this is actually a world model (not just two predictors)

Two stacked predictors alone = a surrogate. It becomes a **world model** because an
**agent acts on it and plans inside it**:

- The agent **acts** (picks knobs) and **imagines rollouts** (knobs → placement → routing → PPA) in seconds.
- **Stage-by-stage** actions mean the agent acts *inside* the rollout (floorplan → pick → placement → pick → routing) — a real multi-step MDP, not one-shot config→PPA.
- The intermediate **placement is a real, inspectable state** — we can look at the imagined RUDY map, and swap in a real placement to continue.
- It's **anchored to reality** by the grounding loop.

This is model-based RL. Maps 1:1 to the reading: **Dreamer** (train a policy by imagining
in a learned model), **MBPO** (ground where unsure), **FastTuner** (RL over PD knobs on a
learned PPA estimator).

---

## 4. Architecture — borrowed, not invented

Predicting routing from placement is a solved problem. We reuse it:

- **`f_route`** (placement → routing/PPA): **MAGNet-style scatter/gather** — netlist GNN ⊗
  placement CNN/U-Net, coupled by projecting the graph onto the grid. Take **Lay-Net's
  features** (MacroMargin, net-to-net bbox-overlap edges) and the **CircuitNet channel set**
  (RUDY long/short/pin, cell density, pin density).
- **Netlist representation**: **DE-HNN** (directed hypergraph + virtual nodes + Laplacian PE).
- **`f_place`** (netlist + knobs → placement stats): **DE-HNN / Net²-style** netlist→congestion
  prediction — the "imagine placement" hop.
- **Agent + grounding**: model-based RL — **FastTuner** is the closest prior; **MBPO** for the
  grounding discipline; **Dreamer** for policy-in-imagination.

Papers in `papers/routing_pred/` and `papers/`.

---

## 5. What's actually novel (be honest)

The nets are borrowed. The contribution is:

- **Staged, inspectable world model** — floorplan→placement→routing with a *visible*
  intermediate placement (FastTuner uses one flat PPA estimator; we imagine the stages).
- **Stage-wise actions** — the agent picks knobs *between* imagined stages (real MDP).
- **The composition problem** ← the real research question. Every routing predictor
  (MAGNet, Lay-Net, RouteNet) is trained and tested on **real** placements. Feeding
  `f_route` an **imagined** placement from `f_place` is out-of-distribution.

  > **CORRECTED 2026-07-16 — "nobody studies this" is NO LONGER TRUE AS WRITTEN, and the
  > direction of the effect was WRONG.**
  > - **PowPrediCT (DAC'24, Yibo Lin)** studies exactly the placement→CTS-power seam with a
  >   real curriculum: pretrain on post-route graphs → swap the input to placement graphs →
  >   fine-tune. Cross-design leave-one-out total-power rel-err: Innovus 9.652%, vanilla GNN
  >   **14.149%** (worse than the tool), **Phase-1-only 5.106% → full 1.981%**. That is
  >   scheduled sampling; they never name it.
  > - **MasterRTL §IV.A (ICCAD'23)** chains a tree on its OWN predictions: vs post-place truth,
  >   the REAL netlist gives TNS MAPE **62%** while CHAINED-ON-PREDICTED gives **4%**.
  >   *"chained predictions achieve similar or even higher accuracy than ground-truth."*
  >
  > ⇒ **Imagination is not only a tax — it is a CALIBRATION CHANNEL.** The downstream model
  > learns to invert the upstream model's systematic bias, which it cannot do if it only ever
  > sees clean input. Our dual-eval must be able to report `imagined > real` as a FINDING.
  >
  > **The honest, still-strong claim:** nobody chains a full imagined **per-node physical
  > state** through **≥2** learned stage models, and **nobody has named it exposure bias**
  > (verified: "exposure bias"/"teacher forcing" appear NOWHERE in this literature; "world
  > model"+"physical design" → 0 hits; Agnesina/Lim NeurIPS'20-wkshp *proposed* our staged
  > formulation and never built it — their environment stayed the real tool).
- **Active grounding** — deciding when to spend a real OpenROAD run.

Honest risk: this overlaps FastTuner. The delta must be the **staged model + stage-wise
actions + composition** — not "RL for DSE," which they already did.

> **VERIFIED 2026-07-16 (FastTuner ISPD'24 read in full):** the overlap is SMALLER than feared.
> FastTuner models **zero stages** — its MDP state is *"the configuration of the parameters
> tuned from time steps 1..t−1"* plus a static design embedding, reward is **terminal-only**,
> and the "stages" are just a naming convention on the knob list. It is a **contextual bandit
> in MDP notation**. Its GNN embedding is a **7-entry lookup table** (7 designs; the netlist is
> identical across knob configs — our F1, confirmed in the wild), its estimator is evaluated
> **within-design, Pearson-only, no cross-design number**, and zero-shot transfer is
> **1.8–2.5× worse** than tuned (and loses to BO on 2/3).
> **AND ITS TABLE 6 IS OUR MOTIVATION:** all > CTS+route > route-only on **7/7 designs, 3/3
> metrics** — locking placement knobs costs **12–34 TNS points**. Independent industrial
> evidence that `f_place` is the load-bearing hop, in the competitor's own data.
>
> **The strongest argument AGAINST us is TimingPredict (DAC'22)**, which exists to replace a
> chained *RF-net-delay + analytic PERT-traversal* pipeline (Barboza) — structurally our design —
> and shows vanilla GNNs collapse cross-design (GCNII R² −0.84/−0.78/−1.51). The resolution:
> they replaced **analytic** propagation with **LEARNED** propagation (topological + max), which
> is what §6 of `docs/model_architecture.md` specifies. **Our defense must be stated in the
> paper: we need the INSPECTABLE intermediate state for stage-wise RL actions — a use case
> TimingPredict does not have.**

---

## 6. Targets = PPA

Predict **P**ower / **P**erformance (WNS/TNS/slack) / **A**rea (emergent buffering the flow
adds) + routed WL + DRC. Die/core area are **inputs** (we set them), not targets. Full
list and metric keys in **`docs/features.md`**.

---

## 7. Features

Three groups (details in `docs/features.md`):
- **Config knobs** — the agent's action (util, density, aspect, synth, routability, clock).
- **Design-intrinsic** — the netlist graph (DE-HNN rep) + scalars; the cross-design backbone.
- **Placement-state** — RUDY/density/pin maps (upgrade to the CircuitNet decomposed channel set).

---

## 8. Data

- **Seed data (now):** `data_gen/sweep_multi.py` — multi-design sweep, ~600 runs across 11
  mid-size designs, randomized knobs. Archives per config: netlist (`synth.v`) +
  `placement.odb` + `routed.def` + `metrics.json` + RUDY maps, then deletes the bulky run.
  This trains the first world model. (`docs/designs.md` — 20 designs routed & verified.)
- **Pretraining:** **CircuitNet 2.0** (10k+ samples, all feature maps) for the perception —
  our sweep is too small alone for a GNN+CNN.
- **Later:** the **grounding loop** decides which real runs to do — active learning. We never
  train on the model's own imagined outputs (circular); only on real, actively-chosen runs.

---

## 9. Build order

1. **`f_route`** — real placement → routing/PPA (MAGNet-style). Validate cross-design
   (**leave-one-design-out**). Proven, tractable.
2. **`f_place`** — netlist + knobs → placement stats (DE-HNN-style). The hard/novel hop.
3. **Compose** `f_place → f_route`; measure how much the imagined placement degrades the
   routing prediction (the **compounding test** — the core experiment).
4. **Agent** (stage-wise knob picking) + **grounding loop**.
5. **CTS** stage.

---

## 10. Status

- ✅ 20 designs routed, stable env, macro recipe (`docs/designs.md`).
- ✅ Features + targets pinned (`docs/features.md`).
- ✅ Seed sweep running (~19% of 600).
- ✅ Literature surveyed; papers pulled; architecture direction set.
- ⬜ `f_route` → `f_place` → compose/compounding test → agent → CTS.

1. graph (cell↔net, driver/sink)          ✓ VERIFIED CORRECT
2. node features                          ← just fixed types + dead dims
                                             PE still broken (agent running)
3. input encoders  (MLP cell→d, MLP net→d, type_emb)     ← NEXT
4. virtual nodes   (METIS partition, VN init, knob/ctx injection)
5. message passing (K=4 × HyperConv)      ← unnormalized-aggregation bug lives here
6. readout heads   (per-net, per-cell, global)  ← knobs dead in the per-net head
7. loss
8. evaluation                             ← the instruments that lie