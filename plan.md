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
- **The composition / compounding problem** ← the real research question. Every routing
  predictor (MAGNet, Lay-Net, RouteNet) is trained and tested on **real** placements.
  Feeding `f_route` an **imagined** placement from `f_place` is out-of-distribution and
  **nobody studies this.** Making the stack hold — and measuring how much imagination
  degrades vs. real placement — is the MBPO error-compounding problem applied to EDA.
- **Active grounding** — deciding when to spend a real OpenROAD run.

Honest risk: this overlaps FastTuner. The delta must be the **staged model + stage-wise
actions + composition** — not "RL for DSE," which they already did.

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
