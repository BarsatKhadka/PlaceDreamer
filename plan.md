# PlaceDreamer

*Living plan — we keep editing and refining this.*

---

## 1. The idea

A **world model predicts the next state of the world.** Here the "world" is the
chip-design flow, and we want to imagine it forward — **placement → CTS →
routing** — and give the final answer, all **without running OpenROAD**.

```
floorplan ──▶ imagine placement ──▶ imagine CTS ──▶ imagine routing ──▶ answer
                        (no OpenROAD — just the learned model)
```

If a stage turns out to need more intermediate steps to predict well, we add them.
Start minimal.

First we **validate that we can learn this flow on a single config** — one design,
fixed settings — i.e. just predict the chain accurately. Once that works, we put a
**planner on top**: it chooses different knobs at each stage, imagines the
outcomes, and picks the best. The learned flow comes first; the planner is the
eventual goal.

---

## 2. A note on MBPO

MBPO (Janner et al. 2019) is about using a learned model to plan. Two lessons we
take from it:

- **Don't trust long imagined rollouts** — errors compound. Imagine **short hops
  from real states**, and re-anchor with real runs now and then. Ensembles help.
- **A planner against a bad model gets exploited** — it finds knobs the model
  loves but reality punishes. So: **learn a trustworthy flow first, plan later.**

That's the whole reason we validate the learned flow before adding the planner.

---

## 3. The staged flow (how we imagine)

How it works, concretely:

1. Feed **floorplan features + placement knobs** → **imagine placement**.
2. Choose **CTS knobs** → **imagine CTS**.
3. Choose **routing knobs** → **imagine routing** → **final answer** (WL, DRC, slack).

The knobs at each stage are inputs. **For now we give them; eventually the model /
planner chooses them.** Each imagined state feeds the next stage. That's it — we
haven't designed beyond this yet.

---

## 4. Cheap proxies in the middle (to guide the imagination)

Partway through, the flow can compute cheap estimates *without* finishing. We use
these to guide/condition the imagination (as inputs, not as the answer):

- **HPWL** — half-perimeter wirelength. A cheap wirelength guess: sum of each net's
  bounding-box half-perimeter. Underestimates real routed wire.
- **RUDY** — rectangular uniform wire density. A cheap congestion guess: each net's
  wire smeared over its bounding box, summed per tile → where routing will be
  crowded, before routing.
- **GRWL** — global-route wirelength. A truer wirelength once *global* route runs
  (still cheaper than full routing). Observed: `routed ≈ 0.727 × GRWL`.

The model predicts the true outcome on top of these guides.

---

## 5. Features

*(To be filled in.)*

- **Floorplan:** die size, utilization, aspect ratio, IO, netlist graph — TBD
- **Placement:** RUDY map, density map, pin map, HPWL — extracted today
- **CTS:** TBD
- **Routing:** TBD
- **Netlist graph:** TBD (not built yet — needed for cross-design)

---

## 6. Data generation

We build training data by running the **real OpenROAD flow** (via LibreLane) and
logging what actually happens.

- **Generator:** `data_gen/sweep.py`, adapted from CTS-Bench's
  `1-gen-placement.py`. It runs the full flow with **randomized knobs** (core util,
  aspect ratio, target density, synthesis strategy, routability/timing-driven),
  then extracts placement features and parses routing labels into `dataset.jsonl`
  (one row per run).
- **Real OpenROAD:** every row is a real, completed flow — real placement, real
  routing, real DRC / WL / slack. That is the ground truth the world model learns
  to imagine.
- **Status:** 39 rows so far, all picorv32 (single design) — currently validating
  the learned flow on one design. Note: *total* routed WL is mostly determined by
  the config/gate-count, so it's an easy target; the interesting signal is spatial
  (congestion/DRC), which is what the placement actually decides.

---

## 7. Architecture (what I envision)

Keep it simple.

The netlist is a **graph** (cells connected by nets). Placement puts each cell at
an **(x,y) on a grid**. So the object is a **graph on a grid.**

Two operations connect the two:

- **Scatter:** spread each net's demand onto the grid tiles its cells cover — this
  *is* RUDY.
- **Gather:** read each tile's value back to the cells sitting there.

The model, a few rounds of:

1. pass messages on the **graph** (cells talk along their nets),
2. **scatter** onto the grid (seed it with RUDY),
3. pass messages on the **grid** (neighboring tiles talk — congestion spreads),
4. **gather** back to the cells.

Output: predict the true congestion/routing as **RUDY + a learned correction**,
plus per-net numbers (routed length, slack). The same block is reused at each
stage of the flow.

Details — graph type, grid resolution, number of rounds — get filled in as we
build and test against simple baselines.
