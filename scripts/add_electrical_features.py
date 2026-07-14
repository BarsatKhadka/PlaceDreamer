#!/usr/bin/env python3
"""Per-net ELECTRICAL features -> cache/elec/{design}.npz

WHY: buffers are inserted for CAP / SLEW violations, NOT for fanout (Kahng ISPD'26 contest §2.2;
GraPhSyM arXiv:2308.03944 Table I). We currently feed the model NOTHING about capacitance, which
is very likely why the buffer targets are our weakest (buf_area knob-response +0.37).

The physical mechanism: a driver with drive strength D driving a total load capacitance C has a
slew ~ C/D. When that exceeds the tool's limit, the resizer INSERTS A BUFFER. So the load/drive
ratio is literally the trigger condition for buffering.

MEASURED before building (aes_core), corr with log HPWL and — crucially — with fanout, since a
feature that just restates fanout is worthless:
    feature            corr w/ HPWL   corr w/ FANOUT
    fanout (have it)      +0.660          --
    driven_cap            +0.687        +0.963   <- 96% REDUNDANT with fanout. NOT ADDED.
    drive_strength        +0.227         (new)
    load/drive ratio      +0.372        +0.623   <- ~38% new information. THE one that matters.

So we add 3 features (not the redundant driven_cap on its own):
    drv_strength   the driver's drive strength
    load_drive     total sink input-capacitance / driver drive strength   <- the buffering trigger
    max_sink_cap   the largest single sink capacitance on the net (a max-cap violation is a max,
                   not a sum — one fat sink can trigger buffering on its own)

The netlist is fixed per design (knobs don't change it), so this is computed ONCE per design and
shared by all 108 flows. No graph rebuild.
"""
import numpy as np, glob, os

ROOT  = "/Users/barsat/PlaceDreamer"
CACHE = f"{ROOT}/cache/graphs"
OUT   = f"{ROOT}/cache/elec"
os.makedirs(OUT, exist_ok=True)

# raw cell_x layout (build_graph.py): 9 drive_strength, 10 input_capacitance_max
I_DRIVE, I_INCAP = 9, 10

designs = sorted({os.path.basename(p).rsplit("-", 1)[0] for p in glob.glob(f"{CACHE}/*.npz")})
for di, dsg in enumerate(designs):
    z = np.load(sorted(glob.glob(f"{CACHE}/{dsg}-*.npz"))[0], allow_pickle=True)
    cx = np.nan_to_num(z["cell_x"])
    N  = len(z["net_x"])
    drv, snk = z["edge_driver"], z["edge_sink"]

    net_driver = {int(n): int(c) for c, n in zip(drv[0], drv[1])}
    net_sinks  = {}
    for c, n in zip(snk[0], snk[1]):
        net_sinks.setdefault(int(n), []).append(int(c))

    drive_s = np.zeros(N, np.float32)
    load_dr = np.zeros(N, np.float32)
    max_cap = np.zeros(N, np.float32)
    for n in range(N):
        sinks = net_sinks.get(n, [])
        d = net_driver.get(n)
        ds = float(cx[d, I_DRIVE]) if d is not None else 0.0
        if sinks:
            caps = cx[sinks, I_INCAP]
            tot, mx = float(caps.sum()), float(caps.max())
        else:
            tot = mx = 0.0
        drive_s[n] = ds
        load_dr[n] = tot / max(ds, 0.1)      # THE buffering trigger: slew ~ C/D
        max_cap[n] = mx                      # a MAX-cap violation is a max, not a sum

    ef = np.stack([drive_s, load_dr, max_cap], 1).astype(np.float32)   # (N, 3)
    np.savez(f"{OUT}/{dsg}.npz", elec_net=ef)
    print(f"  [{di+1}/{len(designs)}] {dsg}: {N} nets -> elec_net {ef.shape}", flush=True)

print(f"\n✓ wrote {len(designs)} designs to {OUT}  (3 electrical net features each)")
