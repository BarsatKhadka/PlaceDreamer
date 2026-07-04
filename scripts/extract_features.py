#!/usr/bin/env python3
"""
PlaceDreamer feature extractor.

Runs under OpenROAD's python:  openroad -python extract_features.py <in.odb> <outdir> [grid]

Computes, from an end-of-global-placement OpenDB, the input-state features for the
world model — all pre-route and deterministic (we do NOT use gui::dump_heatmap,
which only populates in a GUI event loop):
  - HPWL (total, microns)                    the cheap wirelength proxy
  - RUDY map (grid CSV)                       the cheap congestion proxy
  - cell-density map (grid CSV)               placement density
  - pin-density map (grid CSV)
  - summary.json                              scalars + grid metadata

RUDY is a faithful reimplementation of OpenROAD's src/grt/src/Rudy.cpp:
  wire_width = 1.25 / Σ_layers(1/pitch)          (≈ harmonic-mean-pitch / n_layers)
  net_congestion = (dx+dy) * wire_width / bbox_area      per net (getTermBBox)
  tile += net_congestion * (intersect_area/tile_area) * 100
One term is intentionally omitted: getResourceReductions() (the GRT capacity-
reduction baseline from blockages), which needs a live global-route init. It is
negligible for macro-free designs. See features.md B2.
"""
import sys, os, json
import openroad, odb


def routing_wire_width(tech, min_layer="met1", max_layer="met5"):
    """OpenROAD Rudy.cpp: wire_width_ = (1/Σ 1/pitch) * 1.25 over the global
    router's signal routing layers (RT_MIN_LAYER..RT_MAX_LAYER, by name)."""
    lo = tech.findLayer(min_layer).getRoutingLevel()
    hi = tech.findLayer(max_layer).getRoutingLevel()
    pitch_terms = 0.0
    for layer in tech.getLayers():
        if layer.getType() != "ROUTING":
            continue
        lvl = layer.getRoutingLevel()
        if lvl < lo or lvl > hi:
            continue
        pitch = layer.getPitch()
        if pitch == 0:
            pitch = layer.getWidth() + layer.getSpacing()
        if pitch > 0:
            pitch_terms += 1.0 / pitch
    if pitch_terms == 0:
        return 100.0  # OpenROAD default
    return (1.0 / pitch_terms) * 1.25


def main():
    in_odb = sys.argv[1]
    outdir = sys.argv[2]
    G = int(sys.argv[3]) if len(sys.argv) > 3 else 64   # grid resolution (GxG bins)
    os.makedirs(outdir, exist_ok=True)

    db = odb.dbDatabase.create()
    odb.read_db(db, in_odb)
    blk = db.getChip().getBlock()
    tech = db.getTech()
    dbu = blk.getDbUnitsPerMicron()
    die = blk.getDieArea()
    x0, y0, x1, y1 = die.xMin(), die.yMin(), die.xMax(), die.yMax()
    W, H = x1 - x0, y1 - y0
    bw, bh = W / G, H / G

    wire_width = routing_wire_width(tech)   # dbu

    rudy = [[0.0] * G for _ in range(G)]
    hpwl_dbu = 0.0
    nplaced = 0
    for net in blk.getNets():
        if net.getSigType() in ("POWER", "GROUND"):
            continue
        r = net.getTermBBox()               # terminal bounding box (dbu)
        nx0, ny0, nx1, ny1 = r.xMin(), r.yMin(), r.xMax(), r.yMax()
        dx, dy = nx1 - nx0, ny1 - ny0
        net_area = dx * dy
        if net_area <= 0:                   # inverted / zero-area (e.g. 1-pin)
            continue
        hpwl_dbu += (dx + dy)
        nplaced += 1
        net_congestion = (dx + dy) * wire_width / float(net_area)
        # rasterize bbox onto grid, weight by intersect/tile area (OpenROAD formula)
        cxlo = max(0, int((nx0 - x0) / bw)); cxhi = min(G - 1, int((nx1 - x0) / bw))
        cylo = max(0, int((ny0 - y0) / bh)); cyhi = min(G - 1, int((ny1 - y0) / bh))
        for cy in range(cylo, cyhi + 1):
            byl, byh = y0 + cy * bh, y0 + (cy + 1) * bh
            oy = max(0.0, min(ny1, byh) - max(ny0, byl))
            if oy <= 0: continue
            for cx in range(cxlo, cxhi + 1):
                bxl, bxh = x0 + cx * bw, x0 + (cx + 1) * bw
                ox = max(0.0, min(nx1, bxh) - max(nx0, bxl))
                if ox <= 0: continue
                tile_ratio = (ox * oy) / (bw * bh)
                rudy[cy][cx] += net_congestion * tile_ratio * 100.0

    # cell density + pin density
    dens = [[0.0] * G for _ in range(G)]
    pind = [[0.0] * G for _ in range(G)]
    for inst in blk.getInsts():
        bb = inst.getBBox()
        ix0, iy0, ix1, iy1 = bb.xMin(), bb.yMin(), bb.xMax(), bb.yMax()
        cx = min(G - 1, max(0, int(((ix0 + ix1) / 2 - x0) / bw)))
        cy = min(G - 1, max(0, int(((iy0 + iy1) / 2 - y0) / bh)))
        dens[cy][cx] += (ix1 - ix0) * (iy1 - iy0) / (bw * bh)
        pind[cy][cx] += len(inst.getITerms())

    def dump(grid, path):
        with open(path, "w") as f:
            for row in grid:
                f.write(",".join(f"{v:.6g}" for v in row) + "\n")

    dump(rudy, os.path.join(outdir, "rudy.csv"))
    dump(dens, os.path.join(outdir, "density.csv"))
    dump(pind, os.path.join(outdir, "pindensity.csv"))

    maxr = max(max(r) for r in rudy)
    summary = {
        "in_odb": in_odb,
        "design": blk.getName(),
        "dbu_per_micron": dbu,
        "die_um": [x0 / dbu, y0 / dbu, x1 / dbu, y1 / dbu],
        "grid": G,
        "bin_um": [bw / dbu, bh / dbu],
        "wire_width_dbu": wire_width,
        "hpwl_um": hpwl_dbu / dbu,
        "n_nets_counted": nplaced,
        "n_insts": len(blk.getInsts()),
        "rudy_max": maxr,
        "rudy_mean": sum(sum(r) for r in rudy) / (G * G),
    }
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


main()
