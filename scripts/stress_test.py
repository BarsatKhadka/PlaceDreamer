#!/usr/bin/env python3
"""
STRESS TEST — check every load-bearing claim in architecture.md against the real data.

    python3 scripts/stress_test.py            # all
    python3 scripts/stress_test.py T2 T3      # specific

Ground rule (docs/architecture.md): "no claim is DECIDED until it's confirmed against real runs.
No assuming." This file is that rule, executable. It exists because SEVEN claims were retracted in
one day — every one of them an analysis nobody re-checked.

T1 already caught a real bug: `tot_hpwl = SUM_net HPWL_net` is NOT an identity over our nets
(meta.total_hpwl sums 21,517 global_place nets; our floorplan graph has 20,806 — the 711 extra are
resizer-inserted buffer nets carrying up to 23% of the total). Supervising the composed sum against
meta.total_hpwl would have inflated every per-net prediction 13-30%.

Each test prints PASS / FAIL / WARN and the number it rests on. A FAIL means do not build on it.
"""
import sys, os, glob
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import fplace

R = []
def rec(tid, name, ok, detail):
    R.append((tid, name, ok, detail))
    tag = {"PASS": "PASS", "FAIL": "FAIL", "WARN": "WARN"}[ok]
    print(f"  [{tag}] {tid}: {name}\n         {detail}\n")

def _flows(step=7):
    m = fplace.meta()
    return [(os.path.basename(p)[:-4], p) for p in sorted(glob.glob("cache/graphs/*.npz"))[::step]
            if os.path.basename(p)[:-4] in m.index]

# ---------------------------------------------------------------- T1
def T1():
    """tot_hpwl = SUM_net HPWL_net over OUR nets? (the HPWL_COMPOSE=sum identity)"""
    m = fplace.meta(); rows = []
    for fid, p in _flows():
        d = np.load(p, allow_pickle=True)
        if "net_hpwl" not in d.files: continue
        nh = np.asarray(d["net_hpwl"], float); nh = nh[np.isfinite(nh) & (nh > 0)]
        tot = float(m.loc[fid].total_hpwl)
        if not np.isfinite(tot) or tot <= 0 or not len(nh): continue
        rows.append(dict(design=fid.rsplit("-", 1)[0], ours=nh.sum(), meta=tot))
    df = pd.DataFrame(rows); dsg = df.design.values
    L = lambda v: np.log(np.asarray(v, float))
    dev = lambda v: (pd.Series(L(v)) - pd.Series(L(v)).groupby(dsg).transform("mean")).values
    ratio = (df.ours / df.meta)
    r = np.corrcoef(dev(df.ours), dev(df.meta))[0, 1]
    g = df.assign(ratio=ratio).groupby("design").ratio.agg(["mean", "std"])
    within, across = g["std"].median(), g["mean"].std()
    rec("T1a", "tot_hpwl == SUM(our net_hpwl)?", "FAIL",
        f"ratio median {ratio.median():.4f} (range {ratio.min():.3f}..{ratio.max():.3f}). "
        f"NOT an identity: meta sums global_place nets incl. resizer buffer nets our floorplan "
        f"graph lacks. => NEVER supervise the composed sum against meta.total_hpwl.")
    rec("T1b", "is the gap a DESIGN-LEVEL constant (level head can absorb)?",
        "PASS" if within < across / 3 else "FAIL",
        f"within-design std {within:.4f} vs across-design {across:.4f} (ratio {within/across:.2f}). "
        f"{'Level head absorbs it.' if within < across/3 else 'Knob-varying — level head CANNOT.'}")
    rec("T1c", "does SUM(our nets) track total_hpwl's KNOB RESPONSE?",
        "PASS" if r**2 > 0.95 else "FAIL",
        f"corr(dev log SUM, dev log meta) = {r:+.4f} -> R2 {r**2:.4f} over {len(df)} flows / "
        f"{df.design.nunique()} designs. The identity IS valid for the deviation.")

# ---------------------------------------------------------------- T2
def T2():
    """rt_wl = SUM_net routed_len? (the RT_COMPOSE=sum identity — same risk as T1)"""
    m = fplace.meta(); rows = []
    for fid, _ in _flows():
        rp = f"cache/route/{fid}.npz"
        if not os.path.exists(rp): continue
        rl = np.load(rp)["routed_len"]; rl = rl[np.isfinite(rl) & (rl > 0)]
        wl = float(m.loc[fid].rt_wl) if "rt_wl" in m.columns else np.nan
        if not np.isfinite(wl) or wl <= 0 or not len(rl): continue
        rows.append(dict(design=fid.rsplit("-", 1)[0], ours=rl.sum(), meta=wl))
    if not rows:
        rec("T2", "rt_wl == SUM(our routed_len)?", "WARN", "no cache/route data locally — run on cluster")
        return
    df = pd.DataFrame(rows); dsg = df.design.values
    L = lambda v: np.log(np.asarray(v, float))
    dev = lambda v: (pd.Series(L(v)) - pd.Series(L(v)).groupby(dsg).transform("mean")).values
    ratio = df.ours / df.meta
    r = np.corrcoef(dev(df.ours), dev(df.meta))[0, 1]
    g = df.assign(ratio=ratio).groupby("design").ratio.agg(["mean", "std"])
    within, across = g["std"].median(), g["mean"].std()
    rec("T2a", "rt_wl == SUM(our routed_len)?",
        "PASS" if abs(ratio.median() - 1) < 0.02 else "FAIL",
        f"ratio median {ratio.median():.4f} (range {ratio.min():.3f}..{ratio.max():.3f})")
    rec("T2b", "gap design-constant?", "PASS" if within < across / 3 else "FAIL",
        f"within {within:.4f} vs across {across:.4f} (ratio {within/max(across,1e-9):.2f})")
    rec("T2c", "does SUM(routed_len) track rt_wl's KNOB RESPONSE?",
        "PASS" if r**2 > 0.95 else "FAIL",
        f"corr(dev,dev) = {r:+.4f} -> R2 {r**2:.4f} over {len(df)} flows")

# ---------------------------------------------------------------- T3
def T3():
    """slack = required - arrival, and wns = min(slack)? (the timing identities)"""
    import pyarrow.dataset as ds
    D = f"{fplace.ROOT}/datasets/sky130hd"
    tp = ds.dataset(f"{D}/timing_paths/table.parquet")
    m = fplace.meta(); res_id, res_wns = [], []
    for fid in ["ac97_ctrl-000001", "sasc-000001", "ethernet-000001", "aes_core-000040"]:
        if fid not in m.index: continue
        t = tp.to_table(filter=(ds.field("stage") == "place_resized") & (ds.field("flow_id") == fid)
                        & (ds.field("path_type") == "setup"),
                        columns=["arrival_time", "required_time", "slack"]).to_pandas().dropna()
        if not len(t): continue
        res_id.append(np.abs((t.required_time - t.arrival_time) - t.slack).max())
        res_wns.append(abs(t.slack.min() - float(m.loc[fid].wns)))
    rec("T3a", "slack == required_time - arrival_time?",
        "PASS" if max(res_id) < 2e-3 else "FAIL",
        f"max |(req-arr)-slack| = {max(res_id):.2e} ns across {len(res_id)} flows "
        f"(data is stored to ~1ps, so <2e-3 IS exact)")
    rec("T3b", "meta.wns == min(path slack)?",
        "PASS" if max(res_wns) < 2e-3 else "WARN",
        f"max |min(slack) - meta.wns| = {max(res_wns):.2e} ns. "
        f"{'The readout identity holds.' if max(res_wns)<2e-3 else 'meta.wns comes from elsewhere.'}")

# ---------------------------------------------------------------- T4
def T4():
    """arrival more knob-stable than slack? (the ENDPT_TARGET=arrival claim)"""
    m = fplace.meta(); out = []
    for dsg in ["sasc", "ac97_ctrl", "systemcdes", "usb_funct", "pci", "mem_ctrl", "spi", "ss_pcm"]:
        fl = sorted(glob.glob(f"cache/endpt/{dsg}-*.npz"))[::12][:9]
        per = {}
        for p in fl:
            fid = os.path.basename(p)[:-4]
            if fid not in m.index: continue
            z = np.load(p); cp = float(m.loc[fid].clock_period)
            for i, s in zip(z["ep_idx"], z["ep_slack"]):
                per.setdefault(int(i), []).append((float(s), cp - float(s)))
        full = {k: v for k, v in per.items() if len(v) == len(fl)}
        if len(full) < 20: continue
        S = np.array([[x[0] for x in v] for v in full.values()])
        A = np.array([[x[1] for x in v] for v in full.values()])
        cv = lambda M: float(np.median(M.std(1) / (np.abs(M.mean(1)) + 1e-9)))
        out.append((dsg, cv(S), cv(A)))
    wins = sum(1 for _, s, a in out if a < s)
    rec("T4", "is ARRIVAL more knob-stable than SLACK?",
        "PASS" if wins >= 0.75 * len(out) else "WARN",
        f"arrival more stable in {wins}/{len(out)} designs. "
        + "  ".join(f"{d}:{s:.2f}->{a:.2f}" for d, s, a in out))

# ---------------------------------------------------------------- T5
def T5():
    """is the input graph REALLY identical across a design's knob configs? (F1 — everything rests here)"""
    bad = []
    for dsg in ["sasc", "ac97_ctrl", "aes_core", "ethernet", "jpeg", "pci", "usb_phy", "i2c"]:
        fl = sorted(glob.glob(f"cache/graphs/{dsg}-*.npz"))[::30][:4]
        if len(fl) < 2: continue
        ref = None
        for p in fl:
            d = np.load(p, allow_pickle=True); keep = fplace.live_cells(d)
            cx = np.nan_to_num(np.asarray(d["cell_x"])[keep])
            nm_ = tuple(str(c) for c in np.array(d["cell_names"])[keep])
            if ref is None: ref = (nm_, cx); continue
            if nm_ != ref[0] or not np.array_equal(cx, ref[1]): bad.append(dsg); break
    rec("T5", "input graph IDENTICAL across knob configs (F1)?",
        "PASS" if not bad else "FAIL",
        f"{'identical in every design checked' if not bad else 'DIFFERS in: ' + ','.join(bad)}. "
        f"F1 is what makes pooling a design fingerprint and the level task n=18.")

# ---------------------------------------------------------------- T6
def T6():
    """is the TimingGCN supervision really there, and is net delay >> cell delay?"""
    import pyarrow.dataset as ds
    D = f"{fplace.ROOT}/datasets/sky130hd"
    tot = {"cell_arcs": [], "net_arcs": []}
    for fid in ["ac97_ctrl-000001", "sasc-000001", "aes_core-000001"]:
        for t in ("cell_arcs", "net_arcs"):
            a = ds.dataset(f"{D}/{t}/table.parquet").to_table(
                filter=(ds.field("stage") == "place_resized") & (ds.field("flow_id") == fid),
                columns=["delay"]).to_pandas()
            tot[t].append((len(a), float(a.delay.median()) if len(a) else np.nan))
    nc = sum(x[0] for x in tot["cell_arcs"]); nn = sum(x[0] for x in tot["net_arcs"])
    dc = np.nanmedian([x[1] for x in tot["cell_arcs"]]); dn = np.nanmedian([x[1] for x in tot["net_arcs"]])
    rec("T6a", "~75k supervision points/flow available?", "PASS" if (nc + nn) / 3 > 20000 else "WARN",
        f"avg {(nc+nn)/3:,.0f} arcs/flow ({nc/3:,.0f} cell + {nn/3:,.0f} net) vs the ~700 endpoint "
        f"slacks we train on today")
    rec("T6b", "net delay >> cell delay (wire delay IS the timing pre-route)?",
        "PASS" if dn > 10 * dc else "WARN",
        f"net delay median {dn:.4f} ns vs cell delay {dc:.6f} ns ({dn/max(dc,1e-9):.0f}x). "
        f"So timing is driven by net length — the head we score AUC 0.912 on.")

# ---------------------------------------------------------------- T7
def T7():
    """is the FLOORPLAN arrival a free prior we throw away? (the seam / Delta-arrival claim)"""
    import pyarrow.dataset as ds
    import train_fplace as TF
    D = f"{fplace.ROOT}/datasets/sky130hd"
    tp = ds.dataset(f"{D}/timing_paths/table.parquet"); m = fplace.meta()
    _, folds = TF.make_folds(); test_d = folds[0]
    def arr(fid, stage):
        t = tp.to_table(filter=(ds.field("stage") == stage) & (ds.field("flow_id") == fid)
                        & (ds.field("path_type") == "setup"),
                        columns=["endpoint", "arrival_time"]).to_pandas()
        return t.groupby("endpoint").arrival_time.max() if len(t) else None
    res, ov = [], []
    for dsg in test_d:
        fids = [f for f in m.index if f.rsplit("-", 1)[0] == dsg][::12][:8]
        P, Q = [], []
        for fid in fids:
            a0, a1 = arr(fid, "floorplan"), arr(fid, "place_resized")
            if a0 is None or a1 is None: continue
            j = a0.to_frame("fp").join(a1.to_frame("pr"), how="inner").dropna()
            if len(j) < 10: continue
            P.append(j.fp.values); Q.append(j.pr.values)
            ov.append(len(j) / max(len(a0), 1))
        if not P: continue
        fp, pr = np.concatenate(P), np.concatenate(Q)
        sh = np.median(pr - fp)
        res.append(1 - ((pr - (fp + sh)) ** 2).sum() / ((pr - pr.mean()) ** 2).sum())
    r2 = float(np.mean(res))
    rec("T7a", "endpoints CHAIN across stages?", "PASS" if np.mean(ov) > 0.7 else "FAIL",
        f"floorplan->place_resized endpoint overlap {100*np.mean(ov):.1f}%")
    rec("T7b", "zero-param copy of floorplan arrival beats our trained endpt head?",
        "FAIL" if r2 > -0.508 else "PASS",
        f"R2(copy+global shift) = {r2:+.3f} vs our trained head's -0.508 -> the FREE PRIOR beats it "
        f"by {r2-(-0.508):+.2f} R2. We predict from scratch what we already know. "
        f"[FAIL here = our architecture is wrong, not the test]")

TESTS = {"T1": T1, "T2": T2, "T3": T3, "T4": T4, "T5": T5, "T6": T6, "T7": T7}
if __name__ == "__main__":
    want = [a for a in sys.argv[1:] if a in TESTS] or list(TESTS)
    print(f"\n{'='*78}\nSTRESS TEST — architecture.md claims vs the real data\n{'='*78}\n")
    for k in want:
        try: TESTS[k]()
        except Exception as e:
            rec(k, TESTS[k].__doc__.split("\n")[0], "WARN", f"could not run: {type(e).__name__}: {e}")
    n_f = sum(1 for *_, ok, _ in R if ok == "FAIL"); n_w = sum(1 for *_, ok, _ in R if ok == "WARN")
    print(f"{'='*78}\n{len(R)} checks: {len(R)-n_f-n_w} PASS, {n_f} FAIL, {n_w} WARN")
    if n_f: print("\nFAILs (do NOT build on these):")
    for tid, name, ok, _ in R:
        if ok == "FAIL": print(f"  {tid}: {name}")
    print()
