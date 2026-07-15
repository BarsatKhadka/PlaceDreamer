#!/usr/bin/env python3
"""Train f_cts — same size-stratified CV + locked OOD as f_place. Reuses train_fplace's folds,
loss shape (decoupled gnll), and level+deviation eval. Predicts the CTS state from the placement
netlist without running CTS."""
import os, sys, glob, json, time, random
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fcts
from fcts import FCTS, load_graph, set_norm, recon, CTS_GLOBAL, CTS_TIMING
from fplace import gnll, norm
import train_fplace as TF          # reuse make_folds, SIZE_ANCHORS, OOD, flows_of

E = os.environ.get
DEV    = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = int(E("EPOCHS", 200)); LR = float(E("LR", 1e-3))
DIM    = int(E("DIM", 64)); LAYERS = int(E("LAYERS", 4)); ACCUM = int(E("ACCUM", 8))
SEED   = int(E("SEED", 0)); ENCODER = E("ENCODER", "dehnn")
W_DEV  = float(E("W_DEV", 3))       # up-weight the knob deviation — the thing the agent needs
OUT    = E("OUT", "runs/fcts")
ALL    = CTS_GLOBAL + CTS_TIMING
torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED); os.makedirs(OUT, exist_ok=True)

def wloss(o, g):
    L = 0.0
    for k in ALL:
        if g[f"deg_{k}"]: continue                                  # knobs don't move it -> skip dev
        L = L + gnll(o[f"{k}_lvl"], g[f"y_{k}_lvl"])                # design level
        L = L + W_DEV * gnll(o[f"{k}_dev"], g[f"y_{k}_dev"])        # KNOB RESPONSE
    return L

@torch.no_grad()
def evaluate(model, flows):
    model.eval(); nm = norm()
    P = {k: [] for k in ALL}; T = {k: [] for k in ALL}          # reconstructed absolute (log)
    Pd = {k: [] for k in ALL}; Td = {k: [] for k in ALL}        # deviation (knob response)
    for f in flows:
        g = load_graph(f, DEV); o = model(g)
        for k in ALL:
            lv, dv = o[f"{k}_lvl"][0].item(), o[f"{k}_dev"][0].item()
            P[k].append(recon(k, lv, dv, nm, g.get(f"w_{k}"))); T[k].append(g[f"y_{k}"])
            if not g[f"deg_{k}"]:
                Pd[k].append(dv); Td[k].append(g[f"y_{k}_dev"].item())
    res = {}
    for k in ALL:
        p, t = np.array(P[k]), np.array(T[k]); ok = np.isfinite(p) & np.isfinite(t)
        if ok.sum() < 3: continue
        p, t = p[ok], t[ok]
        d = dict(n=int(ok.sum()),
                 med_rel=float(np.median(np.abs(np.expm1(p - t)))),   # log-space -> rel err
                 abs_r2=float(1 - ((t-p)**2).sum()/((t-t.mean())**2).sum()))
        pd_, td_ = np.array(Pd[k]), np.array(Td[k])
        if len(td_) >= 3 and td_.std() > 1e-9:
            d["knob_r2"] = float(1 - ((td_-pd_)**2).sum()/((td_-td_.mean())**2).sum())
        res[k] = d
    return res

def run_fold(fi, test_designs, dev):
    pool = [d for d in dev if d not in test_designs and d not in TF.SIZE_ANCHORS]
    rngd = random.Random(SEED + 100 + fi); shf = pool[:]; rngd.shuffle(shf)
    val_d, train_d = sorted(shf[:2]), sorted(shf[2:] + TF.SIZE_ANCHORS)
    tr, val, te = TF.flows_of(train_d), TF.flows_of(val_d), TF.flows_of(test_designs)
    rng = random.Random(SEED); rng.shuffle(tr)
    print(f"\n=== fold {fi}: test {test_designs}", flush=True)
    print(f"    train {len(tr)} flows/{len(train_d)} designs | val {len(val_d)} | test {test_designs}", flush=True)
    set_norm(train_d)
    model = FCTS(d=DIM, K=LAYERS, encoder=ENCODER).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=10, min_lr=1e-5)
    best, best_state = -1e9, None
    for ep in range(EPOCHS):
        model.train(); rng.shuffle(tr); t0 = time.time(); tot = 0.0; opt.zero_grad()
        for i, f in enumerate(tr):
            l = wloss(model(load_graph(f, DEV)), load_graph(f, DEV)) / ACCUM
            l.backward(); tot += l.item()*ACCUM
            if (i+1) % ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad()
        v = evaluate(model, val)
        score = float(np.mean([v[k]["knob_r2"] for k in ALL if k in v and "knob_r2" in v[k]] or [-9]))
        sched.step(score)
        if score > best + 1e-4:
            best = score; best_state = {k: x.detach().cpu().clone() for k, x in model.state_dict().items()}
        g_ = lambda k, s: v[k].get(s, float("nan")) if k in v else float("nan")
        print(f"  ep {ep:3d} tr {tot/len(tr):7.3f} | VAL knob-R²: buf {g_('cts_buffers','knob_r2'):+.3f} "
              f"pow {g_('cts_power','knob_r2'):+.3f} wns {g_('cts_wns','knob_r2'):+.3f} | "
              f"rel-err buf {g_('cts_buffers','med_rel')*100:.0f}% pow {g_('cts_power','med_rel')*100:.0f}% "
              f"| lr {opt.param_groups[0]['lr']:.1e} ({time.time()-t0:.0f}s)", flush=True)
    if best_state: model.load_state_dict(best_state)
    torch.save(model.state_dict(), f"{OUT}/fold{fi}.pt")
    res = evaluate(model, te)
    print(f"  TEST (unseen designs):", flush=True)
    for k in ALL:
        if k in res:
            r = res[k]
            print(f"      {k:12} knob-R² {r.get('knob_r2',float('nan')):+.3f}  "
                  f"abs-R² {r['abs_r2']:+.3f}  rel-err {r['med_rel']*100:5.1f}%", flush=True)
    return dict(fold=fi, test_designs=test_designs, metrics=res)

if __name__ == "__main__":
    dev, folds = TF.make_folds()
    which = E("FOLD", "all")
    sel = range(len(folds)) if which == "all" else [int(which)]
    print(f"device={DEV} encoder={ENCODER} — f_cts (CTS from placement netlist)")
    print(f"targets: {ALL}  (buffers/power EASY, wns/tns HARD)")
    out = [run_fold(i, folds[i], dev) for i in sel]
    tag = which if which != "all" else "all"
    json.dump(out, open(f"{OUT}/results_fold{tag}.json", "w"), indent=2)
    print(f"\nwrote {OUT}/results_fold{tag}.json")
