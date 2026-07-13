#!/usr/bin/env python3
"""
Train f_place — locked OOD holdout + leave-designs-out CV on the dev set.

  5 LOCKED OOD designs  (jpeg, aes_core, tv80, wb_dma, i2c) — NEVER trained on, NEVER tuned on.
                         Touched exactly once, at the end: `python src/train_fplace.py --eval-ood`.
                         jpeg is the largest design in the dataset → true size EXTRAPOLATION.
 13 DEV designs        → 3 CV folds (hold out 5/4/4), every dev design tested once.
                         ALL development — hyperparams, ablations, loss weights — happens here.

Reports per target: R², median relative error, within-design r (the knob-effect signal).

Env vars (SLURM-friendly):
  FOLD=0|1|2|all     which fold(s)          (default all)
  ENCODER=dehnn|dehnn_novn|dehnn_undirected|sage|gat   (default dehnn)
  EPOCHS=30  LR=1e-3  DIM=64  LAYERS=4  ACCUM=8  SEED=0
  W_NETHPWL / W_NETDEM / W_TOT / W_BUFA / W_BUFC   loss weights (default 1each)
  OUT=runs/<name>    where to write metrics/checkpoints

Usage:  python src/train_fplace.py
"""
import os, sys, glob, json, time, random
import numpy as np, pandas as pd, torch
from scipy.stats import pearsonr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fplace import FPlace, load_graph, loss_fn, gnll, meta, CACHE, set_norm, denorm

E = os.environ.get
DEV     = "cuda" if torch.cuda.is_available() else "cpu"
ENCODER = E("ENCODER", "dehnn")
EPOCHS  = int(E("EPOCHS", 200)); LR = float(E("LR", 1e-3))
DIM     = int(E("DIM", 64));    LAYERS = int(E("LAYERS", 4))
ACCUM   = int(E("ACCUM", 8));   SEED = int(E("SEED", 0))
PATIENCE = int(E("PATIENCE", 0))    # 0 = no early stopping, train all EPOCHS (set >0 to enable)
# LOSS=decoupled (default): mean and variance train TOGETHER from step 0, but the mean's
# gradient never passes through sigma^2 (see fplace.gnll). No warm-up phase needed — the
# phased MSE->NLL switch was what froze the mean and flatlined the run at ep21.
LOSS     = E("LOSS", "decoupled")   # decoupled | beta | nll | mse
WARMUP   = int(E("WARMUP", 0))      # 0 = off. Only meaningful for LOSS=beta/nll.
NLL_LR_MULT = float(E("NLL_LR_MULT", 1.0))
OUT     = E("OUT", f"runs/{ENCODER}")
W = dict(net_hpwl=float(E("W_NETHPWL",1)), net_dem=float(E("W_NETDEM",1)),
         tot_hpwl=float(E("W_TOT",1)), buf_area=float(E("W_BUFA",1)), buf_cnt=float(E("W_BUFC",1)),
         wns=float(E("W_WNS",1)), tns=float(E("W_TNS",1)))
torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
os.makedirs(OUT, exist_ok=True)

# ---------- LOCKED OOD SET — never trained on, never tuned on ----------
# Chosen for size + family diversity (GAN-CTS/FastTuner convention: hold out by family AND
# size extreme). jpeg is the LARGEST design in the whole dataset → true size extrapolation
# (biggest thing the model ever trains on is ethernet, 39k cells).
# These are touched EXACTLY ONCE, at the end, via `python src/train_fplace.py --eval-ood`.
OOD_DESIGNS = ["jpeg",       # 54,900 cells — largest, DSP  → size extrapolation
               "aes_core",   # 15,164 — crypto
               "tv80",       #  6,018 — CPU
               "wb_dma",     #  2,986 — DMA/bus
               "i2c"]        #    908 — tiny, interface

# ---------- dev-set CV (leave-designs-out) ----------
def make_folds():
    """CV folds over the DEV designs only. OOD designs are excluded entirely."""
    all_d = sorted(meta().index.str.replace(r"-\d+$", "", regex=True).unique())
    dev = [d for d in all_d if d not in OOD_DESIGNS]           # 13 dev designs
    rng = random.Random(SEED); d = dev[:]; rng.shuffle(d)
    sizes = [5, 4, 4]                                          # 13 dev designs, each tested once
    folds, i = [], 0
    for s in sizes:
        folds.append(sorted(d[i:i+s])); i += s
    return dev, folds

def flows_of(designs):
    idx = meta().index
    return [f for f in idx if f.rsplit("-", 1)[0] in set(designs)]

def wloss(out, g, nll=True):
    L = (W["net_hpwl"] * gnll(out["net_hpwl"], g["y_net_hpwl"], g["m_net_hpwl"], nll)
       + W["net_dem"]  * gnll(out["net_dem"],  g["y_net_dem"],  g["m_net_dem"],  nll)
       + W["tot_hpwl"] * gnll(out["tot_hpwl"], g["y_tot_hpwl"], None, nll)
       + W["buf_area"] * gnll(out["buf_area"], g["y_buf_area"], None, nll)
       + W["wns"]      * gnll(out["wns"],      g["y_wns"],      None, nll)
       + W["tns"]      * gnll(out["tns"],      g["y_tns"],      None, nll))
    if g["has_bufcnt"]: L = L + W["buf_cnt"] * gnll(out["buf_cnt"], g["y_buf_cnt"], None, nll)
    return L

@torch.no_grad()
def evaluate(model, flows):
    """collect predictions vs truth for every target."""
    model.eval()
    NET = ("net_hpwl","net_dem")                                    # per-net (masked) heads
    GLOB = ("tot_hpwl","buf_area","buf_cnt","wns","tns")            # per-flow scalar heads
    LOGK = ("net_hpwl","net_dem","tot_hpwl","buf_area","buf_cnt")   # log targets -> rel-err via expm1
    P = {k: [] for k in NET+GLOB}
    T = {k: [] for k in P}; sig = {k: [] for k in P}; dz = []
    for f in flows:
        g = load_graph(f, DEV); o = model(g)
        for k in NET:
            m = g[f"m_{k}"]; y = g[f"y_{k}"]
            P[k].append(o[k][m,0].cpu().numpy()); T[k].append(y[m].cpu().numpy())
            sig[k].append(o[k][m,1].cpu().numpy())
        for k in GLOB:
            P[k].append(np.array([o[k][0].item()])); T[k].append(np.array([g[f"y_{k}"].item()]))
            sig[k].append(np.array([o[k][1].item()]))
        dz.append(f.rsplit("-",1)[0])
    res = {}
    for k in P:
        p, t = np.concatenate(P[k]), np.concatenate(T[k])
        ok = np.isfinite(p) & np.isfinite(t)
        p, t = p[ok], t[ok]
        if len(t) < 3 or t.std() < 1e-9: continue
        r2 = 1 - ((t-p)**2).sum()/((t-t.mean())**2).sum()   # affine-invariant: same in std or log space
        if k in LOGK:   # standardized log -> un-standardize, relative error via expm1
            rel = float(np.median(np.abs(np.expm1(denorm(k, p) - denorm(k, t)))))
        else:           # wns/tns: report median abs error in standardized units
            rel = float(np.median(np.abs(p - t)))
        res[k] = dict(r2=float(r2), rel_err=rel, n=int(len(t)))
    # within-design r on the GLOBAL targets (knob-effect signal, size held constant)
    for k in GLOB:
        p, t, dd = np.concatenate(P[k]), np.concatenate(T[k]), np.array(dz)
        rs = [pearsonr(p[dd==d], t[dd==d])[0] for d in np.unique(dd)
              if (dd==d).sum() > 2 and t[dd==d].std() > 1e-9 and p[dd==d].std() > 1e-9]
        if rs and k in res: res[k]["within_r"] = float(np.nanmean(rs))
    return res

def run_fold(fi, test_designs, all_designs):
    train_d = [d for d in all_designs if d not in test_designs]
    tr_all  = flows_of(train_d); te = flows_of(test_designs)
    rng = random.Random(SEED); rng.shuffle(tr_all)
    nval = max(1, int(0.1*len(tr_all))); val, tr = tr_all[:nval], tr_all[nval:]
    print(f"\n=== fold {fi}: test on {len(test_designs)} designs {test_designs}", flush=True)
    print(f"    train {len(tr)} flows / {len(train_d)} designs | val {len(val)} | test {len(te)}", flush=True)

    # normalization from TRAIN designs only — no test/OOD statistics ever touch the model
    set_norm(train_d)

    model = FPlace(d=DIM, K=LAYERS, encoder=ENCODER).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    # long runs plateau; halve the LR when val stalls, floor at 1e-5.
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=10, min_lr=1e-5)
    best, best_state, patience = 1e9, None, 0   # we always KEEP the best-val checkpoint
    for ep in range(EPOCHS):
        nll = ep >= WARMUP          # WARMUP=0 by default -> variance head live from step 0
        if WARMUP and ep == WARMUP:
            for pg in opt.param_groups: pg["lr"] = LR * NLL_LR_MULT
            best, patience = 1e9, 0     # objective changed; loss value not comparable across it
            print(f"  --- epoch {ep}: variance head on, lr -> {LR*NLL_LR_MULT:.1e} ---", flush=True)
        model.train(); rng.shuffle(tr); t0=time.time(); tot=0.0
        opt.zero_grad()
        for i, f in enumerate(tr):
            g = load_graph(f, DEV)
            l = wloss(model(g), g, nll) / ACCUM
            l.backward(); tot += l.item()*ACCUM
            if (i+1) % ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad()
        opt.step(); opt.zero_grad()
        # val — scored the same way the epoch was trained, PLUS per-target R² so we can
        # watch every placement metric (not just the scalar loss) improve in real time.
        NETK  = ("net_hpwl","net_dem")
        GLOBK = ("tot_hpwl","buf_area","buf_cnt","wns","tns")
        model.eval(); vl=0.0
        sse={k:0. for k in NETK}; sst={k:0. for k in NETK}
        gp={k:[] for k in GLOBK}; gt={k:[] for k in GLOBK}
        with torch.no_grad():
            for f in val:
                g = load_graph(f, DEV); o = model(g)
                vl += wloss(o, g, nll).item()
                for k in NETK:
                    m_ = g[f"m_{k}"]
                    if m_.sum() < 3: continue
                    p_, t_ = o[k][m_,0], g[f"y_{k}"][m_]
                    sse[k] += (t_-p_).pow(2).sum().item(); sst[k] += (t_-t_.mean()).pow(2).sum().item()
                for k in GLOBK:
                    gp[k].append(o[k][0].item()); gt[k].append(g[f"y_{k}"].item())
        vl /= max(1,len(val))
        r2 = {k: (1 - sse[k]/sst[k]) if sst[k] > 0 else float("nan") for k in NETK}
        for k in GLOBK:
            p_, t_ = np.array(gp[k]), np.array(gt[k])
            r2[k] = (1 - ((t_-p_)**2).sum()/((t_-t_.mean())**2).sum()) if t_.std() > 1e-9 else float("nan")
        sched.step(vl)
        lr_now = opt.param_groups[0]["lr"]
        print(f"  ep {ep:3d} [{'nll' if nll else 'mse'}] tr {tot/len(tr):7.3f} vl {vl:7.3f} | "
              f"R² hpwl {r2['net_hpwl']:+.3f} tot {r2['tot_hpwl']:+.3f} "
              f"bufA {r2['buf_area']:+.3f} bufC {r2['buf_cnt']:+.3f} "
              f"wns {r2['wns']:+.3f} tns {r2['tns']:+.3f} | lr {lr_now:.1e} ({time.time()-t0:.0f}s)",
              flush=True)
        if vl < best - 1e-4:
            best, best_state, patience = vl, {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}, 0
        else:
            patience += 1
            if PATIENCE and patience >= PATIENCE:      # disabled by default (PATIENCE=0)
                print(f"  early stop (no val improvement in {PATIENCE} epochs)"); break
    print(f"  done — best val {best:.4f}; restoring best checkpoint", flush=True)
    if best_state: model.load_state_dict(best_state)
    torch.save(model.state_dict(), f"{OUT}/fold{fi}.pt")
    res = evaluate(model, te)
    print(f"  TEST (unseen designs): " + " | ".join(
        f"{k}: R²={v['r2']:.3f} rel={v['rel_err']*100:.1f}%" + (f" wr={v['within_r']:.2f}" if 'within_r' in v else "")
        for k,v in res.items()), flush=True)
    return dict(fold=fi, test_designs=test_designs, n_train=len(tr), n_test=len(te), metrics=res)

def eval_ood():
    """FINAL test — run ONCE, at the very end, on the locked OOD designs.
    Loads the fold checkpoints (trained only on dev) and evaluates on designs never seen."""
    dev, folds = make_folds()
    te = flows_of(OOD_DESIGNS)
    print(f"\n########## LOCKED OOD EVALUATION ##########")
    print(f"OOD designs (never trained/tuned on): {OOD_DESIGNS}")
    print(f"{len(te)} test flows\n")
    allres = []
    for ck in sorted(glob.glob(f"{OUT}/fold*.pt")):
        fi = int(os.path.basename(ck)[4:-3])                 # fold<N>.pt
        # each checkpoint was trained with ITS fold's normalization — restore exactly that,
        # or the predictions come back on the wrong scale.
        set_norm([d for d in dev if d not in folds[fi]])
        model = FPlace(d=DIM, K=LAYERS, encoder=ENCODER).to(DEV)
        model.load_state_dict(torch.load(ck, map_location=DEV)); model.eval()
        r = evaluate(model, te); allres.append(r)
        print(f"  {os.path.basename(ck)}: " + " | ".join(
            f"{k}: R²={v['r2']:.3f} rel={v['rel_err']*100:.1f}%" for k, v in r.items()))
    print("\n=== OOD (ensemble across folds) ===")
    for k in ("net_hpwl","net_dem","tot_hpwl","buf_area","buf_cnt","wns","tns"):
        r2 = [r[k]["r2"] for r in allres if k in r]
        if r2: print(f"  {k:10} R² = {np.mean(r2):.3f} ± {np.std(r2):.3f}")
    json.dump(allres, open(f"{OUT}/ood_results.json","w"), indent=2)
    print(f"\nwrote {OUT}/ood_results.json")

def aggregate():
    """Combine per-fold results (written by parallel array tasks) into one CV number."""
    out = []
    for f in sorted(glob.glob(f"{OUT}/results_fold*.json")):
        out += json.load(open(f))
    if not out:
        print(f"no {OUT}/results_fold*.json yet — folds still running?"); return
    print(f"\n=== AGGREGATE across {len(out)} folds (unseen designs) — {OUT} ===")
    for k in ("net_hpwl","net_dem","tot_hpwl","buf_area","buf_cnt","wns","tns"):
        r2 = [f["metrics"][k]["r2"] for f in out if k in f["metrics"]]
        wr = [f["metrics"][k]["within_r"] for f in out
              if k in f["metrics"] and "within_r" in f["metrics"][k]]
        if r2:
            line = f"  {k:10} R² = {np.mean(r2):6.3f} ± {np.std(r2):.3f}"
            if wr: line += f"   within-design r = {np.nanmean(wr):.3f}"
            print(line)
    json.dump(out, open(f"{OUT}/results.json","w"), indent=2)
    print(f"\nwrote {OUT}/results.json")

if __name__ == "__main__":
    dev, folds = make_folds()
    if "--eval-ood" in sys.argv:
        eval_ood(); sys.exit(0)
    if "--aggregate" in sys.argv:
        aggregate(); sys.exit(0)
    which = E("FOLD","all")
    sel = range(len(folds)) if which=="all" else [int(which)]
    print(f"device={DEV} encoder={ENCODER} dim={DIM} layers={LAYERS} lr={LR} "
          f"epochs={EPOCHS} patience={PATIENCE}")
    print(f"loss weights: {W}")
    print(f"LOCKED OOD (excluded from everything): {OOD_DESIGNS}")
    print(f"{len(dev)} dev designs → {len(folds)} CV folds: {folds}")
    out = [run_fold(i, folds[i], dev) for i in sel]
    # per-fold file: parallel array tasks must not clobber each other
    tag = which if which != "all" else "all"
    json.dump(out, open(f"{OUT}/results_fold{tag}.json","w"), indent=2)
    print(f"\nwrote {OUT}/results_fold{tag}.json")
    if which == "all":
        aggregate()
