#!/usr/bin/env python3
"""Train f_cts — same size-stratified CV + locked OOD as f_place. Reuses train_fplace's folds,
loss shape (decoupled gnll), and level+deviation eval. Predicts the CTS state from the placement
netlist without running CTS."""
import os, sys, glob, json, time, random
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fcts
from fcts import FCTS, load_graph, set_norm, recon, CTS_GLOBAL, CTS_TIMING
from fplace import gnll, norm, FPlace
import train_fplace as TF          # reuse make_folds, SIZE_ANCHORS, OOD, flows_of

E = os.environ.get
DEV    = "cuda" if torch.cuda.is_available() else "cpu"
# THE SEAM. "" = standalone; "real" = teacher-forced (real placement state); "imagined" = consume
# f_place's prediction (needs FPLACE_CKPT). We run "real" FIRST — it's the compounding baseline.
SEAM_MODE  = E("SEAM_MODE", "") or None
FPLACE_CKPT = E("FPLACE_CKPT", "")   # required for SEAM_MODE=imagined (a trained f_place fold.pt)
# WARM START — PowPrediCT's (DAC'24) SCHEDULE, which is the part we were missing.
# We train `real` and `imagined` as SEPARATE models; they PRETRAIN teacher-forced on the real
# upstream state and THEN FINE-TUNE on the imagined one, and that step IS their result:
#   cross-design LOO total-power rel-err — Innovus 9.652% | vanilla GNN 14.149% (worse than the
#   tool!) | Phase-1-only 5.106% | full (1->3) 1.981%.   The 1->3 fine-tune more than HALVES it.
# Their Phase1 = pretrain on post-route graphs; Phase3 = swap the input to placement graphs and
# fine-tune. That is teacher forcing -> student forcing; they never name it.
# MasterRTL IV.A says the same thing harder: chained-on-PREDICTED beat chained-on-REAL
# (TNS MAPE 4% vs 62%) — the downstream model learns to INVERT the upstream model's systematic
# bias, which it cannot do if it only ever sees clean input.
# => set WARM_START=<dir with foldN.pt> (a SEAM_MODE=real run) and SEAM_MODE=imagined, with a
#    lower LR. This is the arm the literature says wins, and we did not have it.
WARM_START = E("WARM_START", "")     # dir of a trained (usually SEAM_MODE=real) f_cts to start from
EPOCHS = int(E("EPOCHS", 200)); LR = float(E("LR", 1e-3))
DIM    = int(E("DIM", 64)); LAYERS = int(E("LAYERS", 4)); ACCUM = int(E("ACCUM", 8))
SEED   = int(E("SEED", 0)); ENCODER = E("ENCODER", "dehnn")
W_DEV  = float(E("W_DEV", 3))       # up-weight the knob deviation — the thing the agent needs
FT_LR_MULT = float(E("FT_LR_MULT", 0.1))   # fine-tune LR multiplier when WARM_START is set
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
        # real units: buffers/power are log -> exp; wns/tns are signed-log -> inverse
        if k in ("cts_wns", "cts_tns"):
            rp, rt = np.sign(p)*np.expm1(np.abs(p)), np.sign(t)*np.expm1(np.abs(t))
        else:
            rp, rt = np.exp(p), np.exp(t)
        d = dict(n=int(ok.sum()),
                 med_ae=float(np.median(np.abs(rp - rt))),            # REAL units (buffers/W/ns)
                 med_rel=float(np.median(np.abs(np.expm1(p - t)))),
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
    print(f"\n=== fold {fi}: test {test_designs}  SEAM={SEAM_MODE}", flush=True)
    print(f"    train {len(tr)} flows/{len(train_d)} designs | val {len(val_d)} | test {test_designs}", flush=True)
    set_norm(train_d)                          # norm must exist before the seam builds place_state
    # SEAM: load the same-fold f_place if a ckpt is given — needed to feed IMAGINED placement
    # state (for imagined training AND for the imagined-input eval below). 'real' reads the real
    # placement state straight from EDA-Schema and needs no f_place.
    place_model = None
    if FPLACE_CKPT:
        place_model = FPlace(d=DIM, K=LAYERS, encoder=ENCODER).to(DEV)
        place_model.load_state_dict(torch.load(f"{FPLACE_CKPT}/fold{fi}.pt", map_location=DEV))
        place_model.eval()
    if SEAM_MODE == "imagined" and place_model is None:
        raise SystemExit("SEAM_MODE=imagined needs FPLACE_CKPT=<dir with foldN.pt>")
    fcts.set_seam(SEAM_MODE, place_model)          # TRAINING consumes SEAM_MODE's placement state
    model = FCTS(d=DIM, K=LAYERS, encoder=ENCODER).to(DEV)
    if WARM_START:
        # PowPrediCT Phase1 -> Phase3: keep the teacher-forced weights, swap the INPUT to the
        # imagined state, continue at low LR. The seam dims are identical in real/imagined mode
        # (only the VALUES differ), so the checkpoint loads unchanged.
        sd = torch.load(f"{WARM_START}/fold{fi}.pt", map_location=DEV)
        model.load_state_dict(sd)
        print(f"    WARM START from {WARM_START}/fold{fi}.pt  -> fine-tuning on SEAM={SEAM_MODE}",
              flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=(LR * FT_LR_MULT if WARM_START else LR))
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
        print(f"  ep {ep:3d} tr {tot/len(tr):7.3f} | VAL abs-err: buf {g_('cts_buffers','med_ae'):6.1f}cells "
              f"pow {g_('cts_power','med_ae'):9.0f}W wns {g_('cts_wns','med_ae'):.3f}ns tns {g_('cts_tns','med_ae'):8.1f}ns "
              f"| knob-R2: buf {g_('cts_buffers','knob_r2'):+.2f} pow {g_('cts_power','knob_r2'):+.2f} "
              f"wns {g_('cts_wns','knob_r2'):+.2f} | lr {opt.param_groups[0]['lr']:.1e} ({time.time()-t0:.0f}s)", flush=True)
    if best_state: model.load_state_dict(best_state)
    torch.save(model.state_dict(), f"{OUT}/fold{fi}.pt")
    # DUAL EVAL — score the SAME trained f_cts feeding different placement state at its input:
    #   REAL     -> ceiling: how good f_cts is with a perfect upstream (== teacher forcing at test)
    #   IMAGINED -> deployment: fed f_place's PREDICTION (the real pipeline).
    # REAL - IMAGINED = the compounding. A seam-on model has the extra input dims either way, so we
    # just swap the VALUES fed in; only a standalone (mode=None) model is evaluated as-is.
    if SEAM_MODE is None:
        eval_modes = [(None, None)]
    else:
        eval_modes = [("real", None)] + ([("imagined", place_model)] if place_model is not None else [])
    U = {"cts_buffers": "cells", "cts_power": "W", "cts_wns": "ns", "cts_tns": "ns"}
    all_res = {}
    for emode, pm in eval_modes:
        fcts.set_seam(emode, pm)
        res = evaluate(model, te); all_res[emode or "standalone"] = res
        print(f"  TEST feeding {(emode or 'standalone').upper()} placement state (unseen designs):", flush=True)
        for k in ALL:
            if k in res:
                r = res[k]
                print(f"      {k:12} abs-err {r['med_ae']:>9.3g} {U[k]:5} knob-R² {r.get('knob_r2',float('nan')):+.3f}  "
                      f"abs-R² {r['abs_r2']:+.3f}  rel-err {r['med_rel']*100:5.1f}%", flush=True)
    fcts.set_seam(SEAM_MODE, place_model)          # restore training mode for any later fold
    return dict(fold=fi, test_designs=test_designs, train_mode=SEAM_MODE, metrics=all_res)

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
