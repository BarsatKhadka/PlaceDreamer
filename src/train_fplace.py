#!/usr/bin/env python3
"""
Train f_place — locked OOD holdout + leave-designs-out CV on the dev set.

  5 LOCKED OOD designs  (jpeg, aes_core, tv80, wb_dma, i2c) — NEVER trained on, NEVER tuned on.
                         Touched exactly once, at the end: `python src/train_fplace.py --eval-ood`.
                         jpeg is the largest design in the dataset → true size EXTRAPOLATION.
 13 DEV designs        → 3 CV folds (hold out 5/4/4), every dev design tested once.
                         ALL development — hyperparams, ablations, loss weights — happens here.

VALIDATION IS A HELD-OUT-DESIGN SPLIT (2 designs, never trained on). It used to be a random
10% of FLOWS from the training designs — which, since ~99% of the global targets' variance is
design identity, meant val SHARED that identity with train and could not see cross-design
overfitting at all. Yet it drove both the LR schedule and checkpoint selection.

METRICS: plain absolute error in REAL units (um, um^2, cells, ns) is PRIMARY. Pooled R2 is
reported but is NOT trustworthy for the global targets — ~99% of their variance is merely
"how big is this design", so R2 flatters. `within-R2` (design size held constant) is the number
that measures the knob response, which is the only thing f_place exists to predict.

Env vars (SLURM-friendly):
  FOLD=0|1|2|all     which fold(s)          (default all)
  ENCODER=dehnn|dehnn_novn|dehnn_undirected|sage|gat   (default dehnn)
  EPOCHS=200  LR=1e-3  DIM=64  LAYERS=4  ACCUM=8  SEED=0
  W_NETHPWL=5 W_ENDPT=3 W_TOT=1 W_BUFA=1 W_BUFC=1   loss weights (dense targets up-weighted:
                     they are a .mean() over ~10k nets, so each element got 1/N of the gradient
                     while every global scalar contributed its error undivided — net_hpwl was
                     getting 2.2% of the encoder's gradient, a single scalar 17.8%)
  OUT=runs/<name>    where to write metrics/checkpoints

Usage:  python src/train_fplace.py
"""
import os, sys, glob, json, time, random
import numpy as np, pandas as pd, torch
from scipy.stats import pearsonr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fplace import (FPlace, load_graph, gnll, meta, CACHE, set_norm, denorm,
                    norm, recon, GLOBAL_TARGETS)

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
# LOSS WEIGHTS. The dense (per-node) targets are a .mean() over ~10k nets / ~700 endpoints, so
# each element contributes 1/N of the gradient, while every GLOBAL scalar contributes its error
# undivided. Measured share of the encoder's gradient at all-weights-1.0:
#     net_hpwl  2.2%  (~10,000 supervision points/flow!)   tot_hpwl 17.8%  (1 scalar)
#     buf_area 16.5%  buf_cnt 14.9%  endpt 25.4%           <- nobody designed this
# Up-weight the dense targets so the encoder is actually driven by the per-node structure we
# want it to learn, not by four scalars that are ~99% design size.
W = dict(net_hpwl=float(E("W_NETHPWL", 5)),        # dense: ~10k nets/flow
         endpt=float(E("W_ENDPT", 3)),             # dense: ~700 endpoints/flow
         tot_hpwl=float(E("W_TOT", 1)), buf_area=float(E("W_BUFA", 1)),
         buf_cnt=float(E("W_BUFC", 1)),
         # WNS/TNS as first-class global heads. They used to be READOUTS off the broken endpt
         # head (pooled R2 -0.508) and scored -1.102 / -0.126 knob-response — worse than a
         # constant, while plain OLS on the 3 raw knobs gets 0.649 / 0.657. Weighted like the
         # other globals; the dev head already gets the raw knobs (DIRECT_KNOB), which is
         # exactly where that 0.649 lives.
         wns_g=float(E("W_WNS", 1)), tns_g=float(E("W_TNS", 1)),
         hpwl_sum=float(E("W_HPWL_SUM", 3)),   # calibrates per-net preds to the true total
         pos=float(E("W_POS", 5)),                 # dense: EVERY cell — placement GEOMETRY
         vnbox=float(E("W_VNBOX", 5)),             # per-METIS-cluster bbox — the POSED geometry
         dev=float(E("W_DEV", 3)))                 # knob-DEVIATION weight (the thing we want)
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
# Design sizes (live cells, i.e. AFTER dropping tapcells). Used to stratify the folds.
DESIGN_SIZE = {
    "ss_pcm": 268, "sasc": 275, "usb_phy": 354, "simple_spi": 411, "systemcdes": 1479,
    "spi": 1868, "des3_area": 2512, "ac97_ctrl": 4567, "mem_ctrl": 4729, "systemcaes": 5028,
    "usb_funct": 9221, "pci": 9428, "ethernet": 20596,
    # OOD (never in a dev fold):
    "i2c": 633, "wb_dma": 1964, "tv80": 4426, "aes_core": 11383, "jpeg": 37082,
}
# SIZE ANCHORS — always in TRAIN, never tested. They pin the ENDS of the training size range so
# every test design is bracketed by training designs, i.e. INTERPOLATION, never extrapolation.
# Without this, a random shuffle put ethernet (20,596 cells) in a TEST fold while the largest
# TRAINING design was 9,428 — a 2.2x size extrapolation. That fold's timing scored -5.03 while
# another fold scored +0.49. That was FOLD DIFFICULTY masquerading as model variance, and it
# made every config comparison unreliable.
# jpeg (37,082) stays in the LOCKED OOD set — it is the real extrapolation test, run ONCE at the end.
SIZE_ANCHORS = ["ethernet", "ss_pcm"]        # largest and smallest dev designs

def make_folds():
    """Size-STRATIFIED CV folds over the DEV designs. OOD designs are excluded entirely.

    The size extremes are pinned into TRAIN, and the rest are round-robined by size, so every
    fold's test set spans small->large AND sits inside the training range. Fold difficulty is
    equalized, so a difference between two configs is attributable to the CONFIG, not the fold.
    """
    all_d = sorted(meta().index.str.replace(r"-\d+$", "", regex=True).unique())
    dev = [d for d in all_d if d not in OOD_DESIGNS]           # 13 dev designs
    testable = sorted([d for d in dev if d not in SIZE_ANCHORS],
                      key=lambda d: DESIGN_SIZE.get(d, 0))     # 11, sorted small -> large
    folds = [[] for _ in range(3)]
    for i, d in enumerate(testable):                           # round-robin by size
        folds[i % 3].append(d)
    return dev, [sorted(f) for f in folds]

def flows_of(designs):
    idx = meta().index
    return [f for f in idx if f.rsplit("-", 1)[0] in set(designs)]

# readout_loss (v2b) DELETED — it was the direct cause of the WNS damage.
# It summed predicted slack over ep_idx = the LABELED endpoints only (median coverage 46%),
# but regressed that sum onto g["tns_true"] = the COMPLETE recorded TNS. Its docstring claimed
# it "supervises the endpoints we have no individual label for" — it CANNOT: those endpoints
# have no node in the sum and receive zero gradient. What it actually did was force the labeled
# 46% to inflate their slack ~2.2x (des3_area: 33x), fighting the endpt MSE on the same outputs.
# Simulating exactly that inflation reproduces the failure: wns R2 = -57.5 / -164.4 / -18.1.
# Verified oracle ceiling: a PERFECT endpoint predictor scores tns R2 = 0.34 (label truncation),
# but wns R2 = 0.86-1.0 — so WNS is reachable and this term was what broke it.

def wloss(out, g, nll=True):
    L = (W["net_hpwl"] * gnll(out["net_hpwl"], g["y_net_hpwl"], g["m_net_hpwl"], nll)
       + W["endpt"]    * gnll(out["endpt"],    g["y_endpt"],    g["m_endpt"], nll))
    # PLACEMENT GEOMETRY — the densest target we have (every cell, ~100% coverage). This is what
    # the seam actually needs to carry: CTS is a function of where the sinks landed, not of a
    # scalar total-HPWL. Weighted like net_hpwl (both are dense per-node structure).
    if "y_pos_x" in g:
        L = L + W["pos"] * (gnll(out["pos_x"], g["y_pos_x"], g["m_pos"], nll)
                          + gnll(out["pos_y"], g["y_pos_y"], g["m_pos"], nll))
    # ANALYTIC COMPOSITION: supervise tot_hpwl = SUM_net HPWL_net directly (HPWL_COMPOSE=sum).
    # Plain MSE in log-space, no sigma: this term's job is to CALIBRATE the per-net predictions in
    # absolute terms so their sum reproduces the true total. Ranking alone does not (per-net AUC
    # 0.912 but 43.7% absolute rel-err). Measured on a 182-flow probe: the composed sum holds
    # ~15-20% rel-err while the pooled head diverges to 170%.
    if "tot_hpwl_sum" in out and "y_tot_hpwl" in g:
        L = L + W["hpwl_sum"] * (out["tot_hpwl_sum"] - g["y_tot_hpwl"]) ** 2
    # per-VN bounding box (xmin,ymin,xmax,ymax) — the well-posed geometry target.
    if "y_vnbox" in g and g["m_vnbox"].any():
        for i in range(4):
            L = L + W["vnbox"] * gnll(out["vn_box"][:, i], g["y_vnbox"][:, i], g["m_vnbox"], nll)
    # global targets: LEVEL + knob DEVIATION, both O(1). The deviation is up-weighted because
    # it IS the thing f_place exists to predict — the level is ~n_cells and nearly free.
    for k in GLOBAL_TARGETS:
        if k == "buf_cnt" and not g["has_bufcnt"]: continue
        L = L + W[k] * gnll(out[f"{k}_lvl"], g[f"y_{k}_lvl"], None, nll)
        # skip the DEVIATION term on degenerate designs: some chips are so small the resizer
        # inserts the same handful of buffers regardless of the knobs (usb_phy has TWO distinct
        # buffer_area values across all 108 flows). There is no knob response to learn there,
        # so the term is pure noise.
        if not g.get(f"deg_{k}", False):
            L = L + W["dev"] * W[k] * gnll(out[f"{k}_dev"], g[f"y_{k}_dev"], None, nll)
    return L

@torch.no_grad()
def evaluate(model, flows):
    """collect predictions vs truth for every target."""
    model.eval()
    NET  = ("net_hpwl", "endpt")                       # per-node (masked) heads: net & endpoint
    GLOB = ("tot_hpwl","buf_area","buf_cnt")           # per-flow scalars (level + deviation)
    DEVK = tuple(k+"_dev" for k in GLOB)               # the KNOB RESPONSE, scored on its own
    DERIV = ("wns","tns")                              # NOT heads — read out from endpt per flow
    LOGK = ("net_hpwl","tot_hpwl","buf_area","buf_cnt")# log targets -> real units via exp
    P = {k: [] for k in NET+GLOB+DEVK+DERIV}
    T = {k: [] for k in P}; sig = {k: [] for k in NET+GLOB}; dz = []
    rank_flows = []            # per-FLOW (pred, true) for the ranking metrics — see below
    geo = {"box_m": [], "box_b": [], "pos_m": [], "pos_b": []}   # GEOMETRY: model vs baseline
    BASE = norm()["vnbox_m"]                     # train-only mean box — the honest baseline
    for f in flows:
        g = load_graph(f, DEV); o = model(g)
        # ---- GEOMETRY, scored against the trivial predictor it must beat ----
        # boxes are die-normalized, so every design is on the same [0,1] scale and one mean box
        # is a fair baseline. If the model cannot beat it, it has learned no geometry.
        mv = g["m_vnbox"]
        if mv.any():
            pb = o["vn_box"][mv,:,0].cpu().numpy(); tb = g["y_vnbox"][mv].cpu().numpy()
            geo["box_m"].append(np.abs(pb-tb).mean(1)); geo["box_b"].append(np.abs(BASE[None,:]-tb).mean(1))
        mp = g["m_pos"]
        if mp.any():
            px, py = o["pos_x"][mp,0].cpu().numpy(), o["pos_y"][mp,0].cpu().numpy()
            tx, ty = g["y_pos_x"][mp].cpu().numpy(), g["y_pos_y"][mp].cpu().numpy()
            geo["pos_m"].append(np.sqrt((px-tx)**2+(py-ty)**2))
            geo["pos_b"].append(np.sqrt((tx-.5)**2+(ty-.5)**2))      # baseline: the die centre
        for k, mk, yk in (("net_hpwl","m_net_hpwl","y_net_hpwl"), ("endpt","m_endpt","y_endpt")):
            m = g[mk]
            P[k].append(o[k][m,0].cpu().numpy()); T[k].append(g[yk][m].cpu().numpy())
            sig[k].append(o[k][m,1].cpu().numpy())
        rank_flows.append((o["net_hpwl"][g["m_net_hpwl"],0].cpu().numpy(),
                           g["y_net_hpwl"][g["m_net_hpwl"]].cpu().numpy()))
        nm = norm()
        for k in GLOB:
            # reconstruct absolute log(target) = level + deviation; also score the DEVIATION
            # head on its own — that is the knob response, the thing f_place exists for.
            lv, dv = o[f"{k}_lvl"][0].item(), o[f"{k}_dev"][0].item()
            P[k].append(np.array([recon(k, lv, dv, nm, g.get(f"w_{k}"))]))
            T[k].append(np.array([g[f"y_{k}"].item()]))          # raw log
            sig[k].append(np.array([o[f"{k}_dev"][1].item()]))
            # DEGENERATE designs: the knobs do not move this target AT ALL (usb_phy has TWO
            # distinct buffer_area values across all 108 flows; simple_spi sits at ONE value in
            # 91 of 108). R2 needs the truth to VARY — scoring a constant target is a division
            # by ~zero, and that single degenerate design dragged buf_area_dev to exactly +0.000.
            # There is no knob response to score, so skip it. (The ABSOLUTE metric for k is still
            # valid on these designs — the level is real — so only the DEV metric is skipped.)
            if not g.get(f"deg_{k}", False):
                P[k+"_dev"].append(np.array([dv]))
                T[k+"_dev"].append(np.array([g[f"y_{k}_dev"].item()]))
        # WNS/TNS READOUT from per-endpoint slack (denorm to raw slack), vs recorded truth.
        # append every flow (NaN if no endpoints) so wns/tns stay aligned with dz.
        ep = g["ep_idx"]
        if len(ep):
            slk = denorm("endpt", o["endpt"][ep,0].cpu().numpy())
            wns_p, tns_p = float(slk.min()), float(slk[slk<0].sum())
        else:
            wns_p = tns_p = np.nan
        P["wns"].append([wns_p]); T["wns"].append([g["wns_true"]])
        P["tns"].append([tns_p]); T["tns"].append([g["tns_true"]])
        dz.append(f.rsplit("-",1)[0])
    res = {}
    for k in NET+GLOB+DEVK+DERIV:
        p, t = np.concatenate([np.asarray(x) for x in P[k]]), np.concatenate([np.asarray(x) for x in T[k]])
        ok = np.isfinite(p) & np.isfinite(t); p, t = p[ok], t[ok]
        if len(t) < 3 or t.std() < 1e-9: continue

        # ---- PRIMARY METRIC: plain absolute error in REAL units. No ratios, no baselines.
        # A pooled R2 is a ratio against variance, and ~99% of the variance in the global
        # targets is merely "how big is this design" — so R2 flattered us badly. |target-pred|
        # cannot lie that way.
        if k in GLOB:                      # p,t are RAW LOG (reconstructed level+dev) -> real units
            rp, rt = np.exp(p), np.exp(t)
            rel = np.abs(rp - rt) / np.maximum(np.abs(rt), 1e-9)
        elif k == "net_hpwl":              # standardized log -> real units
            rp, rt = np.exp(denorm(k, p)), np.exp(denorm(k, t))
            rel = np.abs(rp - rt) / np.maximum(np.abs(rt), 1e-9)
        elif k == "endpt":                 # standardized slack -> ns
            rp, rt = denorm("endpt", p), denorm("endpt", t)
            rel = np.abs(rp - rt)
        elif k in DEVK:                    # the KNOB RESPONSE, in within-design std units
            rp, rt = p, t
            rel = np.abs(rp - rt)
        elif k == "wns":                   # already raw ns — absolute error is the honest one
            rp, rt = p, t
            rel = np.abs(rp - rt)
        else:                              # k == "tns": RELATIVE error is MEANINGLESS here.
            # TNS spans 0 .. -35,000 ns, so |err|/|truth| on a near-zero denominator explodes
            # (we printed 9146% and it meant nothing). Normalise by the design's own TNS scale
            # instead, so the number is comparable across designs and finite.
            rp, rt = p, t
            rel = np.abs(rp - rt) / np.maximum(np.abs(rt).mean(), 1.0)
        d = dict(n=int(len(t)),
                 mae=float(np.mean(np.abs(rp - rt))),      # real units
                 med_ae=float(np.median(np.abs(rp - rt))),
                 p90_ae=float(np.percentile(np.abs(rp - rt), 90)),
                 med_rel=float(np.median(rel)),
                 r2=float(1 - ((t-p)**2).sum()/((t-t.mean())**2).sum()))  # kept, but SECONDARY

        # ---- CALIBRATION, fixed. The old form was mean(sigma)/rmse — an ARITHMETIC mean of
        # sigma over an RMS of residuals. By Jensen that is < 1 even for a PERFECT model
        # (log-sigma spread 1.0 -> a perfect model scores 0.61). Every "overconfident" claim
        # made from it was an artifact. Correct: z = r/sigma, and E[z^2] should be 1.0.
        if k in sig:
            s = np.concatenate(sig[k])[ok]
            sd = np.exp(0.5 * np.clip(s, -5, 5))
            d["calib_z2"] = float(np.mean(((t - p) / np.maximum(sd, 1e-9)) ** 2))  # want 1.0
            d["rms_sigma"] = float(np.sqrt(np.mean(sd ** 2)))
            d["rmse"] = float(np.sqrt(np.mean((t - p) ** 2)))
        res[k] = d

    # ---- WITHIN-DESIGN error: hold design size constant, so what's left IS the knob response —
    # the only thing f_place exists to predict. Reported as R2 (not Pearson r: a design can have
    # r=0.9 with a NEGATIVE within-design R2 if the slope or bias is wrong).
    # NOTE: within_r2 is NOT computed for the GLOB targets. It is measured on the RECONSTRUCTED
    # level+dev, and within one design the level is a CONSTANT — so any level bias tanks it even
    # when the knob response is perfect (we saw buf_cnt within_r2 = -1329 while its DEV head
    # scored +0.335). The DEV heads (DEVK) already measure the knob response cleanly and are the
    # honest number. Keep within_r2 only where there is no level/dev split (wns/tns).
    dd_all = np.array(dz)
    for k in DEVK + DERIV:
        if k not in res: continue
        p = np.concatenate([np.asarray(x) for x in P[k]]); t = np.concatenate([np.asarray(x) for x in T[k]])
        dd = dd_all[:len(p)]
        r2s, rels = [], []
        for dsg in np.unique(dd):
            sel = (dd == dsg) & np.isfinite(p) & np.isfinite(t)
            if sel.sum() < 3 or t[sel].std() < 1e-9: continue
            pp, tt = p[sel], t[sel]
            r2s.append(1 - ((tt-pp)**2).sum()/((tt-tt.mean())**2).sum())
            rels.append(np.median(np.abs(pp - tt)))
        if r2s:
            res[k]["within_r2"] = float(np.mean(r2s))       # THE number: knob-response skill
            res[k]["within_med_ae"] = float(np.mean(rels))
            res[k]["n_designs"] = len(r2s)

    # ---- RANKING metrics for per-net HPWL — the axis the FIELD actually reports.
    # Net2 (Xie et al., ASP-DAC'21) found 725 nets in one design with IDENTICAL local features
    # whose post-placement lengths spanned 1um to 100um. Per-net absolute length is
    # UNDER-DETERMINED from a pre-placement netlist: length is set by global placement pressure.
    # So Net2, MacroRank and Huang (DATE'19) all independently ABANDONED absolute regression and
    # report RANKING instead. Net2 never publishes an MAE/RMSE/MAPE at all — they report
    # top-10%-longest-net ROC-AUC (92.2) and a 20-BIN correlation (0.98, with the top 5% of nets
    # EXCLUDED). The binning cancels per-net error; it measures "do long nets come out longer".
    #
    # This is also what the DOWNSTREAM CONSUMER needs: f_route doesn't want "this net is 47.3um",
    # it wants "THESE are the long, congested nets". That is ranking, and it is achievable.
    # Computed PER FLOW then averaged — ranking nets across designs would just rank by chip size.
    if rank_flows:
        aucs, bin_rs, top_recalls = [], [], []
        for pf, tf in rank_flows:
            if len(tf) < 50 or tf.std() < 1e-9: continue
            # (a) ROC-AUC: can we identify the top-10% LONGEST nets?  (Net2's headline metric)
            thr = np.percentile(tf, 90); lab = (tf >= thr).astype(int)
            if 0 < lab.sum() < len(lab):
                order = np.argsort(pf); rk = np.empty(len(pf)); rk[order] = np.arange(len(pf))
                npos, nneg = lab.sum(), len(lab) - lab.sum()
                aucs.append((rk[lab == 1].sum() - npos*(npos-1)/2) / (npos*nneg))
                # (b) recall@10%: of the truly-longest 10%, how many are in our predicted top 10%?
                pred_top = pf >= np.percentile(pf, 90)
                top_recalls.append(float((pred_top & (lab == 1)).sum() / max(npos, 1)))
            # (c) 20-bin correlation (Net2's other metric) — bin by TRUE length, correlate means
            qs = np.quantile(tf, np.linspace(0, 1, 21))
            bp, bt = [], []
            for i in range(20):
                m_ = (tf >= qs[i]) & (tf <= qs[i+1] if i == 19 else tf < qs[i+1])
                if m_.sum() >= 3: bp.append(pf[m_].mean()); bt.append(tf[m_].mean())
            if len(bt) >= 5 and np.std(bt) > 1e-9 and np.std(bp) > 1e-9:
                bin_rs.append(float(np.corrcoef(bp, bt)[0, 1]))
        if aucs and "net_hpwl" in res:
            res["net_hpwl"]["auc_top10"]    = float(np.mean(aucs))        # Net2: 92.2 (=0.922)
            res["net_hpwl"]["recall_top10"] = float(np.mean(top_recalls))
            res["net_hpwl"]["bin20_r"]      = float(np.mean(bin_rs)) if bin_rs else float("nan")
    # GEOMETRY — reported as model-vs-baseline so the run answers "did it learn geometry at all",
    # with no room for me to read a win into a tie. skill > 0 means it beat the trivial predictor.
    for tag, mk, bk in (("vn_box", "box_m", "box_b"), ("cell_pos", "pos_m", "pos_b")):
        if geo[mk]:
            me = float(np.concatenate(geo[mk]).mean()); be = float(np.concatenate(geo[bk]).mean())
            res[tag] = dict(err=me, baseline=be, skill=float(1 - me / be) if be > 0 else float("nan"))
    return res

def run_fold(fi, test_designs, all_designs):
    # VALIDATION IS A HELD-OUT-DESIGN SPLIT.
    # It used to be a random 10% of FLOWS from the training designs. But ~99% of the variance
    # in the global targets is design identity — which val then SHARED with train by
    # construction. So val could not see cross-design overfitting at all, yet it drove BOTH the
    # LR schedule AND checkpoint selection. There was no held-out-design signal anywhere in the
    # training loop. Now: hold out 2 TRAIN designs as val; the model never sees them in training.
    # val = 2 held-out DESIGNS. NEVER take a SIZE ANCHOR for val — the anchors exist to pin the
    # ends of the training size range, so removing one would re-open the extrapolation hole.
    pool     = [d for d in all_designs if d not in test_designs and d not in SIZE_ANCHORS]
    rngd     = random.Random(SEED + 100 + fi)
    shuf     = pool[:]; rngd.shuffle(shuf)
    val_d    = sorted(shuf[:2])                       # 2 held-out DESIGNS for validation
    train_d  = sorted(shuf[2:] + SIZE_ANCHORS)        # anchors ALWAYS in train
    tr, val, te = flows_of(train_d), flows_of(val_d), flows_of(test_designs)
    rng = random.Random(SEED); rng.shuffle(tr)
    print(f"\n=== fold {fi}: test on {len(test_designs)} designs {test_designs}", flush=True)
    print(f"    train {len(tr)} flows / {len(train_d)} designs {train_d}", flush=True)
    print(f"    val   {len(val)} flows / {len(val_d)} HELD-OUT designs {val_d}", flush=True)
    print(f"    test  {len(te)} flows (never seen)", flush=True)

    # normalization from TRAIN designs only — no val/test/OOD statistics ever touch the model
    set_norm(train_d)

    model = FPlace(d=DIM, K=LAYERS, encoder=ENCODER).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    # Select on validation R², NOT val loss.
    # The val LOSS is dominated by the NLL variance terms and the readout term, both of which
    # are spiky per-flow — it swung -2.7 .. +0.35 epoch-to-epoch while every R² sat perfectly
    # still (hpwl 0.751 unchanged for 50 epochs). Checkpointing on that = selecting on NOISE,
    # and it was restoring an early lucky-dip epoch and throwing away the trained model.
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=10, min_lr=1e-5)   # max: higher R² is better
    best, best_state, patience, best_ep = -1e9, None, 0, 0         # best SCORE (R²), not loss
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
        # leftover partial accumulation batch — clip it too (this step used to skip clipping)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad()
        # VAL = held-out DESIGNS -> this IS a generalization signal now.
        model.eval()
        v = evaluate(model, val)
        # SELECTION SCORE: median RELATIVE error on the targets we care about (lower is better).
        # Plain error, not R2 — R2 is a ratio against variance and ~99% of the global targets'
        # variance is design size, so it flattered us. Negated so "higher is better" for sched.
        SEL = ("net_hpwl", "tot_hpwl", "buf_cnt", "wns")
        errs = [v[k]["med_rel"] for k in SEL if k in v and np.isfinite(v[k]["med_rel"])]
        # + the KNOB RESPONSE (dev-head R2, higher is better) — the thing f_place exists for.
        devs = [v[k+"_dev"]["r2"] for k in GLOBAL_TARGETS
                if k+"_dev" in v and np.isfinite(v[k+"_dev"]["r2"])]
        score = (-float(np.mean(errs)) if errs else -1e9) + (float(np.mean(devs)) if devs else 0.0)
        sched.step(score)
        lr_now = opt.param_groups[0]["lr"]
        # ABSOLUTE error in REAL units (um / um2 / cells / ns) — median |pred - true|, so you can
        # watch the physical error shrink. r_ is the knob-response R2 (the thing being optimized).
        a_ = lambda k: v[k]["med_ae"] if k in v else float("nan")
        r_ = lambda k: v[k].get("r2", float("nan")) if k in v else float("nan")
        print(f"  ep {ep:3d} tr {tot/len(tr):7.3f} | VAL abs-err: "
              f"hpwl {a_('net_hpwl'):6.2f}um tot {a_('tot_hpwl'):9.0f}um bufA {a_('buf_area'):7.1f}um2 "
              f"bufC {a_('buf_cnt'):5.1f}cells endpt {a_('endpt'):.2f}ns wns {a_('wns'):.2f}ns tns {a_('tns'):7.1f}ns "
              f"| knob-R2 tot {r_('tot_hpwl_dev'):+.2f} bufA {r_('buf_area_dev'):+.2f} bufC {r_('buf_cnt_dev'):+.2f} "
              f"| GEO skill box {v.get('vn_box',{}).get('skill',float('nan')):+.3f} "
              f"pos {v.get('cell_pos',{}).get('skill',float('nan')):+.3f} "
              f"| lr {lr_now:.1e} ({time.time()-t0:.0f}s)", flush=True)
        if score > best + 1e-4:
            best, best_state, patience = score, {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}, 0
            best_ep = ep
        else:
            patience += 1
            if PATIENCE and patience >= PATIENCE:      # disabled by default (PATIENCE=0)
                print(f"  early stop (no val improvement in {PATIENCE} epochs)"); break
    print(f"  done — best val score {-best:.4f} (mean med-rel-err) @ epoch {best_ep}; "
          f"restoring that checkpoint", flush=True)
    if best_state: model.load_state_dict(best_state)
    torch.save(model.state_dict(), f"{OUT}/fold{fi}.pt")
    res = evaluate(model, te)
    print_metrics(res, f"TEST (unseen designs: {test_designs})")
    return dict(fold=fi, test_designs=test_designs, val_designs=val_d,
                n_train=len(tr), n_test=len(te), metrics=res)

UNITS = dict(net_hpwl="um", tot_hpwl="um", buf_area="um2", buf_cnt="cells",
             endpt="ns", wns="ns", tns="ns")

def print_metrics(res, title):
    """Plain error in real units first; R2 last and explicitly labelled as the weak metric."""
    print(f"\n  {title}", flush=True)
    print(f"  {'target':9} {'med|err|':>10} {'p90|err|':>10} {'med rel':>8} "
          f"{'within-R²':>10} {'calib z²':>9} {'pooled R²':>10}", flush=True)
    for k, v in res.items():
        u = UNITS.get(k, "")
        wr = f"{v['within_r2']:+.3f}" if "within_r2" in v else "     -"
        cz = f"{v['calib_z2']:.2f}" if "calib_z2" in v else "    -"
        print(f"  {k:9} {v['med_ae']:9.3f}{u:>1} {v['p90_ae']:9.3f}{u:>1} "
              f"{v['med_rel']*100:7.1f}% {wr:>10} {cz:>9} {v['r2']:+10.3f}", flush=True)
    if "net_hpwl" in res and "auc_top10" in res["net_hpwl"]:
        n = res["net_hpwl"]
        print(f"\n  RANKING (per-net HPWL) — the axis the field actually reports:")
        print(f"      top-10% longest-net AUC  {n['auc_top10']:.3f}   "
              f"(Net2 ASP-DAC'21, leave-design-out: 0.922)")
        print(f"      recall@10%               {n['recall_top10']:.3f}   "
              f"(of the truly-longest 10%, how many we rank in our top 10%)")
        print(f"      20-bin correlation       {n['bin20_r']:.3f}   (Net2: 0.98, top 5% excluded)")
        print("      NOTE: Net2 (ASP-DAC'21) REGRESSES net length (label = post-placement bbox")
        print("      HPWL) — it does NOT use a ranking loss. What it never publishes is a um-level")
        print("      absolute error: it reports only binned correlation (>0.98) and top-10% AUC")
        print("      (92.5), the standard protocol in net-length works. So the field evaluates")
        print("      this ORDINALLY, which is why we do too — not because regression is")
        print("      impossible. Ranking is chosen for DECISION-THEORETIC reasons: MacroRank")
        print("      (ASP-DAC'23) picks a rank loss because 'the relative relationship ... is")
        print("      noteworthy instead of the absolute value', and shows the dissociation —")
        print("      EHNN has the best MRE but near-zero Kendall tau until the loss is swapped.")
        print("      (Net2's 725 nets with identical local features spanning 1um..100um is real,")
        print("      but they use it to argue for a GLOBAL RECEPTIVE FIELD, not to drop")
        print("      regression.) f_route needs the RANKING — which nets are long/congested.")
    print("\n   (within-R² = knob-response with design size held constant — THE number."
          "  calib z² -> 1.0 = honest sigma.\n"
          "    pooled R² is ~99% design-identity for the global targets — do not trust it.)",
          flush=True)

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
        print_metrics(r, f"OOD — {os.path.basename(ck)}")
    print("\n=== OOD (mean across folds) ===")
    _agg_print(allres)
    json.dump(allres, open(f"{OUT}/ood_results.json","w"), indent=2)
    print(f"\nwrote {OUT}/ood_results.json")

TARGET_ORDER = ("net_hpwl","endpt","tot_hpwl","buf_area","buf_cnt","wns","tns")

def _agg_print(metric_dicts):
    print(f"  {'target':9} {'med|err|':>12} {'med rel':>12} {'within-R²':>14} {'pooled R²':>14}")
    for k in TARGET_ORDER:
        ms = [m[k] for m in metric_dicts if k in m]
        if not ms: continue
        f = lambda key: np.array([m[key] for m in ms if key in m and np.isfinite(m[key])])
        ae, rl, wr, r2 = f("med_ae"), f("med_rel"), f("within_r2"), f("r2")
        u = UNITS.get(k, "")
        s_ae = f"{ae.mean():8.3f}{u:<4}" if len(ae) else " " * 12
        s_rl = f"{rl.mean()*100:9.1f}%  " if len(rl) else " " * 12
        s_wr = f"{wr.mean():+8.3f}±{wr.std():.2f}" if len(wr) else " " * 14
        s_r2 = f"{r2.mean():+8.3f}±{r2.std():.2f}" if len(r2) else " " * 14
        print(f"  {k:9} {s_ae} {s_rl} {s_wr} {s_r2}")
    print("   (within-R² = knob response, size held constant — THE number. "
          "pooled R² is ~99% design-identity.)")

def aggregate():
    """Combine per-fold results (written by parallel array tasks) into one CV number."""
    out = []
    for f in sorted(glob.glob(f"{OUT}/results_fold*.json")):
        out += json.load(open(f))
    if not out:
        print(f"no {OUT}/results_fold*.json yet — folds still running?"); return
    print(f"\n=== AGGREGATE across {len(out)} folds (unseen designs) — {OUT} ===")
    _agg_print([f["metrics"] for f in out])
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
