#!/usr/bin/env python3
"""
VPN / MuZero-model-half PROBE — does a value-equivalent latent ROLLOUT predict final PPA
better than DIRECT prediction, reusing f_place's OWN encoder + states (no hand-rolled loader)?

Model half of MuZero (representation h, dynamics g, reward heads) WITHOUT policy/MCTS = a Value
Prediction Network (Oh 2017) / value-equivalent model (Grimm 2020). The latent is NEVER
reconstructed against the true stage state (that is our wall). It is grounded ONLY through
per-stage REWARD heads that read the real per-stage PPA scalars OFF the latent.

REUSE (per the frame/state infra, not a new data path):
  * h        = fplace.FPlace.encode  (DE-HNN + VN + SignNet PE + Net2/elec net feats), the SAME
               encoder f_place ships. z0 = pool(encode(graph, knobs=0)) -> DESIGN identity, and
               since the graph is byte-identical across a design's 108 configs, it is encoded
               ONCE per design and shared.
  * knobs    = f_place's 6-dim engineered action [clk,util,AR,fp_wns,fp_tns,crit], rebuilt from
               meta+norm exactly as load_graph does (vectorized, no per-flow load).
  * rewards  = the per-stage PPA scalars from cache/meta.parquet (the seam's states are these).

Three arms, SAME encoder + budget + epochs, only the head/rollout differs:
  DIRECT   : z0 -> MLP([z0,k]) -> route metrics           (predict final PPA directly)
  FLAT_MT  : z0 -> MLP([z0,k]) -> ALL stage metrics       (multitask, no rollout = the honest ctrl)
  ROLLOUT  : z1=g(z0,k); z2=g(z1,k); z3=g(z2,k); per-stage reward heads read fp/place/cts/route

Metric = WITHIN-DESIGN Spearman of predicted vs true rt_wl, leave-designs-out.

Run:  PYTHONPATH=src ./venv/bin/python src/muzero_probe.py
Env:  EPOCHS(250) D(128) LR(1e-3) SEED(0) SCALE(0) GHALF(0) TEST("aes_core,i2c,sasc,ethernet")
"""
import os, glob, json, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import spearmanr
import fplace

EPOCHS = int(os.environ.get("EPOCHS", 250)); D = int(os.environ.get("D", 128))
LR = float(os.environ.get("LR", 1e-3)); SEED = int(os.environ.get("SEED", 0))
SCALE = bool(int(os.environ.get("SCALE", 0)))    # MuZero min-max latent scaling (RL trick; off)
GHALF = bool(int(os.environ.get("GHALF", 0)))    # MuZero recurrent grad halving (RL trick; off)
WFINAL = float(os.environ.get("WFINAL", 1.0))    # up-weight route-stage metrics (fix rt_wl dilution)
TEST = os.environ.get("TEST", "aes_core,i2c,sasc,ethernet").split(",")
torch.manual_seed(SEED); np.random.seed(SEED)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- per-stage reward targets (the seam's states, as scalars) + transform ----
LOGT  = {"total_hpwl","buffer_area","buffer_count","cts_power","cts_buffers","rt_wl","rt_power"}
STAGES = [["fp_wns","fp_tns"],
          ["total_hpwl","buffer_area","buffer_count","wns","tns"],
          ["cts_wns","cts_tns","cts_power","cts_buffers"],
          ["rt_wl","rt_wns","rt_tns","rt_power"]]
ALLT  = [t for s in STAGES for t in s]; RT_WL = ALLT.index("rt_wl")
def ttf(name, v):
    v = np.asarray(v, np.float64)
    return np.log(np.maximum(v,1e-6)) if name in LOGT else np.sign(v)*np.log1p(np.abs(v))

# ---------------------------------------------------------------- data (REUSED infra)
meta = fplace.meta().copy(); meta["dsg"] = meta.index.str.replace(r"-\d+$","",regex=True)
DESIGNS = sorted(meta.dsg.unique()); TRAIN_D = [d for d in DESIGNS if d not in TEST]
fplace.set_norm(TRAIN_D); nm = fplace.norm()

# f_place's 6-dim knob vector, rebuilt vectorized from meta+norm (== load_graph lines 719-740)
def build_knobs():
    slog = lambda v: np.sign(v)*np.log1p(np.abs(v))
    kz = lambda c: (meta[c].values.astype(np.float64)-float(nm[f"k_{c}_m"]))/float(nm[f"k_{c}_s"])
    crit = np.maximum(meta.clock_period.values - meta.fp_wns.values, 1e-3)
    K = np.stack([kz("clock_period"), kz("utilization"), kz("aspect_ratio"),
                  (slog(meta.fp_wns.values)-float(nm["a_fp_wns_m"]))/float(nm["a_fp_wns_s"]),
                  (slog(meta.fp_tns.values)-float(nm["a_fp_tns_m"]))/float(nm["a_fp_tns_s"]),
                  (np.log(crit)-float(nm["k_crit_m"]))/float(nm["k_crit_s"])], 1).astype(np.float32)
    return pd.DataFrame(K, index=meta.index)
KZ = build_knobs(); KDIM = KZ.shape[1]

def design_graph(dsg, dev="cpu"):
    """encode-ready graph for a design (one flow; identical across configs), knobs ZEROED."""
    fid = os.path.basename(sorted(glob.glob(f"{fplace.CACHE}/{dsg}-*.npz"))[0])[:-4]
    g = fplace.load_graph(fid, device=dev)
    g["knobs"] = torch.zeros(KDIM, device=dev)          # z0 = design identity, action-free
    return g

def design_batch(dsg, dev="cpu"):
    sub = meta[meta.dsg==dsg]
    k = torch.tensor(KZ.loc[sub.index].values, dtype=torch.float, device=dev)
    Y = np.full((len(sub), len(ALLT)), np.nan, np.float32)
    for j,t in enumerate(ALLT):
        y = ttf(t, sub[t].values)
        mu, sd = ttf(t, meta.loc[meta.dsg.isin(TRAIN_D), t].values), None
        Y[:,j] = (y - np.nanmean(mu)) / (np.nanstd(mu)+1e-6)
    keep = np.isfinite(Y[:, RT_WL])
    return k[torch.tensor(keep, device=dev)], torch.tensor(Y[keep], device=dev)

print(f"[probe] designs={len(DESIGNS)} train={len(TRAIN_D)} test={TEST} KDIM={KDIM} "
      f"D={D} EPOCHS={EPOCHS} SCALE={SCALE} GHALF={GHALF}", flush=True)

# ---------------------------------------------------------------- model
def scale01(z):
    lo=z.min(-1,keepdim=True).values; hi=z.max(-1,keepdim=True).values; return (z-lo)/(hi-lo+1e-5)
def head(d,o): return nn.Sequential(nn.Linear(d,d), nn.LeakyReLU(), nn.Linear(d,o))

class Enc(nn.Module):
    """reuse f_place's encoder; pool cells+nets -> project to D."""
    def __init__(s, d):
        super().__init__(); s.f = fplace.FPlace(d=64, K=4)
        s.proj = nn.Sequential(nn.Linear(256,d), nn.LeakyReLU(), nn.Linear(d,d))
    def forward(s, g):
        h, hn, _ = s.f.encode(g)
        z = s.proj(torch.cat([h.mean(0),h.max(0).values,hn.mean(0),hn.max(0).values]))
        return scale01(z) if SCALE else z

class Dyn(nn.Module):
    def __init__(s,d): super().__init__(); s.ke=nn.Linear(KDIM,d); s.g=head(2*d,d)
    def forward(s,z,k):
        z2 = z + s.g(torch.cat([z, s.ke(k)],1))
        if GHALF and z2.requires_grad: z2.register_hook(lambda gr: gr*0.5)
        return scale01(z2) if SCALE else z2

class Model(nn.Module):
    def __init__(s, arm, d):
        super().__init__(); s.arm=arm; s.enc=Enc(d)
        if arm=="rollout":
            s.dyn=Dyn(d); s.rh=nn.ModuleList([head(d,len(st)) for st in STAGES])
        elif arm=="flat_mt": s.h=head(d+KDIM, len(ALLT))
        else:                s.h=head(d+KDIM, len(STAGES[-1]))
    def forward(s, g, k):
        z0 = s.enc(g).unsqueeze(0).expand(k.size(0),-1)      # shared design latent
        if s.arm=="rollout":
            outs=[s.rh[0](z0)]; z=z0
            for i in range(1,4): z=s.dyn(z,k); outs.append(s.rh[i](z))
            return torch.cat(outs,1)
        return s.h(torch.cat([z0,k],1))

COLS = {"rollout":(list(range(len(ALLT))),RT_WL), "flat_mt":(list(range(len(ALLT))),RT_WL),
        "direct":([ALLT.index(t) for t in STAGES[-1]], STAGES[-1].index("rt_wl"))}

def run(arm):
    m = Model(arm, D).to(DEV); opt = torch.optim.Adam(m.parameters(), lr=LR)
    Gd = {d: design_graph(d, DEV) for d in DESIGNS}
    B  = {d: design_batch(d, DEV) for d in DESIGNS}
    sup, rt = COLS[arm]
    route_cols = {ALLT.index(t) for t in STAGES[-1]}                 # up-weight final stage
    w = torch.tensor([WFINAL if c in route_cols else 1.0 for c in sup], device=DEV)
    for ep in range(EPOCHS):
        m.train(); tot=0.
        for d in np.random.permutation(TRAIN_D):
            k,Y = B[d]; y = Y[:,sup]; msk=torch.isfinite(y)
            opt.zero_grad(); pred = m(Gd[d], k)
            se = (pred - torch.nan_to_num(y))**2 * w                 # weighted MSE, nan-masked
            loss = se[msk].mean(); loss.backward(); opt.step()
            tot += loss.item()
        if ep%50==0 or ep==EPOCHS-1: print(f"  [{arm}] ep{ep:3d} loss {tot/len(TRAIN_D):.4f}", flush=True)
    m.eval()
    def sp(ds):
        o={}
        with torch.no_grad():
            for d in ds:
                k,Y=B[d]; pred=m(Gd[d],k)[:,rt].cpu().numpy(); true=Y[:,RT_WL].cpu().numpy()
                ok=np.isfinite(true); o[d]=spearmanr(pred[ok],true[ok]).correlation
        return o
    return sp(TRAIN_D), sp(TEST)

if __name__=="__main__":
    res={a:run(a) for a in ["direct","flat_mt","rollout"]}
    print("\n============ WITHIN-DESIGN SPEARMAN on rt_wl (leave-designs-out) ============")
    print(f"{'arm':9} | {'TRAIN':>7} | {'TEST':>7} | per-test-design")
    for a in ["direct","flat_mt","rollout"]:
        tr,te=res[a]
        print(f"{a:9} | {np.nanmean(list(tr.values())):7.3f} | {np.nanmean(list(te.values())):7.3f} | "
              + "  ".join(f"{d}:{te[d]:+.3f}" for d in TEST))
    outp = os.environ.get("RESULT", f"{fplace.ROOT}/muzero_probe_result.json")
    json.dump({"test_designs":TEST,"seed":SEED,"epochs":EPOCHS,"wfinal":WFINAL,
               **{a:{"train":res[a][0],"test":res[a][1]} for a in res}},
              open(outp,"w"), indent=2)
    print(f"-> saved {outp}")
