#!/usr/bin/env python3
"""Train f_route — routing state from placement netlist. Same folds/discipline as f_place/f_cts."""
import os, sys, glob, json, time, random
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import froute
from froute import FRoute, load_graph, set_norm, recon, RT_GLOBAL, RT_TIMING
from fplace import gnll, norm
import train_fplace as TF

E = os.environ.get
DEV = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS=int(E("EPOCHS",200)); LR=float(E("LR",1e-3)); DIM=int(E("DIM",64)); LAYERS=int(E("LAYERS",4))
ACCUM=int(E("ACCUM",8)); SEED=int(E("SEED",0)); ENCODER=E("ENCODER","dehnn")
W_NET=float(E("W_NET",5)); W_DEV=float(E("W_DEV",3)); OUT=E("OUT","runs/froute")
W_RT_SUM=float(E("W_RT_SUM",3))   # calibrates per-net routed len to the true total
ALL=RT_GLOBAL+RT_TIMING
torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED); os.makedirs(OUT,exist_ok=True)

def wloss(o,g):
    L = W_NET*gnll(o["rt_len"], g["y_rt_len"], g["m_rt_len"])          # per-net routed length (dense)
    # ANALYTIC COMPOSITION: supervise rt_wl = SUM_net routed_len (RT_COMPOSE=sum).
    # Plain MSE in log space — its job is to CALIBRATE the per-net predictions in
    # ABSOLUTE terms so their sum reproduces the true total. Good ranking alone does not.
    if "rt_wl_sum" in o and np.isfinite(g.get("y_rt_wl", np.nan)):
        L = L + W_RT_SUM*(o["rt_wl_sum"] - float(g["y_rt_wl"]))**2
    for k in ALL:
        if g[f"deg_{k}"]: continue
        L=L+gnll(o[f"{k}_lvl"],g[f"y_{k}_lvl"])+W_DEV*gnll(o[f"{k}_dev"],g[f"y_{k}_dev"])
    return L

@torch.no_grad()
def evaluate(model, flows):
    model.eval(); nm=norm()
    # per-net routed length: ranking (AUC top-10% longest) + rel-err, like f_place net_hpwl
    rank=[]; P={k:[] for k in ALL}; T={k:[] for k in ALL}; Pd={k:[] for k in ALL}; Td={k:[] for k in ALL}
    for f in flows:
        g=load_graph(f,DEV); o=model(g); mk=g["m_rt_len"]
        if mk.sum()>=50:
            rank.append((o["rt_len"][mk,0].cpu().numpy(), g["y_rt_len"][mk].cpu().numpy()))
        for k in ALL:
            lv,dv=o[f"{k}_lvl"][0].item(),o[f"{k}_dev"][0].item()
            P[k].append(recon(k,lv,dv,nm,g.get(f"w_{k}"))); T[k].append(g[f"y_{k}"])
            if not g[f"deg_{k}"]: Pd[k].append(dv); Td[k].append(g[f"y_{k}_dev"].item())
    res={}
    for k in ALL:
        p,t=np.array(P[k]),np.array(T[k]); ok=np.isfinite(p)&np.isfinite(t)
        if ok.sum()<3: continue
        d=dict(med_rel=float(np.median(np.abs(np.expm1(p[ok]-t[ok])))))
        pd_,td_=np.array(Pd[k]),np.array(Td[k])
        if len(td_)>=3 and td_.std()>1e-9: d["knob_r2"]=float(1-((td_-pd_)**2).sum()/((td_-td_.mean())**2).sum())
        res[k]=d
    # routed-length ranking AUC (per flow, averaged)
    aucs=[]
    for pf,tf in rank:
        if tf.std()<1e-9: continue
        lab=(tf>=np.percentile(tf,90)).astype(int)
        if 0<lab.sum()<len(lab):
            order=np.argsort(pf); rk=np.empty(len(pf)); rk[order]=np.arange(len(pf))
            npos=lab.sum(); aucs.append((rk[lab==1].sum()-npos*(npos-1)/2)/(npos*(len(lab)-npos)))
    if aucs: res["rt_len"]={"auc_top10":float(np.mean(aucs))}
    return res

def run_fold(fi, test_designs, dev):
    pool=[d for d in dev if d not in test_designs and d not in TF.SIZE_ANCHORS]
    rngd=random.Random(SEED+100+fi); shf=pool[:]; rngd.shuffle(shf)
    val_d,train_d=sorted(shf[:2]),sorted(shf[2:]+TF.SIZE_ANCHORS)
    tr,val,te=TF.flows_of(train_d),TF.flows_of(val_d),TF.flows_of(test_designs)
    rng=random.Random(SEED); rng.shuffle(tr)
    print(f"\n=== fold {fi}: test {test_designs} | train {len(train_d)} designs ===",flush=True)
    set_norm(train_d)
    model=FRoute(d=DIM,K=LAYERS,encoder=ENCODER).to(DEV); opt=torch.optim.Adam(model.parameters(),lr=LR)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="max",factor=0.5,patience=10,min_lr=1e-5)
    best,best_state=-1e9,None
    for ep in range(EPOCHS):
        model.train(); rng.shuffle(tr); t0=time.time(); tot=0.0; opt.zero_grad()
        for i,f in enumerate(tr):
            l=wloss(model(load_graph(f,DEV)),load_graph(f,DEV))/ACCUM; l.backward(); tot+=l.item()*ACCUM
            if (i+1)%ACCUM==0: torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); opt.zero_grad()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); opt.zero_grad()
        v=evaluate(model,val)
        score=float(np.mean([v[k]["knob_r2"] for k in ALL if k in v and "knob_r2" in v[k]] or [-9]))
        sched.step(score)
        if score>best+1e-4: best=score; best_state={k:x.detach().cpu().clone() for k,x in model.state_dict().items()}
        g_=lambda k,s: v[k].get(s,float('nan')) if k in v else float('nan')
        print(f"  ep {ep:3d} tr {tot/len(tr):7.3f} | VAL routed-WL knob-R² {g_('rt_wl','knob_r2'):+.3f} "
              f"pow {g_('rt_power','knob_r2'):+.3f} | net-len AUC {g_('rt_len','auc_top10'):.3f} "
              f"rel {g_('rt_len','med_rel')*100:.0f}% | lr {opt.param_groups[0]['lr']:.1e} ({time.time()-t0:.0f}s)",flush=True)
    if best_state: model.load_state_dict(best_state)
    torch.save(model.state_dict(),f"{OUT}/fold{fi}.pt")
    res=evaluate(model,te)
    print("  TEST (unseen):",flush=True)
    for k in ("rt_len",)+ALL:
        if k in res: print(f"      {k:9} {res[k]}",flush=True)
    return dict(fold=fi,test_designs=test_designs,metrics=res)

if __name__=="__main__":
    dev,folds=TF.make_folds(); which=E("FOLD","all")
    sel=range(len(folds)) if which=="all" else [int(which)]
    print(f"device={DEV} — f_route. targets: routed-net-length + {ALL}")
    out=[run_fold(i,folds[i],dev) for i in sel]
    tag=which if which!="all" else "all"
    json.dump(out,open(f"{OUT}/results_fold{tag}.json","w"),indent=2)
    print(f"\nwrote {OUT}/results_fold{tag}.json")
