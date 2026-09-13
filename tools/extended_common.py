"""Shared portable post hoc analysis utilities; no original-model mutation."""
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import hashlib,json,csv,sys
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1];D=P/'evidence/extended';O=P/'replayed/extended';O.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(P/'tools/support_lib'))
from run_support_study import load,anchor,subset,macro,evaluate
from sequential_robust_readout_v22 import REGIMES,SUBSPACES,feature_views,fit_ridge,ridge_predict
from canonical_physics_metrics_v7 import score_physics,_actions,_parameters
def write(n,v): (O/n).write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8')
def csvout(n,rows):
    with (O/n).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
def folds(v,k):
    fs=sorted(set(v['family_ids']),key=lambda x:hashlib.sha256(x.encode()).hexdigest());m={f:i%k for i,f in enumerate(fs)}
    return np.array([m[f] for f in v['family_ids']])
def data():return load(P/'evidence/support/development.npz'),load(P/'evidence/support/confirmation.npz')
def action(v,means):
    truth=_actions(_parameters(v['truth'])[0],v['protocol']);res=[];ep=[]
    for r in REGIMES:
        err=np.abs(_actions(_parameters(means[r])[0],v['protocol'])-truth).mean(1)
        res.append(macro(err,v));ep.append(err)
    return np.mean(res,0),ep
def contrast(v,a,b):
    delta=a-b;n=len(delta)
    draw=load(P/'evidence/results/com_observation_v23/confirmation_v1/FAMILY_ACTION_ERRORS.npz')['bootstrap_draw'] if n==72 else np.random.default_rng(20260913).integers(n,size=(20000,n))
    boot=delta[draw].mean(1)*1000;lo,hi=np.quantile(boot,[.025,.975])
    return {'difference_mm':float(delta.mean()*1000),'low95_mm':float(lo),'high95_mm':float(hi),'one_sided_upper_mm':float(np.quantile(boot,.95)),'relative_percent':float((b.mean()-a.mean())/b.mean()*100),'wins':int((delta<0).sum()),'n':n}
def output(v,means):
    out={}
    for r in REGIMES:
        a,c=anchor(v,r);d=means[r]-a;cov=c.copy();cov[:,np.arange(3),np.arange(3)]+=(.1*d)**2
        out[r]={'mean':means[r],'covariance':cov}
    return out
def predict_variant(train,test,view='full',penalty=10,shrink=.25,crossfit=False,numeric_penalty=1.,dedup=False,fallback=True,diagnostics=None):
    nx,vx=feature_views(train['features'],24);nt,vt=feature_views(test['features'],24)
    if dedup:
        nx=np.concatenate((train['features'][:,:,:9].reshape(len(nx),72),train['features'][:,0,9:24]),1)
        nt=np.concatenate((test['features'][:,:,:9].reshape(len(nt),72),test['features'][:,0,9:24]),1)
    sel=slice(None) if view=='full' else slice(0,32) if view=='pooled' else slice(32,None)
    vx=vx[:,sel];vt=vt[:,sel];out={};inner=folds(train,5)
    for r in REGIMES:
        a,c=anchor(train,r);at,ct=anchor(test,r);s=np.sqrt(np.diagonal(c,axis1=1,axis2=2));st=np.sqrt(np.diagonal(ct,axis1=1,axis2=2));target=(train['truth']-a)/s;dz=np.zeros_like(at)
        for p,cols in SUBSPACES.items():
            tr=np.flatnonzero(train['protocol']==p);te=np.flatnonzero(test['protocol']==p)
            nm=fit_ridge(nx[tr],target[tr][:,cols],numeric_penalty)
            rawtrain=ridge_predict(nm,nx[tr]);ntrain=np.clip(rawtrain,-2,2);rawtest=ridge_predict(nm,nt[te]);n=np.clip(rawtest,-2,2)
            if crossfit:
                for f in range(5):
                    aidx=tr[inner[tr]!=f];bidx=np.flatnonzero(inner[tr]==f)
                    model=fit_ridge(nx[aidx],target[aidx][:,cols],numeric_penalty)
                    ntrain[bidx]=np.clip(ridge_predict(model,nx[tr[bidx]]),-2,2)
            residual=(train['truth'][tr][:,cols]-(a[tr][:,cols]+ntrain*s[tr][:,cols]))/s[tr][:,cols]
            vm=fit_ridge(vx[tr],residual,penalty);rawv=ridge_predict(vm,vt[te]);vis=np.clip(rawv,-1,1)
            dz[np.ix_(te,cols)]=n+shrink*vis
            if diagnostics is not None and p=='bounce':
                from scipy.stats import pearsonr
                true=(test['truth'][te][:,cols]-at[te][:,cols])/st[te][:,cols]
                diagnostics.append({'regime':r,'n':len(te),'residual_correlation':float(pearsonr((n+shrink*vis).ravel(),true.ravel()).statistic),
                 'training_numeric_clip_rate':float((np.abs(rawtrain)>2).mean()),'prediction_numeric_clip_rate':float((np.abs(rawtest)>2).mean()),'prediction_visual_clip_rate':float((np.abs(rawv)>1).mean()),
                 'anchor_sd_median':float(np.median(st[te,2])),'anchor_sd_q25':float(np.quantile(st[te,2],.25)),'anchor_sd_q75':float(np.quantile(st[te,2],.75)),
                 'target_sd':float(np.std(true))})
        delta=st*dz*test['active']
        if fallback and r=='generalized_translation':delta[test['protocol']=='bounce']=0
        out[r]=at+delta
    return out
def cv_variant(v,**kwargs):
    f=folds(v,6);out={r:np.empty_like(v['truth']) for r in REGIMES}
    for fold in range(6):
        tr=np.flatnonzero(f!=fold);te=np.flatnonzero(f==fold);pred=predict_variant(subset(v,tr),subset(v,te),**kwargs)
        for r in REGIMES:out[r][te]=pred[r]
    return out
