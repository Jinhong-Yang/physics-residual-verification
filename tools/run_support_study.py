"""Reproduce all post hoc support experiments from bundled cached features.

The original model and primary results are never replaced or re-selected.
CPU only; Python, NumPy and SciPy. No image extraction or new sample generation.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
from scipy.stats import chi2

P=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(P/'tools/support_lib'))
from sequential_robust_readout_v22 import REGIMES,SUBSPACES,feature_views,fit_ridge,ridge_predict,fit_checkpoint,predict
from canonical_physics_metrics_v7 import score_physics
D=P/'evidence/support'; O=P/'replayed/support'

def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def write(name,value): (O/name).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
def csvout(name,rows):
    with (O/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
def anchor(v,r):return v[f'huber_1.345_rgb__{r}__mean'],v[f'huber_1.345_rgb__{r}__covariance']
def macro(values,v):
    return np.array([np.mean([values[(v['family_ids']==f)&(v['protocol']==p)].mean() for p in SUBSPACES]) for f in sorted(set(v['family_ids']))])

def fit_predict(train,test,variant):
    nx,vx=feature_views(train['features'],24);nt,vt=feature_views(test['features'],24)
    if variant=='pooled_only':vx[:,32:]=0;vt[:,32:]=0
    if variant=='motion_only':vx[:,:32]=0;vt[:,:32]=0
    clipping=variant!='no_clipping'; out={}
    for r in REGIMES:
        a,c=anchor(train,r);at,ct=anchor(test,r)
        s=np.sqrt(np.diagonal(c,axis1=1,axis2=2));st=np.sqrt(np.diagonal(ct,axis1=1,axis2=2))
        target=(train['truth']-a)/s;dz=np.zeros_like(at)
        for p,cols in SUBSPACES.items():
            tr=np.flatnonzero(train['protocol']==p);te=np.flatnonzero(test['protocol']==p)
            model=fit_ridge(nx[tr],target[tr][:,cols],1.)
            ntrain=ridge_predict(model,nx[tr]);n=ridge_predict(model,nt[te])
            if clipping:ntrain=np.clip(ntrain,-2,2);n=np.clip(n,-2,2)
            z=n.copy()
            if variant!='numeric':
                # Match production arithmetic, including physical-space subtraction.
                after=a[tr][:,cols]+ntrain*s[tr][:,cols]
                residual=(train['truth'][tr][:,cols]-after)/s[tr][:,cols]
                if variant=='direct_visual_target':residual=target[tr][:,cols]
                vm=fit_ridge(vx[tr],residual,10.)
                visual=ridge_predict(vm,vt[te])
                if clipping:visual=np.clip(visual,-1,1)
                z=z+.25*visual
            dz[np.ix_(te,cols)]=z
        delta=st*dz*test['active']
        if r=='generalized_translation' and variant!='no_bounce_fallback':delta[test['protocol']=='bounce']=0
        covariance=ct.copy();covariance[:,np.arange(3),np.arange(3)]+=(.1*delta)**2
        out[r]={'mean':at+delta,'covariance':covariance}
    return out

def subset(v,ix):return {k:a[ix] for k,a in v.items() if isinstance(a,np.ndarray) and a.ndim and a.shape[0]==len(v['ids'])}

def evaluate(v,out):
    metric=[];loss=[];coverage=[]
    for r in REGIMES:
        x=out[r];s=score_physics(x['mean'],x['covariance'],v['truth'],v['active'],v['family_ids'],v['protocol'])
        g=s['metrics']['active_subspace_gaussian']['family_macro']
        width=np.empty(len(v['ids']));m=s['per_episode']['active_subspace_gaussian']['mahalanobis_squared']
        levels={q:np.empty(len(v['ids'])) for q in (.5,.6,.7,.8,.9,.95,.99)}
        for p,cols in SUBSPACES.items():
            ix=v['protocol']==p
            diagonal=np.diagonal(x['covariance'][ix][:,cols][:,:,cols],axis1=1,axis2=2)
            width[ix]=(2*np.sqrt(chi2.ppf(.9,len(cols))*diagonal)).mean(1)
            for q,arr in levels.items():arr[ix]=m[ix]<=chi2.ppf(q,len(cols))
        metric.append({'action_mm':s['metrics']['action']['family_macro_mae_m']*1000,
                       'nll':g['joint_nll_per_dimension'],'crps':g['mean_marginal_crps'],
                       'coverage90':g['joint90_coverage'],'projected_diameter':macro(width,v).mean()})
        loss.append(macro(s['per_episode']['action_error_m'],v))
        coverage.append({str(q):float(macro(arr,v).mean()) for q,arr in levels.items()})
    return ({k:float(np.mean([a[k] for a in metric])) for k in metric[0]},
            np.mean(loss,axis=0),{k:float(np.mean([a[k] for a in coverage])) for k in coverage[0]})

def main():
    started=time.perf_counter();O.mkdir(parents=True,exist_ok=True)
    plan=json.loads((D/'ANALYSIS_PLAN.json').read_text());dev=load(D/'development.npz');conf=load(D/'confirmation.npz')
    assert set(dev['ids']).isdisjoint(conf['ids']) and set(dev['family_ids']).isdisjoint(conf['family_ids'])
    # Verify a newly fitted full model against the existing locked checkpoint.
    anch={r:dict(zip(('mean','covariance'),anchor(dev,r))) for r in REGIMES}
    fitted=fit_checkpoint(dev['features'],24,dev['truth'],dev['active'],dev['protocol'],anch)
    locked=load(D/'locked_checkpoint.npz')
    for k in fitted:
        if fitted[k].dtype.kind in 'fiu':np.testing.assert_allclose(fitted[k],locked[k],rtol=0,atol=1e-12)
        else:assert np.array_equal(fitted[k],locked[k])
    families=sorted(set(dev['family_ids']),key=lambda f:hashlib.sha256(f.encode()).hexdigest())
    mapping={f:i%6 for i,f in enumerate(families)};fold=np.array([mapping[f] for f in dev['family_ids']])
    results=[];family_archive={'development_families':np.array(sorted(set(dev['family_ids']))),'confirmation_families':np.array(sorted(set(conf['family_ids']))),'development_folds':fold}
    reference=load(P/'evidence/results/com_observation_v23/confirmation_v1/FAMILY_ACTION_ERRORS.npz')
    original=json.loads((P/'evidence/results/com_observation_v23/confirmation_v1/RESULT.json').read_text())
    for variant in plan['variants']:
        oof={r:{'mean':np.empty((len(dev['ids']),3)),'covariance':np.empty((len(dev['ids']),3,3))} for r in REGIMES}
        for f in range(6):
            tr=np.flatnonzero(fold!=f);te=np.flatnonzero(fold==f)
            assert set(dev['family_ids'][tr]).isdisjoint(dev['family_ids'][te])
            pred=fit_predict(subset(dev,tr),subset(dev,te),variant)
            for r in REGIMES:
                for k in ('mean','covariance'):oof[r][k][te]=pred[r][k]
        final=fit_predict(dev,conf,variant)
        for cohort,v,pred in (('development',dev,oof),('confirmation',conf,final)):
            metrics,loss,coverage=evaluate(v,pred)
            results.append({'cohort':cohort,'variant':variant,**metrics})
            family_archive[f'{cohort}__{variant}__family_error']=loss
            for r in REGIMES:
                for k in ('mean','covariance'):family_archive[f'{cohort}__{variant}__{r}__{k}']=pred[r][k]
            if cohort=='confirmation' and variant in ('full','numeric'):
                name='v22_vlm' if variant=='full' else 'v22_numeric'
                for r in REGIMES:
                    for k in ('mean','covariance'):np.testing.assert_allclose(pred[r][k],conf[f'{name}__{r}__{k}'],rtol=0,atol=1e-12)
                np.testing.assert_allclose(loss,reference[name+'__family_action_error'],rtol=0,atol=1e-12)
                expected=original['scores'][name]['mean_two_regimes']
                for k,oldkey in (('nll','active_nll_per_dimension'),('crps','active_crps'),('coverage90','active_joint90_coverage')):
                    np.testing.assert_allclose(metrics[k],expected[oldkey],rtol=0,atol=1e-12)
        print(json.dumps({'completed_variant':variant,'elapsed_seconds':round(time.perf_counter()-started,1)}),flush=True)
    np.savez_compressed(O/'PREDICTIONS_AND_FAMILY_ERRORS.npz',**family_archive)
    csvout('readout_ablation.csv',results)
    contrasts=[]
    for cohort in ('development','confirmation'):
        base=family_archive[f'{cohort}__numeric__family_error'];n=len(base)
        draws=reference['bootstrap_draw'] if cohort=='confirmation' else np.random.default_rng(20260913).integers(n,size=(20000,n))
        for variant in plan['variants'][1:]:
            delta=family_archive[f'{cohort}__{variant}__family_error']-base
            lo,hi=np.quantile(delta[draws].mean(1),[.025,.975])*1000
            contrasts.append({'cohort':cohort,'variant':variant,'difference_vs_numeric_mm':float(delta.mean()*1000),'low95_mm':float(lo),'high95_mm':float(hi),'analysis':'post hoc, unadjusted; development also reuses model-selection data'})
    csvout('readout_contrasts.csv',contrasts)
    component=[]
    for cohort in ('development','confirmation'):
        base=family_archive[f'{cohort}__full__family_error'];n=len(base)
        draws=reference['bootstrap_draw'] if cohort=='confirmation' else np.random.default_rng(20260913).integers(n,size=(20000,n))
        for variant in plan['variants']:
            if variant=='full':continue
            delta=family_archive[f'{cohort}__{variant}__family_error']-base
            lo,hi=np.quantile(delta[draws].mean(1),[.025,.975])*1000
            component.append({'cohort':cohort,'variant':variant,'difference_vs_full_mm':float(delta.mean()*1000),'low95_mm':float(lo),'high95_mm':float(hi),'analysis':'post hoc, unadjusted; fixed variant, no model reselection'})
    csvout('component_contrasts.csv',component)
    sensitivity=[];curves=[]
    baseline={r:dict(zip(('mean','covariance'),anchor(conf,r))) for r in REGIMES}
    ref_metrics,_,_=evaluate(conf,baseline)
    for scale in plan['covariance_scales']:
        out={}
        for r in REGIMES:
            a,c=anchor(conf,r);mean=conf[f'v22_vlm__{r}__mean'];d=mean-a;cov=c.copy();cov[:,np.arange(3),np.arange(3)]+=(scale*d)**2
            out[r]={'mean':mean,'covariance':cov}
        m,_,curve=evaluate(conf,out)
        sensitivity.append({'scale':scale,**m,'width_ratio_to_huber':m['projected_diameter']/ref_metrics['projected_diameter']})
        for q,cov in curve.items():curves.append({'scale':scale,'nominal':float(q),'empirical_coverage':cov})
    csvout('covariance_sensitivity.csv',sensitivity);csvout('coverage_curve.csv',curves)
    # Initialization derivative audit: the cache is standardized, not raw SI Jacobian.
    jac=[];projection_error=[]
    for p,cols in SUBSPACES.items():
        smallest=[];conditions=[];ranks=[]
        for i in np.flatnonzero(conf['protocol']==p):
            valid=conf['valid'][i];j=conf['J'][i][valid];jn=conf['Jn'][i][valid,:int(conf['nuisance_dim'][i])]
            projected=j-jn@np.linalg.pinv(jn,rcond=1e-8)@j
            projection_error.append(float(np.max(np.abs(projected-conf['source_projected_J'][i][valid]))))
            sv=np.linalg.svd(projected[:,cols],compute_uv=False)
            rank=int((sv>max(1e-10,1e-8*sv[0])).sum());ranks.append(rank);smallest.append(float(sv[-1]))
            if rank==len(cols):conditions.append(float(sv[0]/sv[-1]))
        jac.append({'protocol':p,'episodes':len(ranks),'active_dimensions':len(cols),'full_rank_episodes':int(sum(r==len(cols) for r in ranks)),
                    'minimum_singular_value_median':float(np.median(smallest)),'minimum_singular_value_p10':float(np.quantile(smallest,.1)),
                    'finite_condition_median':float(np.median(conditions)) if conditions else None})
    csvout('initial_jacobian.csv',jac)
    # Draw-wise comparator selection, independently inspected rather than assumed.
    names=['unit_rgb','huber_1.345_rgb','cauchy_2.385_rgb'];draw=reference['bootstrap_draw']
    sampled=np.stack([reference[n+'__family_action_error'][draw].mean(1) for n in names],1)
    wins=np.argmin(sampled,axis=1);counts={n:int((wins==i).sum()) for i,n in enumerate(names)}
    write('SUPPORT_RESULT.json',{'status':'completed','analysis_plan':plan,'variants':results,'covariance_sensitivity':sensitivity,
          'initial_jacobian':jac,'max_projection_reconstruction_error':max(projection_error),'draw_winner_counts':counts,
          'locked_checkpoint_refit_matches':True,'original_full_numeric_predictions_and_scores_match_atol':1e-12,
          'new_independent_confirmation':False,'final_model_changed':False,'elapsed_seconds':time.perf_counter()-started})
    print(json.dumps({'completed':True,'seconds':round(time.perf_counter()-started,1),'draw_winner_counts':counts,'covariance_sensitivity':sensitivity,'jacobian':jac},indent=2))

if __name__=='__main__':main()
