"""Fixed-plan frozen-video comparison; PCA and regression fit on training families only."""
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='2'
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

R=Path(__file__).resolve().parents[1]
P=R if (R/'main.tex').exists() else R/'paper/ieee_access_initial_submission/overleaf'
I=P/'evidence/video_control' if (R/'main.tex').exists() else R/'results/access_video_control_v1'
O=P/'replayed/video_control' if (R/'main.tex').exists() else I
O.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(P/'tools'))
import run_support_study as s

def pack(v,projected):
    out=dict(v);x=v['features'].astype(np.float64).copy()
    x[:,:,24:56]=projected[:,:32,None].transpose(0,2,1)
    x[:,:,56:58]=projected[:,32:].reshape(-1,8,2)
    out['features']=x
    np.testing.assert_allclose(s.feature_views(x,24)[1],projected,rtol=0,atol=1e-12)
    np.testing.assert_array_equal(x[:,:,:24],v['features'][:,:,:24])
    return out

def projection(x,train_ix,test_ix):
    center=x[train_ix].mean(0)
    _,_,vt=np.linalg.svd(x[train_ix]-center,full_matrices=False)
    components=vt[:48].T
    return (x[train_ix]-center)@components,(x[test_ix]-center)@components,center,components

def main():
    extraction=json.loads((I/'EXTRACTION.json').read_text());assert extraction['status']=='completed'
    dev=s.load(P/'evidence/support/development.npz');conf=s.load(P/'evidence/support/confirmation.npz')
    features=[]
    for cohort,values in (('development',dev),('confirmation',conf)):
        z=s.load(I/f'{cohort}_features.npz');assert np.array_equal(values['ids'],z['ids'])
        features.append(z['features'].astype(np.float64))
    x=np.concatenate(features)
    original=s.load(P/'replayed/support/PREDICTIONS_AND_FAMILY_ERRORS.npz')
    fold=original['development_folds'];oof={r:{'mean':np.empty((len(dev['ids']),3)),'covariance':np.empty((len(dev['ids']),3,3))} for r in s.REGIMES}
    projections={}
    for f in range(6):
        tr=np.flatnonzero(fold!=f);te=np.flatnonzero(fold==f)
        assert set(dev['family_ids'][tr]).isdisjoint(dev['family_ids'][te])
        a,b,c,w=projection(x,tr,te)
        projections[f'fold{f}_mean']=c;projections[f'fold{f}_components']=w
        pred=s.fit_predict(pack(s.subset(dev,tr),a),pack(s.subset(dev,te),b),'full')
        for r in s.REGIMES:
            for k in ('mean','covariance'):oof[r][k][te]=pred[r][k]
        print('FOLD_DONE '+str(f),flush=True)
    nd=len(dev['ids']);tr=np.arange(nd);te=np.arange(nd,len(x))
    a,b,c,w=projection(x,tr,te);projections['all_development_mean']=c;projections['all_development_components']=w
    train=pack(dev,a);test=pack(conf,b);final=s.fit_predict(train,test,'full')
    anchored={r:dict(zip(('mean','covariance'),s.anchor(dev,r))) for r in s.REGIMES}
    checkpoint=s.fit_checkpoint(train['features'],24,dev['truth'],dev['active'],dev['protocol'],anchored)
    np.savez_compressed(O/'CHECKPOINT.npz',**checkpoint,**projections)
    draws=s.load(P/'evidence/results/com_observation_v23/confirmation_v1/FAMILY_ACTION_ERRORS.npz')['bootstrap_draw']
    rows=[];contrasts=[];archive={}
    for cohort,values,pred in (('development',dev,oof),('confirmation',conf,final)):
        metrics,errors,_=s.evaluate(values,pred);rows.append({'cohort':cohort,'arm':'vjepa2_48',**metrics})
        archive[cohort+'__family_error']=errors
        for r in s.REGIMES:
            for k in ('mean','covariance'):archive[f'{cohort}__{r}__{k}']=pred[r][k]
        boot=draws if cohort=='confirmation' else np.random.default_rng(20260913).integers(len(errors),size=(20000,len(errors)))
        for name in ('numeric','full'):
            d=errors-original[f'{cohort}__{name}__family_error'];lo,hi=np.quantile(d[boot].mean(1),[.025,.975])*1000
            contrasts.append({'cohort':cohort,'comparison':'vjepa2_48 minus '+name,'difference_mm':float(d.mean()*1000),'low95_mm':float(lo),'high95_mm':float(hi)})
    for name,records in (('SCORES.csv',rows),('CONTRASTS.csv',contrasts)):
        with (O/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=records[0]);writer.writeheader();writer.writerows(records)
    np.savez_compressed(O/'PREDICTIONS.npz',**archive)
    result={'status':'completed','scores':rows,'contrasts':contrasts,'PCA_fit_on_training_families_only':True,
            'all_planned_arms_reported':True,'original_control_scores':'overleaf/replayed/support/readout_ablation.csv',
            'primary_model_changed':False,'new_independent_confirmation':False,
            'interpretation':'post hoc frozen representation comparison; unequal upstream supervision/preprocessing, no causal attribution to language'}
    (O/'RESULT.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
