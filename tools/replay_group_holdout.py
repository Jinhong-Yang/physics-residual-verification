"""Refit and score the material-held-out study from bundled numeric views.

This replay uses archived RGB-derived features. It does not download videos or
rerun the foundation encoder. Evaluation targets are loaded only after fitting.
"""
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'evidence/group_holdout'
sys.path.insert(0,str(D/'code'))
from develop_physprobe_vector_readout_v35 import fit,predict
from develop_physprobe_confidence_gate_v38 import gate_features,optimal_gate,clipped_update
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def main():
    data=load(D/'VIEWS.npz');target=load(D/'TRAIN_TARGETS.npz')
    train=np.flatnonzero(data['roles']=='train');test=np.flatnonzero(data['roles']=='evaluation')
    assert np.array_equal(data['ids'][train],target['ids'])
    y=target['target'];nx=data['numeric'];vx=data['visual'];base=data['quality']
    obj=data['object_material'][train];surf=data['surface_material'][train]
    for k in ('object_material','surface_material'):
        assert not set(data[k][train])&set(data[k][test])
    n_oof=np.empty_like(y);u_oof=np.empty_like(y)
    for a,b in sorted(set(zip(obj.tolist(),surf.tolist()))):
        held=(obj==a)&(surf==b);keep=(obj!=a)&(surf!=b)
        n=fit(nx[train][keep],y[keep],10.,True)
        v=fit(vx[train][keep],y[keep]-predict(n,nx[train][keep]),1000.,False)
        cap=max(float(np.std(np.linalg.norm(y[keep],axis=1))),1e-6)
        n_oof[held]=predict(n,nx[train][held]);u_oof[held]=clipped_update(v,vx[train][held],cap)
    gate=fit(gate_features(base[train],n_oof,u_oof),optimal_gate(n_oof,u_oof,y),.1,False)
    n=fit(nx[train],y,10.,True);v=fit(vx[train],y-predict(n,nx[train]),1000.,False)
    cap=max(float(np.std(np.linalg.norm(y,axis=1))),1e-6)
    numeric=predict(n,nx[test]);update=clipped_update(v,vx[test],cap)
    g=np.clip(predict(gate,gate_features(base[test],numeric,update)).reshape(-1),0,1)
    computed={'numeric':numeric,'raw':numeric+update,'gated':numeric+g[:,None]*update}
    expected=load(D/'PREDICTIONS.npz')
    assert np.array_equal(expected['ids'],data['ids'][test])
    for k,value in computed.items():np.testing.assert_allclose(value,expected[k],rtol=0,atol=1e-10)
    outcome=load(D/'EVALUATION_TARGETS.npz')
    assert np.array_equal(outcome['ids'],data['ids'][test])
    errors={k:np.linalg.norm(v-outcome['target'],axis=1)*1000 for k,v in computed.items()}
    obj=data['object_material'][test];surf=data['surface_material'][test]
    objects=sorted(set(obj.tolist()));surfaces=sorted(set(surf.tolist()));cells=sorted(set(zip(obj.tolist(),surf.tolist())))
    cell_errors={k:np.array([v[(obj==a)&(surf==b)].mean() for a,b in cells]) for k,v in errors.items()}
    delta=cell_errors['gated']-cell_errors['numeric']
    plan=json.loads((D/'PLAN.json').read_text());rng=np.random.default_rng(plan['seed']+1)
    ow=rng.multinomial(8,np.ones(8)/8,size=20000);sw=rng.multinomial(8,np.ones(8)/8,size=20000)
    weights=ow[:,[objects.index(a) for a,b in cells]]*sw[:,[surfaces.index(b) for a,b in cells]]
    draws=weights@delta/weights.sum(1)
    result=json.loads((D/'RESULT.json').read_text())
    np.testing.assert_allclose(np.quantile(draws,[.025,.975]),result['two_way_bootstrap_95_mm'],rtol=0,atol=1e-8)
    for k,v in cell_errors.items():np.testing.assert_allclose(v.mean(),result['macro_cell_mae_mm'][k],rtol=0,atol=1e-8)
    report={'refit_and_predictions_match':True,'material_overlap':0,'two_way_interval_matches':True,
            'scientific_pass':result['pass'],'macro_cell_mae_mm':{k:float(v.mean()) for k,v in cell_errors.items()},
            'two_way_bootstrap_95_mm':np.quantile(draws,[.025,.975]).tolist()}
    out=ROOT/'replayed/group_holdout';out.mkdir(parents=True,exist_ok=True)
    (out/'REPLAY.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
