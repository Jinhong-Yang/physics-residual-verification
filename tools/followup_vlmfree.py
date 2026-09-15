"""VLM-free nonlinear estimator using RGB tracks and a development population prior."""
from followup_common import *
from converged_anchor_baselines import solve_job
from concurrent.futures import ProcessPoolExecutor
import time
def main():
    d,v=verify_original();obs=load(D/'observation_inputs.npz');mu=d['truth'].mean(0);sd=d['truth'].std(0).clip(1e-3)
    jobs=[];saved={};diag=[];start=time.perf_counter()
    for r in REGIMES:saved[r]=np.tile(mu,(len(v['ids']),1))
    for i in range(len(obs['ids'])):
        b={k:a[i].copy() if isinstance(a[i],np.ndarray) else a[i] for k,a in obs.items() if a.ndim and a.shape[0]==len(obs['ids'])}
        b['prior_mean']=mu.copy();b['prior_scale']=sd.copy()
        axis=int(b['active'][2]);t=b['times'][b['valid']];y=b['xz'][b['valid'],axis]
        vel=float((y[1]-y[0])/max(t[1]-t[0],1e-6));q=np.zeros_like(b['nuisance'])
        if axis:q[:3]=[y[0],np.clip(vel,-1.99,1.99),0.]
        else:q[:2]=[y[0],np.clip(vel,.0001,9.999)]
        b['nuisance']=q
        for r in REGIMES:
            if r==REGIMES[1] and not axis:continue
            jobs.append((i,r,b,mu.copy()))
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i,r,answers in pool.map(solve_job,jobs,chunksize=8):
            a=answers['profile_map'];saved[r][i]=a['mean']
            if not obs['active'][i,2]:saved[REGIMES[1]][i]=a['mean']
            diag.append({'id':str(obs['ids'][i]),'regime':r,'converged':a['converged'],'calls':a['calls']})
    loss,_=action(v,saved);saved['family_loss']=loss;np.savez_compressed(O/'vlmfree_predictions.npz',**saved)
    result={'action_mm':float(loss.mean()*1000),'elapsed_fit_seconds_4workers':time.perf_counter()-start,
      'population_prior_mean':mu.tolist(),'population_prior_scale':sd.tolist(),'prior_fitted_on':'development truth only',
      'VLM_used':False,'input':'cached calibrated RGB tracking positions/sigmas; no original prior or anchor initialization',
      'converged_fraction':float(np.mean([a['converged'] for a in diag])),'residual_adapter':False}
    csvsave('vlmfree_diagnostics.csv',diag);save('VLMFREE.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
