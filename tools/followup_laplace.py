"""Local joint Gaussian MAP and nuisance-marginalized Laplace posterior.

The Gaussian marginal mean equals the joint mode. Marginalization changes
covariance via the Schur complement; no claim of non-Gaussian integration.
Bound-contact and rank-deficient rows are retained and explicitly diagnosed.
"""
from followup_common import *
from converged_anchor_baselines import vertical
from main3d_numeric import slide
from scipy.optimize import least_squares
from scipy.special import expit
from concurrent.futures import ProcessPoolExecutor
import time

def solve(job):
    i,r,b,a=job;cols=np.flatnonzero(b['active']);k=len(cols);bounce=bool(b['active'][2]);setup=bounce and r==REGIMES[1]
    prior=b['prior_mean'];scale=b['prior_scale'];t=b['times'][b['valid']];axis=int(bounce)
    y=b['xz'][b['valid'],axis];sigma=b['sigma_xz'][b['valid'],axis]
    if setup:q0=np.array([b['nuisance'][2]]);lo=np.array([-np.inf]);hi=np.array([np.inf])
    elif bounce:q0=b['nuisance'][:3];lo=np.array([-.5,-2.,-.4]);hi=np.array([2.,2.,.5])
    else:q0=b['nuisance'][:2];lo=np.array([-5.,0.]);hi=np.array([5.,10.])
    def residual(x):
        th=prior.copy();th[cols]+=scale[cols]*x[:k];q=x[k:]
        if setup:curve=vertical(t,.65,0,0,expit(th[2]))+q[0]
        elif bounce:curve=vertical(t,*q,expit(th[2]))
        else:curve=slide(t,*q,np.exp(np.clip(th[1],-30,30)),np.exp(np.clip(-th[0],-30,30)),b['force'],b['force_switch'])
        return np.r_[x[:k],8*(curve-y)/sigma]
    candidates=[]
    for start in ((a[cols]-prior[cols])/scale[cols],np.zeros(k)):
        fit=least_squares(residual,np.r_[start,np.clip(q0,lo+1e-10,hi-1e-10)],bounds=(np.r_[[-np.inf]*k,lo],np.r_[[np.inf]*k,hi]),ftol=1e-12,xtol=1e-12,gtol=1e-10,max_nfev=500)
        candidates.append(fit)
    fit=min(candidates,key=lambda f:f.cost);th=prior.copy();th[cols]+=scale[cols]*fit.x[:k]
    h=fit.jac.T@fit.jac;hn=h[k:,k:];rank=int(np.linalg.matrix_rank(hn));schur=h[:k,:k]-h[:k,k:]@np.linalg.pinv(hn)@h[k:,:k]
    cov=np.diag(scale**2);cov[np.ix_(cols,cols)]=np.linalg.pinv(schur)*scale[cols,None]*scale[None,cols]
    boundary=bool(np.any(np.minimum(fit.x[k:]-lo,hi-fit.x[k:])<1e-6))
    return i,r,th,cov,{'episode':str(b['ids']),'regime':r,'protocol':str(b['protocol']),'converged':bool(fit.success),'calls':fit.nfev,'boundary':boundary,'nuisance_rank':rank,'nuisance_dim':len(q0),'objective':float(fit.cost)}
def main():
    _,v=verify_original();obs=load(D/'observation_inputs.npz');jobs=[];saved={};diag=[]
    for r in REGIMES:
        saved[r+'__mean']=anchor(v,r)[0].copy();saved[r+'__covariance']=anchor(v,r)[1].copy()
    for i in range(len(obs['ids'])):
        b={k:a[i] for k,a in obs.items() if a.ndim and a.shape[0]==len(obs['ids'])}
        for r in REGIMES:
            if r==REGIMES[1] and not b['active'][2]:continue
            jobs.append((i,r,b,anchor(v,r)[0][i]))
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i,r,mean,cov,d in pool.map(solve,jobs,chunksize=8):
            saved[r+'__mean'][i]=mean;saved[r+'__covariance'][i]=cov;diag.append(d)
            if not obs['active'][i,2]:
                saved[REGIMES[1]+'__mean'][i]=mean;saved[REGIMES[1]+'__covariance'][i]=cov
    loss,ep=action(v,{r:saved[r+'__mean'] for r in REGIMES});saved['family_loss']=loss
    full,_=action(v,{r:v[f'v22_vlm__{r}__mean'] for r in REGIMES})
    result={'method':'Gaussian joint MAP with Gauss-Newton Laplace nuisance marginal', 'action_mm':float(loss.mean()*1000),
       'contrast_full_minus_laplace':paired(full,loss),'converged_fraction':float(np.mean([d['converged'] for d in diag])),
       'boundary_rows':sum(d['boundary'] for d in diag),'rank_deficient_nuisance_rows':sum(d['nuisance_rank']<d['nuisance_dim'] for d in diag),
       'fitted_regime_rows':len(diag),'non_gaussian_integral':False,'bound_truncation_corrected':False}
    np.savez_compressed(O/'laplace_predictions.npz',**saved);csvsave('laplace_diagnostics.csv',diag);save('LAPLACE.json',result)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
