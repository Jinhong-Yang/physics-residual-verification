"""Post hoc nonlinear robust fits with original prior and nuisance constraints.

Weights use residual/original observation sigma; fixed precision alpha=64 is
retained. IRLS solves a penalized local M-estimator, not globally convex bounce.
Each episode uses anchor and prior starts, selected by its own objective only.
No truth is supplied to fitting. Profiled Gaussian scale is ML-profiled, not
given a new prior; its numerical floor and convergence limits are reported.
"""
from extended_common import *
from scipy.optimize import least_squares,minimize
from scipy.special import expit
from concurrent.futures import ProcessPoolExecutor
import time,argparse
from main3d_numeric import slide

def vertical(t,z0,v0,floor,e):
    g=9.81;z0=max(z0,floor);speed=np.sqrt(v0*v0+2*g*(z0-floor));hit=(v0+speed)/g
    result=np.full_like(t,floor);pre=t<=hit;result[pre]=z0+v0*t[pre]-.5*g*t[pre]**2
    period=2*e*speed/g
    if period>0 and e>0:
        tau=np.maximum(t-hit,0);total=period/max(1-e,1e-16);live=(~pre)&(tau<total)
        if e>=1-1e-15:n=np.floor(tau[live]/period);start=n*period;power=np.ones_like(n)
        else:
            n=np.floor(np.log1p(-np.minimum(tau[live]/total,1-np.finfo(float).eps))/np.log(e)).clip(0)
            power=e**n;start=period*(1-power)/(1-e)
        dt=tau[live]-start;result[live]=floor+e*speed*power*dt-.5*g*dt*dt
    return result

def solve_job(job):
    i,r,b,anchor_mean=job;cols=np.flatnonzero(b['active']);k=len(cols);bounce=bool(b['active'][2]);setup=bounce and r=='setup_translation'
    prior=b['prior_mean'];scale=b['prior_scale'];t=b['times'][b['valid']];axis=int(bounce);observed=b['xz'][b['valid'],axis];sigma=b['sigma_xz'][b['valid'],axis]
    if setup:q0=np.array([b['nuisance'][2]]);lo=[-np.inf];hi=[np.inf]
    elif bounce:q0=b['nuisance'][:3];lo=[-.5,-2.,-.4];hi=[2.,2.,.5]
    else:q0=b['nuisance'][:2];lo=[-5.,0.];hi=[5.,10.]
    low=np.r_[np.full(k,-np.inf),lo];high=np.r_[np.full(k,np.inf),hi]
    def raw(x):
        theta=prior.copy();theta[cols]+=scale[cols]*x[:k]
        q=x[k:]
        if setup:curve=vertical(t,.65,0,0,expit(theta[2]))+q[0]
        elif bounce:curve=vertical(t,*q,expit(theta[2]))
        else:curve=slide(t,*q,np.exp(np.clip(theta[1],-30,30)),np.exp(np.clip(-theta[0],-30,30)),b['force'],b['force_switch'])
        return (curve-observed)/sigma
    starts=[np.r_[(anchor_mean[cols]-prior[cols])/scale[cols],np.clip(q0,np.array(lo)+1e-10,np.array(hi)-1e-10)],np.r_[np.zeros(k),np.clip(q0,np.array(lo)+1e-10,np.array(hi)-1e-10)]]
    answers={}
    for kind in ('huber','cauchy','profile_map'):
        candidates=[]
        for start in starts:
            x=start.copy();converged=False;inner_success=True;floorhit=False;calls=0
            if kind=='profile_map':
                def objective(x):
                    nonlocal calls,floorhit
                    calls+=1;ss=float(raw(x)@raw(x));floorhit|=ss<1e-12
                    return .5*(x[:k]@x[:k])+len(t)/2*np.log(max(ss/len(t),1e-12))
                # Bound only the original nuisance coordinates. The physical prior is unchanged.
                fit=minimize(objective,x,method='L-BFGS-B',bounds=list(zip(low,high)),options={'maxiter':500,'ftol':1e-13,'gtol':1e-8,'maxls':40})
                x=fit.x;converged=bool(fit.success);it=fit.nit;cost=float(fit.fun);scale_hat=float(np.sqrt(np.mean(raw(x)**2)))
            else:
                for it in range(1,51):
                    nu=raw(x);w=np.minimum(1,1.345/np.maximum(abs(nu),1e-300)) if kind=='huber' else 1/(1+(nu/2.385)**2)
                    def residual(z):return np.r_[z[:k],8*np.sqrt(w)*raw(z)]
                    fit=least_squares(residual,x,bounds=(low,high),ftol=1e-12,xtol=1e-12,gtol=1e-10,max_nfev=150)
                    calls+=fit.nfev;inner_success &= bool(fit.success);step=float(np.linalg.norm(fit.x[:k]-x[:k]));nuisance_step=float(np.linalg.norm(fit.x[k:]-x[k:]));x=fit.x
                    if step<1e-8 and nuisance_step<1e-8:converged=True;break
                nu=raw(x);rho=np.where(abs(nu)<=1.345,.5*nu**2,1.345*(abs(nu)-.5*1.345)) if kind=='huber' else .5*2.385**2*np.log1p((nu/2.385)**2)
                cost=float(.5*(x[:k]@x[:k])+64*rho.sum());scale_hat=1/8
            theta=prior.copy();theta[cols]+=scale[cols]*x[:k]
            candidates.append({'mean':theta,'iterations':int(it),'converged':converged,'inner_success':inner_success,'objective':cost,'calls':calls,'scale_hat':scale_hat,'floor_hit':floorhit,'movement_prior_units':float(np.linalg.norm((theta-anchor_mean)/scale))})
        answers[kind]=min(candidates,key=lambda z:z['objective'])
    return i,r,answers

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);ap.add_argument('--limit',type=int,default=0);args=ap.parse_args();start=time.perf_counter()
    obs=load(D/'observation_inputs.npz');_,conf=data();jobs=[]
    for i in range(len(obs['ids']) if not args.limit else args.limit):
        b={k:v[i] for k,v in obs.items() if v.ndim and v.shape[0]==len(obs['ids'])}
        for r in REGIMES:
            if r==REGIMES[1] and not b['active'][2]:continue
            jobs.append((i,r,b,conf[f'huber_1.345_rgb__{r}__mean'][i]))
    saved={kind+'__'+r:conf[f'huber_1.345_rgb__{r}__mean'].copy() for kind in ('huber','cauchy','profile_map') for r in REGIMES};diagnostic=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for n,(i,r,ans) in enumerate(pool.map(solve_job,jobs,chunksize=6),1):
            for kind,result in ans.items():
                saved[kind+'__'+r][i]=result['mean']
                if r==REGIMES[0] and not obs['active'][i,2]:saved[kind+'__'+REGIMES[1]][i]=result['mean']
                diagnostic.append({'episode':str(obs['ids'][i]),'regime':r,'protocol':str(obs['protocol'][i]),'method':kind,**{k:v for k,v in result.items() if k!='mean'}})
            if n%60==0:print(json.dumps({'completed_jobs':n,'total':len(jobs),'seconds':round(time.perf_counter()-start)}),flush=True)
    suffix='_pilot' if args.limit else '';np.savez_compressed(O/f'anchor_predictions{suffix}.npz',**saved);csvout(f'anchor_diagnostics{suffix}.csv',diagnostic)
    if not args.limit:
        full,_=action(conf,{r:conf[f'v22_vlm__{r}__mean'] for r in REGIMES});rows=[]
        for kind in ('huber','cauchy','profile_map'):
            loss,ep=action(conf,{r:saved[kind+'__'+r] for r in REGIMES});ds=[d for d in diagnostic if d['method']==kind]
            rows.append({'method':kind,'action_mm':float(loss.mean()*1000),'full_minus_baseline':contrast(conf,full,loss),'converged_fraction':float(np.mean([d['converged'] for d in ds])),'mean_iterations':float(np.mean([d['iterations'] for d in ds])),'median_movement_prior_units':float(np.median([d['movement_prior_units'] for d in ds])),'scale_floor_episodes':sum(d['floor_hit'] for d in ds)})
        write('ANCHORS.json',{'analysis':'post hoc, local nonlinear fits; capped rows retained','results':rows,'seconds':time.perf_counter()-start});print(rows,flush=True)
if __name__=='__main__':main()
