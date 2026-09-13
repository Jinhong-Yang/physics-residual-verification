"""Separate known-H/v0 bounce control; no replacement of frozen V7 primary.

Original logit-e prior, analytically profiled translation, fixed75 calls/row.
All optimizer branches use the same original-prior objective. No data work at
import, no learned weights, no material-support clamp, no adaptive extra fit.
"""
import torch
from reliability_canonical_v7 import vertical
H=.65
G=9.81


def profile_curve(height,z,sigma,valid,alpha):
    """Profile free translation at fixed alpha/sigma², without a c prior."""
    precision=torch.where(valid,alpha[:,None]/sigma.square(),torch.zeros_like(sigma))
    total=precision.sum(-1);denominator=torch.where(total>0,total,torch.ones_like(total))
    c=(precision*(z-height)).sum(-1)/denominator
    root=torch.where(valid,alpha[:,None].sqrt()/sigma,torch.zeros_like(sigma))
    residual=root*(height+c[:,None]-z)
    return c,residual


def trajectory(times,logit_e):
    one=torch.ones_like(logit_e)
    if not bool(torch.isfinite(logit_e).all()):raise ValueError('Nonfinite logit input, no hidden clamp')
    return vertical(times,one*H,one*0,one*0,logit_e.sigmoid(),one*G)


def fit_setup_bounce(z,sigma,times,prior_mean,prior_scale,valid,alpha):
    b,t=z.shape
    if b<1 or t!=8 or prior_mean.shape!=(b,3) or prior_scale.shape!=(b,3) or valid.shape!=(b,t) or valid.dtype!=torch.bool:
        raise ValueError('Require Bx8 observations, Bx3 prior, bool mask')
    for name,v,shape in [('z',z,(b,8)),('sigma',sigma,(b,8)),('times',times,(b,8)),('prior_mean',prior_mean,(b,3)),('prior_scale',prior_scale,(b,3)),('alpha',alpha,(b,))]:
        if v.shape!=shape or v.dtype!=torch.float64 or not bool(torch.isfinite(v).all()):raise ValueError('Invalid float64 '+name)
    if not bool((sigma>0).all() and (prior_scale>0).all() and (alpha>0).all()) or bool((times<0).any()):
        raise ValueError('Require positive noise/scale/alpha and nonnegative times')
    meaningful=valid.any(-1)
    calls=0;invocations=0
    def evaluate(u,index):
        nonlocal calls,invocations
        invocations+=1;calls+=len(index)
        theta=prior_mean[index,2]+prior_scale[index,2]*u
        curve=trajectory(times[index],theta)
        c,residual=profile_curve(curve,z[index],sigma[index],valid[index],alpha[index])
        cost=.5*(u.square()+residual.square().sum(-1))
        return c,residual,cost
    def linear(u,index):
        c,r,cost=evaluate(u,index)
        _,plus,_=evaluate(u+1e-3,index);_,minus,_=evaluate(u-1e-3,index)
        J=(plus-minus)/2e-3
        return c,r,J,cost
    def step(u,index):
        _,r,J,before=linear(u,index)
        curvature=1+J.square().sum(-1)
        update=(-(u+(J*r).sum(-1))/curvature).clamp(-1,1)
        best,bestcost=u,before
        accepted=torch.zeros_like(u)
        for fraction in [1.,.5,.25]:
            trial=u+fraction*update
            _,_,cost=evaluate(trial,index)
            better=torch.isfinite(cost)&(cost<bestcost)&meaningful[index]
            best=torch.where(better,trial,best);bestcost=torch.where(better,cost,bestcost)
            accepted=torch.where(better,torch.full_like(u,fraction),accepted)
        return best,bestcost,accepted
    # Center first yields a deterministic center-preferred tie.
    starts=z.new_tensor([0.,-1.,1.])
    index=torch.arange(b,device=z.device).repeat(3)
    u=starts.repeat_interleave(b);first_history=[]
    for _ in range(3):
        u,cachedcost,accepted=step(u,index);first_history.append(accepted.reshape(3,b).T)
    branch_costs=cachedcost.reshape(3,b).T
    chosen=branch_costs.argmin(-1)
    winner=u[chosen*b+torch.arange(b,device=z.device)]
    index=torch.arange(b,device=z.device);winner_history=[]
    for _ in range(3):
        winner,cachedcost,accepted=step(winner,index);winner_history.append(accepted)
    winner=torch.where(meaningful,winner,torch.zeros_like(winner))
    c,r,J,finalcost=linear(winner,index)
    var=prior_scale[:,2].square()/(1+J.square().sum(-1))
    mean=prior_mean.clone();mean[:,2]=prior_mean[:,2]+prior_scale[:,2]*winner
    covariance=torch.diag_embed(prior_scale.square());covariance[:,2,2]=var
    if calls!=75*b or invocations!=39:raise RuntimeError('Fixed75-call ledger changed')
    if not all(bool(torch.isfinite(v).all()) for v in [mean,covariance,c,r,J,finalcost]):raise ValueError('Nonfinite final fit')
    nuisance=torch.stack([c+H,torch.zeros_like(c),c,torch.full_like(c,G)],-1)
    return dict(mean=mean,covariance=covariance,nuisance=nuisance,offset=c,weights=valid.to(z.dtype),
        alpha=alpha.clone(),observation_precision=torch.where(valid,alpha[:,None]/sigma.square(),torch.zeros_like(sigma)),
        selected_start_index=chosen,branch_costs=branch_costs,final_cost=finalcost,
        final_profiled_residual=r,final_prior_whitened_J=J,
        first_accepted_fraction=torch.stack(first_history,1),winner_accepted_fraction=torch.stack(winner_history,1),
        no_evidence=~meaningful,forward_calls_per_observation=torch.full((b,),75,dtype=torch.int64,device=z.device),
        forward_calls=calls,kernel_invocations=invocations,source_unused_initial_nuisance_calls=0,
        covariance_scope='conditional Gauss-Newton Schur complement includes local free-c uncertainty; omits nonlinear/contact/start/line-search/noise-fit uncertainty',
        original_prior_reused=True,information_contract='knownH.65/v0zero/g9.81; unknowntranslation only; extra setup input versus primary')


def fit_training_dispersion(z,sigma,times,valid,true_logit_e):
    """TRAIN supervisor only: e fixed, c dimension1, alpha=df/SSE clipped."""
    if z.ndim!=2 or z.shape[1]!=8 or sigma.shape!=z.shape or times.shape!=z.shape or valid.shape!=z.shape or valid.dtype!=torch.bool:
        raise ValueError('Invalid training observation shapes')
    if true_logit_e.shape!=(len(z),) or any(v.dtype!=torch.float64 for v in [z,sigma,times,true_logit_e]):
        raise ValueError('Require float64 training tensors and one true logit per row')
    if not all(bool(torch.isfinite(v).all()) for v in [z,sigma,times,true_logit_e]) or not bool((sigma>0).all()) or bool((times<0).any()):
        raise ValueError('Nonfinite/nonpositive training inputs')
    if bool((valid.sum(-1)<=1).any()):raise ValueError('Nonpositive calibration df; no row exclusion')
    with torch.no_grad():
        h=trajectory(times,true_logit_e)
        c,r=profile_curve(h,z,sigma,valid,torch.ones(len(z),dtype=z.dtype,device=z.device))
        sse=r.square().sum();df=(valid.sum(-1)-1).sum()
        raw=df/sse
        alpha=raw.clamp(1/64,64)
    if not bool(torch.isfinite(sse)&torch.isfinite(alpha)) or bool(torch.isnan(raw)):
        raise ValueError('Nonfinite dispersion inputs/result; no hidden fallback')
    infinite=bool(torch.isposinf(raw))
    return dict(alpha=alpha.item(),alpha_raw=None if infinite else raw.item(),alpha_raw_positive_infinity=infinite,
        SSE=sse.item(),n_eff=int(df),n=len(z),
        nuisance_dim=1,offset=c.cpu().numpy().tolist(),row_SSE=r.square().sum(-1).cpu().numpy().tolist(),
        physical_kernel_calls=len(z),excluded=0,supervision='truee TRAIN only; e not a fitted nuisance in this stage')
