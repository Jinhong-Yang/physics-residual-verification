"""Prospective V7 exact-Zeno observation physics with unchanged M95 step algebra.

No old kernel/global is patched. General vertical API supports independent
tests; observation inference fixes known gravity9.81 and fits z0/v0/floor.
The canonical data are the v0=0 subset. Conditional local
curvature is still an approximation around contact events and selected starts.
"""
from types import SimpleNamespace
import ast
import torch
import reliability_nonlinear_v2 as reference
import reliability_nonlinear_efficient_v2 as efficient
import reliability_nonlinear_fast_v2 as fast
import reliability_contact_multistart_v2 as multistart
import reliability_contact_wall_budget_v2 as wall
from reliability_experiment_v2 import torch_slide


def vertical(times,z0,v0,floor,restitution,gravity):
    """B x T exact piecewise Newton trajectory, no speed/count cutoff.

    Geometric flight lengths locate the current flight analytically. At and
    after the finite Zeno limit the object rests at floor. e=0 and e=1 are
    handled explicitly, including floating-point sigmoid saturation. Like the
    invalid negative initial clearance is projected to the contact plane for
    both the first flight and impact time. This physical projection is not a
    material-parameter support clamp; callers record its activation separately.
    """
    if not all(bool(torch.isfinite(v).all()) for v in [times,z0,v0,floor,restitution,gravity]) or bool((times<0).any()):
        raise ValueError('Require finite physical inputs and nonnegative times')
    if bool((gravity<=0).any() or (restitution<0).any() or (restitution>1).any()):
        raise ValueError('Require gravity>0 and 0<=e<=1')
    tiny=torch.finfo(times.dtype).tiny
    initial_height=torch.maximum(z0,floor)
    disc=v0.square()+2*gravity*(initial_height-floor)
    speed=torch.where(disc>0,disc.clamp_min(tiny).sqrt(),torch.zeros_like(disc))
    hit=(v0+speed)/gravity
    tau=(times-hit[:,None]).clamp_min(0)
    launch=restitution*speed
    period=2*launch/gravity
    # Replace inactive operands BEFORE divisions/logarithms, not merely their
    # output. where(live, bad_division,0) still backpropagates 0*NaN otherwise.
    inelastic=(restitution>0)&(restitution<1)&(period>0)
    one_minus=torch.where(inelastic,1-restitution,torch.ones_like(restitution))
    total=period/one_minus
    live=(times>hit[:,None])&(tau<total[:,None])&inelastic[:,None]
    ep=torch.where(live,restitution[:,None],torch.full_like(tau,.5))
    per=torch.where(live,period[:,None],torch.ones_like(tau))
    elapsed=torch.where(live,tau,torch.zeros_like(tau))
    loge=torch.log(ep)
    fraction=((1-ep)*elapsed/per).clamp(0,1-torch.finfo(times.dtype).eps)
    n=torch.floor(torch.log1p(-fraction)/loge).clamp_min(0)
    power=torch.exp(n*loge)
    start=per*(-torch.expm1(n*loge))/(1-ep)
    within=elapsed-start
    rebound=floor[:,None]+torch.where(live,launch[:,None],torch.zeros_like(tau))*power*within-.5*gravity[:,None]*within.square()
    after=torch.where(live,rebound,floor[:,None])
    elastic_live=(times>hit[:,None])&(restitution==1)[:,None]&(period>0)[:,None]
    elastic_period=torch.where(elastic_live,period[:,None],torch.ones_like(tau))
    elastic_elapsed=torch.where(elastic_live,tau,torch.zeros_like(tau))
    elastic_n=torch.floor(elastic_elapsed/elastic_period)
    elastic_t=elastic_elapsed-elastic_n*elastic_period
    elastic=floor[:,None]+torch.where(elastic_live,launch[:,None],torch.zeros_like(tau))*elastic_t-.5*gravity[:,None]*elastic_t.square()
    after=torch.where(elastic_live,elastic,after)
    before=times<=hit[:,None]
    initial_time=torch.where(before,times,torch.zeros_like(times))
    initial=initial_height[:,None]+v0[:,None]*initial_time-.5*gravity[:,None]*initial_time.square()
    return torch.where(before,initial,after)


def observation_residual(batch,theta,nuisance):
    bouncing=batch['active'][:,2]
    if not torch.equal(bouncing,bouncing[:1].expand_as(bouncing)):
        raise ValueError('Requires homogeneous protocol')
    if bool(bouncing[0]):
        prediction=vertical(batch['times'],nuisance[:,0],nuisance[:,1],nuisance[:,2],theta[:,2].sigmoid(),torch.full_like(nuisance[:,3],9.81))
    else:
        prediction=reference.slide(batch['times'],nuisance[:,0],nuisance[:,1],theta[:,1].clamp(-30,30).exp(),
            1/theta[:,0].clamp(-30,30).exp(),batch['force'],batch['force_switch'])
    axis=bouncing.long()[:,None,None].expand(-1,batch['times'].shape[1],1)
    observed=batch['xz'].gather(2,axis).squeeze(-1);sigma=batch['sigma_xz'].gather(2,axis).squeeze(-1)
    if not bool((sigma>0).all()):raise ValueError('Invalid observation noise')
    valid=batch['valid']&batch['active'].any(-1)[:,None]
    return torch.where(valid,(prediction-observed)/sigma,torch.zeros_like(prediction))


def nuisance_bounds(batch,q):
    bounded=reference._nuisance_bounds(batch,q)
    return torch.cat([bounded[:,:3],torch.where(batch['active'][:,2],torch.full_like(q[:,3],9.81),torch.zeros_like(q[:,3]))[:,None]],-1)


def _known_g_columns(tree):
    count=0
    for node in ast.walk(tree):
        if isinstance(node,ast.IfExp) and isinstance(node.body,ast.Constant) and node.body.value==4 and isinstance(node.orelse,ast.Constant) and node.orelse.value==2:
            node.body=ast.Constant(3);count+=1
    assert count==1
    return tree


_reference=SimpleNamespace(**dict(vars(reference),observation_residual=observation_residual,_nuisance_bounds=nuisance_bounds))
relinearize=fast._copy_function(efficient.relinearize,{'reference':_reference},_known_g_columns)
_core=fast._copy_function(reference.refine,{'relinearize':relinearize,'observation_residual':observation_residual,
    '_nuisance_bounds':nuisance_bounds,'_solve_no_covariance':fast._solve_no_covariance,'_weights':fast._weights_no_covariance},fast._candidate_no_covariance)
_grouped=fast._copy_function(efficient.refine,{'_same_algebra_core':_core})
_efficient=SimpleNamespace(relinearize=relinearize,refine=_grouped)
_step_reference=SimpleNamespace(**dict(vars(_reference),_solve=fast._solve_no_covariance))
one_step=fast._copy_function(multistart.one_step,{'reference':_step_reference,'efficient':_efficient})
_fast=SimpleNamespace(one_step=one_step,relinearize=relinearize)
def _known_g_budget(tree):
    count=0
    for node in ast.walk(tree):
        if isinstance(node,ast.Compare) and isinstance(node.left,ast.Name) and node.left.id=='calls':
            assert len(node.comparators)==1 and node.comparators[0].value==95
            node.comparators[0]=ast.Constant(81);count+=1
    assert count==1
    return tree


bounce_beam=fast._copy_function(wall.bounce_beam,{'fast':_fast,'reference':_reference},_known_g_budget)
_outer=fast._copy_function(multistart.refine,{'bounce_beam':bounce_beam,'efficient':_efficient})


def refine(batch,trust_radius=1.):
    result=_outer(batch,trust_radius)
    result['physical_kernel_calls']=result['forward_calls']
    result['physics_contract']='V7 exact Newton/Zeno bounce known g9.81; original stopped sliding; same bounded-step algebra; bounce81 residual calls'
    return result


def analytic_actions(mean,active,training=False):
    """Same training/new action parameters, exact V7 bounce trajectory."""
    mass=mean[:,0].clamp(-30,30).exp();mu=mean[:,1].clamp(-30,30).exp()
    times=torch.linspace(.125,1.,8,dtype=mean.dtype,device=mean.device)
    forced=torch_slide(times,mass,mu,(.6,.9) if training else (.45,1.1),.4 if training else .35,.3 if training else .25)
    free=torch_slide(times,mass,mu,(0.,0.),.4 if training else .35,.8 if training else 1.)
    scalar=torch.ones_like(mass)
    bounce=vertical(times[None].expand(len(mean),-1),scalar*(.65 if training else .85),scalar*0,scalar*0,mean[:,2].sigmoid(),scalar*9.81)
    return torch.where(active[:,2,None],bounce,torch.where(active[:,0,None],forced,free))
