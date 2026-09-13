"""Equal-budget, nuisance-projected, observation-only point refinement."""
import time
import numpy as np
from scipy.optimize import least_squares
from main3d_numeric import center,world_xz,slide,vertical
from main3d_evaluation import natural,operational

LOW=np.array([np.log(.15),np.log(.05),np.log(.25/.75)])
HIGH=np.array([np.log(1.5),np.log(.35),np.log(.9/.1)])

def transformed_point(p):return np.array([np.log(p[0]),np.log(p[1]),np.log(p[2]/(1-p[2]))])

def tracking(row,base):
    if row['protocol'] not in ['multiple_forces','unforced_slide','bounce'] or not row['camera_metric_calibrated'] or row['camera'] is None or row['occlusion']:
        return {'status':'unobservable'}
    if row['protocol']=='multiple_forces' and row['force_N'] is None:return {'status':'missing_force'}
    pixels=[center(base/path) for path in row['frames']]
    if any(x is None for x in pixels):return {'status':'tracking_failed'}
    pixels=np.array(pixels);xz=np.array([world_xz(x,row['camera']) for x in pixels]);jac=[]
    for pixel in pixels:
        jac.append(np.stack([(world_xz(pixel+delta,row['camera'])-world_xz(pixel-delta,row['camera']))/2 for delta in np.eye(2)],axis=-1))
    sigma=np.sqrt((np.array(jac)**2).sum(-1));assert np.isfinite(sigma).all() and (sigma>0).all()
    return {'status':'tracked','pixels':pixels.tolist(),'xz':xz.tolist(),'sigma_xz_m':sigma.tolist()}

class BudgetExceeded(Exception):pass

def refine(row,base,initial_point,common_scale,tracked=None):
    started=time.perf_counter();point=operational(np.array(initial_point,dtype=float));theta0=transformed_point(point);scale=np.array(common_scale,dtype=float)
    assert point.shape==scale.shape==(3,) and np.isfinite(point).all() and (scale>0).all()
    tracked=tracking(row,base) if tracked is None else tracked
    result={'point':point.copy(),'initial_operational_point':point.copy(),'status':tracked['status'],'forward_calls':0,'rank':0,'singular_values':[],
      'nullspace_update_norm':0.,'common_scale':scale.copy(),'regularizer':'common_training_statistical_scale_not_model_uncertainty'}
    def finish():
        result['elapsed_s']=time.perf_counter()-started;return result
    if tracked['status']!='tracked':return finish()
    bounce=row['protocol']=='bounce';active=np.array([2] if bounce else [0,1] if row['protocol']=='multiple_forces' else [1])
    axis=1 if bounce else 0;t=np.array(row['timestamps_s']);xz=np.array(tracked['xz']);data=xz[:,axis];sigma=np.array(tracked['sigma_xz_m'])[:,axis]
    if bounce:
        q0=np.array([data[0],0.,data.min()-.04,9.81]);qlo=np.array([-.5,-2.,-.4,5.]);qhi=np.array([2.,2.,.5,15.])
    else:
        q0=np.array([data[0],(data[1]-data[0])/(t[1]-t[0])]);qlo=np.array([-5.,0.]);qhi=np.array([5.,10.])
    q0=np.clip(q0,qlo+1e-8,qhi-1e-8);best_nuisance=[float('inf'),q0.copy()];joint_best=[float('inf'),None,None];basis=None;rank=0
    def residual(theta,q):
        if result['forward_calls']>=400:raise BudgetExceeded()
        result['forward_calls']+=1;m,mu,e=natural(theta)
        if bounce:prediction=vertical(t,q[0],q[1],q[2],e,q[3])
        else:
            force=row['force_N'] if row['protocol']=='multiple_forces' else [0.,0.]
            switch=row['force_switch_s'] if row['protocol']=='multiple_forces' else 1.
            prediction=slide(t,q[0],q[1],mu,1/m,force,switch)
        value=(prediction-data)/sigma;assert np.isfinite(value).all();return value
    def nuisance_fun(q):
        rr=residual(theta0,q);cost=float(rr@rr)
        if cost<best_nuisance[0]:best_nuisance[:]=[cost,q.copy()]
        return rr
    try:
        fit=least_squares(nuisance_fun,q0,bounds=(qlo,qhi),max_nfev=20,ftol=1e-8,xtol=1e-8,gtol=1e-8)
        q=best_nuisance[1];result['nuisance_nfev']=fit.nfev
        # Physical derivatives are in standardized prior coordinates. Nuisance
        # derivatives need only span their column space, so their units cancel.
        J=[]
        for k in active:
            delta=np.zeros(3);delta[k]=scale[k]*1e-3
            J.append((residual(theta0+delta,q)-residual(theta0-delta,q))/(2e-3))
        J=np.stack(J,axis=1);Jn=[]
        for k in range(len(q)):
            delta=np.zeros(len(q));delta[k]=1e-4
            Jn.append((residual(theta0,q+delta)-residual(theta0,q-delta))/(2e-4))
        Jn=np.stack(Jn,axis=1);projected=J-Jn@np.linalg.pinv(Jn,rcond=1e-8)@J
        _,singular,vh=np.linalg.svd(projected,full_matrices=False);keep=singular>max(1.,1e-3*singular[0]);rank=int(keep.sum())
        result.update(rank=rank,singular_values=singular.tolist(),initial_nuisance_fit_normalized_rmse=float(np.sqrt(best_nuisance[0]/len(data))))
        if rank==0:result['status']='no_resolved_local_direction';return finish()
        local_basis=np.eye(len(active)) if rank==len(active) else vh[keep].T
        basis=np.zeros((3,rank));basis[active]=local_basis
        if rank==len(active):
            clo=(LOW[active]-theta0[active])/scale[active];chi=(HIGH[active]-theta0[active])/scale[active]
        else:
            assert rank==1;clo=np.array([-np.inf]);chi=np.array([np.inf])
            for k in active:
                coefficient=scale[k]*basis[k,0]
                if abs(coefficient)<1e-14:continue
                a,b=sorted([(LOW[k]-theta0[k])/coefficient,(HIGH[k]-theta0[k])/coefficient]);clo[0]=max(clo[0],a);chi[0]=min(chi[0],b)
        if np.any(chi-clo<1e-12):result['status']='support_blocks_resolved_direction';return finish()
        lo=np.r_[clo,qlo];hi=np.r_[chi,qhi];initial=np.r_[np.zeros(rank),q]
        def joint_fun(value):
            c=value[:rank];nuisance=value[rank:];theta=theta0+scale*(basis@c)
            assert np.all(theta>=LOW-1e-8) and np.all(theta<=HIGH+1e-8)
            rr=residual(theta,nuisance);combined=np.r_[rr,c];cost=float(combined@combined)
            if cost<joint_best[0]:joint_best[:]=[cost,theta.copy(),rr.copy()]
            return combined
        fit=least_squares(joint_fun,initial,bounds=(lo,hi),max_nfev=80,ftol=1e-8,xtol=1e-8,gtol=1e-8)
        result['joint_nfev']=fit.nfev;result['status']='converged' if fit.success else 'optimizer_nfev_cap'
    except BudgetExceeded:result['status']='forward_call_cap'
    if joint_best[1] is not None:
        theta=joint_best[1];updated=natural(theta);inactive=[k for k in range(3) if k not in active];updated[inactive]=point[inactive]
        assert np.allclose(updated,operational(updated),atol=1e-12,rtol=0)
        # Preserve operational endpoints against exp/log rounding only.
        updated=operational(updated);delta=(theta-theta0)/scale;null=delta-basis@(basis.T@delta)
        result.update(point=updated,nullspace_update_norm=float(np.linalg.norm(null)),joint_objective=joint_best[0],
          final_normalized_fit_rmse=float(np.sqrt(np.mean(joint_best[2]**2))))
        assert result['nullspace_update_norm']<1e-8 and np.array_equal(updated[inactive],point[inactive])
    assert result['forward_calls']<=400
    return finish()
