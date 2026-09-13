"""Observation-only local physical Jacobian for the posterior supplement.

The current transformed point and appearance prior scale are explicit inputs;
no simulator truth, target manifest, or learned model output labels are read.
"""
import numpy as np
from scipy.optimize import least_squares
from main3d_numeric import slide,vertical
from main3d_refinement import tracking,transformed_point
from main3d_evaluation import natural,operational


def local_jacobian(row,base,current_theta,prior_scale,tracked=None):
    theta=np.asarray(current_theta,dtype=float);scale=np.asarray(prior_scale,dtype=float)
    if theta.shape!=(3,) or scale.shape!=(3,) or not np.isfinite(theta).all() or not np.isfinite(scale).all() or not (scale>0).all():raise ValueError('Invalid current estimate or prior scale')
    result={'status':'unobservable','jacobian':np.zeros((1,3)),'calls':0,'uses_target_values':False,'linearization_theta':None,
      'threshold_absolute':1.,'threshold_relative':1e-3,'prior_scale':scale.copy(),'singular_values':[0.,0.,0.],'rank':0}
    protocol=row['protocol'];times=np.asarray(row.get('timestamps_s') or [],dtype=float)
    if protocol not in ['unforced_slide','multiple_forces','bounce'] or not row.get('camera_metric_calibrated') or row.get('camera') is None or row.get('occlusion'):
        return result
    if len(times)<3 or not np.isfinite(times).all() or not (np.diff(times)>0).all():result['status']='missing_or_invalid_time';return result
    if protocol=='multiple_forces':
        if row.get('force_N') is None or row.get('force_switch_s') is None:result['status']='missing_force';return result
        force=np.asarray(row['force_N'],dtype=float)
        if force.shape!=(2,) or not np.isfinite(force).all() or not np.isfinite(row['force_switch_s']):raise ValueError('Invalid supplied force')
    tracked=tracking(row,base) if tracked is None else tracked
    if tracked['status']!='tracked':result['status']=tracked['status'];return result
    bounce=protocol=='bounce';active=[2] if bounce else [0,1] if protocol=='multiple_forces' else [1];axis=1 if bounce else 0
    xz=np.asarray(tracked['xz'],dtype=float);sigma_xz=np.asarray(tracked['sigma_xz_m'],dtype=float)
    if xz.shape!=(len(times),2) or sigma_xz.shape!=xz.shape or not np.isfinite(xz).all() or not np.isfinite(sigma_xz).all() or not (sigma_xz>0).all():raise ValueError('Invalid tracked positions or observation noise')
    data=xz[:,axis];sigma=sigma_xz[:,axis]
    point=operational(natural(theta));linear=transformed_point(point)
    result.update(linearization_theta=linear.copy(),linearization_point_projected_to_ID_support=bool(not np.allclose(linear,theta,atol=1e-12,rtol=0)))
    if bounce:q0=np.array([data[0],0.,data.min()-.04,9.81]);lo=np.array([-.5,-2.,-.4,5.]);hi=np.array([2.,2.,.5,15.])
    else:q0=np.array([data[0],(data[1]-data[0])/(times[1]-times[0])]);lo=np.array([-5.,0.]);hi=np.array([5.,10.])
    q0=np.clip(q0,lo+1e-8,hi-1e-8);best=[float('inf'),q0.copy()]
    class BudgetExceeded(Exception):pass
    def residual(value,q):
        if result['calls']>=200:raise BudgetExceeded()
        result['calls']+=1;m,mu,e=natural(value)
        prediction=vertical(times,q[0],q[1],q[2],e,q[3]) if bounce else slide(times,q[0],q[1],mu,1/m,row['force_N'] if protocol=='multiple_forces' else [0,0],row['force_switch_s'] if protocol=='multiple_forces' else 1.)
        rr=(prediction-data)/sigma
        if not np.isfinite(rr).all():raise ValueError('Nonfinite simulated observation')
        return rr
    def nuisance(q):
        rr=residual(linear,q);cost=float(rr@rr)
        if cost<best[0]:best[:]=[cost,q.copy()]
        return rr
    try:
        fit=least_squares(nuisance,q0,bounds=(lo,hi),max_nfev=20,ftol=1e-8,xtol=1e-8,gtol=1e-8)
        q=best[1];J=np.zeros((len(times),3));Jn=[]
        for k in active:
            delta=np.zeros(3);delta[k]=scale[k]*1e-3
            J[:,k]=(residual(linear+delta,q)-residual(linear-delta,q))/(2e-3)
        for k in range(len(q)):
            delta=np.zeros(len(q));delta[k]=1e-4
            Jn.append((residual(linear,q+delta)-residual(linear,q-delta))/(2e-4))
        Jn=np.stack(Jn,axis=1);projected=J-Jn@np.linalg.pinv(Jn,rcond=1e-8)@J
        # Numerical nuisance projection can leave tiny entries in zero columns.
        # They are structurally zero: explicitly retain physical protocol zeros.
        projected[:,[k for k in range(3) if k not in active]]=0
        singular=np.linalg.svd(projected,compute_uv=False);rank=int((singular>max(1.,1e-3*singular[0])).sum())
        result.update(status='estimated' if fit.success else 'nuisance_fit_nfev_cap',jacobian=projected,nuisance_parameters=q,
          nuisance_normalized_rmse=float(np.sqrt(best[0]/len(times))),singular_values=singular.tolist(),rank=rank,
          nuisance_projection_residual_norm=float(np.linalg.norm(Jn.T@projected)),nuisance_nfev=int(fit.nfev))
    except BudgetExceeded:result.update(status='linearization_forward_cap',jacobian=np.zeros((1,3)),rank=0)
    return result
