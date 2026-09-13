"""RGB-only numerical control for the fixed-background 3D experiment."""
import numpy as np
import cv2
from PIL import Image
from scipy.optimize import least_squares
cv2.setNumThreads(1)

def center(path):
    a=np.asarray(Image.open(path).convert('RGB')).astype(float)
    # Fixed camera, uniform unshadowed background: row medians are an
    # observation-derived background estimate, not simulator segmentation.
    bg=np.median(a,axis=1,keepdims=True);mask=(np.max(np.abs(a-bg),axis=2)>12).astype(np.uint8)
    n,labels,stats,centroids=cv2.connectedComponentsWithStats(mask,8)
    if n<=1:return None
    index=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]));area=int(stats[index,cv2.CC_STAT_AREA])
    if area<8 or area>a.shape[0]*a.shape[1]*.15:return None
    return centroids[index]

def world_xz(pixel,camera):
    matrix=np.array(camera['projection_matrix']).reshape(4,4).T@np.array(camera['view_matrix']).reshape(4,4).T
    u,v=pixel;nx=2*(u+.5)/camera['width']-1;ny=1-2*(v+.5)/camera['height']
    a=np.array([[matrix[0,0]-nx*matrix[3,0],matrix[0,2]-nx*matrix[3,2]],
                [matrix[1,0]-ny*matrix[3,0],matrix[1,2]-ny*matrix[3,2]]])
    b=np.array([nx*matrix[3,3]-matrix[0,3],ny*matrix[3,3]-matrix[1,3]])
    return np.linalg.solve(a,b)

def slide(times,x0,v0,mu,invm,forces,switch):
    # Exact piecewise constant acceleration with a non-reversing/static stop.
    def advance(x,v,a,dt):
        duration=np.asarray(dt);moving=np.minimum(duration,v/max(-a,1e-15)) if a<0 else duration
        xx=x+v*moving+.5*a*moving*moving;vv=np.maximum(0,v+a*moving)
        return xx,vv
    t=np.asarray(times);a0=forces[0]*invm-mu*9.81;a1=forces[1]*invm-mu*9.81
    before=np.minimum(t,switch);x,v=advance(x0,v0,a0,before);xx,vv=advance(x,v,a1,np.maximum(0,t-switch))
    return xx

def vertical(times,z0,v0,floor,e,gravity):
    t=np.asarray(times);out=np.empty_like(t);disc=max(0,v0*v0+2*gravity*max(z0-floor,0));hit=(v0+np.sqrt(disc))/gravity
    before=t<=hit;out[before]=z0+v0*t[before]-.5*gravity*t[before]**2
    remaining=t[~before]-hit;velocity=e*np.sqrt(disc);z=np.zeros_like(remaining);active=np.ones(len(remaining),bool)
    for _ in range(20):
        period=2*velocity/gravity
        inside=active&(remaining<=period);z[inside]=floor+velocity*remaining[inside]-.5*gravity*remaining[inside]**2;active[inside]=False
        if not active.any():break
        remaining[active]-=period;velocity*=e
        if velocity<.01:break
    z[active]=floor;out[~before]=z;return out

def estimate(row,base,prior_mean,bounds):
    mean=np.array(prior_mean,dtype=float).copy()
    if row['protocol'] not in ['unforced_slide','multiple_forces','bounce'] or not row['camera_metric_calibrated'] or row['occlusion']:
        return {'mean':mean,'status':'prior_unobservable'}
    centers=[center(base/p) for p in row['frames']]
    if any(p is None for p in centers):return {'mean':mean,'status':'prior_tracking_failed'}
    xz=np.array([world_xz(p,row['camera']) for p in centers]);t=np.array(row['timestamps_s']);status='fit';residual=None
    if row['protocol'] in ['unforced_slide','multiple_forces']:
        x=xz[:,0];v0=float(np.clip((x[1]-x[0])/(t[1]-t[0]),.01,9.99));multiple=row['protocol']=='multiple_forces'
        if multiple and row['force_N'] is None:return {'mean':mean,'status':'prior_missing_force'}
        f=row['force_N'] if multiple else [0,0];switch=row['force_switch_s'] if multiple else 1.
        lo=[-5,0,bounds['mu'][0]];hi=[5,10,bounds['mu'][1]];initial=[x[0],v0,np.exp(mean[1])]
        if multiple:lo.append(1/bounds['mass'][1]);hi.append(1/bounds['mass'][0]);initial.append(np.exp(-mean[0]))
        fun=lambda q:slide(t,q[0],q[1],q[2],q[3] if multiple else 1.,f,switch)-x
        fit=least_squares(fun,np.clip(initial,np.array(lo)+1e-8,np.array(hi)-1e-8),bounds=(lo,hi),max_nfev=200)
        mean[1]=np.log(fit.x[2])
        if multiple:mean[0]=-np.log(fit.x[3])
        residual=fit.fun;status='fit' if fit.success else 'fit_nonconverged'
    else:
        z=xz[:,1];fits=[]
        # Effective gravity absorbs the small constant depth/centroid scale
        # mismatch of the observed object point. It is a nuisance, not a label.
        lo=[-.5,-2,-.4,bounds['e'][0],5];hi=[2,2,.5,bounds['e'][1],15]
        for start_e in [.3,.6,.85]:
            initial=[z[0],0,min(z)-.04,start_e,9.81];initial=np.clip(initial,np.array(lo)+1e-8,np.array(hi)-1e-8)
            fit=least_squares(lambda q:vertical(t,*q)-z,initial,bounds=(lo,hi),max_nfev=250)
            fits.append(fit)
        fit=min(fits,key=lambda f:np.mean(f.fun**2));ee=fit.x[3];mean[2]=np.log(ee/(1-ee));residual=fit.fun;status='fit' if fit.success else 'fit_nonconverged'
    return {'mean':mean,'status':status,'tracking_xz':xz,'fit_rmse_m':float(np.sqrt(np.mean(residual**2))),
      'limitation':'Uniform-background RGB tracking, planar center approximation, point estimate without calibrated posterior.'}
