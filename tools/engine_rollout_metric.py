"""Post hoc PyBullet controlled-sphere proxy, not original-asset validation."""
from extended_common import *
from canonical_physics_metrics_v7 import _parameters
import pybullet as pb
from scipy.stats import pearsonr,spearmanr
import time
def main():
    start=time.perf_counter();_,v=data();pb.connect(pb.DIRECT);pb.setGravity(0,0,-9.81);pb.setTimeStep(1/480)
    pb.setPhysicsEngineParameter(numSolverIterations=100,restitutionVelocityThreshold=0,deterministicOverlappingPairs=1)
    plane=pb.createMultiBody(0,pb.createCollisionShape(pb.GEOM_PLANE));pb.changeDynamics(plane,-1,lateralFriction=1,restitution=1)
    radius=.01;shape=pb.createCollisionShape(pb.GEOM_SPHERE,radius=radius);body=pb.createMultiBody(1,shape,basePosition=[0,0,1]);saved={};cache={}
    def rollout(theta,protocol):
        nonlocal body
        key=(tuple(theta),protocol)
        if key in cache:return cache[key]
        # Recreate the body so contact warm-start state cannot cross episodes.
        pb.removeBody(body);body=pb.createMultiBody(1,shape,basePosition=[0,0,1])
        m,mu,e=_parameters(np.asarray(theta)[None])[0][0];bounce=protocol=='bounce'
        pb.changeDynamics(body,-1,mass=float(m),localInertiaDiagonal=[0,0,0],lateralFriction=float(mu),restitution=float(e),linearDamping=0,angularDamping=0,rollingFriction=0,spinningFriction=0,activationState=pb.ACTIVATION_STATE_DISABLE_SLEEPING)
        pb.resetBasePositionAndOrientation(body,[0,0,radius+(.85 if bounce else 0)],[0,0,0,1]);pb.resetBaseVelocity(body,[0 if bounce else .25 if protocol=='multiple_forces' else 1.,0,0],[0,0,0]);points=[]
        for step in range(480):
            if protocol=='multiple_forces':
                pos=pb.getBasePositionAndOrientation(body)[0];pb.applyExternalForce(body,-1,[.45 if step<168 else 1.1,0,0],pos,pb.WORLD_FRAME)
            pb.stepSimulation()
            if (step+1)%60==0:
                pos=pb.getBasePositionAndOrientation(body)[0];points.append(pos[2]-radius if bounce else pos[0])
        cache[key]=np.array(points);return cache[key]
    truth=np.stack([rollout(theta,p) for theta,p in zip(v['truth'],v['protocol'])]);rows=[];protocols=[]
    for name in ('huber_1.345_rgb','v22_numeric','v22_vlm'):
        eps=[]
        for r in REGIMES:
            pred=np.stack([rollout(theta,p) for theta,p in zip(v[f'{name}__{r}__mean'],v['protocol'])]);err=abs(pred-truth).mean(1);eps.append(err);saved[name+'__'+r]=pred
            for p in SUBSPACES:
                val=np.array([err[(v['family_ids']==f)&(v['protocol']==p)].mean() for f in sorted(set(v['family_ids']))]);protocols.append({'method':name,'regime':r,'protocol':p,'engine_mm':float(val.mean()*1000)})
        family=np.mean([macro(x,v) for x in eps],0);analytic,_=action(v,{r:v[f'{name}__{r}__mean'] for r in REGIMES});saved[name+'__family_engine']=family;saved[name+'__family_analytic']=analytic
        rows.append({'method':name,'engine_mm':float(family.mean()*1000),'analytic_mm':float(analytic.mean()*1000),'pearson':float(pearsonr(family,analytic).statistic),'spearman':float(spearmanr(family,analytic).statistic)})
        print(rows[-1],flush=True)
    saved['truth_trajectory']=truth;np.savez_compressed(O/'engine_predictions.npz',**saved);csvout('engine_protocols.csv',protocols)
    write('ENGINE.json',{'scope':'same actions and coordinates, controlled nonrotating sphere; no original-asset rollout, no real-world validation','pybullet_api':pb.getAPIVersion(),'dt':1/480,'solver_iterations':100,'results':rows,'full_minus_huber':contrast(v,saved['v22_vlm__family_engine'],saved['huber_1.345_rgb__family_engine']),'seconds':time.perf_counter()-start});pb.disconnect()
if __name__=='__main__':main()
