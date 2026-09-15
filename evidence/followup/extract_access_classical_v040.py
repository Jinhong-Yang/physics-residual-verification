"""Hash-checked eight-frame Lucas-Kanade and pixel controls; no target access."""
from pathlib import Path
import json,hashlib,time
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
R=Path(__file__).resolve().parents[1];P=R/'paper/ieee_access_initial_submission/overleaf'
O=P/'evidence/followup';cv2.setNumThreads(1)

def extract(row):
    frames=[]
    for p,h in zip(row['frame_paths'],row['frame_sha256']):
        buf=(R/p).read_bytes();assert hashlib.sha256(buf).hexdigest()==h
        frames.append(cv2.imdecode(np.frombuffer(buf,np.uint8),cv2.IMREAD_GRAYSCALE))
    start=time.perf_counter();h,w=frames[0].shape
    x0,y0,x1,y1=np.asarray(row['initial_cue']['bbox_xyxy'],int)
    mask=np.zeros((h,w),np.uint8);mask[max(y0,0):min(y1,h),max(x0,0):min(x1,w)]=255
    points=cv2.goodFeaturesToTrack(frames[0],maxCorners=64,qualityLevel=.01,minDistance=2,mask=mask)
    if points is None:points=np.array([[[.5*(x0+x1),.5*(y0+y1)]]],np.float32)
    n0=len(points);centers=[np.array([.5*(x0+x1),.5*(y0+y1)])];scales=[];flow=[];failure=0
    for a,b in zip(frames[:-1],frames[1:]):
        new,status,error=cv2.calcOpticalFlowPyrLK(a,b,points,None,winSize=(21,21),maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
        ok=status.ravel().astype(bool)&np.isfinite(new).all((1,2))
        if not ok.any():
            failure+=1;shift=np.zeros(2);scales.append(1.)
        else:
            old=points[ok,0];nxt=new[ok,0];delta=nxt-old;shift=np.median(delta,0)
            spread=np.mean(np.linalg.norm(old-old.mean(0),axis=1));spread2=np.mean(np.linalg.norm(nxt-nxt.mean(0),axis=1))
            scales.append(spread2/max(spread,1.));flow.extend(np.linalg.norm(delta,axis=1)/np.hypot(h,w));points=new[ok]
        centers.append(centers[-1]+shift)
    pos=np.array(centers)/np.array([w,h]);d=np.diff(pos,axis=0);dd=np.diff(d,axis=0)
    cos=d[:,0]/np.maximum(np.linalg.norm(d,axis=1),1e-12)
    f=np.array(flow or [0.]);extra=[f.mean(),f.std(),np.quantile(f,.25),np.quantile(f,.75),failure/7,n0/64,len(points)/n0,np.mean(np.abs(frames[-1].astype(float)-frames[0]))/255]
    out=np.r_[d[:,0],d[:,1],dd[:,0],dd[:,1],cos,scales,extra]
    assert out.shape==(48,) and np.isfinite(out).all()
    duration=(time.perf_counter()-start)*1000
    pixel=np.stack([cv2.resize(im,(16,16),interpolation=cv2.INTER_AREA) for im in frames]).astype(np.float64).ravel()/255
    return out,pixel,duration,failure

def main():
    paths={'old900':R/'cache/com_observation_v7/INPUT_INDEX.json','v11':R/'data/canonical_confirmation_v11/INPUT_INDEX.json','confirmation':R/'data/canonical_confirmation_v23/INPUT_INDEX.json'}
    lookup={k:{e['id']:e for e in json.loads(p.read_text())['episodes']} for k,p in paths.items()}
    for cohort in ('development','confirmation'):
        dest=O/f'{cohort}_classical.npz'
        if dest.exists():print('existing',dest);continue
        # Only identifier array is accessed; physical labels are not loaded.
        with np.load(P/f'evidence/support/{cohort}.npz') as z:ids=z['ids']
        rows=[]
        for identifier in ids:
            src,raw=identifier.split('::') if '::' in identifier else ('confirmation',identifier)
            rows.append(lookup[src][raw])
        ans=[]
        with ThreadPoolExecutor(max_workers=4) as pool:
            for i,a in enumerate(pool.map(extract,rows),1):
                ans.append(a)
                if i%200==0:print(cohort,i,len(rows),flush=True)
        np.savez_compressed(dest,ids=ids,flow=np.stack([a[0] for a in ans]),pixels=np.stack([a[1] for a in ans]),milliseconds=np.array([a[2] for a in ans]),failed_transitions=np.array([a[3] for a in ans]))
        (O/f'{cohort}_classical_manifest.json').write_text(json.dumps({'episodes':len(rows),'frames':len(rows)*8,'frame_hashes_verified':True,'labels_loaded':False,'failed_transitions_retained':sum(a[3] for a in ans),'source_rows':rows},indent=2))
        print('saved',cohort,flush=True)
if __name__=='__main__':main()
