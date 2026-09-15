"""Reconstruct frozen real-video features and correct their pixel/noise units."""
from pathlib import Path
import sys,json,hashlib,os
os.environ.update(HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts import stream_com_internal_features_v16 as stream
from scripts.extract_idpp_real_visual_scores_v24 import overlay_geometry,video_frames
from sequential_robust_readout_v22 import load_checkpoint,load_model,ridge_predict
P=R/'paper/ieee_access_initial_submission/overleaf';F=P/'evidence/followup'
src=R/'results/com_observation_v25/idpp_real_friction_test3_v1'
def main():
    plan=json.loads((src/'PLAN.json').read_text());old=np.load(src/'VISUAL_SCORES.npz');episodes=[]
    for record in plan['records']:
        rgb=video_frames(R/record['rgb_path'],[0])[0];cue=video_frames(R/record['cue_path'],[0])[0]
        _,ellipse=overlay_geometry(rgb,cue);x0,y0,x1,y1=ellipse;w=x1-x0;h=y1-y0
        box=np.array([int(x0+.25*w),int(y0+.20*h),max(2,int(.5*w)),max(2,int(.6*h))],float)
        bbox=np.r_[box[:2],box[:2]+box[2:]]*np.array([280/rgb.shape[1],280/rgb.shape[0]]*2)
        paths=[(src/'observation_frames'/record['id']/f'{i}.png').relative_to(R).as_posix() for i in range(8)]
        episodes.append({'id':record['id'],'frame_paths':paths,'frame_sha256':[stream.sha(R/p) for p in paths],'initial_cue':{'bbox_xyxy':bbox.tolist()}})
    values={'pixels':old['cue_centers'],'times':np.tile(plan['frames']['timestamps_s'],(33,1))}
    result=stream.run_rows(episodes,values,stream.load_models(),progress=True)
    dynamic=np.einsum('nti,ni->nt',result['pixels']-result['static_pixels'],old['axes'])
    static=np.einsum('nti,ni->nt',result['static_pixels']-old['cue_centers'],old['axes'])
    visual=np.c_[result['pooled'].mean(1),np.stack((dynamic,static),axis=2).reshape(33,16)].astype(float)
    model=load_model(load_checkpoint(R/'results/com_observation_v22/final_development_fit_v1/CHECKPOINT.npz'),'visual','generalized_translation','unforced_slide')
    original_replay=.25*ridge_predict(model,visual).ravel()
    t=np.array(plan['frames']['timestamps_s']);design=np.c_[np.ones(8),t,.5*t*t]
    position=np.einsum('nti,ni->nt',old['cue_centers'],old['axes'])
    residual=position-position@np.linalg.pinv(design).T@design.T
    sigma=np.maximum(1.,1.4826*np.median(abs(residual-np.median(residual,axis=1)[:,None]),axis=1))
    corrected=visual.copy();corrected[:,32:]/=sigma[:,None]
    pooled=corrected.copy();pooled[:,32:]=model['mean_x'][32:]
    motion=corrected.copy();motion[:,:32]=model['mean_x'][:32]
    saved={'ids':old['ids'],'family_ids':old['family_ids'],'visual_raw':visual,'visual_corrected':corrected,'sigma_pixels':sigma,
           'combined':.25*ridge_predict(model,corrected).ravel(),'pooled':.25*ridge_predict(model,pooled).ravel(),
           'position':.25*ridge_predict(model,motion).ravel(),'raw_replay':original_replay,
           'kinematic_dimensionless':old['kinematic_deceleration_score']/sigma}
    np.savez_compressed(F/'real_unit_corrected.npz',**saved)
    error=float(np.max(abs(original_replay-old['v22_visual_score'])))
    (F/'REAL_EXTRACTION.json').write_text(json.dumps({'videos':33,'original_raw_score_max_error':error,'same_saved_tracker_centers':True,
        'tracking_sigma':'quadratic detrended projected centers, MAD x1.4826, floor 1 pixel',
        'target_loaded':False,'unit_correction':'both correction channels divided by estimated pixel noise',
        'existing_cohort_post_hoc':True,'sigma_min':float(sigma.min()),'sigma_max':float(sigma.max()),
        'feature_sha256':stream.sha(F/'real_unit_corrected.npz')},indent=2))
    print('real features reconstructed; raw score max difference',error,flush=True)
if __name__=='__main__':main()
