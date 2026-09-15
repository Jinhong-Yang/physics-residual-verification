"""Independent 40-family RGB calibration with separate generation/inference/scoring."""
from pathlib import Path
import sys,json,hashlib,datetime,os,argparse,time
os.environ.update(HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R));P=R/'paper/ieee_access_initial_submission/overleaf'
D=R/'data/access_calibration_v040';F=P/'evidence/followup';D.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2)
def plan():
    p=D/'PLAN.json'
    if not p.exists():
        spec={'seed':20260916,'families':40,'episodes_per_family':15,'purpose':'calibration only; no model or hyperparameter fit',
          'geometry_seeds':list(range(940000,940040)),'physical_seeds':list(range(950000,950040)),
          'prediction':'original frozen full/numeric; beta0 covariance alternative fixed by development Pareto, no test coverage selection',
          'calibration_score':'sqrt(active Mahalanobis / chi2_0.90(k)); one uniformly seed-selected episode/regime per family',
          'quantile':'37th of 40 scores (ceil((40+1)*0.9)); additionally family-maximum 37th order statistic',
          'evaluation':'previous 72 families, descriptive; report protocol and marginal coverage and width; no targeting coverage',
          'seed_exclusion':'geometry hashes checked against existing manifests before rendering',
          'stages':['generate isolated labels','predict with target-open guard','seal predictions','open calibration labels'],
          'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
        write(p,spec);(D/'PLAN.sha256').write_text(sha(p)+'  PLAN.json\n')
    return json.loads(p.read_text())
def generate():
    spec=plan()
    import pybullet as bullet
    from bullet_scene import geometry,episode,camera
    from canonical_confirmation_data_v11 import sample_parameters,sampled_slide_states,times,observation
    from scripts.prepare_canonical_bounce_overlay_v7 import render_bounce
    from scripts.extract_com_observation_features_v4 import initial_cue
    from scripts.inventory_existing_procedural_geometry_v11 import shape_signatures
    excluded=set()
    for name in ('canonical_confirmation_v11','canonical_confirmation_v16','canonical_confirmation_v23'):
        for p in (R/'data'/name).rglob('geometry.json'):
            excluded.add(shape_signatures(json.loads(p.read_text()))['physical_shape_sha256'])
    for p in (R/'data/main3d_v1').rglob('geometry.json'):
        excluded.add(shape_signatures(json.loads(p.read_text()))['physical_shape_sha256'])
    obs=[];labels=[];shapes=[];started=time.perf_counter();bullet.connect(bullet.DIRECT)
    try:
        for fi,(gs,ps) in enumerate(zip(spec['geometry_seeds'],spec['physical_seeds'])):
            family=f'access_cal40_f{fi:03d}';shape=geometry(gs);sig=shape_signatures(shape)
            assert sig['physical_shape_sha256'] not in excluded
            excluded.add(sig['physical_shape_sha256']);shapes.append({'family':family,'geometry':shape,**sig})
            parameters=sample_parameters(ps);cam=camera()
            for variant in range(5):
                theta=parameters[variant];color=list(map(float,np.roll(shape['rgba'][:3],1)))+[1.] if variant==4 else None
                for protocol in ('multiple_forces','unforced_slide','bounce'):
                    ident=hashlib.sha256(f'cal040/{family}/{variant}/{protocol}'.encode()).hexdigest()[:24]
                    group={'id':ident,'family':family,'variant':variant,'appearance_variant':int(variant==4),'object':f'{family}_physical_{0 if variant==4 else variant}','protocol':protocol,'split':'calibration'}
                    if protocol=='bounce':frames,_,visible=render_bounce(shape,theta,cam,times(protocol),rgba=color)
                    else:frames=episode(shape,theta['mass_kg'],theta['dynamic_friction'],theta['restitution'],protocol,theta['forces'],cam=cam,rgba=color)['frames']
                    paths=[];hashes=[]
                    for j,im in enumerate(frames):
                        p=D/'frames'/ident/f'{j}.png';p.parent.mkdir(parents=True,exist_ok=True)
                        with p.open('xb') as f:im.save(f,format='PNG')
                        paths.append(p.relative_to(R).as_posix());hashes.append(sha(p))
                    obs.append(observation(group,paths,hashes,cam,initial_cue(np.asarray(frames[0])),theta['forces']))
                    labels.append({'id':ident,**{k:theta[k] for k in ('mass_kg','dynamic_friction','restitution')}})
            print('calibration generated family',fi+1,'seconds',round(time.perf_counter()-started),flush=True)
    finally:bullet.disconnect()
    obs.sort(key=lambda x:x['id']);labels.sort(key=lambda x:x['id'])
    write(D/'INPUT_INDEX.json',{'episodes':obs,'plan_sha256':sha(D/'PLAN.json')})
    (D/'evaluator_only').mkdir(exist_ok=True);write(D/'evaluator_only/targets.json',labels)
    write(D/'GEOMETRY.json',shapes)
    write(D/'OBSERVATION_SEAL.json',{'plan_sha256':sha(D/'PLAN.json'),'index_sha256':sha(D/'INPUT_INDEX.json'),'geometry_disjoint':True,'episodes':len(obs),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
def predict():
    spec=plan();episodes=json.loads((D/'INPUT_INDEX.json').read_text())['episodes']
    # Generator-only targets cannot be opened by this process.
    def guard(event,args):
        if event=='open' and args and not isinstance(args[0],int) and 'evaluator_only' in Path(args[0]).parts:raise PermissionError('calibration targets sealed during inference')
    sys.addaudithook(guard)
    import torch
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    from scripts.extract_canonical_confirmation_inputs_v11 import forward_first_image,MODEL,PRIOR_CHECKPOINT
    from projected_posterior import AppearancePrior
    from canonical_confirmation_inputs_v11 import tracking_row,pack_inputs
    from reliability_observation_v7 import tracking,extract_observation
    from scripts import stream_com_internal_features_v16 as stream
    from scripts.predict_com_physics_v16 import project_inverse,fixed_robust_fits
    from scripts import train_com_physics_internal_readout_v12 as base
    from sequential_robust_readout_v22 import load_checkpoint,predict as residual_predict
    torch.set_num_threads(2)
    model=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda().eval().requires_grad_(False)
    proc=AutoProcessor.from_pretrained(MODEL,local_files_only=True);cache={};features=[]
    for i,e in enumerate(episodes):
        h=e['frame_sha256'][0]
        if h not in cache:cache[h]=forward_first_image(torch,model,proc,R/e['frame_paths'][0])['feature']
        features.append(cache[h])
        if i%100==0:print('calibration prior',i,len(episodes),flush=True)
    del model;torch.cuda.empty_cache()
    x=torch.stack(features).float();x=torch.nn.functional.layer_norm(x,(x.shape[1],),eps=1e-5)
    prior=AppearancePrior(x.shape[1]);prior.load_state_dict(torch.load(PRIOR_CHECKPOINT,map_location='cpu',weights_only=True)['state']);prior.eval()
    with torch.no_grad():means,scales=prior(x)
    records=[];tracks=[]
    for i,(e,m,s) in enumerate(zip(episodes,means.numpy(),scales.numpy())):
        row=tracking_row(e);track=tracking(row,R);records.append(extract_observation(row,R,m,s,track));tracks.append(track)
        if i%100==0:print('calibration RGB',i,len(episodes),flush=True)
    values,diagnostic=pack_inputs(episodes,records,tracks)
    if not (D/'INPUT_DIAGNOSTICS.json').exists():write(D/'INPUT_DIAGNOSTICS.json',diagnostic)
    assert diagnostic['ready_for_frozen_head'], 'Retain failure; cannot exclude bad calibration rows'
    z=stream.run_rows(episodes,values,stream.load_models(),progress=True)
    np.savez_compressed(D/'observation_values.npz',**values)
    np.savez_compressed(D/'streamed_features.npz',**z)
    qwen={'xz':project_inverse(z['pixels'],episodes)};static={'xz':project_inverse(z['static_pixels'],episodes)}
    feat,_=base.raw_features(values,{'pooled':z['pooled']},qwen,static)
    batch=base.tensor_batch(values)
    anchors,_=fixed_robust_fits(batch,'huber')
    ck=load_checkpoint(R/'results/com_observation_v22/final_development_fit_v1/CHECKPOINT.npz')
    saved={'ids':values['ids'],'family_ids':np.array([e['family'] for e in episodes]),'protocol':values['protocol'],'active':values['active']}
    for mode in (False,True):
        out=residual_predict(ck,feat.numpy(),values['active'],values['protocol'],anchors,include_vlm=mode)
        for r,a in out.items():
            for k in ('mean','covariance'):saved[('full' if mode else 'numeric')+'__'+r+'__'+k]=a[k]
    for r,a in anchors.items():
        for k in ('mean','covariance'):saved['anchor__'+r+'__'+k]=a[k]
    np.savez_compressed(F/'calibration_predictions.npz',**saved)
    write(F/'CALIBRATION_PREDICTION_SEAL.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'prediction_sha256':sha(F/'calibration_predictions.npz'),'targets_opened':False,'episodes':len(episodes),'plan_sha256':sha(D/'PLAN.json')})
    print('calibration predictions sealed',flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['plan','generate','predict']);a=ap.parse_args()
    {'plan':plan,'generate':generate,'predict':predict}[a.stage]()
