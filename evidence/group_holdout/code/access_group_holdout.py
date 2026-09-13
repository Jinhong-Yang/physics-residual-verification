"""Prospective new-episode, double-material-held-out PhysProbe evaluation.

Historical model design used the same material bank; independence here concerns
the new parameter-fitting/evaluation partitions, not unseen object assets.
"""
import argparse
import collections
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from urllib.request import urlopen
import numpy as np

R=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(R),str(R/'scripts')]
O=R/'results/access_group_holdout_v1'
D=R/'data/access_group_holdout_v1'
REV='b679fcf0a800d60f0d5064304e0e8777392f34dd'
SEED=2026091307
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def material(row,prefix):return [row[prefix+'_static_friction'],row[prefix+'_dynamic_friction']]
def prepare():
    if (O/'PLAN.json').exists():raise FileExistsError('Preserve the first group split')
    metadata=R/'data/physprobe_push_v31/evaluator_only/episodes.jsonl'
    seal=read(R/'data/physprobe_push_v31/DOWNLOAD_SEAL.json')
    assert sha(metadata)==seal['target_sha256']
    rows=[json.loads(s) for s in metadata.read_text().splitlines()]
    old=read(R/'data/physprobe_push_v31/PLAN.json')
    excluded={r['episode_index'] for r in old['records']}|set(range(10))
    objects=sorted({tuple(material(r,'object_0')) for r in rows})
    surfaces=sorted({tuple(material(r,'surface')) for r in rows})
    assert len(objects)==len(surfaces)==16
    def assignment(values,name):
        order=sorted(range(len(values)),key=lambda i:hashlib.sha256(f'{SEED}:{name}:{values[i]}'.encode()).hexdigest())
        return order[:8],order[8:]
    objtrain,objtest=assignment(objects,'object')
    surftrain,surftest=assignment(surfaces,'surface')
    records=[];discarded=collections.Counter()
    oldrole=collections.defaultdict(lambda:{'objects':set(),'surfaces':set()})
    for e in read(R/'data/physprobe_push_v31/observations_v1/INPUT_INDEX.json')['episodes']:
        row=next(v for v in rows if v['episode_index']==e['episode_index'])
        oldrole[e['role']]['objects'].add(objects.index(tuple(material(row,'object_0'))))
        oldrole[e['role']]['surfaces'].add(surfaces.index(tuple(material(row,'surface'))))
    for row in rows:
        i=row['episode_index'];a=objects.index(tuple(material(row,'object_0')));b=surfaces.index(tuple(material(row,'surface')))
        if i in excluded:discarded['previous_episode']+=1;continue
        if a in objtrain and b in surftrain:role='train'
        elif a in objtest and b in surftest:role='evaluation'
        else:discarded['mixed_material_partition']+=1;continue
        stem=f'episode_{i:06d}'
        records.append({'episode_index':i,'id':stem,'role':role,'object_material':a,'surface_material':b,
                        'video_remote':f'push/videos/chunk-{i//1000:03d}/observation.images.image_0/{stem}.mp4',
                        'parquet_remote':f'push/data/chunk-{i//1000:03d}/{stem}.parquet'})
    counts=dict(collections.Counter(x['role'] for x in records))
    assert min(counts.values())>=100
    plan={'schema':'access_double_material_holdout_v1','dataset_repository':'leesangoh/dynamics-probing','revision':REV,
          'seed':SEED,'group_unit':'exact (static friction, dynamic friction) material tuple, separately for cube and surface',
          'not_object_asset_holdout':True,'object_types':dict(collections.Counter(x['object_0_type'] for x in rows)),
          'object_materials':objects,'surface_materials':surfaces,'object_train':objtrain,'object_evaluation':objtest,
          'surface_train':surftrain,'surface_evaluation':surftest,'counts':counts,'discarded':dict(discarded),
          'records':records,'old_eligible_material_counts':{k:{a:len(b) for a,b in v.items()} for k,v in oldrole.items()},
          'old_train_evaluation_object_overlap':len(oldrole['development']['objects']&oldrole['confirmation']['objects']),
          'old_train_evaluation_surface_overlap':len(oldrole['development']['surfaces']&oldrole['confirmation']['surfaces']),
          'metadata_sha256':sha(metadata),'old_plan_sha256':sha(R/'data/physprobe_push_v31/PLAN.json'),
          'source_sha256':sha(Path(__file__)),
          'features':'unchanged eight-frame original V31 observation and frozen Qwen/decoder pipeline; full-video visibility selection retained',
          'model':{'numeric_view':'numeric_summary','numeric_lambda':10.0,'robust':True,
                   'visual_view':'visual_temporal_stats','visual_lambda':1000.0,'gate_lambda':0.1,
                   'gate_crossfit':'all 8x8 material pairs; train components after excluding BOTH held pair materials',
                   'tuning':False,'original_checkpoint_replaced':False},
          'historical_design_disclosure':'Architecture/hyperparameters were selected earlier using episodes from this material bank; new fitting partitions are disjoint, not historically untouched material discovery.',
          'primary':'gated minus numeric vector MAE, equal weight to each observed object-surface material cell',
          'interval':'20000 two-way pigeonhole resamples of eight object and eight surface materials; existing cell means, equal cell weighting',
          'pass_rule':'negative primary difference AND two-sided 95% upper endpoint below zero AND negative macro difference in at least 5/8 object and 5/8 surface groups',
          'minimum_eligible_evaluation':100,'minimum_eligible_train':100,'minimum_test_levels_each_axis':8,
          'reference_diagnostics':['episode micro MAE','raw residual MAE','per-material marginal differences','coverage of material pairs','exclusion counts by material'],
          'outcomes_read_before_plan':False,'metadata_read_for_grouping':True,'new_videos_read_before_plan':False,
          'evaluation_trajectories_downloaded_only_after_prediction_seal':True,
          'no_reselection_on_evaluation':True,'new_target_definition':'xy object position at onset+90 minus onset; same 50 fps offline horizon'}
    save(O/'PLAN.json',plan)
    print(json.dumps({k:plan[k] for k in ('counts','discarded','old_eligible_material_counts','old_train_evaluation_object_overlap','old_train_evaluation_surface_overlap')}),flush=True)
def download_file(remote,destination):
    if destination.exists():return
    destination.parent.mkdir(parents=True,exist_ok=True)
    url=f'https://huggingface.co/datasets/leesangoh/dynamics-probing/resolve/{REV}/{remote}'
    for attempt in range(3):
        try:
            with urlopen(url,timeout=45) as response,destination.with_suffix('.partial').open('wb') as f:
                while block:=response.read(1024*1024):f.write(block)
            destination.with_suffix('.partial').replace(destination);return
        except Exception:
            if attempt==2:raise
            time.sleep(1+attempt)
def download_videos():
    plan=read(O/'PLAN.json');records=plan['records'];receipts=[]
    def job(e):
        p=D/'videos'/(e['id']+'.mp4');download_file(e['video_remote'],p)
        return {'id':e['id'],'path':p.relative_to(R).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i,future in enumerate(as_completed([pool.submit(job,e) for e in records]),1):
            receipts.append(future.result())
            if i%64==0 or i==len(records):print(f'VIDEOS {i}/{len(records)}',flush=True)
    save(D/'DOWNLOAD.json',{'plan_sha256':sha(O/'PLAN.json'),'files':sorted(receipts,key=lambda x:x['id'])})
def install_no_targets():
    def guard(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0]))
            if p.suffix=='.parquet' or 'evaluator_only' in p.parts:raise PermissionError('No trajectory/target access in observation/feature extraction')
    sys.addaudithook(guard)
def observe():
    from PIL import Image
    import cv2
    from extract_physprobe_observations_v31 import read_video,locate
    install_no_targets();plan=read(O/'PLAN.json');receipts={x['id']:x for x in read(D/'DOWNLOAD.json')['files']}
    episodes=[];excluded=[];arrays=collections.defaultdict(list)
    for i,e in enumerate(plan['records'],1):
        path=R/receipts[e['id']]['path'];assert sha(path)==receipts[e['id']]['sha256']
        try:
            frames=read_video(path);onset,ix,centers,boxes,areas,axis=locate(frames,[-21,-16,-11,-6,-1,4,9,14],90)
        except ValueError as err:excluded.append({**e,'reason':str(err)});continue
        scale=np.array([280/frames[0].shape[1],280/frames[0].shape[0]])
        pixels=centers[ix]*scale;box=boxes[ix]*np.tile(scale,2);axis=axis*scale;axis/=np.linalg.norm(axis)
        paths=[];hashes=[]
        for k,j in enumerate(ix):
            p=D/'frames'/e['id']/f'{k}.png';p.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(cv2.cvtColor(frames[j],cv2.COLOR_BGR2RGB)).resize((280,280),Image.Resampling.BILINEAR).save(p)
            paths.append(p.relative_to(R).as_posix());hashes.append(sha(p))
        episodes.append({**e,'onset_index':int(onset),'observation_indices':ix.tolist(),'target_index':int(onset+90),
                         'frame_paths':paths,'frame_sha256':hashes,'initial_cue':{'bbox_xyxy':box[0].tolist()}})
        for k,v in {'pixels':pixels,'times':np.arange(8)*.1,'tracked_along':(pixels-pixels[0])@axis,
                    'tracked_perpendicular':(pixels-pixels[0])@np.array([-axis[1],axis[0]]),
                    'object_size_px':np.sqrt(np.maximum((box[:,2]-box[:,0])*(box[:,3]-box[:,1]),1))}.items():arrays[k].append(v)
        if i%64==0:print(f'OBSERVATIONS {i}/{len(plan["records"])}',flush=True)
    for k,field in [('ids','id'),('roles','role'),('episode_indices','episode_index'),('object_material','object_material'),('surface_material','surface_material')]:arrays[k]=[e[field] for e in episodes]
    np.savez_compressed(D/'inputs.npz',**{k:np.asarray(v) for k,v in arrays.items()})
    save(D/'INPUT_INDEX.json',{'episodes':episodes,'excluded':excluded})
    report={'plan_sha256':sha(O/'PLAN.json'),'inputs_sha256':sha(D/'inputs.npz'),'index_sha256':sha(D/'INPUT_INDEX.json'),
            'eligible':dict(collections.Counter(e['role'] for e in episodes)),
            'excluded':dict(collections.Counter(e['role'] for e in excluded)),'targets_read':False}
    save(D/'OBSERVATION_SEAL.json',report);print(json.dumps(report),flush=True)
def features():
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    install_no_targets()
    import torch
    from scripts import stream_com_internal_features_v16 as stream
    torch.set_num_threads(2)
    obs=read(D/'INPUT_INDEX.json')['episodes'];inputs=load(D/'inputs.npz')
    result=stream.run_rows(obs,inputs,stream.load_models(),progress=True)
    axes=inputs['pixels'][:,-1]-inputs['pixels'][:,0];axes/=np.linalg.norm(axes,axis=1,keepdims=True)
    dynamic=np.einsum('nti,ni->nt',result['pixels']-result['static_pixels'],axes)
    static=np.einsum('nti,ni->nt',result['static_pixels']-inputs['pixels'],axes)
    np.savez_compressed(D/'FEATURES.npz',ids=inputs['ids'],roles=inputs['roles'],episode_indices=inputs['episode_indices'],
                        visual_features=np.concatenate((result['pooled'].mean(1),dynamic,static),axis=1).astype(np.float32),
                        pooled_evidence=result['pooled'],qwen_pixels=result['pixels'],qwen_static_pixels=result['static_pixels'])
    save(D/'FEATURE_SEAL.json',{'features_sha256':sha(D/'FEATURES.npz'),'inputs_sha256':sha(D/'inputs.npz'),
                              'frames':len(obs)*8,'targets_read':False,'elapsed_s':result['elapsed_s'],
                              'decoder_sha256':sha(R/'results/com_observation_v9/decoder/continuation/qwen_warp_joint/epoch10.pt')})
    print('FEATURES SEALED',flush=True)
def targets(role):
    import pyarrow.parquet as pq
    obs=[x for x in read(D/'INPUT_INDEX.json')['episodes'] if x['role']==role]
    values=[];receipts=[]
    def fetch(e):
        p=D/'trajectories'/role/(e['id']+'.parquet');download_file(e['parquet_remote'],p);return e,p
    with ThreadPoolExecutor(max_workers=8) as pool:downloaded=dict((e['id'],p) for e,p in pool.map(fetch,obs))
    for e in obs:
        p=downloaded[e['id']];table=pq.read_table(p,columns=['physics_gt.object_position'])
        position=np.asarray(table['physics_gt.object_position'].to_pylist(),dtype=np.float64)
        values.append(position[e['target_index'],:2]-position[e['onset_index'],:2])
        receipts.append({'id':e['id'],'sha256':sha(p)})
    return np.asarray(values),receipts
def fit_predict():
    from develop_physprobe_vector_readout_v35 import fit,predict,make_views
    from develop_physprobe_confidence_gate_v38 import quality_base,gate_features,optimal_gate,clipped_update
    plan=read(O/'PLAN.json');inputs=load(D/'inputs.npz');feature=load(D/'FEATURES.npz')
    assert np.array_equal(inputs['ids'],feature['ids'])
    train=np.flatnonzero(inputs['roles']=='train');test=np.flatnonzero(inputs['roles']=='evaluation')
    assert len(train)>=100 and len(test)>=100
    for field in ('object_material','surface_material'):
        assert not set(inputs[field][train])&set(inputs[field][test])
        assert len(set(inputs[field][test]))==8
    y,receipts=targets('train')
    np.savez_compressed(O/'TRAIN_TARGETS.npz',ids=inputs['ids'][train],target=y)
    # No code in fitting can read any evaluation parquet, even if one exists.
    def guard(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0]))
            if p.suffix=='.parquet' and 'evaluation' in p.parts:raise PermissionError('Evaluation targets forbidden until predictions sealed')
    sys.addaudithook(guard)
    view=make_views(feature,inputs,np.arange(len(inputs['ids'])))
    nx=view['numeric_summary'];vx=view['visual_temporal_stats'];base=quality_base(feature,inputs,np.arange(len(inputs['ids'])))
    n_oof=np.empty_like(y);u_oof=np.empty_like(y);fitted=[]
    obj=inputs['object_material'][train];surf=inputs['surface_material'][train]
    for a,b in sorted(set(zip(obj.tolist(),surf.tolist()))):
        held=(obj==a)&(surf==b);keep=(obj!=a)&(surf!=b)
        assert keep.sum()>=30
        n=fit(nx[train][keep],y[keep],10.,True);nfit=predict(n,nx[train][keep])
        v=fit(vx[train][keep],y[keep]-nfit,1000.,False)
        cap=max(float(np.std(np.linalg.norm(y[keep],axis=1))),1e-6)
        n_oof[held]=predict(n,nx[train][held]);u_oof[held]=clipped_update(v,vx[train][held],cap)
        fitted.append({'held_object':int(a),'held_surface':int(b),'train_rows':int(keep.sum()),'held_rows':int(held.sum())})
    gate=fit(gate_features(base[train],n_oof,u_oof),optimal_gate(n_oof,u_oof,y),.1,False)
    n=fit(nx[train],y,10.,True);v=fit(vx[train],y-predict(n,nx[train]),1000.,False)
    cap=max(float(np.std(np.linalg.norm(y,axis=1))),1e-6)
    numeric=predict(n,nx[test]);u=clipped_update(v,vx[test],cap)
    g=np.clip(predict(gate,gate_features(base[test],numeric,u)).reshape(-1),0,1)
    np.savez_compressed(O/'CHECKPOINT.npz',clip=np.asarray(cap),**{f'{name}_{k}':a for name,m in [('numeric',n),('visual',v),('gate',gate)] for k,a in m.items()})
    np.savez_compressed(O/'PREDICTIONS.npz',ids=inputs['ids'][test],numeric=numeric,raw=numeric+u,gated=numeric+g[:,None]*u,gate=g,
                        object_material=inputs['object_material'][test],surface_material=inputs['surface_material'][test])
    np.savez_compressed(O/'TRAIN_OOF.npz',ids=inputs['ids'][train],numeric=n_oof,update=u_oof)
    save(O/'PREDICTION_SEAL.json',{'plan_sha256':sha(O/'PLAN.json'),'checkpoint_sha256':sha(O/'CHECKPOINT.npz'),
          'predictions_sha256':sha(O/'PREDICTIONS.npz'),'feature_seal_sha256':sha(D/'FEATURE_SEAL.json'),
          'train_target_receipts':receipts,'material_crossfits':fitted,'training_count':len(train),'evaluation_count':len(test),
          'evaluation_targets_read':False,'evaluation_trajectory_directory_exists':(D/'trajectories/evaluation').exists(),
          'object_and_surface_fit_eval_overlap':0,'original_model_replaced':False})
    print(json.dumps({'predictions_sealed':len(test),'train':len(train),'material_crossfits':len(fitted)}),flush=True)
def score():
    seal=read(O/'PREDICTION_SEAL.json');assert sha(O/'PREDICTIONS.npz')==seal['predictions_sha256']
    assert not seal['evaluation_targets_read'] and not seal['evaluation_trajectory_directory_exists']
    pred=load(O/'PREDICTIONS.npz');y,receipts=targets('evaluation')
    np.savez_compressed(O/'EVALUATION_TARGETS.npz',ids=pred['ids'],target=y)
    errors={k:np.linalg.norm(pred[k]-y,axis=1)*1000 for k in ('numeric','raw','gated')}
    objects=sorted(set(pred['object_material'].tolist()));surfaces=sorted(set(pred['surface_material'].tolist()))
    cells=sorted(set(zip(pred['object_material'].tolist(),pred['surface_material'].tolist())))
    cell_errors={k:np.asarray([v[(pred['object_material']==a)&(pred['surface_material']==b)].mean() for a,b in cells]) for k,v in errors.items()}
    delta=cell_errors['gated']-cell_errors['numeric'];point=float(delta.mean())
    rng=np.random.default_rng(SEED+1);B=20000
    ow=rng.multinomial(8,np.ones(8)/8,size=B);sw=rng.multinomial(8,np.ones(8)/8,size=B)
    weights=ow[:,[objects.index(a) for a,b in cells]]*sw[:,[surfaces.index(b) for a,b in cells]]
    assert np.all(weights.sum(1)>0)
    draws=weights@delta/weights.sum(1);ci=np.quantile(draws,[.025,.975])
    objdelta={str(a):float(delta[[c[0]==a for c in cells]].mean()) for a in objects}
    surfdelta={str(b):float(delta[[c[1]==b for c in cells]].mean()) for b in surfaces}
    owins=sum(v<0 for v in objdelta.values());swins=sum(v<0 for v in surfdelta.values())
    result={'status':'completed','evaluation_count':len(y),'training_count':seal['training_count'],'material_cells':len(cells),
            'object_levels':len(objects),'surface_levels':len(surfaces),
            'macro_cell_mae_mm':{k:float(v.mean()) for k,v in cell_errors.items()},
            'micro_episode_mae_mm':{k:float(v.mean()) for k,v in errors.items()},
            'gated_minus_numeric_mm':point,'two_way_bootstrap_95_mm':ci.tolist(),
            'relative_macro_improvement_percent':float(100*(1-cell_errors['gated'].mean()/cell_errors['numeric'].mean())),
            'object_marginal_differences_mm':objdelta,'surface_marginal_differences_mm':surfdelta,
            'object_wins':owins,'surface_wins':swins,
            'pass':bool(point<0 and ci[1]<0 and owins>=5 and swins>=5),
            'evaluation_target_receipts':receipts,'prediction_seal_sha256':sha(O/'PREDICTION_SEAL.json'),
            'historical_design_material_exposure':True,'new_object_asset_generalization_tested':False,
            'caveats':['eight levels per resampling axis','offline visibility-selected cohort','same cube asset and scene','historical architecture selection exposed to material bank']}
    np.savez_compressed(O/'ERRORS.npz',ids=pred['ids'],**errors,cells=np.asarray(cells),**{f'cell_{k}':v for k,v in cell_errors.items()},bootstrap_difference_mm=draws)
    save(O/'RESULT.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('evaluation_target_receipts','object_marginal_differences_mm','surface_marginal_differences_mm')},indent=2))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','download','observe','features','fit','score']);args=parser.parse_args()
    {'prepare':prepare,'download':download_videos,'observe':observe,'features':features,'fit':fit_predict,'score':score}[args.stage]()
