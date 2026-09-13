"""Spatial pre-merge vision features and matched fixed-query tracking diagnostic.

Only the local frozen Qwen vision tower is loaded. This is no language-decoder
reasoning or physical inference. Prediction reads RGB and the existing explicit
single-frame bbox cues; evaluation is a separate process after prediction seal.
"""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'sources/public_movi_sample_20260912'
MODEL=ROOT/'models/qwen3-vl-2b'
CUES=ROOT/'results/public_movi_tracking_v2/INITIALIZATION_CUES.json'
DEST=ROOT/'results/spatial_vlm_observation_v3'
CONFIG={'resized_size':256,'patch_size':16,'merge_size':2,'temporal_patch_size':2,'expected_grid':[1,16,16],
    'query':'single nearest patch center to initial bbox pixel center, fixed descriptor; bbox-center offset retained',
    'descriptors':{'rgb_patch':'full normalized RGB16x16x3=768, L2 normalization','qwen_premerge':'full last vision block1024, L2 normalization; no pooling or dimensional projection'},
    'search_radius_original_px':24.,'minimum_cosine':.65,'minimum_peak_gap':.05,'alternate_peak_exclusion_radius_original_px':8.,
    'global_reacquire':'same frame after local rejection; after abstention use global','template_update':False,
    'scale_search':False,'rotation_search':False,'visible_eval_minimum_pixels':8,'success_threshold_px':5.}


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def write(name,value):(DEST/name).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def declare():
    if DEST.exists():raise RuntimeError('Preserve prior diagnostic')
    DEST.mkdir()
    package=ROOT/'.venv/Lib/site-packages/transformers'
    paths=[Path(__file__),SOURCE/'observations/inputs.npz',CUES,MODEL/'config.json',MODEL/'preprocessor_config.json',MODEL/'model.safetensors',
        package/'models/qwen3_vl/modeling_qwen3_vl.py',package/'models/qwen2_vl/image_processing_qwen2_vl.py',package/'vision_utils.py']
    plan={'status':'frozen_before_spatial_extraction_tracking_or_new_output_score','scope':'one previously selected public MOVi-C scene3835 development diagnostic;24frames,3initialized targets',
        'config':CONFIG,'source_sha256':{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in paths},
        'privileged_initialization':'reuse existing first-visible single-frame bbox/frame cues; no masks, pose, depth or physical properties available to predictor',
        'spatial_API':'Qwen3VLVisionModel.forward(...).last_hidden_state before final merger; merger prehook must match bitexact',
        'grid_order':'get_vision_position_ids(grid_thw,2) produces2x2block-major (row,col); synthetic patch-color test verifies raw preprocessing correspondence',
        'original_pixel_center_mapping':'x=(col+.5)*16*original_width/256-.5, y analogous',
        'temporal':'images separately encoded; processor repeats identical image patch2times, no other frame/future input',
        'receptive_field':'16x16 resized pixels=8x8original patch anchor;24full-attention blocks yield whole-frame receptive field, not a strictly local descriptor',
        'matched_tracker':'same grid,query patch,offset,cosine,local/global policy and thresholds; only descriptor differs',
        'limitations':['single-center patch may include background, especially existing2px initialization cue','resizing128to256adds no image detail',
            'same numeric cosine thresholds are not calibrated equally across descriptor spaces','single fixed-scale/rotation query is not a mature tracker',
            'no pooling ablation here: success or failure does not isolate pooling as sole cause','one scene is not independent physics or general tracking evidence'],
        'resources':'local vision-only406957056parameters;BF16~0.758GiB weights, actualGPUpeak/time recorded; no download,training or textdecoder',
        'model_revision':'89644892e4d85e24eaac8bacfd4f463576704203; local file hash authoritative',
        'root_GPU_authorized':True,'prediction_seal_before_evaluation':True,'goal_complete':False}
    write('PLAN.json',plan);print(json.dumps({'declared':str(DEST/'PLAN.json'),'sha256':sha(DEST/'PLAN.json')}),flush=True)


def verify():
    plan=read(DEST/'PLAN.json');assert plan['config']==CONFIG
    for name,digest in plan['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Changed source: '+name)
    return plan


def processor_and_mapping_test():
    import torch
    from transformers import AutoImageProcessor
    from transformers.vision_utils import get_vision_position_ids
    processor=AutoImageProcessor.from_pretrained(MODEL,local_files_only=True)
    # Every 16x16 patch has a distinct row/column RGB code. No interpolation.
    y,x=np.indices((256,256));synthetic=np.stack([(y//16)*13,(x//16)*13,((y//16)+(x//16))*7],-1).astype(np.uint8)
    payload=processor(images=synthetic,do_resize=False,return_tensors='pt')
    grid=payload['image_grid_thw'];assert grid.tolist()==[[1,16,16]]
    position=get_vision_position_ids(grid,2)
    patches=payload['pixel_values'].reshape(256,3,2,16,16)
    assert torch.equal(patches[:,:,0],patches[:,:,1])
    codes=torch.stack([position[:,0]*13,position[:,1]*13,(position[:,0]+position[:,1])*7],-1).float()/255*2-1
    torch.testing.assert_close(patches[:,:,0,0,0],codes,atol=2e-7,rtol=0)
    assert len(torch.unique(position,dim=0))==256
    manual=torch.arange(256).reshape(8,2,8,2).permute(0,2,1,3).reshape(-1)
    assert torch.equal(position[:,0]*16+position[:,1],manual)
    write('SPATIAL_MAPPING_TEST.json',{'status':'passed','positions_first16':position[:16].tolist(),'grid':grid.tolist(),
        'raw_pixel_shape':list(payload['pixel_values'].shape),'block_order_matches_position_ids':True,
        'synthetic_patch_color_matches_reported_coordinates':True,'duplicated_temporal_patch_verified':True})
    return processor,position


def load_vision_only():
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel,Qwen3VLVisionRotaryEmbedding
    config=AutoConfig.from_pretrained(MODEL,local_files_only=True).vision_config
    config._attn_implementation='sdpa'
    with torch.device('meta'):vision=Qwen3VLVisionModel(config)
    prefix='model.visual.'
    with safe_open(MODEL/'model.safetensors',framework='pt',device='cpu') as reader:
        state={key[len(prefix):]:reader.get_tensor(key) for key in reader.keys() if key.startswith(prefix)}
    vision.load_state_dict(state,strict=True,assign=True)
    # Nonpersistent rotary frequencies are not in the checkpoint; reconstruct
    # by the same source constructor instead of leaving a meta buffer.
    vision.rotary_pos_emb=Qwen3VLVisionRotaryEmbedding(config.hidden_size//config.num_heads//2)
    assert not any(p.is_meta for p in vision.parameters())
    assert not any(p.is_meta for p in vision.buffers())
    vision=vision.to(device='cuda',dtype=torch.bfloat16).eval()
    vision.requires_grad_(False)
    assert sum(p.numel() for p in vision.parameters())==406957056
    return vision


def fixed_query_tracker(features,centers,cues,timestamps):
    import torch
    normalized=torch.nn.functional.normalize(features.float(),dim=-1).numpy()
    results=[]
    for cue in cues['targets']:
        first=cue['initialize_frame'];x0,y0,x1,y1=cue['initialize_bbox_xyxy']
        original_center=np.array([(x0+x1-1)/2,(y0+y1-1)/2])
        query_index=int(np.argmin(np.linalg.norm(centers-original_center,axis=-1)))
        offset=original_center-centers[query_index];query=normalized[first,query_index].copy()
        trajectory=[];previous=None
        def match(t,indices):
            if not len(indices):return None
            scores=normalized[t,indices]@query;best=int(np.argmax(scores));index=int(indices[best]);score=float(scores[best])
            excluded=(np.abs(centers[indices]-centers[index])<=CONFIG['alternate_peak_exclusion_radius_original_px']).all(-1)
            alternate=scores[~excluded];second=float(np.max(alternate)) if len(alternate) else -1.
            return index,score,score-second
        okay=lambda result:result is not None and result[1]>=CONFIG['minimum_cosine'] and result[2]>=CONFIG['minimum_peak_gap']
        for t in range(len(features)):
            if t<first:
                trajectory.append(dict(frame=t,status='not_initialized',accepted=False,center_xy=None,search=None,correlation=None,peak_gap=None));continue
            if t==first:
                previous=centers[query_index]
                trajectory.append(dict(frame=t,status='initialization_cue',accepted=True,center_xy=original_center.tolist(),search='cue',correlation=1.,peak_gap=None));continue
            found=None;search='global'
            if previous is not None:
                local=np.flatnonzero((np.abs(centers-previous)<=CONFIG['search_radius_original_px']).all(-1));found=match(t,local);search='local'
            if not okay(found):found=match(t,np.arange(len(centers)));search='global'
            accepted=okay(found);previous=centers[found[0]] if accepted else None
            trajectory.append(dict(frame=t,status='tracked' if accepted else 'abstained',accepted=bool(accepted),
                center_xy=(previous+offset).tolist() if accepted else None,search=search,
                correlation=None if found is None else found[1],peak_gap=None if found is None else found[2],
                matched_patch_index=None if found is None else found[0]))
        results.append(dict(target_index=cue['target_index'],initialize_frame=first,initialize_bbox_xyxy=cue['initialize_bbox_xyxy'],
            query_patch_index=query_index,query_patch_center_xy=centers[query_index].tolist(),fixed_bbox_center_offset_xy=offset.tolist(),
            normalized_query_sha256=hashlib.sha256(query.tobytes()).hexdigest(),trajectory=trajectory))
    return dict(frames=len(features),relative_time_s=timestamps.tolist(),targets=results,
        initialization_cues_sha256=sha(CUES),plan_sha256=sha(DEST/'PLAN.json'))


def predict():
    opened=[]
    def audit(event,args):
        if event!='open' or not args or not isinstance(args[0],str):return
        path=Path(args[0])
        if 'evaluator_only' in path.parts or path.name in ['SAMPLE_VALIDATION.json','EVALUATION.json','CUE_RECEIPT.json']:
            raise PermissionError('Predictor cannot read evaluator information')
        if path.suffix in ['.npz','.json','.png']:opened.append(str(path))
    sys.addaudithook(audit)
    import torch
    from transformers.vision_utils import get_vision_position_ids
    torch.set_num_threads(2)
    if (DEST/'PREDICTION_SEAL.json').exists():raise RuntimeError('Preserve sealed outputs')
    verify();processor,position=processor_and_mapping_test()
    with np.load(SOURCE/'observations/inputs.npz',allow_pickle=False) as obj:rgb=obj['rgb'];timestamps=obj['relative_time_s']
    assert rgb.shape==(24,128,128,3)
    cues=read(CUES)
    centers=(position[:,[1,0]].double()+.5)*16*128/256-.5
    payload=processor(images=list(rgb),size={'shortest_edge':65536,'longest_edge':65536},return_tensors='pt')
    assert payload['image_grid_thw'].tolist()==[[1,16,16]]*24
    pixels=payload['pixel_values'].reshape(24,256,3,2,16,16)
    assert torch.equal(pixels[:,:,:,0],pixels[:,:,:,1])
    rgb_features=pixels[:,:,:,0].reshape(24,256,768).contiguous()
    torch.cuda.reset_peak_memory_stats();loading=time.perf_counter();vision=load_vision_only();torch.cuda.synchronize();load_seconds=time.perf_counter()-loading
    features=[];times=[];merger_inputs=[]
    handle=vision.merger.register_forward_pre_hook(lambda module,args:merger_inputs.append(args[0]))
    with torch.no_grad():
        for t in range(24):
            raw=payload['pixel_values'][t*256:(t+1)*256].to('cuda',dtype=torch.bfloat16)
            grid=payload['image_grid_thw'][t:t+1].to('cuda')
            assert torch.equal(get_vision_position_ids(grid,2).cpu(),position)
            torch.cuda.synchronize();start=time.perf_counter();output=vision(raw,grid_thw=grid,return_dict=True);torch.cuda.synchronize()
            times.append(time.perf_counter()-start)
            dense=output.last_hidden_state
            assert dense.shape==(256,1024) and output.pooler_output.shape==(64,2048)
            assert torch.equal(dense,merger_inputs.pop()) and bool(torch.isfinite(dense).all())
            features.append(dense.float().cpu())
            if t==0:
                write('GPU_SMOKE.json',{'status':'passed','grid':[1,16,16],'premerge_shape':list(dense.shape),'postmerge_shape':list(output.pooler_output.shape),
                    'API_output_matches_merger_input_bitexact':True,'model_parameters':sum(p.numel() for p in vision.parameters()),
                    'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'forward_seconds':times[0],'loading_seconds':load_seconds,
                    'text_decoder_loaded':False,'training':False})
            print(json.dumps({'frame':t,'seconds':times[-1],'dense_shape':list(dense.shape)}),flush=True)
    handle.remove();dense_features=torch.stack(features)
    torch.save({'qwen_premerge':dense_features,'rgb_patch':rgb_features,'grid_thw':payload['image_grid_thw'],
        'position_ids':position,'centers_original_xy':centers,'plan_sha256':sha(DEST/'PLAN.json')},DEST/'SPATIAL_FEATURES.pt')
    predictions={}
    for name,descriptor in [('rgb_patch',rgb_features),('qwen_premerge',dense_features)]:
        start=time.perf_counter();predictions[name]=fixed_query_tracker(descriptor,centers.numpy(),cues,timestamps)
        write(name+'_PREDICTIONS.json',predictions[name])
        predictions[name]['tracker_seconds']=time.perf_counter()-start
    cost={'vision_parameters':406957056,'loading_seconds':load_seconds,'vision_forward_seconds_each':times,
        'vision_forward_seconds_total':sum(times),'vision_forward_seconds_median_excluding_first':float(np.median(times[1:])),
        'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved(),
        'tracker_cpu_seconds':{name:p['tracker_seconds'] for name,p in predictions.items()},
        'scope':'vision forward excludes preprocessing/IO/loading/tracking; tracker cached CPU cosine; not end-to-end latency','torch_version':torch.__version__}
    write('COST.json',cost);verify()
    write('PREDICTION_SEAL.json',{'status':'sealed_before_evaluation','plan_sha256':sha(DEST/'PLAN.json'),
        'feature_sha256':sha(DEST/'SPATIAL_FEATURES.pt'),'prediction_sha256':{name:sha(DEST/(name+'_PREDICTIONS.json')) for name in predictions},
        'initialization_cues_sha256':sha(CUES),'evaluator_files_opened':False,'open_log':sorted(set(opened)),'training':False})
    print(json.dumps({'status':'sealed','methods':list(predictions),'cost':cost}),flush=True)


def evaluate():
    verify();seal=read(DEST/'PREDICTION_SEAL.json')
    assert seal['plan_sha256']==sha(DEST/'PLAN.json')
    for name,digest in seal['prediction_sha256'].items():assert sha(DEST/(name+'_PREDICTIONS.json'))==digest
    with np.load(SOURCE/'evaluator_only/labels.npz',allow_pickle=False) as obj:seg=obj['segmentations'];positions=obj['image_positions']
    def aggregate(rows):
        eligible=[r for r in rows if r['eligible']];accepted=[r for r in eligible if r['accepted']];errors=[r['error_px'] for r in accepted]
        return {'eligible':len(eligible),'covered':len(accepted),'abstained':len(eligible)-len(accepted),
            'coverage':len(accepted)/len(eligible) if eligible else None,'covered_mean_error_px':float(np.mean(errors)) if errors else None,
            'covered_median_error_px':float(np.median(errors)) if errors else None,'success_5px_count':sum(r['success'] for r in eligible),
            'success_5px_fraction_all_eligible':sum(r['success'] for r in eligible)/len(eligible) if eligible else None}
    reports={}
    for method in seal['prediction_sha256']:
        prediction=read(DEST/(method+'_PREDICTIONS.json'));rows=[]
        for target in prediction['targets']:
            index=target['target_index']
            for item in target['trajectory']:
                t=item['frame'];mask=seg[t]==index+1;yy,xx=np.nonzero(mask);pixels=len(xx)
                center=None if not pixels else np.array([(xx.min()+xx.max())/2,(yy.min()+yy.max())/2])
                error=None if center is None or not item['accepted'] else float(np.linalg.norm(np.array(item['center_xy'])-center))
                eligible=t>target['initialize_frame'] and pixels>=8
                state=('visible_ge8' if pixels>=8 else 'tiny_visible_1to7' if pixels else
                    'zero_pixels_projected_center_inside' if bool(((positions[index,t]>=0)&(positions[index,t]<=1)).all()) else 'zero_pixels_projected_center_outside')
                rows.append({'target':index,'frame':t,'pixels':pixels,'gt_state':state,'accepted':item['accepted'],'eligible':eligible,
                    'error_px':error,'success':bool(eligible and item['accepted'] and error<=5)})
        reports[method]={'overall':aggregate(rows),'per_target':{str(i):aggregate([r for r in rows if r['target']==i]) for i in [0,1,2]},
            'accepted_zero_pixel_postinitialization':sum(r['pixels']==0 and r['accepted'] and prediction['targets'][r['target']]['initialize_frame']<r['frame'] for r in rows),
            'rows':rows}
    result={'scope':'one public development scene; matched fixed patch-query cosine diagnostic; no threshold tuning, training,decoder reasoning,physics or generalization claim',
        'prediction_seal_sha256':sha(DEST/'PREDICTION_SEAL.json'),'labels_sha256':sha(SOURCE/'evaluator_only/labels.npz'),
        'metric':'visible segmentation bbox pixel center, not center of mass; initial cue frames excluded; visible>=8pixels, success<=5px',
        'methods':reports,'goal_complete':False}
    write('EVALUATION.json',result);print(json.dumps({k:{key:value for key,value in v.items() if key!='rows'} for k,v in reports.items()}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['declare','predict','evaluate']);args=parser.parse_args()
    globals()[args.stage]()
