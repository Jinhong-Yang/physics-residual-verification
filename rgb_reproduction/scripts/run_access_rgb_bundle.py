"""Portable RGB inference runner. Explicit model directory or pinned download."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode=True
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'scripts')]
import numpy as np
import torch
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model-dir',type=Path,required=True);args=parser.parse_args()
    modeldir=args.model_dir.resolve();out=R/'reproduced';out.mkdir(exist_ok=True)
    plan={'selection':'first eight cases in original index, not outcome-selected',
          'prior_atol':1e-6,'input_atol':1e-4,'decoder_atol':1e-4,'posterior_atol':1e-5,
          'recompute_first_image_prior':True,'upstream_training_reproduced':False}
    (out/'PLAN.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    manifest=json.loads((R/'MANIFEST.json').read_text())
    for row in manifest['files']:assert digest(R/row['path'])==row['sha256'],row['path']
    mm=json.loads((R/'sources/model_download_manifest.json').read_text())
    for row in mm['files']:assert digest(modeldir/row['path'])==row['sha256'],row['path']
    # Runtime reads may use package files, explicit base-model inputs, and installed libraries.
    # Any attempt to fall back to the original research checkout is rejected.
    allowed=(R,modeldir,Path(sys.prefix).resolve(),Path(sys.base_prefix).resolve())
    opened=set()
    def guard(event,items):
        if event!='open' or not items or not isinstance(items[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(items[0])).resolve()
        if any(p.is_relative_to(root) for root in allowed):
            if p.is_relative_to(R):opened.add(p.relative_to(R).as_posix())
            return
        # System device and timezone/runtime metadata are not research inputs.
        if p.name.lower() in ('nul','localtime','tzdata.zi') or 'windows' in [v.lower() for v in p.parts]:return
        raise PermissionError('Unbundled file access: '+str(p))
    sys.addaudithook(guard)
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    from scripts.extract_canonical_confirmation_inputs_v11 import forward_first_image
    from projected_posterior import AppearancePrior
    from scripts import stream_com_internal_features_v16 as stream
    from scripts import extract_spatial_vlm_observations_v3 as visionloader
    from scripts import train_com_physics_internal_readout_v12 as base
    from scripts import predict_com_physics_v16 as frozen
    from canonical_confirmation_inputs_v11 import tracking_row,pack_inputs
    from reliability_observation_v7 import tracking,extract_observation
    from sequential_robust_readout_v22 import predict,REGIMES
    stream.MODEL=modeldir;visionloader.MODEL=modeldir
    torch.set_num_threads(4);started=time.perf_counter()
    episodes=json.loads((R/'data/canonical_confirmation_v23/INPUT_INDEX.json').read_text())['episodes']
    model=Qwen3VLForConditionalGeneration.from_pretrained(modeldir,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda().eval().requires_grad_(False)
    processor=AutoProcessor.from_pretrained(modeldir,local_files_only=True)
    vectors=[]
    for e in episodes:vectors.append(forward_first_image(torch,model,processor,R/e['frame_paths'][0])['feature'])
    del model,processor;gc.collect();torch.cuda.empty_cache()
    x=torch.stack(vectors).float();x=torch.nn.functional.layer_norm(x,(x.shape[1],),eps=1e-5)
    head=AppearancePrior(x.shape[1]);head.load_state_dict(torch.load(R/'results/projection_prior/seed17/best.pt',map_location='cpu',weights_only=True)['state']);head.eval()
    # Preserve the original CPU linear-layer batch shape; padding has no cross-row operation.
    padding=torch.zeros((256-len(x),x.shape[1]),dtype=x.dtype)
    with torch.inference_mode():m,sc=head(torch.cat([x,padding]));m=m[:len(x)].numpy();sc=sc[:len(x)].numpy()
    prior=load(R/'cache/canonical_confirmation_v23/prior/priors_seed17.npz')
    errors={'prior_mean':float(np.max(np.abs(m-prior['mean']))),'prior_scale':float(np.max(np.abs(sc-prior['scale'])))}
    np.testing.assert_allclose(m,prior['mean'],rtol=0,atol=plan['prior_atol']);np.testing.assert_allclose(sc,prior['scale'],rtol=0,atol=plan['prior_atol'])
    print('FIRST_IMAGE_PRIOR_REGENERATED '+json.dumps(errors),flush=True)
    records=[];tracks=[]
    for e,mean,scale in zip(episodes,m,sc):
        row=tracking_row(e);track=tracking(row,R);tracks.append(track);records.append(extract_observation(row,R,mean,scale,track))
    inputs,_=pack_inputs(episodes,records,tracks);old=load(R/'cache/canonical_confirmation_v23/input/inputs.npz')
    input_errors={}
    for k,a in inputs.items():
        if a.dtype.kind in 'fiu':
            np.testing.assert_allclose(a,old[k],rtol=0,atol=plan['input_atol']);input_errors[k]=float(np.max(np.abs(a-old[k])))
        else:assert np.array_equal(a,old[k])
    torch.set_num_threads(2)
    models=stream.load_models();result=stream.run_rows(episodes,inputs,models,progress=True,decoder_batch_size=8)
    pooled={'pooled':result['pooled']};qwen={'xz':stream.project_inverse(result['pixels'],episodes)};static={'xz':stream.project_inverse(result['static_pixels'],episodes)}
    feature_errors={}
    for field,file,key in (('pooled','pooled_evidence.npz','pooled'),('pixels','qwen_warp_joint.npz','pixels'),('static_pixels','qwen_warp_joint_static.npz','pixels')):
        ref=load(R/'cache/canonical_confirmation_v23/vlm_stream_v1'/file)[key]
        np.testing.assert_allclose(result[field],ref,rtol=0,atol=plan['decoder_atol']);feature_errors[field]=float(np.max(np.abs(result[field]-ref)))
    del models;gc.collect();torch.cuda.empty_cache()
    features,_=base.raw_features(inputs,pooled,qwen,static)
    fits,_=frozen.fixed_robust_fits(base.tensor_batch(inputs),'huber')
    anchors={r:{k:fits[r][k] for k in ('mean','covariance')} for r in REGIMES}
    candidate=predict(load(R/'results/com_observation_v22/final_development_fit_v1/CHECKPOINT.npz'),features.numpy(),inputs['active'],inputs['protocol'],anchors)
    expected=load(R/'results/com_observation_v23/frozen_predictions_v1/methods/v22_vlm.npz');posterior_errors={};arrays={}
    for regime,prefix in (('generalized_translation','generalized'),('setup_translation','setup')):
        for key in ('mean','covariance'):
            a=candidate[regime][key];b=expected[f'{prefix}_{key}'];np.testing.assert_allclose(a,b,rtol=0,atol=plan['posterior_atol'])
            posterior_errors[regime+'__'+key]=float(np.max(np.abs(a-b)));arrays[regime+'__'+key]=a
    np.savez_compressed(out/'PREDICTIONS.npz',ids=inputs['ids'],**arrays)
    report={'status':'passed','plan':plan,'cases':len(episodes),'protocols':sorted(set(inputs['protocol'])),
            'prior_errors':errors,'input_errors':input_errors,'feature_errors':feature_errors,'posterior_errors':posterior_errors,
            'original_research_data_used':False,'explicit_base_model_used':True,'targets_used':False,
            'seconds':time.perf_counter()-started,'read_files':sorted(opened)}
    (out/'RESULT.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in report.items() if k!='read_files'},indent=2))
if __name__=='__main__':main()
