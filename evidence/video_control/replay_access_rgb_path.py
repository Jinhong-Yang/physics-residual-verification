"""Replay the first original eight-case batch from RGB with fixed recorded priors.

Recomputes tracking, observation initialization, frozen Qwen/decoder features,
robust physical solves and locked residual predictions. It does not retrain
upstream models or regenerate the first-image prior network output.
"""
import os
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'scripts')]
from scripts import stream_com_internal_features_v16 as stream
from scripts import train_com_physics_internal_readout_v12 as base
from scripts import predict_com_physics_v16 as frozen
from canonical_confirmation_inputs_v11 import tracking_row,pack_inputs
from reliability_observation_v7 import tracking,extract_observation
from sequential_robust_readout_v22 import predict,REGIMES

O=R/'results/access_rgb_replay_v1';O.mkdir(exist_ok=True)
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def main():
    plan={'cases':'first eight episodes in original V23 index; same decoder batch size',
          'input_tolerance':1e-10,'feature_pixel_tolerance':1e-4,'posterior_tolerance':1e-7,
          'selection_uses_outcomes':False,'prior_source':'recorded first-image prior means/scales, not regenerated',
          'upstream_training_reproduced':False}
    (O/'PLAN.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    torch.set_num_threads(2);started=time.perf_counter()
    episodes=json.loads((R/'data/canonical_confirmation_v23/INPUT_INDEX.json').read_text())['episodes'][:8]
    prior=load(R/'cache/canonical_confirmation_v23/prior/priors_seed17.npz')
    assert prior['ids'][:8].tolist()==[e['id'] for e in episodes]
    records=[];tracks=[]
    for e,mean,scale in zip(episodes,prior['mean'][:8],prior['scale'][:8]):
        row=tracking_row(e);track=tracking(row,R)
        records.append(extract_observation(row,R,mean,scale,track));tracks.append(track)
    inputs,diagnostic=pack_inputs(episodes,records,tracks)
    old=load(R/'cache/canonical_confirmation_v23/input/inputs.npz')
    input_errors={}
    for k,x in inputs.items():
        ref=old[k][:8]
        if x.dtype.kind in 'fiu':
            np.testing.assert_allclose(x,ref,rtol=0,atol=plan['input_tolerance'])
            input_errors[k]=float(np.max(np.abs(x-ref)))
        else:assert np.array_equal(x,ref)
    print('RGB_TRACKING_AND_INITIALIZATION_MATCH',flush=True)
    models=stream.load_models();result=stream.run_rows(episodes,inputs,models,progress=True,decoder_batch_size=8)
    streamdir=R/'cache/canonical_confirmation_v23/vlm_stream_v1'
    qwen={'xz':stream.project_inverse(result['pixels'],episodes)}
    static={'xz':stream.project_inverse(result['static_pixels'],episodes)}
    pooled={'pooled':result['pooled']}
    feature_errors={}
    for field,name,key in (('pooled','pooled_evidence.npz','pooled'),('pixels','qwen_warp_joint.npz','pixels'),('static_pixels','qwen_warp_joint_static.npz','pixels')):
        ref=load(streamdir/name)[key][:8]
        feature_errors[field]=float(np.max(np.abs(result[field]-ref)))
        np.testing.assert_allclose(result[field],ref,rtol=0,atol=plan['feature_pixel_tolerance'])
    del models;torch.cuda.empty_cache()
    features,dim=base.raw_features(inputs,pooled,qwen,static)
    fits,weights=frozen.fixed_robust_fits(base.tensor_batch(inputs),'huber')
    anchors={r:{k:fits[r][k] for k in ('mean','covariance')} for r in REGIMES}
    checkpoint=load(R/'results/com_observation_v22/final_development_fit_v1/CHECKPOINT.npz')
    candidate=predict(checkpoint,features.numpy(),inputs['active'],inputs['protocol'],anchors)
    expected=load(R/'results/com_observation_v23/frozen_predictions_v1/methods/v22_vlm.npz')
    errors={}
    for regime,prefix in (('generalized_translation','generalized'),('setup_translation','setup')):
        for key in ('mean','covariance'):
            a=candidate[regime][key];b=expected[f'{prefix}_{key}'][:8]
            errors[regime+'__'+key]=float(np.max(np.abs(a-b)))
            np.testing.assert_allclose(a,b,rtol=0,atol=plan['posterior_tolerance'])
    report={'status':'passed','plan':plan,'episodes':8,'protocols':sorted(set(inputs['protocol'])),
            'tracking_and_initialization_errors':input_errors,'feature_errors':feature_errors,
            'posterior_errors':errors,'elapsed_seconds':time.perf_counter()-started,
            'targets_opened':False,'full_training_or_prior_network_reproduced':False}
    (O/'RESULT.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
