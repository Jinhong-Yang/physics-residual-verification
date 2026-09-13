"""New first-image prior and exact-canonical RGB input stages; draft only.

No declaration or work on import. Future Root PLAN/AUTH plus independent old900
T8 regression is required. Dense premerge and final learned-head prediction
are separate stages, not provided or silently substituted here.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/canonical_confirmation_v11'
OUT=ROOT/'cache/canonical_confirmation_v11'
MODEL=ROOT/'models/qwen3-vl-2b'
PRIOR_CHECKPOINT=ROOT/'results/projection_prior/seed17/best.pt'


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))


def write(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:json.dump(obj,f,indent=2,allow_nan=False)


def npwrite(p,arrays):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:np.savez_compressed(f,**arrays)


def relative(p):return Path(p).resolve().relative_to(ROOT.resolve()).as_posix()


def ref(name):
    from canonical_confirmation_inputs_v11 import root_reference
    p=root_reference(ROOT,name)
    if any(v in p.parts for v in ('evaluator_only','sealed_targets','target_traces')) or ('target' in p.name.lower() or 'quality' in p.name.lower() or 'labels' in p.name.lower()) and p.suffix in ('.json','.jsonl','.npz','.npy','.pt'):
        raise PermissionError('Supervised dependency forbidden even during source/hash preflight')
    return p


class InputGuard:
    """Research data/cache/model reads exact; runtime imports remain available."""
    def __init__(self,allowed):self.allowed={Path(p).resolve() for p in allowed};self.reads=set()
    def __call__(self,event,args):
        if event!='open' or not args or isinstance(args[0],int):return
        p=Path(args[0]).resolve();mode=str(args[1] or '')
        flags=args[2] if len(args)>2 and isinstance(args[2],int) else 0
        writing=any(v in mode for v in 'wax+') or bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
        if writing:
            if not p.is_relative_to(OUT.resolve()):raise PermissionError('Only new input output/cache writes')
            return
        if any(v in p.parts for v in ('evaluator_only','sealed_targets','target_traces')) or ('target' in p.name.lower() or 'quality' in p.name.lower() or 'labels' in p.name.lower()) and p.suffix in ('.json','.jsonl','.npz','.npy','.pt'):
            raise PermissionError('Observation stage refuses supervised bytes')
        if p.is_relative_to(ROOT.resolve()):
            rel=p.relative_to(ROOT.resolve())
            if rel.parts[0] in ('data','cache','models','results') and p not in self.allowed:
                raise PermissionError('Unlisted research artifact read: '+str(p))
            self.reads.add(rel.as_posix())
        else:
            runtime=[Path(sys.base_prefix).resolve(),Path(sys.prefix).resolve()]
            if not any(p.is_relative_to(base) for base in runtime):raise PermissionError('External repository/nonruntime access')


def context(stage):
    from canonical_confirmation_inputs_v11 import require_regression,first_image_index,PROMPT
    from canonical_confirmation_data_v11 import index_contract
    # No generation PLAN is parsed: it contains generation-side physical RNG seeds.
    plan=read(OUT/'PLAN.json');auth=read(OUT/'AUTHORIZATION.json')
    if stage not in auth['allowed_stages'] or auth['plan_sha256']!=sha(OUT/'PLAN.json'):
        raise PermissionError('Stage not authorized by Root final input PLAN')
    if plan['prior_seed']!=17 or plan['prompt']!=PROMPT or plan['layer_norm_epsilon']!=1e-5:
        raise ValueError('Fixed first-image prior contract changed')
    seal=read(DATA/'OBSERVATION_SEAL.json')
    if seal['status']!='sealed' or plan['observation_seal_sha256']!=sha(DATA/'OBSERVATION_SEAL.json'):
        raise ValueError('Observation seal mismatch')
    if seal['files']['INPUT_INDEX.json']!=sha(DATA/'INPUT_INDEX.json'):raise ValueError('Index changed')
    index=read(DATA/'INPUT_INDEX.json');episodes=index['episodes'];index_contract(episodes)
    if index['plan_sha256']!=seal['plan_sha256'] or plan['ids']!=[e['id'] for e in episodes]:raise ValueError('Dataset/order changed')
    images,joins=first_image_index(episodes)
    if images!=plan['first_images'] or joins!=plan['row_features']:raise ValueError('First-image identity changed')
    regression=ref(plan['regression']['path'])
    if sha(regression)!=plan['regression']['sha256']:raise ValueError('Regression receipt changed')
    require_regression(read(regression))
    allowed=[OUT/'PLAN.json',OUT/'AUTHORIZATION.json',DATA/'OBSERVATION_SEAL.json',DATA/'INPUT_INDEX.json',regression]
    required={'canonical_confirmation_inputs_v11.py','canonical_confirmation_data_v11.py',relative(Path(__file__)),
        'reliability_observation_v7.py','reliability_canonical_v7.py','reliability_data_v2.py',
        'projected_observation.py','numerical_observation_likelihood.py','main3d_numeric.py',
        'main3d_refinement.py','main3d_evaluation.py','projection_prior.py','projected_posterior.py'}
    if not required<=set(plan['source_sha256']):raise ValueError('Input source closure incomplete')
    for mapping in ('source_sha256','model_sha256'):
        for name,h in plan[mapping].items():
            p=ref(name)
            if sha(p)!=h:raise ValueError('Frozen source/model changed: '+name)
            allowed.append(p)
    if relative(PRIOR_CHECKPOINT) not in plan['model_sha256']:raise ValueError('Fixed seed17 prior checkpoint missing')
    for e in episodes:
        for name,h in zip(e['frame_paths'],e['frame_sha256']):
            p=ref(name)
            if sha(p)!=h:raise ValueError('Frame changed')
            allowed.append(p)
    # Each stage reads only its sealed immediate predecessor.
    if stage=='prior_cpu':
        receipt=OUT/'prior/FEATURE_SEAL.json';payload=read(receipt)
        if payload['plan_sha256']!=sha(OUT/'PLAN.json'):raise ValueError('Prior feature plan')
        allowed.append(receipt)
        for r in payload['records']:
            p=ref(r['path'])
            if sha(p)!=r['sha256']:raise ValueError('Prior feature bytes changed')
            allowed.append(p)
    if stage=='rgb_inputs':
        receipt=OUT/'prior/PRIOR_SEAL.json';payload=read(receipt)
        if payload['plan_sha256']!=sha(OUT/'PLAN.json'):raise ValueError('Prior plan')
        p=OUT/'prior/priors_seed17.npz'
        if payload['prediction_sha256']!=sha(p):raise ValueError('Prior bytes changed')
        allowed += [receipt,p]
    guard=InputGuard(allowed);sys.addaudithook(guard)
    # Atomic stage lock before any retained forward/extraction; never auto-resume.
    write(OUT/f'RUN_{stage}.json',{'stage':stage,'plan_sha256':sha(OUT/'PLAN.json'),'restart':False})
    return plan,episodes,images,joins,guard


def forward_first_image(torch,model,proc,source):
    """B1 original full-final-token computation, shared with regression runner."""
    from canonical_confirmation_inputs_v11 import PROMPT
    from PIL import Image
    with Image.open(source) as im:rgb=im.convert('RGB')
    messages=[{'role':'user','content':[{'type':'image','image':rgb},{'type':'text','text':PROMPT}]}]
    with torch.inference_mode():
        inputs=proc.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to('cuda')
        feature=model.model(**inputs,use_cache=False).last_hidden_state[0,-1].cpu()
    if feature.dtype!=torch.bfloat16 or feature.shape!=(model.config.text_config.hidden_size,) or not torch.isfinite(feature).all():
        raise ValueError('Invalid first-image final token')
    return {'feature':feature,'input_tokens':int(inputs.attention_mask.sum()),
        'image_tokens':int((inputs.input_ids==model.config.image_token_id).sum())}


def prior_gpu():
    # Equivalent B1 full-decoder first-image forward; never mean-pooled premerge.
    import torch
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    plan,episodes,images,joins,guard=context('prior_gpu')
    torch.set_num_threads(4)
    model=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,dtype=torch.bfloat16,
        attn_implementation='sdpa',local_files_only=True).cuda().eval().requires_grad_(False)
    proc=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
    rows=[];started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for item in images:
            source=ref(item['source_paths'][0])
            value=forward_first_image(torch,model,proc,source)
            path=OUT/'prior/features'/f"{item['feature_id']}.pt";path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as f:torch.save({'feature_id':item['feature_id'],'plan_sha256':sha(OUT/'PLAN.json'),
                **value},f)
            guard.allowed.add(path.resolve());rows.append({'feature_id':item['feature_id'],'path':relative(path),'sha256':sha(path)})
    write(OUT/'prior/FEATURE_SEAL.json',{'status':'sealed','plan_sha256':sha(OUT/'PLAN.json'),'records':rows,
        'elapsed_s':time.perf_counter()-started,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,
        'scope':'one full frozen Qwen final token per unique first PNG; excludes premerge extraction',
        'targets_opened':False,'read_log':sorted(guard.reads)})


def prior_cpu():
    import torch
    from projected_posterior import AppearancePrior
    from canonical_confirmation_inputs_v11 import join_priors,prior_batches
    plan,episodes,images,joins,guard=context('prior_cpu');torch.set_num_threads(4)
    receipt=read(OUT/'prior/FEATURE_SEAL.json')
    if [r['feature_id'] for r in receipt['records']]!=[r['feature_id'] for r in images]:raise ValueError('Prior features reordered')
    values=[]
    for r in receipt['records']:
        z=torch.load(ref(r['path']),map_location='cpu',weights_only=True)
        if z['feature_id']!=r['feature_id'] or z['plan_sha256']!=sha(OUT/'PLAN.json'):raise ValueError('Feature payload identity')
        values.append(z['feature'])
    features=torch.stack(values).float();features=torch.nn.functional.layer_norm(features,(features.shape[1],),eps=1e-5)
    payload=torch.load(PRIOR_CHECKPOINT,map_location='cpu',weights_only=True)
    model=AppearancePrior(features.shape[1]);model.load_state_dict(payload['state'],strict=True);model.eval()
    with torch.no_grad():
        results=[model(batch) for batch in prior_batches(features)]
        mean=torch.cat([r[0] for r in results]);scale=torch.cat([r[1] for r in results])
    m,s=join_priors(plan['ids'],[r['feature_id'] for r in images],mean.numpy(),scale.numpy(),joins)
    path=OUT/'prior/priors_seed17.npz';npwrite(path,{'ids':np.array(plan['ids']),'mean':m,'scale':s})
    guard.allowed.add(path.resolve())
    write(OUT/'prior/PRIOR_SEAL.json',{'status':'sealed','plan_sha256':sha(OUT/'PLAN.json'),
        'feature_seal_sha256':sha(OUT/'prior/FEATURE_SEAL.json'),'prediction_sha256':sha(path),
        'checkpoint_sha256':sha(PRIOR_CHECKPOINT),'seed':17,'layer_norm_epsilon':1e-5,
        'targets_opened':False,'fit':False,'read_log':sorted(guard.reads)})


def rgb_inputs():
    import torch
    # All initializer layers use canonical known-g vertical, not old V2 vertical.
    from reliability_observation_v7 import tracking,extract_observation
    from canonical_confirmation_inputs_v11 import tracking_row,pack_inputs
    plan,episodes,_,_,guard=context('rgb_inputs');torch.set_num_threads(2)
    with np.load(OUT/'prior/priors_seed17.npz',allow_pickle=False) as z:
        if z['ids'].tolist()!=plan['ids']:raise ValueError('Prior ID order changed')
        means=z['mean'].copy();scales=z['scale'].copy()
    records=[];tracks=[];started=time.perf_counter()
    for e,mean,scale in zip(episodes,means,scales):
        row=tracking_row(e);track=tracking(row,ROOT)
        record=extract_observation(row,ROOT,mean,scale,track)
        records.append(record);tracks.append(track)
    arrays,diagnostic=pack_inputs(episodes,records,tracks)
    path=OUT/'input/inputs.npz';npwrite(path,arrays);guard.allowed.add(path.resolve())
    write(OUT/'input/EXTRACTION_DIAGNOSTIC.json',dict(diagnostic,inputs_sha256=sha(path),
        ids=plan['ids'],elapsed_s=time.perf_counter()-started,targets_opened=False,
        initialization_forward_calls_total=int(arrays['forward_calls'].sum())))
    if not diagnostic['ready_for_frozen_head']:
        raise ValueError('Retained all rows; tracking/valid failure forbids READY seal, no new fallback')
    write(OUT/'input/INPUT_SEAL.json',{'status':'sealed','inputs_sha256':sha(path),'ids':plan['ids'],
        'plan_sha256':sha(OUT/'PLAN.json'),'observation_seal_sha256':plan['observation_seal_sha256'],
        'prior_seal_sha256':sha(OUT/'prior/PRIOR_SEAL.json'),'tracking_failures':0,'excluded':0,
        'raw_sigma_preserved':True,'alpha_attached_or_refit':False,'K_or_normalization_fit':False,
        'initialization_forward_calls_total':int(arrays['forward_calls'].sum()),
        'source_schema':{k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in arrays.items()},
        'targets_opened':False,'read_log':sorted(guard.reads)})


if __name__=='__main__':
    sys.path.insert(0,str(ROOT))
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
        HF_HOME=str(OUT/'runtime/hf'),PYTHONDONTWRITEBYTECODE='1')
    sys.dont_write_bytecode=True
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prior_gpu','prior_cpu','rgb_inputs']);args=parser.parse_args()
    try:{'prior_gpu':prior_gpu,'prior_cpu':prior_cpu,'rgb_inputs':rgb_inputs}[args.stage]()
    except BaseException as exc:
        write(OUT/f'FAILURE_{time.time_ns()}.json',{'stage':args.stage,'type':type(exc).__name__,'message':str(exc),'preserved':True,'auto_restart':False})
        raise
