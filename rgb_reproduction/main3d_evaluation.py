"""Shared immutable inputs and label-free utilities for main evaluation."""
import hashlib,json,math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'data/main3d_v1'
DEST=ROOT/'results/main3d_evaluation'
SEEDS=[17,43,101]
NAMES=['D','S','DG','SG','DG_no_rollout','SG_no_rollout','FrozenHead','LoRA_last2_qv','ResidualMLP']
BOUNDS={'mass':[.15,1.5],'mu':[.05,.35],'e':[.25,.9]}
ZERO_PROMPT=('Return only one JSON object with keys mass_kg, dynamic_friction, restitution.\n'
 'Use a positive number for mass_kg, a nonnegative number for dynamic_friction,\n'
 'and a number from0to1 for restitution. Use null for a property when the supplied\n'
 'observations do not provide enough information to estimate it. Do not include\n'
 'an explanation or additional keys.')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.partial');tmp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)

def entries():
    out=[]
    for seed in SEEDS:
        for name in NAMES+['FrozenFeatureMLP']:
            folder=ROOT/('results/main3d_probe' if name=='FrozenFeatureMLP' else 'results/main3d')/f'{name}_seed{seed}'
            out.append({'id':f'{name}_seed{seed}','name':name,'seed':seed,'folder':folder})
    return out

def verify_lock():
    lock=read_json(DEST/'LOCK.json');assert lock['status']=='locked'
    for path,digest in lock['files'].items():assert sha(ROOT/path)==digest,f'Changed locked file: {path}'
    assert lock['zero_shot_prompt']==ZERO_PROMPT
    return lock

def verify_refinement_lock():
    lock=read_json(DEST/'refinement/LOCK.json');assert lock['status']=='locked'
    for path,digest in lock['files'].items():assert sha(ROOT/path)==digest,f'Changed refinement file: {path}'
    return lock

def observation_rows(splits=('calibration','test')):
    lock=read_json(BASE/'LOCK.json');assert lock['status']=='locked'
    for name in ['groups','observations']:assert sha(BASE/f'{name}.jsonl')==lock['files'][name]
    groups=[g for g in map(json.loads,(BASE/'groups.jsonl').read_text().splitlines()) if g['split'] in splits]
    ids={g['id'] for g in groups}
    obs={r['id']:r for r in map(json.loads,(BASE/'observations.jsonl').read_text().splitlines()) if r['id'] in ids}
    assert len(obs)==len(groups)
    return groups,obs

def natural(mean):
    x=np.asarray(mean,dtype=float)
    # Retain native out-of-support values; cap only the exponent machine range.
    return np.stack([np.exp(np.clip(x[...,0],-700,700)),np.exp(np.clip(x[...,1],-700,700)),
      np.exp(-np.logaddexp(0,-x[...,2]))],axis=-1)

def operational(values):
    return np.clip(values,[.15,.05,.25],[1.5,.35,.9])

def parse_zero(text,prior_mean):
    fallback=natural(np.asarray(prior_mean));status=['unparsed']*3;native=[None]*3;obj=None
    try:
        cleaned=text.strip()
        if cleaned.startswith('```'):
            lines=cleaned.splitlines();cleaned='\n'.join(lines[1:-1]) if lines[-1].strip()=='```' else cleaned
        obj=json.loads(cleaned)
        if not isinstance(obj,dict) or set(obj)!={'mass_kg','dynamic_friction','restitution'}:raise ValueError('keys')
    except (ValueError,TypeError):return {'native':native,'point':fallback.tolist(),'status':status,'parse_ok':False}
    for i,key in enumerate(['mass_kg','dynamic_friction','restitution']):
        v=obj[key]
        if v is None:status[i]='abstained';continue
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v):status[i]='invalid_type';continue
        if (i==0 and v<=0) or (i==1 and v<0) or (i==2 and not 0<=v<=1):status[i]='invalid_range';continue
        native[i]=float(v);fallback[i]=v;status[i]='valid'
    return {'native':native,'point':fallback.tolist(),'status':status,'parse_ok':True}
