"""Extract official frozen V-JEPA 2 features from the original eight-frame RGB inputs."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
import transformers
from transformers import AutoVideoProcessor, VJEPA2Model

R=Path(__file__).resolve().parents[1]
O=R/'results/access_video_control_v1'
C=R/'cache/access_video_control_v1';C.mkdir(exist_ok=True)
P=R/'paper/ieee_access_initial_submission/overleaf'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int,default=0);args=parser.parse_args()
    plan=read(O/'PLAN.json')
    paths={'old900':R/'cache/com_observation_v7/INPUT_INDEX.json',
           'v11':R/'data/canonical_confirmation_v11/INPUT_INDEX.json',
           'confirmation':R/'data/canonical_confirmation_v23/INPUT_INDEX.json'}
    lookup={k:{e['id']:e for e in read(p)['episodes']} for k,p in paths.items()}
    rows=[]
    for cohort in ('development','confirmation'):
        with np.load(P/f'evidence/support/{cohort}.npz',allow_pickle=False) as z: ids=z['ids'].tolist()
        for identifier in ids:
            src,raw=identifier.split('::') if '::' in identifier else ('confirmation',identifier)
            e=lookup[src][raw]
            assert len(e['frame_paths'])==8
            rows.append({'cohort':cohort,'id':identifier,'frame_paths':e['frame_paths'],'frame_sha256':e['frame_sha256']})
    assert len(rows)==3060
    if args.limit:rows=rows[:args.limit]
    signature=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
    torch.set_num_threads(2);torch.manual_seed(20260913)
    model_dir=R/'models/vjepa2-vitl-fpc64-256'
    processor=AutoVideoProcessor.from_pretrained(model_dir,local_files_only=True)
    model=VJEPA2Model.from_pretrained(model_dir,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
    model.requires_grad_(False)
    torch.cuda.reset_peak_memory_stats();started=time.perf_counter();done=0
    with torch.inference_mode():
        for offset in range(0,len(rows),4):
            batch=rows[offset:offset+4];dest=C/f'batch_{offset:05d}_{len(batch)}.npz'
            expected=np.array([r['id'] for r in batch])
            if dest.exists():
                with np.load(dest,allow_pickle=False) as z:
                    assert z['plan_sha256'].item()==signature and np.array_equal(z['ids'],expected)
                done+=len(batch);continue
            videos=[]
            for row in batch:
                frames=[]
                for path,digest in zip(row['frame_paths'],row['frame_sha256']):
                    file=R/path
                    assert sha(file)==digest,file
                    with Image.open(file) as im:frames.append(np.asarray(im.convert('RGB')).copy())
                videos.append(torch.from_numpy(np.stack(frames)).permute(0,3,1,2))
            inp=processor(videos=videos,do_sample_frames=False,return_tensors='pt')
            x=inp['pixel_values_videos'].cuda().to(torch.bfloat16)
            assert x.shape[1]==8,x.shape
            features=model.get_vision_features(x).float().mean(1).cpu().numpy()
            assert features.shape==(len(batch),1024) and np.isfinite(features).all()
            np.savez_compressed(dest,ids=expected,features=features,plan_sha256=np.asarray(signature))
            done+=len(batch)
            if offset%80==0 or done==len(rows):print(json.dumps({'extracted':done,'total':len(rows),'seconds':round(time.perf_counter()-started,1)}),flush=True)
    if args.limit:return
    features=np.concatenate([np.load(C/f'batch_{i:05d}_{len(rows[i:i+4])}.npz')['features'] for i in range(0,len(rows),4)])
    for cohort in ('development','confirmation'):
        ix=np.array([r['cohort']==cohort for r in rows])
        np.savez_compressed(O/f'{cohort}_features.npz',ids=np.array([r['id'] for r in rows])[ix],features=features[ix])
    manifest={'status':'completed','plan_sha256':signature,'revision':plan['revision'],
              'episodes':len(rows),'frames':len(rows)*8,'feature_dimensions':1024,
              'original_frame_hashes_verified':True,'model_training':False,'targets_loaded':False,
              'torch':torch.__version__,'transformers':transformers.__version__,'gpu':torch.cuda.get_device_name(),
              'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'elapsed_seconds_this_run':time.perf_counter()-started,
              'model_sha256':sha(model_dir/'model.safetensors'),'index_hashes':{k:sha(p) for k,p in paths.items()},
              'features':{k:sha(O/f'{k}_features.npz') for k in ('development','confirmation')}}
    (O/'EXTRACTION.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest),flush=True)
if __name__=='__main__':main()
