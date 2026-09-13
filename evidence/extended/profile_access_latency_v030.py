"""Measure exact local inference stages on 100 fixed, outcome-free cases."""
import os
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
import sys,json,time,gc,platform,subprocess
from pathlib import Path
import numpy as np
import torch
from PIL import Image
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'scripts')]
from scripts import stream_com_internal_features_v16 as stream
from scripts import train_com_physics_internal_readout_v12 as base
from scripts import evaluate_canonical_reliability_transfer_v12 as robust
from scripts.extract_canonical_confirmation_inputs_v11 import forward_first_image
from reliability_update_v2 import robust_weights
from sequential_robust_readout_v22 import predict,REGIMES
from projection_prior import AppearancePrior
from canonical_confirmation_inputs_v11 import tracking_row
from reliability_observation_v7 import tracking,extract_observation
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def timed(fn,gpu=False):
    if gpu:torch.cuda.synchronize()
    t=time.perf_counter();v=fn()
    if gpu:torch.cuda.synchronize()
    return v,(time.perf_counter()-t)*1000
def main():
    torch.set_num_threads(2);out=R/'paper/ieee_access_initial_submission/overleaf/replayed/extended';out.mkdir(exist_ok=True)
    episodes=json.loads((R/'data/canonical_confirmation_v23/INPUT_INDEX.json').read_text())['episodes'];inp=load(R/'cache/canonical_confirmation_v23/input/inputs.npz')
    indices=np.concatenate([np.flatnonzero(inp['protocol']==p)[:n] for p,n in [('multiple_forces',34),('unforced_slide',33),('bounce',33)]]);episodes=[episodes[i] for i in indices];values={k:v[indices] for k,v in inp.items() if v.ndim and len(v)==len(inp['ids'])};rows=[{'id':e['id'],'protocol':str(inp['protocol'][i])} for e,i in zip(episodes,indices)]
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    model=Qwen3VLForConditionalGeneration.from_pretrained(stream.MODEL,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda().eval().requires_grad_(False);proc=AutoProcessor.from_pretrained(stream.MODEL,local_files_only=True)
    first=forward_first_image(torch,model,proc,R/episodes[0]['frame_paths'][0]);head=AppearancePrior(len(first['feature']));head.load_state_dict(torch.load(R/'results/projection_prior/seed17/best.pt',map_location='cpu',weights_only=True)['state']);head.eval()
    for j,e in enumerate(episodes):
        def prior():
            z=forward_first_image(torch,model,proc,R/e['frame_paths'][0])['feature'].float()[None]
            z=torch.nn.functional.layer_norm(z,(z.shape[1],),eps=1e-5)
            with torch.inference_mode():return head(z)
        prior_output,rows[j]['first_image_prior_ms']=timed(prior,True)
        np.testing.assert_allclose(prior_output[0].numpy()[0],values['prior_mean'][j],atol=1e-5,rtol=0)
        np.testing.assert_allclose(prior_output[1].numpy()[0],values['prior_scale'][j],atol=1e-5,rtol=0)
    del model,proc,head;gc.collect();torch.cuda.empty_cache();models=stream.load_models();_,processor,order,vision,decoder=models
    # Separate hooks synchronize GPU work; preprocessing/transfer overhead remains explicit.
    for j,e in enumerate(episodes):
        def preprocess():
            images=[np.asarray(Image.open(R/f).convert('RGB')).copy() for f in e['frame_paths']]
            payload=processor(images=images,size={'shortest_edge':512**2,'longest_edge':512**2},return_tensors='pt');return images,payload
        (images,payload),rows[j]['frame_preprocess_ms']=timed(preprocess)
        with torch.inference_mode():
            def encode():return vision(payload['pixel_values'].to('cuda',dtype=torch.bfloat16),grid_thw=payload['image_grid_thw'].to('cuda'),return_dict=True).last_hidden_state.reshape(8,1024,1024)[:,order.to('cuda')].contiguous()
            dense,rows[j]['vision_encoder_ms']=timed(encode,True)
            def decode():
                rgb=torch.stack([torch.from_numpy(im).permute(2,0,1).float()/255 for im in images])[None].cuda();bbox=torch.tensor([e['initial_cue']['bbox_xyxy']],dtype=torch.float32,device='cuda')
                return decoder(rgb,dense[None],bbox,torch.from_numpy(values['pixels'][j:j+1]).double().cuda(),torch.from_numpy(values['times'][j:j+1]).float().cuda(),return_diagnostics=True)
            _,rows[j]['observation_decoder_ms']=timed(decode,True)
        if (j+1)%25==0:print('GPU stages',j+1,flush=True)
    del models,vision,decoder,dense;gc.collect();torch.cuda.empty_cache()
    support=load(R/'paper/ieee_access_initial_submission/overleaf/evidence/support/confirmation.npz');ck=load(R/'results/com_observation_v22/final_development_fit_v1/CHECKPOINT.npz')
    for j,e in enumerate(episodes):
        i=indices[j]
        def observe():
            row=tracking_row(e);tr=tracking(row,R);return extract_observation(row,R,inp['prior_mean'][i],inp['prior_scale'][i],tr)
        _,rows[j]['tracking_initialization_ms']=timed(observe)
        batch=base.tensor_batch({k:v[j:j+1] for k,v in values.items()})
        def physical():
            translated,_,innovation=robust.initial_feature_state(batch,batch['xz']);w=robust_weights(innovation,translated['valid'],kind='huber').double();adjusted=dict(translated,sigma_xz=robust.weighted_sigma(translated,w))
            with torch.no_grad():general=base.refine(adjusted)
            fits={r:{k:general[k].numpy() for k in ('mean','covariance')} for r in REGIMES};setup_ms=0.
            if bool(batch['active'][0,2]):
                def setup():
                    with torch.no_grad():return base.fit_setup_bounce(adjusted['xz'][:,:,1],adjusted['sigma_xz'][:,:,1],adjusted['times'],adjusted['prior_mean'],adjusted['prior_scale'],adjusted['valid'],torch.full((1,),base.ALPHA,dtype=torch.float64))
                special,setup_ms=timed(setup);fits[REGIMES[1]]={k:special[k].numpy() for k in ('mean','covariance')}
            return fits,setup_ms
        (fits,setup_ms),rows[j]['robust_both_conventions_ms']=timed(physical);rows[j]['setup_bounce_75_calls_ms']=setup_ms
        _,rows[j]['residual_heads_ms']=timed(lambda:predict(ck,support['features'][i:i+1],support['active'][i:i+1],support['protocol'][i:i+1],fits))
        rows[j]['shared_prior_anchor_total_ms']=sum(rows[j][k] for k in ('first_image_prior_ms','tracking_initialization_ms','robust_both_conventions_ms'))
        rows[j]['full_component_total_ms']=rows[j]['shared_prior_anchor_total_ms']+sum(rows[j][k] for k in ('frame_preprocess_ms','vision_encoder_ms','observation_decoder_ms','residual_heads_ms'))
    summary={k:{'median':float(np.median([row[k] for row in rows])),'q25':float(np.quantile([row[k] for row in rows],.25)),'q75':float(np.quantile([row[k] for row in rows],.75))} for k in rows[0] if k.endswith('_ms')}
    setup=[row['setup_bounce_75_calls_ms'] for row in rows if row['protocol']=='bounce'];summary['setup_bounce_75_calls_ms']={k:float(np.quantile(setup,q)) for k,q in [('median',.5),('q25',.25),('q75',.75)]}
    result={'n':100,'batch':1,'cuda_synchronized':True,'model_load_excluded':True,'totals':'component sums across staged, warmed runs; not contiguous cold-start latency','baseline':'shared first-image VLM prior + RGB tracking + anchor; visual-free pipeline is not the compared model',
        'hardware':{'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda,'python':platform.python_version(),'platform':platform.platform(),'driver':subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip()},'summary_ms':summary,'episodes':rows}
    (out/'LATENCY.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
