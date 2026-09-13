"""Recompute paired statistics from bundled loss archives; Python + NumPy only.

This replays saved predictions, not GPU feature extraction or model training.
Public raw/gate decompositions and fixed-reference main intervals are post hoc.
"""
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def main():
    out=ROOT/'replayed';out.mkdir(exist_ok=True)
    main_root=ROOT/'evidence/results/com_observation_v23/confirmation_v1'
    pub_root=ROOT/'evidence/results/com_observation_v39/physprobe_independent_confirmation_v1'
    main_values=np.load(main_root/'FAMILY_ACTION_ERRORS.npz',allow_pickle=False)
    main_result=json.loads((main_root/'RESULT.json').read_text())
    public=np.load(pub_root/'CONFIRMATION_ERRORS.npz',allow_pickle=False)
    public_result=json.loads((pub_root/'RESULT.json').read_text())
    rows=[]
    draws=main_values['bootstrap_draw']
    losses={k.split('__')[0]:main_values[k] for k in main_values.files if k.endswith('__family_action_error')}
    for name,loss in losses.items():
        np.testing.assert_allclose(loss.mean(),main_result['scores'][name]['mean_two_regimes']['action_mae_m'],rtol=0,atol=1e-12)
    candidate=losses['v22_vlm']
    for ref in ('unit_rgb','huber_1.345_rgb','cauchy_2.385_rgb','v22_numeric','draw_wise_strongest'):
        if ref=='draw_wise_strongest':
            reference=np.stack([losses[k][draws].mean(1) for k in ('unit_rgb','huber_1.345_rgb','cauchy_2.385_rgb')]).min(0)
            boot=candidate[draws].mean(1)-reference
            point=candidate.mean()-min(losses[k].mean() for k in ('unit_rgb','huber_1.345_rgb','cauchy_2.385_rgb'))
        else:
            delta=candidate-losses[ref];boot=delta[draws].mean(1);point=delta.mean()
        lo,hi=np.quantile(boot,[.025,.975])
        rows.append(dict(task='main',comparison='full minus '+ref,n=len(candidate),delta_mm=point*1000,low95_mm=lo*1000,high95_mm=hi*1000,upper95_mm=np.quantile(boot,.95)*1000,
                         analysis='original replay' if ref in ('draw_wise_strongest','v22_numeric') else 'post hoc fixed-reference diagnostic'))
    np.testing.assert_allclose(rows[-1]['upper95_mm']/1000,main_result['gates']['primary']['candidate_minus_drawwise_strongest_numeric_one_sided95_upper_m'],rtol=0,atol=1e-12)
    draws=public['bootstrap_draw']
    for name in ('constant_prior','numeric','raw_vlm','gated_vlm'):
        err=public['error_'+name]
        np.testing.assert_allclose([err.mean(),np.sqrt(np.mean(err**2)),np.median(err)],
            [public_result['methods'][name][k] for k in ('vector_mae_m','vector_rmse_m','median_error_m')],rtol=0,atol=1e-12)
    for candidate,ref in (('raw_vlm','numeric'),('gated_vlm','raw_vlm'),('gated_vlm','numeric')):
        delta=public['error_'+candidate]-public['error_'+ref]
        boot=delta[draws].mean(1);lo,hi=np.quantile(boot,[.025,.975])
        rows.append(dict(task='public',comparison=candidate+' minus '+ref,n=len(delta),delta_mm=delta.mean()*1000,low95_mm=lo*1000,high95_mm=hi*1000,upper95_mm=np.quantile(boot,.95)*1000,
                         analysis='original replay' if candidate=='gated_vlm' and ref=='numeric' else 'post hoc unadjusted diagnostic'))
    np.testing.assert_allclose([rows[-1]['low95_mm']/1000,rows[-1]['high95_mm']/1000],public_result['primary_vlm_increment']['two_sided_95pct_bootstrap_interval_m'],rtol=0,atol=1e-12)
    with (out/'paired_statistics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    result={'saved_main_losses_replay':True,'saved_public_mae_rmse_median_replay':True,
            'original_primary_intervals_replay':True,'absolute_tolerance_m':1e-12,
            'full_training_reproduced':False,'posterior_proper_scores_recomputed':False,'contrasts':rows}
    (out/'REPLAY_RESULT.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='contrasts'},indent=2))

if __name__=='__main__':main()
