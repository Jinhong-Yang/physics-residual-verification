"""Enumerate available post hoc mean-error comparisons with global Holm correction."""
from followup_common import *
def main():
    d,v=verify_original();models={};ref=load(P/'evidence/results/com_observation_v23/confirmation_v1/FAMILY_ACTION_ERRORS.npz')
    for k,a in ref.items():
        if k.endswith('__family_action_error'):models[k.split('__')[0]]=a
    s=load(P/'replayed/support/PREDICTIONS_AND_FAMILY_ERRORS.npz')
    for k,a in s.items():
        if k.startswith('confirmation__') and k.endswith('__family_error'):models['support_'+k.split('__')[1]]=a
    for filename,prefixes in [('anchor_predictions.npz',['huber','cauchy','profile_map']),('channel_predictions.npz',['pooled','position','deduplicated_numeric'])]:
        z=load(D/'reference_results'/filename)
        for name in prefixes:models[name]=action(v,{r:z[name+'__'+r] for r in REGIMES})[0]
    z=load(D/'reference_results/crossfit_predictions.npz');models['crossfit_visual']=action(v,{r:z['confirmation__True__'+r] for r in REGIMES})[0]
    z=load(D/'reference_results/direct_predictions.npz')
    for name in ['direct_ridge','direct_mlp']:models[name]=action(v,{r:z[name+'__confirmation'] for r in REGIMES})[0]
    models['vjepa_original']=load(P/'replayed/video_control/PREDICTIONS.npz')['confirmation__family_error']
    z=load(O/'representation_predictions.npz')
    for k,a in z.items():
        if k.startswith('evaluation__') and k.endswith('__family_loss'):models[k.split('__')[1]]=a
    models['laplace']=load(O/'laplace_predictions.npz')['family_loss'];models['vlmfree_map']=load(O/'vlmfree_predictions.npz')['family_loss']
    full=models['v22_vlm'];numeric=models['v22_numeric'];rows=[]
    # Exact duplicate prediction vectors need not inflate the comparison family.
    unique={}
    for name,a in models.items():
        duplicate=next((n for n,b in unique.items() if np.allclose(a,b,atol=1e-12,rtol=0)),None)
        if duplicate is None:unique[name]=a
    for name,a in unique.items():
        for label,b in [('original_full',full),('original_numeric',numeric)]:
            if not np.allclose(a,b,atol=1e-12,rtol=0):rows.append({'metric':'family action MAE','comparison':name+' minus '+label,**paired(a,b)})
    for a,b in [('qwen87','flow87'),('qwen87','numeric87'),('flow87','numeric87'),('pixels87','numeric87'),('vjepa87','numeric87')]:
        rows.append({'metric':'family action MAE','comparison':a+' minus '+b,**paired(models[a],models[b])})
    holm(rows);csvsave('global_action_holm.csv',rows)
    save('STATISTICS.json',{'tests':len(rows),'adjustment':'Holm across all listed unique mean-action comparisons',
      'rows':rows,'scope':'72-family action losses only; historical public episode bootstrap, correlations and covariance descriptive summaries use different estimands',
      'limitations':'Multiplicity correction does not undo reuse of evaluated cohorts, selected penalties, or adaptive prior analyses. No confirmatory superiority or equivalence conclusion.'})
    print('globally adjusted comparisons',len(rows))
if __name__=='__main__':main()
