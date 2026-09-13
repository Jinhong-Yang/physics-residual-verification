from extended_common import *
from scipy.stats import binomtest
def main():
    _,v=data();means={n:{r:v[f'{n}__{r}__mean'] for r in REGIMES} for n in ('unit_rgb','huber_1.345_rgb','cauchy_2.385_rgb','v22_numeric','v22_vlm')}
    for n in ('legacy_qwen_warp_joint','rgb_warp_joint'):
        z=load(D/(n+'.npz'));means[n]={r:z[p+'mean'] for r,p in zip(REGIMES,('generalized_','setup_'))}
    if (O/'anchor_predictions.npz').exists():
        z=load(O/'anchor_predictions.npz')
        for n in ('huber','cauchy','profile_map'):means[n]={r:z[n+'__'+r] for r in REGIMES}
    rows=[];coords=[];families=sorted(set(v['family_ids']))
    for n,pred in means.items():
        fam,eps=action(v,pred);cells=[]
        for r,ep in zip(REGIMES,eps):
            for p,cols in SUBSPACES.items():
                mask=v['protocol']==p;val=np.array([ep[mask&(v['family_ids']==f)].mean() for f in families]);cells.append(val.mean())
                rows.append({'method':n,'regime':r,'protocol':p,'action_mm':float(val.mean()*1000)})
                error=np.abs(pred[r]-v['truth'])
                for c in range(3):
                    values=np.array([error[mask&(v['family_ids']==f),c].mean() for f in families])
                    coords.append({'method':n,'regime':r,'protocol':p,'coordinate':c,'family_macro_mean':float(values.mean()) if c in cols else 'N/A','family_macro_median':float(np.median(values)) if c in cols else 'N/A'})
        np.testing.assert_allclose(np.mean(cells),fam.mean(),atol=1e-12,rtol=0)
    full,_=action(v,means['v22_vlm']);num,_=action(v,means['v22_numeric']);wins=int((full<num).sum());sign=binomtest(wins,len(full),.5)
    csvout('protocols.csv',rows);csvout('coordinates.csv',coords);write('SIGN_TEST.json',{'wins':wins,'families':len(full),'two_sided_p':float(sign.pvalue),'bootstrap':contrast(v,full,num),'interpretation':'sign test tests directional frequency; mean bootstrap tests mean magnitude; hypotheses differ'})
    print('protocol aggregation matches; exact sign p',sign.pvalue)
if __name__=='__main__':main()
