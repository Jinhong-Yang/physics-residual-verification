from followup_common import *
from scipy.stats import spearmanr,rankdata
def correlation(x,y,draw):
    a=rankdata(x[draw],axis=1);b=rankdata(y[draw],axis=1);a-=a.mean(1)[:,None];b-=b.mean(1)[:,None]
    den=np.sqrt((a*a).sum(1)*(b*b).sum(1));keep=den>0;boot=(a*b).sum(1)[keep]/den[keep]
    return {'rho':float(spearmanr(x,y).statistic),'low95':float(np.quantile(boot,.025)),'high95':float(np.quantile(boot,.975)),'degenerate_bootstrap_draws':int((~keep).sum())}
def main():
    verify_original();old=load(D/'real_diagnostic_inputs.npz');new=load(F/'real_unit_corrected.npz')
    assert np.array_equal(old['ids'],new['ids']);families=sorted(set(old['family_ids']))
    agg=lambda a:np.array([a[old['family_ids']==f].mean() for f in families])
    target=agg(old['target']);draw=np.random.default_rng(20260915).integers(len(families),size=(20000,len(families)))
    rows=[]
    for label,key,nkey in [('combined','v22_visual_score','combined'),('pooled','v22_pooled_only_score','pooled'),('position','v22_motion_only_score','position'),('kinematic_short','kinematic_deceleration_score','kinematic_dimensionless')]:
        for stage,a in [('original',old[key]),('noise_normalized',new[nkey])]:rows.append({'arm':label,'stage':stage,**correlation(agg(a),target,draw)})
    if (F/'long_kinematic_replay.npz').exists():
        z=load(F/'long_kinematic_replay.npz');assert np.array_equal(z['ids'],old['ids'])
        for arm in ('bridge','scalar'):
            for stage in ('original','normalized'):rows.append({'arm':arm+'_152frame','stage':stage,**correlation(agg(z[arm+'_'+stage]),target,draw)})
    csvsave('real_unit_correlations.csv',rows)
    save('REAL_FIXED.json',{'rows':rows,'families':len(families),'videos':len(new['ids']),
       'interpretation':'post hoc unit-normalized residual score, not friction estimation; orientation/domain mismatch not eliminated',
       'raw_reconstruction_error':float(np.max(abs(new['raw_replay']-old['v22_visual_score'])))})
    print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
