"""Development Pareto and independently generated family calibration."""
from followup_common import *
from scipy.stats import chi2
import argparse
def out_cov(v,means,beta):
    out={}
    for r in REGIMES:
        a,c=anchor(v,r);delta=means[r]-a;cov=c.copy();cov[:,np.arange(3),np.arange(3)]+=(beta*delta)**2
        out[r]={'mean':means[r],'covariance':cov}
    return out
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pareto-only',action='store_true');args=ap.parse_args()
    d,v=verify_original();z=load(O/'representation_predictions.npz');rows=[]
    for cohort,dat in [('development',d),('evaluation',v)]:
        means={r:z[cohort+'__full192__'+r] for r in REGIMES}
        for beta in (0,.025,.05,.1,.2):
            met,_,_=evaluate(dat,out_cov(dat,means,beta));rows.append({'cohort':cohort,'beta':beta,**met})
    csvsave('covariance_pareto.csv',rows)
    # Choice uses development CRPS, with beta as fixed tie-break. No test coverage target.
    selected=min([r for r in rows if r['cohort']=='development'],key=lambda r:(r['crps'],r['beta']))['beta']
    save('CALIBRATION_SELECTION.json',{'criterion':'minimum development OOF CRPS, beta tie-break','selected_beta':selected,'rows':rows})
    print('development-selected beta',selected,flush=True)
    if args.pareto_only:return
    cal=load(F/'calibration_predictions.npz');targets=load(F/'calibration_targets.npz');assert np.array_equal(cal['ids'],targets['ids'])
    cal['truth']=targets['truth'];fs=sorted(set(cal['family_ids']));assert len(fs)==40
    scores=[]
    for r in REGIMES:
        mean=cal['full__'+r+'__mean'];cov=cal['anchor__'+r+'__covariance'].copy()
        delta=mean-cal['anchor__'+r+'__mean'];cov[:,np.arange(3),np.arange(3)]+=(selected*delta)**2
        score=np.empty(len(mean))
        for p,cols in SUBSPACES.items():
            ix=np.flatnonzero(cal['protocol']==p);err=(cal['truth']-mean)[ix][:,cols];c=cov[ix][:,cols][:,:,cols]
            mahal=np.einsum('ni,ni->n',err,np.linalg.solve(c,err[...,None])[...,0]);score[ix]=np.sqrt(mahal/chi2.ppf(.9,len(cols)))
        scores.append(score)
    scores=np.stack(scores);rng=np.random.default_rng(20260916);representative=[];maxima=[];sample=[]
    for f in fs:
        ix=np.flatnonzero(cal['family_ids']==f);flat=scores[:,ix].ravel();j=int(rng.integers(len(flat)))
        representative.append(flat[j]);maxima.append(flat.max());sample.append({'family':f,'flat_index':j,'score':float(flat[j]),'family_max':float(flat.max())})
    q=float(np.sort(representative)[36]);qmax=float(np.sort(maxima)[36]);results=[]
    means={r:z['evaluation__full192__'+r] for r in REGIMES};base=out_cov(v,means,selected)
    for mode,scale in [('beta_selected',1.),('representative_family_conformal',q),('family_max_conformal',qmax)]:
        pred={r:{'mean':base[r]['mean'],'covariance':base[r]['covariance']*scale**2} for r in REGIMES}
        met,_,_=evaluate(v,pred)
        for protocol in ['all',*SUBSPACES]:
            cover=[];family_all=[]
            for r in REGIMES:
                raw=score_physics(pred[r]['mean'],pred[r]['covariance'],v['truth'],v['active'],v['family_ids'],v['protocol'])['per_episode']
                # Use the actual active principal submatrix directly.
                flags=np.empty(len(v['ids']),bool)
                for p,cols in SUBSPACES.items():
                    ix=np.flatnonzero(v['protocol']==p);err=(v['truth']-pred[r]['mean'])[ix][:,cols];c=pred[r]['covariance'][ix][:,cols][:,:,cols]
                    m=np.einsum('ni,ni->n',err,np.linalg.solve(c,err[...,None])[...,0]);flags[ix]=m<=chi2.ppf(.9,len(cols))
                mask=np.ones(len(flags),bool) if protocol=='all' else v['protocol']==protocol
                cover.append(float(flags[mask].mean()));family_all.append(flags)
            joint=np.logical_and.reduce(family_all)
            famcov=np.mean([joint[(v['family_ids']==f)&(np.ones(len(joint),bool) if protocol=='all' else v['protocol']==protocol)].all() for f in sorted(set(v['family_ids']))])
            results.append({'mode':mode,'q':scale,'protocol':protocol,'coverage90':float(np.mean(cover)),
               'whole_family_coverage':float(famcov),'overall_width':met['projected_diameter'],'overall_nll':met['nll'],'overall_crps':met['crps']})
    csvsave('conformal_coverage.csv',results);csvsave('calibration_family_scores.csv',sample)
    save('CALIBRATION.json',{'calibration_families':40,'quantile_rank':37,'q':q,'q_familymax':qmax,'selected_beta':selected,
        'selection_uses_evaluation':False,'rows':results,'marginal_target':'one uniformly sampled episode/setup per random family',
        'guarantee_assumption':'exchangeable new families; family-max score targets simultaneous coverage instead',
        'evaluation_previously_inspected':True})
    print(json.dumps({'q':q,'qmax':qmax,'rows':results},indent=2))
if __name__=='__main__':main()
