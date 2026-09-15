"""Development-only selection and matched deduplicated representation controls."""
from followup_common import *
import time
from scipy.linalg import svd

def pack(v,x):
    out=dict(v);feat=v['features'].astype(float).copy()
    feat[:,:,24:56]=x[:,:32,None].transpose(0,2,1);feat[:,:,56:58]=x[:,32:].reshape(-1,8,2)
    out['features']=feat
    np.testing.assert_allclose(feature_views(feat,24)[1],x,atol=1e-12,rtol=0)
    return out
def project(a,b):
    center=a.mean(0);_,_,vt=svd(a-center,full_matrices=False,check_finite=False)
    return (a-center)@vt[:48].T,(b-center)@vt[:48].T
def main():
    d,v=verify_original();fold=folds(d,6);devrows=[];scores=[];saved={}
    candidates=[]
    for pen in (.1,1.,10.):
        pred=cv_variant(d,dedup=True,numeric_penalty=pen,shrink=0)
        loss,_=action(d,pred);candidates.append((float(loss.mean()),pen))
        devrows.append({'arm':'numeric87','numeric_penalty':pen,'visual_penalty':0,'dev_mm':float(loss.mean()*1000)})
    npen=min(candidates)[1];print('selected numeric penalty',npen,flush=True)
    for name,kwargs in [('numeric192',dict(shrink=0)),('full192',{}),('numeric87',dict(dedup=True,numeric_penalty=npen,shrink=0)),('full87_legacy',dict(dedup=True,numeric_penalty=npen))]:
        for cohort,dat,pred in [('development',d,cv_variant(d,**kwargs)),('evaluation',v,predict_variant(d,v,**kwargs))]:
            loss,_=action(dat,pred);scores.append({'arm':name,'cohort':cohort,'mae_mm':float(loss.mean()*1000),'numeric_penalty':npen if '87' in name else 1,'visual_penalty':0 if name.startswith('numeric') else 10})
            saved[cohort+'__'+name+'__family_loss']=loss
            for r in REGIMES:saved[cohort+'__'+name+'__'+r]=pred[r]
    cd,cv=load(F/'development_classical.npz'),load(F/'confirmation_classical.npz')
    assert np.array_equal(cd['ids'],d['ids']) and np.array_equal(cv['ids'],v['ids'])
    raw={'qwen':(feature_views(d['features'],24)[1],feature_views(v['features'],24)[1]),
         'flow':(cd['flow'],cv['flow']),'pixels':(cd['pixels'],cv['pixels'])}
    jd,jv=load(P/'evidence/video_control/development_features.npz'),load(P/'evidence/video_control/confirmation_features.npz')
    assert np.array_equal(jd['ids'],d['ids']) and np.array_equal(jv['ids'],v['ids'])
    raw['vjepa']=(jd['features'],jv['features'])
    for arm,(x,y) in raw.items():
        cached=[]
        for f in range(6):
            tr=np.flatnonzero(fold!=f);te=np.flatnonzero(fold==f)
            a,b=project(x[tr],x[te]) if x.shape[1]!=48 else (x[tr],x[te])
            cached.append((tr,te,pack(subset(d,tr),a),pack(subset(d,te),b)))
        selection=[]
        for pen in (1.,10.,100.):
            pred={r:np.empty_like(d['truth']) for r in REGIMES}
            for tr,te,train,test in cached:
                p=predict_variant(train,test,dedup=True,numeric_penalty=npen,penalty=pen)
                for r in REGIMES:pred[r][te]=p[r]
            loss,_=action(d,pred);selection.append((float(loss.mean()),pen,pred,loss))
            devrows.append({'arm':arm+'87','numeric_penalty':npen,'visual_penalty':pen,'dev_mm':float(loss.mean()*1000)})
        _,pen,pred,dloss=min(selection,key=lambda x:(x[0],x[1]))
        a,b=project(x,y) if x.shape[1]!=48 else (x,y)
        final=predict_variant(pack(d,a),pack(v,b),dedup=True,numeric_penalty=npen,penalty=pen)
        for cohort,dat,out in [('development',d,pred),('evaluation',v,final)]:
            loss,_=action(dat,out)
            scores.append({'arm':arm+'87','cohort':cohort,'mae_mm':float(loss.mean()*1000),'numeric_penalty':npen,'visual_penalty':pen})
            saved[cohort+'__'+arm+'87__family_loss']=loss
            for r in REGIMES:saved[cohort+'__'+arm+'87__'+r]=out[r]
        print('completed',arm,'penalty',pen,flush=True)
    # Strong anchor is a descriptive existing benchmark; retain all original arrays.
    anchors=load(P/'evidence/extended/reference_results/anchor_predictions.npz')
    ladder={}
    for name in ('huber','cauchy','profile_map'):
        loss,_=action(v,{r:anchors[name+'__'+r] for r in REGIMES});ladder[name]=loss
    a=saved['evaluation__full192__family_loss'];b=ladder['profile_map']
    focal=[{'comparison':'full192 minus profiled MAP',**paired(a,b)},
           {'comparison':'qwen87 minus flow87',**paired(saved['evaluation__qwen87__family_loss'],saved['evaluation__flow87__family_loss'])},
           {'comparison':'qwen87 minus numeric87',**paired(saved['evaluation__qwen87__family_loss'],saved['evaluation__numeric87__family_loss'])}]
    holm(focal)
    exploratory=[]
    for arm in ('full87_legacy','flow87','pixels87','vjepa87'):
        exploratory.append({'comparison':arm+' minus numeric87',**paired(saved['evaluation__'+arm+'__family_loss'],saved['evaluation__numeric87__family_loss'])})
    holm(exploratory)
    csvsave('representation_scores.csv',scores);csvsave('representation_development.csv',devrows)
    csvsave('focal_statistics.csv',focal);csvsave('representation_adjusted.csv',exploratory)
    np.savez_compressed(O/'representation_predictions.npz',**saved)
    result={'selected_numeric_penalty':npen,'scores':scores,'focal_contrasts':focal,'exploratory_contrasts':exploratory,
       'flow_extraction_median_ms_concurrent':float(np.median(cv['milliseconds'])),'flow_failed_transitions':int(cv['failed_transitions'].sum()),
       'original_arrays_verified_atol':1e-12,'new_independent_confirmation':False,'supervision_matched':False,
       'prior_free':False,'claims_equivalence':False}
    save('REPRESENTATIONS.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
