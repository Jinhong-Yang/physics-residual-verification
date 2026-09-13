from extended_common import *
def main():
    dev,conf=data();rows=[];saved={};diag=[]
    for cross in (False,True):
        for cohort,v,pred in [('development',dev,cv_variant(dev,crossfit=cross)),('confirmation',conf,predict_variant(dev,conf,crossfit=cross))]:
            loss,_=action(v,pred);base,_=action(v,{r:v[f'v22_numeric__{r}__mean'] for r in REGIMES}) if cohort=='confirmation' else action(v,cv_variant(dev,shrink=0))
            rows.append({'cohort':cohort,'target':'crossfit' if cross else 'in_sample','action_mm':float(loss.mean()*1000),**contrast(v,loss,base)})
            for r in REGIMES:saved[f'{cohort}__{cross}__{r}']=pred[r]
            if not cross and cohort=='confirmation':
                for r in REGIMES:np.testing.assert_allclose(pred[r],v[f'v22_vlm__{r}__mean'],rtol=0,atol=1e-12)
    for f in range(6):
        mask=folds(dev,6)==f;ds=[];predict_variant(subset(dev,np.flatnonzero(~mask)),subset(dev,np.flatnonzero(mask)),fallback=False,diagnostics=ds)
        diag.extend([{'fold':f,**row} for row in ds])
    ds=[];predict_variant(dev,conf,fallback=False,diagnostics=ds);diag.extend([{'fold':'confirmation',**row} for row in ds])
    csvout('crossfit.csv',rows);csvout('bounce_diagnostics.csv',diag);np.savez_compressed(O/'crossfit_predictions.npz',**saved);write('CROSSFIT.json',rows);print(rows,flush=True)
if __name__=='__main__':main()
