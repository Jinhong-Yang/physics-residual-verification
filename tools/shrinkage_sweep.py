from extended_common import *
def main():
    dev,conf=data();rows=[]
    fulls=[cv_variant(dev),predict_variant(dev,conf)];nums=[cv_variant(dev,shrink=0),predict_variant(dev,conf,shrink=0)]
    for name,v,full,num in zip(('development','confirmation'),(dev,conf),fulls,nums):
        for s in (0,.1,.25,.5,1.):
            pred={r:num[r]+(s/.25)*(full[r]-num[r]) for r in REGIMES};m,_,_=evaluate(v,output(v,pred))
            if s==.25 and name=='confirmation':
                for r in REGIMES:np.testing.assert_allclose(pred[r],v[f'v22_vlm__{r}__mean'],rtol=0,atol=1e-12)
            rows.append({'cohort':name,'shrinkage':s,**m})
    csvout('shrinkage.csv',rows);print(rows,flush=True)
if __name__=='__main__':main()
