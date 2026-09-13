from extended_common import *
def main():
    dev,conf=data();grid=[];rows=[];saved={}
    for view in ('full','pooled','position','deduplicated_numeric'):
        candidates=[]
        for p in ((.1,1,10) if view=='deduplicated_numeric' else (1,10,100)):
            kw={'dedup':True,'numeric_penalty':p} if view=='deduplicated_numeric' else {'view':view,'penalty':p}
            cv=cv_variant(dev,**kw);loss,_=action(dev,cv);val=float(loss.mean()*1000);grid.append({'view':view,'penalty':p,'development_mm':val});candidates.append((val,p,kw))
        val,p,kw=min(candidates,key=lambda a:(a[0],a[1]));pred=predict_variant(dev,conf,**kw);loss,_=action(conf,pred);base,_=action(conf,{r:conf[f'v22_numeric__{r}__mean'] for r in REGIMES})
        rows.append({'view':view,'selected_penalty':p,'development_mm':val,'confirmation_mm':float(loss.mean()*1000),**contrast(conf,loss,base)})
        for r in REGIMES:saved[view+'__'+r]=pred[r]
        print(rows[-1],flush=True)
    csvout('channel_grid.csv',grid);csvout('channels.csv',rows);np.savez_compressed(O/'channel_predictions.npz',**saved)
if __name__=='__main__':main()
