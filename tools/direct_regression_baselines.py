"""Post hoc direct-coordinate baselines. All selection uses development families."""
from extended_common import *
import torch,time
torch.set_num_threads(1)
def view(v):return np.concatenate(feature_views(v['features'],24),1)
def train_mlp(x,y,epochs,seed):
    torch.manual_seed(seed);net=torch.nn.Sequential(torch.nn.Linear(240,64),torch.nn.ReLU(),torch.nn.Linear(64,64),torch.nn.ReLU(),torch.nn.Linear(64,3))
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=1e-4)
    for epoch in range(epochs):
        opt.zero_grad();loss=(net(x)-y).square().mean();loss.backward();opt.step()
    return net
def mlp(train,test):
    x=view(train);xt=view(test);y=train['truth'];f=folds(train,5);tr=f!=0;va=~tr;preds=[];ledger=[]
    for seed in (17,43,101):
        mx=x[tr].mean(0);sx=x[tr].std(0).clip(1e-6);my=y[tr].mean(0);sy=y[tr].std(0).clip(1e-6)
        a=torch.tensor((x[tr]-mx)/sx,dtype=torch.float32);b=torch.tensor((y[tr]-my)/sy,dtype=torch.float32)
        av=torch.tensor((x[va]-mx)/sx,dtype=torch.float32);bv=torch.tensor((y[va]-my)/sy,dtype=torch.float32)
        net=train_mlp(a,b,0,seed);opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=1e-4);best=float('inf');chosen=1
        for epoch in range(1,301):
            opt.zero_grad();loss=(net(a)-b).square().mean();loss.backward();opt.step()
            with torch.no_grad():val=float((net(av)-bv).square().mean())
            if val<best-1e-7:best=val;chosen=epoch
            if epoch-chosen>=25:break
        mx=x.mean(0);sx=x.std(0).clip(1e-6);my=y.mean(0);sy=y.std(0).clip(1e-6)
        net=train_mlp(torch.tensor((x-mx)/sx,dtype=torch.float32),torch.tensor((y-my)/sy,dtype=torch.float32),chosen,seed)
        with torch.no_grad():pred=net(torch.tensor((xt-mx)/sx,dtype=torch.float32)).numpy().astype(float)*sy+my
        preds.append(pred);ledger.append({'seed':seed,'epochs':chosen,'inner_validation_mse_standardized':best})
    return np.mean(preds,0),ledger
def select_ridge(train):
    f=folds(train,5);scores=[]
    for p in (.1,1,10,100,1000):
        pred=np.empty_like(train['truth'])
        for fold in range(5):
            tr=f!=fold;te=~tr;m=fit_ridge(view(train)[tr],train['truth'][tr],p);pred[te]=ridge_predict(m,view(train)[te])
        loss,_=action(train,{r:pred for r in REGIMES});scores.append((float(loss.mean()),p))
    return min(scores)[1],scores
def main():
    dev,conf=data();rows=[];saved={};ledger=[];start=time.perf_counter()
    for method in ('direct_ridge','direct_mlp'):
        oof=np.empty_like(dev['truth']);f=folds(dev,6)
        for fold in range(6):
            tr=np.flatnonzero(f!=fold);te=np.flatnonzero(f==fold);train=subset(dev,tr);test=subset(dev,te)
            if method=='direct_ridge':
                penalty,scores=select_ridge(train);pred=ridge_predict(fit_ridge(view(train),train['truth'],penalty),view(test));entry={'penalty':penalty,'grid':scores}
            else:pred,entry=mlp(train,test)
            oof[te]=pred;ledger.append({'method':method,'fold':fold,'selection':entry})
        if method=='direct_ridge':
            penalty,scores=select_ridge(dev);final=ridge_predict(fit_ridge(view(dev),dev['truth'],penalty),view(conf));entry={'penalty':penalty,'grid':scores}
        else:final,entry=mlp(dev,conf)
        ledger.append({'method':method,'fold':'final','selection':entry})
        for cohort,v,pred in [('development',dev,oof),('confirmation',conf,final)]:
            loss,_=action(v,{r:pred for r in REGIMES});base,_=action(v,{r:v[f'v22_vlm__{r}__mean'] for r in REGIMES}) if cohort=='confirmation' else action(v,cv_variant(dev))
            rows.append({'method':method,'cohort':cohort,'action_mm':float(loss.mean()*1000),**contrast(v,loss,base)});saved[method+'__'+cohort]=pred
        print(rows[-2:],flush=True)
    csvout('direct.csv',rows);write('DIRECT_SELECTION.json',ledger);np.savez_compressed(O/'direct_predictions.npz',**saved);print('seconds',time.perf_counter()-start)
if __name__=='__main__':main()
