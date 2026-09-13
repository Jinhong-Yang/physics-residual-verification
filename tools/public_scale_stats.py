from extended_common import *
def summarize(y,pred):
    norm=np.linalg.norm(y,axis=1);den=((y-y.mean(0))**2).sum();result={'n':len(y),'mean_displacement_mm':float(norm.mean()*1000),'median_displacement_mm':float(np.median(norm)*1000),'sd_displacement_mm':float(norm.std(ddof=1)*1000),'q25_mm':float(np.quantile(norm,.25)*1000),'q75_mm':float(np.quantile(norm,.75)*1000),'methods':{}}
    for name,est in pred.items():
        mae=np.linalg.norm(est-y,axis=1).mean();result['methods'][name]={'episode_mae_mm':float(mae*1000),'normalized_mae':float(mae/norm.mean()),'vector_R2':float(1-((est-y)**2).sum()/den)}
    result['evaluation_mean_oracle_R2']=float(1-((y-y.mean(0))**2).sum()/den)
    return result
def main():
    z=load(D/'public_original_vectors.npz');rows={'episode_split':summarize(z['target'],{k:z[k] for k in z if k!='target'})}
    y=load(P/'evidence/group_holdout/EVALUATION_TARGETS.npz');p=load(P/'evidence/group_holdout/PREDICTIONS.npz')
    assert np.array_equal(y['ids'],p['ids'])
    rows['material_split']=summarize(y['target'],{k:p[k] for k in ('numeric','raw','gated')})
    write('PUBLIC_SCALES.json',rows);print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
