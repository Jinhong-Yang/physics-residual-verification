"""Pure fixed1080 inference/evaluation contracts. No files, models, or fitting."""
import numpy as np
from com_confirmation_statistics_v11 import METHODS, REGIMES, PROTOCOLS

N=1080
FAMILIES=tuple(f'canonical_v11_f{i:03d}' for i in range(1,73))
HEADS=('qwen_warp_joint','rgb_warp_joint')
COUNTS={REGIMES[0]:dict(zip(PROTOCOLS,(57,47,81))),REGIMES[1]:dict(zip(PROTOCOLS,(57,47,75)))}
TOTAL_CALLS=468000
BOOTSTRAP_SEED=202609130732
GAIN=.6177441562124601
CORE=('mean','covariance','nuisance','weights','forward_calls_per_observation','numerical_log_guard_evaluations_per_observation')
DIAGNOSTICS=('static_bias','dynamic_residual','warp_support_fraction','empty_region_fallback','initial_query_fallback')
BATCH_KEYS=('prior_mean','prior_scale','linearization_mean','J','Jn','residual','times','xz','sigma_xz','force','force_switch','valid','active','nuisance')


def require(ok,message):
    if not bool(ok):raise ValueError(message)


def batches():return tuple(np.arange(i,i+8) for i in range(0,N,8))


def numeric_contract(a,episodes):
    from canonical_confirmation_data_v11 import index_contract
    index_contract(episodes)
    require(len(episodes)==N and set(e['family'] for e in episodes)==set(FAMILIES),'Fixed72 families/1080 clips')
    ids=np.array([e['id'] for e in episodes]);require(np.array_equal(a['ids'],ids),'Numeric ID order')
    require(ids.tolist()==sorted(ids.tolist()),'Original sorted1080 order')
    for k,v in [('family_ids',[e['family'] for e in episodes]),('split',[e['split'] for e in episodes]),('protocol',[e['protocol'] for e in episodes])]:
        require(np.array_equal(a[k],v),'Numeric metadata '+k)
    for k in BATCH_KEYS:
        require(len(a[k])==N and np.isfinite(a[k]).all(),'Finite1080 numeric field '+k)
    require(a['valid'].shape==(N,8) and a['valid'].dtype==np.bool_ and a['valid'].all(),'All8 valid, no filtering')
    require(a['pixels'].shape==a['xz'].shape==a['sigma_xz'].shape==(N,8,2),'Raw observation shape')
    require(a['pixels'].dtype==a['xz'].dtype==np.float64 and np.isfinite(a['pixels']).all(),'Original double coordinates')
    require((a['sigma_xz']>0).all() and (a['prior_scale']>0).all(),'Strictly positive original noise/prior scale')
    require(np.array_equal(a['times'],np.asarray([e['timestamps_s'] for e in episodes],np.float64)),'Actual time identity')
    active=np.array([[p=='multiple_forces',p!='bounce',p=='bounce'] for p in a['protocol']],bool)
    require(np.array_equal(a['active'],active),'Fixed physical active dimensions')
    require(a['nuisance'].shape==(N,4),'Shared original nuisance shape')


def validate_head(v,a,episodes):
    for k,expected in [('ids',a['ids']),('family',a['family_ids']),('split',a['split']),('cue_failure',[e['initial_cue']['failure'] for e in episodes])]:
        require(np.array_equal(v[k],expected),'Head metadata '+k)
    for k,shape,dtype in [('pixels',(N,8,2),np.float64),('static_bias',(N,2),np.float32),('dynamic_residual',(N,8,2),np.float32),
                          ('warp_support_fraction',(N,8),np.float32),('empty_region_fallback',(N,8),np.bool_),('initial_query_fallback',(N,),np.bool_)]:
        require(v[k].shape==shape and v[k].dtype==dtype and np.isfinite(v[k]).all(),'Head output '+k)
    require((v['dynamic_residual'][:,0]==0).all(),'First dynamic residual zero')
    require(np.array_equal(v['pixels'],a['pixels']+v['static_bias'].astype(np.float64)[:,None]+v['dynamic_residual'].astype(np.float64)),'Original float64 skip arithmetic')
    require(((v['warp_support_fraction']>=-1e-6)&(v['warp_support_fraction']<=1+1e-6)).all(),'Warp support range')


def validate_mean(v,a):
    require(np.array_equal(v['ids'],a['ids']),'Mean IDs')
    for k in ('pixels','xz'):
        require(v[k].shape==(N,8,2) and v[k].dtype==np.float64 and np.isfinite(v[k]).all(),'Finite raw mean '+k)


def validate_physics(v,a,regime,alpha_general,alpha_setup):
    validate_mean(v,a)
    for k,shape in [('mean',(N,3)),('covariance',(N,3,3)),('nuisance',(N,4)),('weights',(N,8))]:
        require(v[k].shape==shape and v[k].dtype==np.float64 and np.isfinite(v[k]).all(),'Physics shape/finite '+k)
    for k in CORE[-2:]:require(v[k].shape==(N,) and v[k].dtype.kind in 'iu' and (v[k]>=0).all(),'Actual call ledger '+k)
    require(np.array_equal(v[CORE[-2]],[COUNTS[regime][str(p)] for p in a['protocol']]),'Fixed per-row solver calls')
    np.testing.assert_allclose(v['covariance'],v['covariance'].swapaxes(-1,-2),atol=1e-12,rtol=1e-12)
    np.linalg.cholesky(v['covariance'])
    require(np.array_equal(v['mean'][~a['active']],a['prior_mean'][~a['active']]),'Inactive prior mean exact')
    prior=np.eye(3)[None]*a['prior_scale'][:,:,None]**2
    mask=~(a['active'][:,:,None]&a['active'][:,None,:])
    require(np.array_equal(v['covariance'][mask],prior[mask]),'Inactive prior covariance exact')
    alpha=np.array([alpha_setup if regime==REGIMES[1] and p=='bounce' else alpha_general[str(p)] for p in a['protocol']])
    require(np.array_equal(v['weights'],alpha[:,None]*a['valid']),'Original alpha times unit-valid weights')
    require('world_covariance' not in v and 'confidence_q' not in v,'No V10 covariance correction')


def all_outputs_before_truth(values,means,a,alpha_general,alpha_setup):
    require(set(values)==set(REGIMES) and set(means)==set(METHODS),'Both regimes/all means required')
    slide=a['protocol']!='bounce';calls=0
    for name in METHODS:
        validate_mean(means[name],a)
        for regime in REGIMES:
            require(set(values[regime])==set(METHODS),'All5 before any target')
            v=values[regime][name];validate_physics(v,a,regime,alpha_general,alpha_setup)
            for k in ('pixels','xz'):require(np.array_equal(v[k],means[name][k]),'Same sealed coordinates')
        for k in values[REGIMES[0]][name]:
            require(np.array_equal(values[REGIMES[0]][name][k][slide],values[REGIMES[1]][name][k][slide]),'Setup slide exact reuse '+k)
        calls+=int(values[REGIMES[0]][name][CORE[-2]].sum())+int(values[REGIMES[1]][name][CORE[-2]][~slide].sum())
    require(calls==TOTAL_CALLS,'468000 new calls, reused setup slides not charged twice')
    return calls


def family_cube(row_errors,episodes):
    """Join original ID order to family/protocol/variant; never reshape sorted IDs."""
    require(len(row_errors)==len(episodes)==N,'All1080 row errors')
    out=np.full((72,3,5),np.nan,np.float64);seen=set()
    for e,value in zip(episodes,row_errors):
        key=(FAMILIES.index(e['family']),PROTOCOLS.index(e['protocol']),e['variant'])
        require(type(key[2]) is int and 0<=key[2]<5 and key not in seen,'Unique family/protocol/variant')
        out[key]=value;seen.add(key)
    require(len(seen)==N,'No dropped cells')
    return out


def com_descriptive(pixels,target,episodes):
    require(pixels.shape==target.shape==(N,8,2),'All1080 COM shapes')
    error=np.linalg.norm(pixels[:,1:]-target[:,1:],axis=-1)
    require(np.isfinite(error).all(),'No COM row filtering')
    cube=family_cube(error.mean(1),episodes)
    return dict(frames=7560,excluded=0,scope='descriptive; no candidate or confirmation gate',
        family_macro_postinit_L2_px=float(cube.mean()),
        by_protocol={p:float(cube[:,j].mean()) for j,p in enumerate(PROTOCOLS)},
        by_family={f:float(cube[i].mean()) for i,f in enumerate(FAMILIES)},
        p95_postinit_L2_px=float(np.quantile(error,.95))),error


def natural_truth(rows,ids):
    """Join sealed physical rows and convert to the solver's coordinates."""
    require(len(rows)==N and len(ids)==N,'All1080 physical targets required')
    by={r['id']:r for r in rows};require(len(by)==N and set(by)==set(ids),'Physical target ID join')
    ordered=[by[str(i)] for i in ids]
    values=np.asarray([[r['mass_kg'],r['dynamic_friction'],r['restitution']] for r in ordered],np.float64)
    require(values.shape==(N,3) and np.isfinite(values).all(),'Finite physical truth')
    require((values[:,:2]>0).all() and ((values[:,2]>0)&(values[:,2]<1)).all(),'Natural truth domain')
    return np.column_stack((np.log(values[:,0]),np.log(values[:,1]),np.log(values[:,2]/(1-values[:,2]))))


def action_error_cube(per_episode,episodes):
    """Use the canonical scorer's eight-action MAE without reimplementing it."""
    required={'action_error_m','action_absolute_error_by_time_m','predicted_action_m','truth_action_m'}
    require(required<=set(per_episode),'Canonical per-episode action fields required')
    row=np.asarray(per_episode['action_error_m'])
    bytime=np.asarray(per_episode['action_absolute_error_by_time_m'])
    require(row.dtype==np.float64 and row.shape==(N,) and np.isfinite(row).all() and (row>=0).all(),'Finite1080 action MAE')
    require(bytime.dtype==np.float64 and bytime.shape==(N,8) and np.isfinite(bytime).all(),'Finite eight-time action errors')
    np.testing.assert_allclose(row,bytime.mean(1),atol=0,rtol=0)
    return family_cube(row,episodes)


def seal_records(records,names):
    """Pure schema check used before callers open any registered payload."""
    require(set(records)==set(names),'Complete fixed artifact registry')
    for name in names:
        r=records[name]
        require(set(r)=={'path','sha256'} and isinstance(r['path'],str),'Exact path/hash artifact entry')
        require(isinstance(r['sha256'],str) and len(r['sha256'])==64 and set(r['sha256'])<={'0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f'},'Lowercase SHA256')
    return True
