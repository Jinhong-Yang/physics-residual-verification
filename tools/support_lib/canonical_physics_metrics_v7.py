"""Pure NumPy/SciPy V7 physical scoring; no I/O, inference, fit, or filtering.

mean/truth are [log(mass/kg), log(dynamic friction), logit(restitution)].
Full-three-coordinate Gaussian scores and protocol-permitted marginal scores
are distinct. Active covariance is a principal submatrix, never conditioned on
inactive truth. Every family must have all three protocols; all supplied rows
contribute. Family macro means: rows within family/protocol, then three equally
weighted protocols, then equally weighted families. Input row order is retained.
"""
import math
import numpy as np
from scipy.special import expit, ndtr
from scipy.stats import chi2
from main3d_numeric import slide
from canonical_bounce_reference_v7 import canonical_bounce_state

COORDINATES=('log_mass','log_friction','logit_restitution')
SUBSPACES={'multiple_forces':(0,1),'unforced_slide':(1,),'bounce':(2,)}
PROTOCOLS=tuple(SUBSPACES)
ACTION_TIMES=np.arange(1,9,dtype=np.float64)/8
Z90=1.6448536269514722
SCALAR_FIELDS=('joint_nll','joint_nll_per_dimension','mean_marginal_crps',
               'joint90_coverage','mean_marginal90_coverage','mean_marginal90_width','mean_transformed_mae')


def _array(value,shape,name):
    a=np.asarray(value)
    if a.shape!=shape or a.dtype.kind not in 'fiu' or not np.isfinite(a).all():
        raise ValueError(f'{name}: expected finite real array {shape}')
    return a.astype(np.float64)


def _validate(mean,covariance,truth,active,family_ids,protocols):
    mean=np.asarray(mean)
    if mean.ndim!=2 or mean.shape[1]!=3 or len(mean)==0:raise ValueError('mean must be nonempty N x3')
    n=len(mean);mean=_array(mean,(n,3),'mean');truth=_array(truth,(n,3),'truth')
    covariance=_array(covariance,(n,3,3),'covariance');active=np.asarray(active)
    if active.shape!=(n,3) or active.dtype.kind!='b':raise ValueError('active must be boolean N x3')
    family=np.asarray(family_ids);protocol=np.asarray(protocols)
    for name,a in [('family_ids',family),('protocols',protocol)]:
        if a.shape!=(n,) or a.dtype.kind not in 'US' or np.any(a==''):raise ValueError(name+': expected nonempty strings N')
    family=family.astype(str);protocol=protocol.astype(str)
    if set(protocol)!=set(PROTOCOLS):raise ValueError('All three supported protocols required; no unknown protocol')
    for p,cols in SUBSPACES.items():
        expected=np.zeros(3,bool);expected[list(cols)]=True
        if not np.all(active[protocol==p]==expected):raise ValueError('active/protocol mismatch: '+p)
    if not np.allclose(covariance,covariance.swapaxes(1,2),atol=1e-12,rtol=1e-12):raise ValueError('Covariance must be symmetric')
    try:np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:raise ValueError('Covariance must be positive definite; no jitter/filter') from exc
    for f in np.unique(family):
        if set(protocol[family==f])!=set(PROTOCOLS):raise ValueError('Every family must have every protocol: '+f)
    return mean,covariance,truth,active,family,protocol


def gaussian_rows(mean,covariance,truth):
    """Unaggregated scores for one fixed-dimensional marginal Gaussian."""
    n,k=mean.shape
    try:
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            factor=np.linalg.cholesky(covariance);error=truth-mean
            whitened=np.linalg.solve(factor,error[...,None])[...,0]
            mahalanobis=np.square(whitened).sum(1)
            logdet=2*np.log(np.diagonal(factor,axis1=1,axis2=2)).sum(1)
            joint_nll=.5*(mahalanobis+logdet+k*math.log(2*math.pi))
            sigma=np.sqrt(np.diagonal(covariance,axis1=1,axis2=2));z=error/sigma
            crps=sigma*(z*(2*ndtr(z)-1)+2*np.exp(-.5*z*z)/math.sqrt(2*math.pi)-1/math.sqrt(math.pi))
            width=2*Z90*sigma;marginal=(np.abs(z)<=Z90).astype(float);mae=np.abs(error)
            rows=dict(joint_nll=joint_nll,joint_nll_per_dimension=joint_nll/k,
                mean_marginal_crps=crps.mean(1),joint90_coverage=(mahalanobis<=chi2.ppf(.9,k)).astype(float),
                mean_marginal90_coverage=marginal.mean(1),mean_marginal90_width=width.mean(1),mean_transformed_mae=mae.mean(1),
                mahalanobis_squared=mahalanobis,marginal_crps=crps,marginal90_coverage=marginal,
                marginal90_width=width,marginal_transformed_mae=mae)
    except (FloatingPointError,np.linalg.LinAlgError) as exc:raise ValueError('Nonfinite/invalid Gaussian score; no row removal') from exc
    if not all(np.isfinite(v).all() for v in rows.values()):raise ValueError('Nonfinite Gaussian score')
    return rows


def _summary(rows,mask):
    return {k:(float(v[mask].mean()) if v.ndim==1 else v[mask].mean(0).tolist()) for k,v in rows.items()}


def _average_dict(values):
    return {k:(float(np.mean([v[k] for v in values])) if np.ndim(values[0][k])==0 else np.mean([v[k] for v in values],axis=0).tolist()) for k in values[0]}


def _fixed_dimension_summary(rows,family,protocol,columns):
    families=sorted(set(family));by_protocol={};cells={};family_summary={}
    for p in sorted(set(protocol)):
        mask=protocol==p;by_family={f:_summary(rows,mask&(family==f)) for f in families}
        by_protocol[p]=dict(observations=int(mask.sum()),dimensions=len(columns),coordinates=[COORDINATES[c] for c in columns],
            pooled=_summary(rows,mask),family_macro=_average_dict(list(by_family.values())),by_family=by_family)
        for f in families:cells[f,p]=by_family[f]
    for f in families:family_summary[f]=_average_dict([cells[f,p] for p in by_protocol])
    return dict(dimensions=len(columns),coordinates=[COORDINATES[c] for c in columns],
        chi_square90_threshold=float(chi2.ppf(.9,len(columns))),pooled=_summary(rows,np.ones(len(family),bool)),
        family_macro=_average_dict(list(family_summary.values())),by_protocol=by_protocol,by_family=family_summary)


def _parameters(transformed):
    clipped=np.clip(transformed[:,:2],-30,30);natural=np.column_stack([np.exp(clipped),expit(transformed[:,2])])
    hit=transformed[:,:2]!=clipped
    receipt=dict(exp_guard_limit=[-30,30],rows_with_exp_guard=int(hit.any(1).sum()),exp_guard_values=int(hit.sum()),
        exp_guard_by_coordinate={'log_mass':int(hit[:,0].sum()),'log_friction':int(hit[:,1].sum())},
        sigmoid_e_zero_rows=int((natural[:,2]==0).sum()),sigmoid_e_one_rows=int((natural[:,2]==1).sum()))
    return natural,receipt


def _actions(natural,protocol):
    """Absolute x displacement for slides, height above floor for bounce.

    e=1 uses a separate exact elastic period, since the independent canonical
    NumPy oracle intentionally accepts only e<1. No parameter is discarded.
    """
    output=np.empty((len(natural),8),float)
    for i,((mass,mu,e),p) in enumerate(zip(natural,protocol)):
        if p=='multiple_forces':output[i]=slide(ACTION_TIMES,0.,.25,mu,1/mass,(.45,1.1),.35)
        elif p=='unforced_slide':output[i]=slide(ACTION_TIMES,0.,1.,mu,1/mass,(0.,0.),.35)
        elif p=='bounce':
            if e<1:output[i]=canonical_bounce_state(ACTION_TIMES,.85,e,9.81)[0]
            else:
                hit=math.sqrt(2*.85/9.81);speed=9.81*hit;period=2*hit
                before=ACTION_TIMES<=hit;output[i,before]=.85-.5*9.81*ACTION_TIMES[before]**2
                elapsed=np.remainder(ACTION_TIMES[~before]-hit,period)
                output[i,~before]=speed*elapsed-.5*9.81*elapsed**2
        else:raise ValueError('Unsupported action protocol')
    if not np.isfinite(output).all():raise ValueError('Nonfinite action prediction')
    return output


def score_physics(mean,covariance,truth,active,family_ids,protocols):
    """Return {'metrics': JSON-compatible dict, 'per_episode': numpy arrays}.

    No identity joining or split selection occurs here: caller must first seal
    and verify row alignment. No quality/visibility mask is accepted. Gaussian
    MAE/CRPS/widths are in the transformed coordinates, NOT kg or natural e.
    Action evaluation is the independent analytic canonical contract, not an
    actual-engine rollout. Bounce heights are not initial-height-subtracted.
    """
    mean,cov,truth,active,family,protocol=_validate(mean,covariance,truth,active,family_ids,protocols)
    n=len(mean);families=sorted(set(family));full=gaussian_rows(mean,cov,truth)
    full_summary=_fixed_dimension_summary(full,family,protocol,(0,1,2))
    active_rows={k:np.empty(n,float) for k in (*SCALAR_FIELDS,'mahalanobis_squared')};active_by_protocol={}
    active_dimension=np.empty(n,np.int64)
    for p,cols in SUBSPACES.items():
        ix=protocol==p;columns=list(cols)
        rows=gaussian_rows(mean[ix][:,columns],cov[ix][:,columns][:,:,columns],truth[ix][:,columns])
        active_by_protocol[p]=_fixed_dimension_summary(rows,family[ix],protocol[ix],cols)
        for k in active_rows:active_rows[k][ix]=rows[k]
        active_dimension[ix]=len(cols)
    active_families={f:_average_dict([_summary(active_rows,(family==f)&(protocol==p)) for p in PROTOCOLS]) for f in families}
    active_summary=dict(scope='Protocol-permitted marginal subspaces; not proof of local identifiability; inactive truth never conditioned on',
        aggregation_note='Pooled counts each supplied row once; macro counts each family/protocol equally. Joint NLL across varying dimensions is descriptive; per-dimension NLL is explicit.',
        pooled=_summary(active_rows,np.ones(n,bool)),family_macro=_average_dict(list(active_families.values())),
        by_protocol=active_by_protocol,by_family=active_families)
    predicted,pg=_parameters(mean);target,tg=_parameters(truth)
    pred_action=_actions(predicted,protocol);truth_action=_actions(target,protocol)
    error=np.abs(pred_action-truth_action);episode_error=error.mean(1)
    cells={f:{p:float(episode_error[(family==f)&(protocol==p)].mean()) for p in PROTOCOLS} for f in families}
    fam={f:float(np.mean(list(cells[f].values()))) for f in families}
    action=dict(scope='Independent canonical analytic new-action error; not actual engine',times_s=ACTION_TIMES.tolist(),
        units='m',bounce_coordinate='height above floor; both truth and prediction start atH=.85m',
        contract={'multiple_forces':{'forces_N':[.45,1.1],'switch_s':.35,'initial_v_m_s':.25},
                  'unforced_slide':{'forces_N':[0,0],'initial_v_m_s':1.},'bounce':{'H_m':.85,'gravity_m_s2':9.81,'initial_v_m_s':0.}},
        pooled_mae_m=float(episode_error.mean()),family_macro_mae_m=float(np.mean(list(fam.values()))),
        by_protocol={p:dict(observations=int((protocol==p).sum()),pooled_mae_m=float(episode_error[protocol==p].mean()),
            family_macro_mae_m=float(np.mean([cells[f][p] for f in families]))) for p in PROTOCOLS},
        by_family_mae_m=fam,by_family_protocol_mae_m=cells,parameter_guard_counts={'prediction':pg,'truth':tg})
    metrics=dict(schema_version='canonical_physics_metrics_v7_v1',observations=n,families=len(families),excluded=0,
        coordinate_space=list(COORDINATES),gaussian_covariance_policy='SPD checked, no jitter/recalibration/row filtering',
        aggregation='equal rows within family/protocol; equal3protocols; equal families',
        family_protocol_counts={f:{p:int(((family==f)&(protocol==p)).sum()) for p in PROTOCOLS} for f in families},
        full_3d_gaussian=full_summary,active_subspace_gaussian=active_summary,action=action)
    per_episode=dict(action_error_m=episode_error,action_absolute_error_by_time_m=error,predicted_action_m=pred_action,
        truth_action_m=truth_action,active_dimensions=active_dimension,full_3d_gaussian=full,active_subspace_gaussian=active_rows)
    return dict(metrics=metrics,per_episode=per_episode)
