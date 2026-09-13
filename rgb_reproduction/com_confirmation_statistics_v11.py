"""Pure prospective 72-family confirmation statistics; no I/O or data fitting.

One shared family resample covers all 12 prespecified margins. Original-sample
SEs are fixed: this is a standardized centered max-absolute bootstrap, not a
bootstrap-t procedure with a newly estimated SE in every replicate.
"""
import math
import numpy as np

FAMILIES=72
VARIANTS=5
REPLICATES=10000
CONFIDENCE=.95
CHECKPOINT_SEED=17
REGIMES=('generalized_translation','setup_translation')
PROTOCOLS=('multiple_forces','unforced_slide','bounce')
METHODS=('qwen_centered_attenuation','qwen_warp_joint','rgb_centroid','rgb_warp_joint','qwen_warp_joint_static')
CANDIDATE=METHODS[0]
KINDS=('mean_vs_0.9_RGB','force_vs_1.05_RGB','free_vs_1.05_RGB','bounce_vs_1.05_RGB','mean_vs_RGBjoint','mean_vs_old_Qwen')
LABELS=tuple(r+'/'+k for r in REGIMES for k in KINDS)


def require(condition,message):
    if not bool(condition):raise ValueError(message)


def family_contract(family_ids,expected_family_ids,checkpoint_seed):
    actual=tuple(family_ids);expected=tuple(expected_family_ids)
    require(checkpoint_seed==CHECKPOINT_SEED and type(checkpoint_seed) is int,'Fixed seed17 checkpoint; bootstrap families are not training-seed replicates')
    for value in [actual,expected]:
        require(len(value)==FAMILIES and len(set(value))==FAMILIES,'Exactly72 unique families; no augmentation or stopping change')
        require(all(isinstance(x,str) and x for x in value),'Nonempty family string IDs')
    require(actual==expected,'Family order/identity must equal the separately frozen confirmation manifest')
    return actual


def family_margins(errors,family_ids,expected_family_ids,checkpoint_seed=CHECKPOINT_SEED):
    """errors[regime][method]: float64[72,3protocols,5variants] action MAE in m.

    Every supplied episode contributes. Episodes within a family are averaged,
    never treated as independent model seeds or bootstrap observations.
    """
    ids=family_contract(family_ids,expected_family_ids,checkpoint_seed)
    require(set(errors)==set(REGIMES),'Exactly two fixed information conditions')
    invalid={};affected=np.zeros(FAMILIES,bool)
    for r in REGIMES:
        require(set(errors[r])==set(METHODS),'Fixed five methods; no arm substitution')
        invalid[r]={}
        for name in METHODS:
            x=errors[r][name]
            require(isinstance(x,np.ndarray) and x.dtype==np.float64 and x.shape==(FAMILIES,3,VARIANTS),'Each method requires float64[72,3,5]; no model-seed axis')
            bad=~np.isfinite(x)|(x<0)
            invalid[r][name]=int(bad.sum());affected|=bad.any(axis=(1,2))
    # These slides are the same saved observation/physics/truth quantities.
    # Equality checks preserve this dependency rather than inventing replicates.
    for name in METHODS:
        require(np.array_equal(errors[REGIMES[0]][name][:,:2],errors[REGIMES[1]][name][:,:2],equal_nan=True),'General/setup slides must be exact reuse within each family')
    blocks=[]
    with np.errstate(over='ignore',invalid='ignore'):
        for r in REGIMES:
            v={name:errors[r][name].mean(axis=2) for name in METHODS}
            c,rgb,matched,old=(v[name] for name in [CANDIDATE,'rgb_centroid','rgb_warp_joint','qwen_warp_joint'])
            blocks.append(np.column_stack((c.mean(1)-.9*rgb.mean(1),c-1.05*rgb,c.mean(1)-matched.mean(1),c.mean(1)-old.mean(1))))
    margins=np.concatenate(blocks,axis=1)
    require(margins.shape==(FAMILIES,12),'Twelve fixed family margins')
    return margins,dict(families=FAMILIES,clips=FAMILIES*3*VARIANTS,variants_per_family_protocol=VARIANTS,
        ordered_family_ids=list(ids),checkpoint_seed=checkpoint_seed,invalid_values_by_method=invalid,
        invalid_input_values=sum(sum(x.values()) for x in invalid.values()),affected_families=int(affected.sum()),
        duplicated_slide_margins_preserved=True,excluded_families=0,excluded_margins=0)


def point_decisions(estimates):
    require(estimates.shape==(12,) and np.isfinite(estimates).all(),'Finite twelve marginal estimates')
    by={}
    for j,r in enumerate(REGIMES):
        m=estimates[j*6:(j+1)*6]
        p=bool(np.all(m[:4]<=0) and m[4]<0)
        by[r]=dict(practical_P=p,added_A=bool(m[5]<0),
            mean_10pct=bool(m[0]<=0),protocol_guard={pr:bool(m[k+1]<=0) for k,pr in enumerate(PROTOCOLS)},
            strict_matched_RGB=bool(m[4]<0),strict_old_Qwen=bool(m[5]<0))
    P=all(v['practical_P'] for v in by.values());A=all(v['added_A'] for v in by.values())
    return dict(P=P,A=A,both_point_gates=bool(P and A),by_regime=by)


def _nullable(x):return float(x) if np.isfinite(x) else None


def _validate_seed(bootstrap_seed):
    require(type(bootstrap_seed) is int and 0<=bootstrap_seed<2**64,'An explicit prespecified bootstrap seed is required')


def _center(margins):
    # Identical represented values give exactly zero variance, even when their
    # value is not exactly representable in decimal (e.g. a constant -0.1).
    relative=margins-margins[:1]
    relative_mean=relative.mean(axis=0)
    return margins[0]+relative_mean,relative-relative_mean[None]


def simultaneous_band(margins,bootstrap_seed):
    """Approximate simultaneous95% two-sided intervals using fixed family SE.

    No zero-SE coordinate is dropped. One undefined coordinate makes the joint
    inference inconclusive while retaining all estimates/SEs and denominators.
    """
    _validate_seed(bootstrap_seed)
    require(isinstance(margins,np.ndarray) and margins.dtype==np.float64 and margins.shape==(FAMILIES,12),'Fixed72×12 float64 margins; sample size cannot be changed')
    finite=np.isfinite(margins)
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        estimate,centered=_center(margins)
        se=np.sqrt(np.sum(centered*centered,axis=0)/(FAMILIES-1)/FAMILIES)
    bad=~np.isfinite(estimate)|~np.isfinite(se)|(se<=0)|~finite.all(axis=0)
    rows=[dict(label=LABELS[j],estimate_m=_nullable(estimate[j]),SE_m=_nullable(se[j]),
               lower_m=None,upper_m=None,upper_strictly_negative=None,SE_valid=bool(not bad[j])) for j in range(12)]
    result=dict(status='inconclusive' if bad.any() else 'computed',families=FAMILIES,margins=12,
        planned_bootstrap_replicates=REPLICATES,executed_bootstrap_replicates=0,bootstrap_seed=bootstrap_seed,
        confidence_level=CONFIDENCE,critical_value=None,intervals=rows,joint_confirmation_pass=None,
        invalid_margin_values=int((~finite).sum()),invalid_SE_labels=[LABELS[j] for j in np.flatnonzero(bad)],
        excluded_families=0,excluded_margins=0,
        procedure='single shared family resample; centered max-absolute statistic divided by original sample SE',
        scope='approximate simultaneous family-sampling intervals, conditional fixed seed17 pipeline; not exact finite-sample coverage')
    if bad.any():
        result['reason']='At least one nonfinite margin/estimate/SE or zero SE; no coordinate removal, jitter, or additional sampling'
        return result
    rng=np.random.Generator(np.random.PCG64(bootstrap_seed))
    draws=rng.integers(0,FAMILIES,size=(REPLICATES,FAMILIES),dtype=np.int64)
    statistics=np.empty(REPLICATES,np.float64)
    # A block of twelve coordinates uses exactly the same family indices.
    # Centering first avoids subtracting large nearly equal bootstrap means.
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        for start in range(0,REPLICATES,250):
            mean=centered[draws[start:start+250]].mean(axis=1)
            statistics[start:start+250]=np.max(np.abs(mean/se[None]),axis=1)
    result['executed_bootstrap_replicates']=REPLICATES
    if not np.isfinite(statistics).all():
        result.update(status='inconclusive',reason='Nonfinite bootstrap statistic; all observations retained')
        return result
    # Fixed empirical inverse CDF: 9500th order statistic among10000 draws.
    order=math.ceil(CONFIDENCE*REPLICATES)-1
    critical=float(np.sort(statistics)[order])
    lower=estimate-critical*se;upper=estimate+critical*se
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        result.update(status='inconclusive',reason='Nonfinite interval endpoint; no repair')
        return result
    for j,row in enumerate(rows):row.update(lower_m=float(lower[j]),upper_m=float(upper[j]),upper_strictly_negative=bool(upper[j]<0))
    result.update(critical_value=critical,critical_order_one_based=order+1,joint_confirmation_pass=bool((upper<0).all()))
    return result


def analyze_confirmation(errors,family_ids,expected_family_ids,bootstrap_seed,checkpoint_seed=CHECKPOINT_SEED):
    """Caller verifies artifact seals/episode joins before passing these arrays."""
    _validate_seed(bootstrap_seed)
    margins,integrity=family_margins(errors,family_ids,expected_family_ids,checkpoint_seed)
    if integrity['invalid_input_values']:
        # Even finite derived margins from an invalid action input cannot pass.
        with np.errstate(over='ignore',invalid='ignore'):estimates,_=_center(margins)
        report=dict(status='inconclusive',reason='Nonfinite or negative action MAE input; no row removal',
            families=FAMILIES,margins=12,planned_bootstrap_replicates=REPLICATES,executed_bootstrap_replicates=0,
            bootstrap_seed=bootstrap_seed,joint_confirmation_pass=None,
            interval_estimates_m=[_nullable(x) for x in estimates],excluded_families=0,excluded_margins=0)
        point=None
    else:
        report=simultaneous_band(margins,bootstrap_seed)
        with np.errstate(over='ignore',invalid='ignore'):estimates,_=_center(margins)
        point=point_decisions(estimates) if np.isfinite(estimates).all() else None
    overall=None if report['status']=='inconclusive' or point is None else bool(point['both_point_gates'] and report['joint_confirmation_pass'])
    return dict(schema='fixed72_family_confirmation_statistics_v11_draft_v1',integrity=integrity,point_decision=point,
        simultaneous_inference=report,overall_joint_confirmation=overall,
        original_development_decisions_unchanged=True,model_seed_replication=False,
        independent_upstream_validation=False,no_post_result_sample_increase=True,goal_complete=False)
