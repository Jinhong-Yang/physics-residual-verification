from extended_common import *
from scipy.stats import spearmanr,rankdata
def main():
    z=load(D/'real_diagnostic_inputs.npz');fs=sorted(set(z['family_ids']));target=np.array([z['target'][z['family_ids']==f].mean() for f in fs]);rows={}
    for key in ('v22_visual_score','v22_pooled_only_score','v22_motion_only_score','kinematic_deceleration_score'):
        x=z[key];family=np.array([x[z['family_ids']==f].mean() for f in fs]);rows[key]={'mean':float(x.mean()),'sd':float(x.std(ddof=1)),'q25':float(np.quantile(x,.25)),'median':float(np.median(x)),'q75':float(np.quantile(x,.75)),'family_spearman':float(spearmanr(family,target).statistic)}
    x=z['synthetic_score'];rows['synthetic_unforced_visual_score']={'mean':float(x.mean()),'sd':float(x.std(ddof=1)),'q25':float(np.quantile(x,.25)),'median':float(np.median(x)),'q75':float(np.quantile(x,.75))}
    # Observation-only path smoothness proxy; this is not labelled tracking error.
    center=z['cue_centers'];rough=np.linalg.norm(np.diff(center,n=2,axis=1),axis=2).mean(1);frough=np.array([rough[z['family_ids']==f].mean() for f in fs]);score=np.array([z['v22_visual_score'][z['family_ids']==f].mean() for f in fs]);rankerr=abs(rankdata(score)-rankdata(target))
    result={'score_statistics':rows,'roughness_vs_absolute_rank_error_rho':float(spearmanr(frough,rankerr).statistic),
      'contracts':{'synthetic_motion_channels':'dynamic and static position shifts divided by tracking sigma','real_motion_channels':'axis-projected pixel shifts; no matching noise normalization','real_score_is_physical_estimate':False,'appearance_prior_used_in_real_diagnostic':False,'tracking':'MIL tracker initialized from overlay; not red-component tracking','reported_tracker_failures':0,'videos':33,'size_stability':'MIL box does not estimate changing scale; cannot validate size accuracy','causal_failure_attribution':'not identified; feature-unit/domain mismatch is a documented difference, not an isolated intervention'}}
    write('REAL_DIAGNOSTICS.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
