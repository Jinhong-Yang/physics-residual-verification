"""Refit/replay all CPU extensions and compare against release reference arrays."""
from extended_common import *
import subprocess,argparse
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--with-engine',action='store_true');args=ap.parse_args()
    names=['converged_anchor_baselines','crossfit_visual_target','shrinkage_sweep','channel_group_controls','direct_regression_baselines','per_protocol_breakdown','public_scale_stats','real_video_diagnostics','latency_profile']
    if args.with_engine:names+=['engine_rollout_metric']
    for name in names:
        run=subprocess.run([sys.executable,str(P/'tools'/f'{name}.py')],capture_output=True,text=True)
        if run.returncode:raise RuntimeError(name+'\n'+run.stderr[-3000:])
        print(name+': passed',flush=True)
    checked={}
    for name in ('anchor_predictions.npz','crossfit_predictions.npz','channel_predictions.npz','direct_predictions.npz')+ (('engine_predictions.npz',) if args.with_engine else ()):
        expected=load(D/'reference_results'/name);actual=load(O/name);assert actual.keys()==expected.keys();err=0
        for k in actual:
            np.testing.assert_allclose(actual[k],expected[k],rtol=0,atol=1e-7);err=max(err,float(np.max(abs(actual[k]-expected[k]))))
        checked[name]=err
    # Recompute the engine metric from saved rollout arrays even without PyBullet.
    _,v=data();z=load(D/'reference_results/engine_predictions.npz')
    for m in ('huber_1.345_rgb','v22_numeric','v22_vlm'):
        e=np.mean([macro(abs(z[m+'__'+r]-z['truth_trajectory']).mean(1),v) for r in REGIMES],0)
        np.testing.assert_allclose(e,z[m+'__family_engine'],atol=1e-12,rtol=0)
    result={'all_cpu_extensions_refitted':True,'reference_prediction_max_abs_errors':checked,'reference_tolerance':1e-7,'saved_engine_losses_recomputed':True,'fresh_engine_rollout':args.with_engine,'fresh_latency_measurement':False,'original_model_reselected':False}
    write('EXTENDED_REPLAY.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
