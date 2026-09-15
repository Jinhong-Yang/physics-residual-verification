"""Recompute the v0.4 follow-up and verify archived predictions and statistics."""
from followup_common import *
import subprocess
def main():
    refs=F/'reference_results'
    for script in ['followup_laplace','followup_vlmfree','followup_representations','followup_real','followup_calibration','followup_statistics']:
        p=subprocess.run([sys.executable,str(P/'tools'/f'{script}.py')],capture_output=True,text=True)
        if p.returncode:raise RuntimeError(script+'\n'+p.stderr[-2000:])
        print(script+' passed',flush=True)
    checked={}
    for name in ['laplace_predictions.npz','vlmfree_predictions.npz','representation_predictions.npz']:
        a=load(refs/name);b=load(O/name);assert a.keys()==b.keys();err=0.
        for k in a:
            np.testing.assert_allclose(a[k],b[k],atol=1e-12,rtol=0);err=max(err,float(np.max(abs(a[k]-b[k]))))
        checked[name]=err
    for name in ['REAL_FIXED.json','CALIBRATION.json','STATISTICS.json']:
        assert json.loads((refs/name).read_text())==json.loads((O/name).read_text())
    verify_original()
    save('FOLLOWUP_REPLAY.json',{'status':'passed','maximum_prediction_errors':checked,'absolute_tolerance':1e-12,
       'real_calibration_statistics_match':True,'original_evidence_hashes_unchanged':True,
       'fresh_RGB_extraction_performed_by_this_runner':False,'upstream_training_reproduced':False})
if __name__=='__main__':main()
