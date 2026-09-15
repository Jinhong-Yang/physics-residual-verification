"""Utilities for the sealed, explicitly post hoc follow-up study."""
from extended_common import *
O=P/'replayed/followup';O.mkdir(exist_ok=True)
F=P/'evidence/followup'
def save(name,value): (O/name).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
def csvsave(name,rows):
    with (O/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def verify_original():
    seal=json.loads((F/'BASELINE_SEAL.json').read_text())
    for name,h in seal['original_evidence_sha256'].items():
        assert hashlib.sha256((P/name.replace('\\','/')).read_bytes()).hexdigest()==h,name
    from run_support_study import fit_predict
    d,v=data()
    for mode,name in [('numeric','v22_numeric'),('full','v22_vlm')]:
        pred=fit_predict(d,v,mode)
        for r in REGIMES:
            for k in ('mean','covariance'):np.testing.assert_allclose(pred[r][k],v[f'{name}__{r}__{k}'],atol=1e-12,rtol=0)
    return d,v
def paired(a,b):
    delta=(a-b)*1000;n=len(delta);rng=np.random.default_rng(20260915)
    means=delta[rng.integers(n,size=(20000,n))].mean(1)
    perm=(delta*rng.choice([-1.,1.],size=(20000,n))).mean(1)
    p=(1+np.sum(abs(perm)>=abs(delta.mean())))/20001
    return {'difference_mm':float(delta.mean()),'low95_mm':float(np.quantile(means,.025)),
        'high95_mm':float(np.quantile(means,.975)),'p_signflip':float(p),'n_families':n}
def holm(rows):
    order=np.argsort([r['p_signflip'] for r in rows]);last=0
    for j,idx in enumerate(order):
        last=max(last,min(1,(len(rows)-j)*rows[idx]['p_signflip']));rows[idx]['p_holm']=last
    return rows
