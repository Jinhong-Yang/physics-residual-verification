"""Replay recorded synchronized timing summaries, not new hardware measurement.

The exact local measurement source is evidence/extended/profile_access_latency_v030.py.
Fresh measurement needs its original research checkout, 100 RGB episodes, the
frozen Qwen weights and checkpoints. Hardware timings are not portable constants.
"""
from extended_common import *
def main():
    z=json.loads((D/'LATENCY.json').read_text());rows=z['episodes'];assert len(rows)==100
    for key,expected in z['summary_ms'].items():
        values=[r[key] for r in rows if key!='setup_bounce_75_calls_ms' or r['protocol']=='bounce']
        for name,q in [('median',.5),('q25',.25),('q75',.75)]:np.testing.assert_allclose(np.quantile(values,q),expected[name],rtol=0,atol=1e-10)
    write('LATENCY_REPLAY.json',{'recorded_100_episode_quantiles_match':True,'fresh_measurement':False,'scope':z['totals']});print('100 recorded timing rows and all quantiles match')
if __name__=='__main__':main()
