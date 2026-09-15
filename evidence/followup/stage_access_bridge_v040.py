from pathlib import Path
import sys,json
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R));P=R/'paper/ieee_access_initial_submission/overleaf';F=P/'evidence/followup'
from scripts.predict_idpp_physics_bridge_v30 import feature_map
def load(p):
    with np.load(p) as z:return {k:z[k] for k in z.files}
v=load(R/'results/com_observation_v28/idpp_long_kinematics_v1/prior_test_secondary.npz')
m=load(R/'results/com_observation_v29/idpp_physics_bridge_development_v1/BRIDGE_CHECKPOINT.npz')
old=load(R/'results/com_observation_v30/idpp_physics_bridge_secondary_test_v1/PREDICTIONS.npz')
new=load(F/'real_unit_corrected.npz');assert np.array_equal(v['ids'],new['ids'])
features=feature_map(v);x=np.column_stack([features[n] for n in m['feature_names']])
pred=m['intercept']+(x-m['mean_x'])/m['scale_x']@m['weight'];np.testing.assert_allclose(pred,old['selected_bridge'],atol=1e-12,rtol=0)
names=v['kinematic_feature_names'].tolist();raw=v['kinematic_features'];sigma=new['sigma_pixels']
# The bridge's only kinematic input is a ratio. Pixel-to-noise conversion of BOTH
# numerator and denominator cancels exactly; camera identity is unitless.
acc=raw[:,names.index('deceleration_px_s2')];speed=raw[:,names.index('initial_speed_px_s')]
scaled=(acc/sigma)/np.maximum(abs(speed/sigma),1e-6/sigma)
np.testing.assert_allclose(scaled,features['deceleration_over_initial_speed'],atol=1e-12,rtol=0)
xn=x.copy();xn[:,0]=scaled;pn=m['intercept']+(xn-m['mean_x'])/m['scale_x']@m['weight']
np.savez_compressed(F/'long_kinematic_replay.npz',ids=v['ids'],bridge_original=pred,bridge_normalized=pn,
    scalar_original=features['deceleration_over_initial_speed'],scalar_normalized=scaled)
(F/'real_unit_contract.md').write_text('''# Real-video unit contract

| Input | Original synthetic | Original real | Corrected real |
|---|---|---|---|
| Dynamic decoder correction | (dynamic-static) / tracker sigma | axis-projected pixels | projected pixels / RGB noise estimate |
| Static decoder correction | (static-anchor) / tracker sigma | axis-projected pixels | projected pixels / same noise estimate |
| Pooled32 | decoder hidden states | same hidden-state computation | unchanged |
| Long bridge | not part of synthetic readout | deceleration / initial speed, camera dummy | numerator and denominator both rescaled; ratio invariant |
| Scalar long proxy | not part of synthetic readout | deceleration / initial speed | same invariance |

The real noise estimate is 1.4826 times the median absolute deviation of eight
tracked centers about a quadratic trend, floored at one resized-image pixel.
It is an observation-only proxy, not measured ground-truth tracking error.
The raw 33-video scores reproduce exactly before applying this correction.
Normalizing units does not match camera geometry, decoder training distribution,
or the physical meaning of the score. In particular the score is not friction.
The long bridge uses all 152 frames; its ratio already cancels pixel units and
must not be divided by noise again after forming the ratio.
''',encoding='utf-8')
print('152-frame bridge/scalar reconstruction and scale invariance verified')
