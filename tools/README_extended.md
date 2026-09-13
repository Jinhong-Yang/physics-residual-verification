# Additional post hoc analyses (v0.3.0)

These analyses use previously evaluated data descriptively. Original primary
models, confirmation rules and input records are preserved. Selection uses
development families only. They are not a new independent confirmation.

Install NumPy, SciPy and PyTorch, then run `python tools/run_extended_study.py`.
It refits and checks release reference predictions. Outputs go to
`replayed/extended/`. The reference tolerance is 1e-7 in transformed coordinates;
the original full configuration additionally matches its cache at 1e-12.

| Script | Analysis |
| --- | --- |
| converged_anchor_baselines.py | Iterated Huber/Cauchy and profiled-scale MAP |
| per_protocol_breakdown.py | Protocol/coordinate errors and exact sign test |
| crossfit_visual_target.py | Five-inner-fold visual targets and bounce diagnosis |
| shrinkage_sweep.py | Five fixed visual multipliers |
| direct_regression_baselines.py | Nested direct ridge and three-seed MLP |
| channel_group_controls.py | Development-selected channels and 87-D view |
| engine_rollout_metric.py | New PyBullet controlled-sphere rollouts |
| public_scale_stats.py | Displacement scale, normalized errors and vector R2 |
| real_video_diagnostics.py | Score distributions and documented input mismatch |
| latency_profile.py | Recalculate 100 recorded timing quantiles |

Fresh engine execution additionally requires PyBullet. Run
`python tools/engine_rollout_metric.py`, or add `--with-engine` to the runner.
Tested with the existing Linux PyBullet January 29, 2025 build (API 202010061);
the Windows Python environment was used for other calculations. The engine
object is a nonrotating sphere, not the original geometry assets.

Fresh hardware timing is not part of portable numerical replay. The exact local
measurement source is `evidence/extended/profile_access_latency_v030.py` and
expects the original research checkout, its 100 selected RGB episodes, original
upstream modules and frozen weights. The recorded timings are component sums
from staged model-resident runs, not contiguous cold-start latency. The shared
first-image VLM prior belongs to both compared paths.

The profiled-MAP optimizer flags two nonconverged cases, which are retained.
Robust fits converge locally; no global nonlinear optimality is claimed. Direct
regressors are specified small-model controls, not state-of-the-art baselines.
