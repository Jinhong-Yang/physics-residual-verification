# Verification materials for rigid-body residual adaptation

Verification release **v0.3.0**.

Protocol-Constrained Residual Adaptation for Rigid-Body Inference: A Bounded Evaluation of Frozen Visual Features

Hyojin Park, Nam-Hyun Yoo, and Jinhong Yang. Initial-submission Research Article;
publication acceptance is not claimed. This release does not report external
journal review or a response to reviewers.

## Measured evidence

The original frozen models and primary rules are preserved. On 72 new synthetic
geometry families (1,080 episodes), full error is 1.453122 mm, compared with
2.074269 mm for the one-step-weight Huber anchor: **29.95%** reduction.
New post hoc local iterated Huber fitting gives 1.914820 mm; full improvement is
**24.11%**, with a paired difference interval **[-0.66342, -0.26975] mm**.
Iterated Cauchy gives 1.913327 mm and profiled Gaussian MAP 1.823148 mm.
Both IRLS methods converge locally on every distinct fit. Two profiled-MAP
fits are flagged nonconverged and retained; global optimality is not claimed.

The original visual increment is **1.07%**, with a two-sided interval including
zero. Cross-fitted numerical targets retain **1.17%**, also with an interval
including zero. Development-selected channel controls favor position shifts
(1.451117 mm) over pooled-only states (1.467461 mm); the combined view retains
its original penalty and model. The design does not isolate language pretraining.
The direct ridge and three-seed MLP controls yield 33.402860 and 26.489233 mm.
Their hyperparameters and normalization are fitted on development families.

A PyBullet **nonrotating-sphere proxy** gives 2.236155 mm for fixed Huber and
1.605715 mm for full (**28.19%** reduction). Analytic/engine family rank
correlations exceed 0.96. This is not an original-geometry-asset rollout or
validation in real manipulation. Protocol and active-coordinate breakdowns,
shrinkage and covariance sensitivities, bounce diagnostics, and 100 measured
inference-component timing rows are supplied.

The prior independent friction-condition study remains unchanged: 285 fitting
episodes and 329 evaluation episodes, with no shared object-friction or
surface-friction conditions or complete video hash. Equal weighting over 63
evaluation material pairs yields **4.38%** improvement, difference **-2.778 mm**,
and two-way bootstrap interval **[-6.044, -0.116] mm**. Its upper endpoint is near
zero. Every episode uses the same red cube/scene assets. Historical architecture
selection used the material bank. This is a material-condition holdout during
refitting, not transfer to new object/scene assets or material recognition from
appearance. The original episode-split improvement remains **4.27%**; the
development result **3.90%** still fails its original 10% criterion.

The real-video score has inverse family correlation -0.70. The supplied source
audit identifies a mismatch between synthetic noise-normalized shifts and real
pixel shifts. This diagnostic outputs a score, uses no appearance prior, and
does not identify the causal reason for failure. Calibrated property transfer,
independent uncertainty calibration, and matched-supervision encoder controls
remain outside the evidence.

## Reproduction

Upload the Overleaf ZIP and select `main.tex`, `supplementary.tex`, or
`cover_letter.tex`, using pdfLaTeX. Execute numerical tools outside Overleaf.

```text
python tools/replay_tables.py
python tools/run_support_study.py
python tools/score_access_video_control.py
python tools/replay_group_holdout.py
python tools/run_extended_study.py
```

The extended runner refits the CPU comparisons from bundled inputs and checks
reference predictions. Install NumPy, SciPy, and PyTorch. Fresh PyBullet execution
is optional: `python tools/engine_rollout_metric.py`; the runner also recomputes
metrics from stored engine trajectories without PyBullet. See
`tools/README_extended.md` for all individual commands and numerical limits.
Recorded timing quantiles replay without GPU work. Fresh hardware measurement
requires the original 100 RGB episodes and upstream source environment; reported
totals are sums across staged, model-resident passes, not cold-start benchmarks.

The separate `IEEE_Access_RGB_Reproduction.zip` supplies 64 original frames,
small learned checkpoints, source closure, and an eight-case full-inference
runner including the common first-image prior. Foundation weights are separate
inputs pinned by revision and hashes. See the companion README before running.
Full upstream training and a fresh environment installation are not reproduced.

## Archives and provenance

Release assets include the Overleaf ZIP, the RGB reproduction ZIP, and
`SHA256SUMS.txt`. The repository root mirrors the Overleaf project; the RGB
companion is under `rgb_reproduction/`. New plans, numerical inputs, diagnostic
records and reference outputs are under `evidence/extended/`. Original version
v0.2.0 remains available unchanged in its own release.

Third-party dataset/model terms remain with their original providers; this
release does not grant rights beyond those terms. No access tokens or foundation
weights are included. The original source provenance may contain local paths;
those paths are not public download links.
