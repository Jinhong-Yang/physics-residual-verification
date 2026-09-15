# Follow-up analysis specification (2026-09-15)

This is a timestamped local analysis specification, not a public preregistration.
The 72-family and public cohorts have already been inspected: all new comparisons
on them are exploratory. No reclassification as independent confirmation is allowed.

Priority: complete the instruction's minimum set WP0/1/2(R0d,R6)/6/7/9/10.
Preserve all original models, predictions, author metadata and acknowledgment.
The positive estimator improvement is the principal claim; failure to reject a
visual difference is neither equivalence nor an upper bound on representation value.

Three focal descriptive contrasts: original full versus strongest reported classical
anchor; original visual versus classical tracker on the deduplicated numerical view;
visual versus no visual on the same deduplicated numerical view. Report paired
20,000 family-bootstrap percentile intervals, paired sign-flip mean-test p values,
and Holm-adjusted p values jointly for these three tests. Additional comparisons
form an explicitly enumerated exploratory family; do not suppress failures.
Seeds: 20260915 (resampling/features), 20260916 (new calibration).

R0d uses 72 dynamic plus 15 single-copy static numerical channels; numeric ridge
penalty {0.1,1,10} chosen exclusively by original six development family folds.
Matched visual arms use the same selected numerical penalty, 48 dimensions,
visual ridge {1,10,100} selected on the same development folds, clipping 1,
shrinkage .25, original protocol mask and bounce fallback. Report original locked
192-dimensional arms separately. R6: original eight frames, initial RGB object cue,
classical Lucas-Kanade ROI tracking plus frame-difference motion summaries; no labels
or neural feature extractor. R5: 16x16 grayscale frames, training-fold PCA-48.
Record image hashes and extraction latency separately from previously recorded
GPU component sums; a common appearance prior remains in matched residual arms.

L4: nuisance Laplace marginal likelihood with original physical prior, alpha=64
observation precision and original nuisance bounds; original anchor/prior starts,
bounded local optimization, failed/constrained fits retained and counted.
Use the strongest observed anchor only as a descriptive reference, not a selected
confirmatory hypothesis. Existing L0-L3 and full predictions must remain byte-identical.

Calibration: report development beta=0/.1 Pareto, retain locked original model.
Any reused-family recalibration is descriptive and may not be called independent.
New calibration data, if generated, must have an independent family specification,
sealed prediction stage and target-access log. Never target the requested coverage
range by selecting on evaluation coverage. Family conformal controls a declared
family score, not automatically all episode/protocol coverage.

Real-unit correction: choose direction A before re-running. Estimate RGB tracker
noise in pixels with a quadratic trend residual and robust MAD, lower bound 1 pixel;
divide both dynamic/static projected pixel corrections by this scale. Report
combined/pooled/position variants, original scores, and family bootstrap intervals.
This fixes dimensionless scale only: camera orientation/decoder domain differences
remain. Also report coordinate-wise x/z correction when reconstructable; do not
claim full distributional compatibility from units alone. The existing 152-frame
bridge is a distinct kinematic estimator, not an eight-frame residual feature;
re-evaluate dimensionless proxy only without changing its development-trained input contract.

Bibliography: verify every cited item against primary records; retain unverified
status explicitly. IROS acceptance is attributed to author arXiv metadata and is
not invented proceedings publication. Preserve initial-submission presentation.
