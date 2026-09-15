# IEEE Access initial-submission package v0.4.0

Protocol-Constrained Residual Estimation for Rigid-Body Inference: Strong Baselines and Visual Controls

Hyojin Park, Nam-Hyun Yoo, and Jinhong Yang. This is an initial-submission research manuscript; journal submission or acceptance is not claimed.

## Main evidence

- The original full estimator reaches 1.453122 mm versus 1.823148 mm for the strongest tested profiled-dispersion MAP: 20.30% lower held-out-action trajectory error. The paired family difference is -0.370027 mm, with a descriptive 95% interval [-0.561991, -0.180243] mm. Two of 1,440 MAP fits did not converge and are retained.
- Removing duplicated setup channels yields numeric-only 87-dimensional error 1.442132 mm, Qwen-assisted 1.426315 mm, and matched 48-dimensional Lucas-Kanade summaries 1.416953 mm. The Qwen increment is 1.10%, with an interval including zero. This is not an equivalence test or proof of a VLM benefit.
- A VLM-free population-prior MAP control gives 1.866472 mm. Local Gaussian Laplace nuisance marginalization gives 1.918031 mm; this shares the joint-MAP mean and is not numerical integration of a non-Gaussian posterior.
- Noise-normalized real-video re-extraction preserves the negative combined rank correlation, -0.70. Pixel-noise estimates mostly meet the one-pixel floor; the correction is not real-property validation.
- Forty new geometry families / 600 RGB episodes provide separate calibration inputs. Predictions were sealed before targets were opened. Development CRPS selects beta=0. The fixed representative-family calibration reaches only 81.94% coverage on the existing evaluation set, failing the 88--93% target. It was not tuned to that set. A family-maximum region answers a different simultaneous-coverage question.
- The prior 4.38% material-pair-disjoint public-task result remains restricted to one object asset and scene. The failed 3.90% development target is retained.

New analyses on previously examined evaluation families are exploratory. The timestamped analysis plan is local, not a public preregistration. Fifty-nine action contrasts receive a pooled Holm audit; other estimands remain explicitly descriptive. The release does not establish language-pretraining benefit, equivalence with tracking, robustness to physical-model misspecification, or cross-object real-world transfer.

## Files and compilation

Upload IEEE_Access_Initial_Submission_Overleaf.zip to Overleaf and select main.tex with pdfLaTeX. Compile supplementary.tex and cover_letter.tex separately. A local equivalent is pdfLaTeX, BibTeX for main, then two pdfLaTeX passes. The full submission bundle includes all three PDFs and three source/RGB archives. Overleaf's web service itself was not exercised.

The manuscript preserves supplied author information, biographies, photographs, ORCIDs, funding and AI acknowledgment. Table 10 is placed on main-text page 10; Data and Code Availability follows Discussion and Conclusion.

## Verification

Use Python 3.11 with NumPy, SciPy, PyTorch and the versions in evidence/environment. Run:

    python tools/run_support_study.py
    python tools/run_extended_study.py
    python tools/run_followup_study.py

The follow-up runner refits controls from cached observations/features, compares saved predictions at absolute tolerance 1e-12, recomputes real/calibration/statistical results, and checks original evidence hashes. It does not rerun foundation-model training or GPU extraction. See tools/README_followup.md.

IEEE_Access_RGB_Reproduction.zip preserves the earlier upstream RGB reproduction material. IEEE_Access_Followup_RGB.zip supplies 29,280 original/fresh-calibration RGB frames, indexes, geometry records and 47 extraction/generation sources, with frame hashes. Its README explains original paths and the required research environment; it is not a clean-install-tested one-command GPU pipeline. Real-video inputs remain with their source dataset; foundation weights are obtained separately. SHA256SUMS.txt covers release archives.

Reference audit records and source snapshots are in evidence/followup/reference_audit.csv and evidence/followup/references. Metadata verification is distinguished from full-text methodological verification and from unverified proceedings-page fields.

Release: https://github.com/Jinhong-Yang/physics-residual-verification/releases/tag/v0.4.0

## Remaining scope

Matched-supervision encoder 2-by-2 experiments, an automatic Fisher-information mask, external-method ports, independent public calibration, and three-axis asset/physics/sensor stress tests are not implemented. No omitted experiment is represented as completed. These limits remain relevant to editorial assessment.
