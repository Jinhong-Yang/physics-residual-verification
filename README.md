# Verification materials for rigid-body residual adaptation

Verification release **v0.2.0**.

Protocol-Constrained Residual Adaptation with Frozen Vision–Language Features for Rigid-Body Inference

Hyojin Park, Nam-Hyun Yoo, and Jinhong Yang. Initial-submission research manuscript; publication acceptance is not claimed.

## Evidence and scope

The primary analytic-action experiment reports 29.95% lower error than the strongest fixed robust baseline on 72 later synthetic geometry families (1,080 episodes). Most improvement comes from numerical residual adaptation. All primary paths share a learned first-image appearance prior. The additional visual residual improves error by 1.07%; its two-sided interval includes zero. A separate public-task result is 4.27%, while its original development result of 3.90% failed the 10% threshold. These distinct results must not be combined or relabeled as a universal performance pass.

The supplementary studies refit seven components, evaluate covariance sensitivity and initial Jacobians, and compare a frozen V-JEPA 2 representation under fixed residual-stage rules. They are post hoc, use already evaluated data, and do not reselect the original model. Real-world property transfer, general semantic-physics superiority, and guaranteed uncertainty calibration remain unestablished.

A separate follow-up uses newly selected episodes and disjoint object/surface friction conditions during refitting. After excluding every formerly selected/feasibility episode, it retains 285 training and 329 evaluation cases. No friction condition on either axis or complete video hash is shared across these fitting/evaluation roles. The same architecture and penalties are used. Evaluation trajectories were downloaded only after predictions were sealed.

Across 63 held-out material pairs, equally weighted mean errors are **63.419 mm numerical**, **61.943 mm raw residual**, and **60.641 mm gated residual**. The gated reduction is **4.38%**, with paired difference **-2.778 mm** and a two-way material-bootstrap 95% interval **[-6.044, -0.116] mm**. Object/surface marginal improvements occur in **7/8** and **5/8** groups, passing the fixed study rule. The interval is close to zero at its upper endpoint and has only eight levels per axis. This is a friction-condition holdout within the same red-cube/scene setting. Earlier architecture selection exposed the material bank; new object/scene assets and historically untouched material discovery are not claimed. The original 3.90% development failure remains unchanged.

## Files

- Repository root: Overleaf paper source/PDFs, `evidence/`, cached features, checkpoints, and `tools/`. Compile `main.tex` with pdfLaTeX/BibTeX; select `supplementary.tex` for the supplement.
- `rgb_reproduction/`: 64 original RGB frames, small learned checkpoints and required modules for eight primary inference cases. A pinned Qwen base model is downloaded separately. See its README for the recorded CUDA environment and offline command.
- Release assets: `IEEE_Access_Initial_Submission_Overleaf.zip`, `IEEE_Access_RGB_Reproduction.zip`, and `SHA256SUMS.txt`. The checksums apply to these downloadable archives.

After extracting the Overleaf archive, run the following with Python, NumPy and SciPy:

```text
python tools/replay_tables.py
python tools/effect_sensitivity.py
python tools/run_support_study.py
python tools/score_access_video_control.py
python tools/replay_prior_lineage.py
python tools/replay_group_holdout.py
```

The original saved statistics, locked numerical/full refits, frozen-video comparison, and material-held-out refit were reproduced from independently extracted archives. The RGB example also passed after extraction with reads against the original research checkout blocked. It regenerates the first-image prior through the final posterior. This uses the existing recorded environment, not a newly installed environment. Full upstream training and the complete RGB cohort are not reproduced. `tools/replay_group_holdout.py` checks new refitted predictions and crossed-material intervals from bundled numeric views; the new 614-episode RGB extraction is documented by source/video hashes and observation records, not rerun by that cached-feature tool.

The archives contain provenance records; hashes identify content and are not independent proof of preregistration time. Foundation weights are not redistributed. Their official repository and pinned commit are included in the RGB README and model manifest. Any historical local paths in source/provenance are not public download links.
