# RGB inference reproduction

This companion to the IEEE Access initial-submission manuscript reproduces the first eight indexed confirmation cases (64 original RGB frames; all three protocols), selected without outcomes. It recomputes the first-image appearance prior, tracking and physical initialization, frozen Qwen visual features and learned observation outputs, robust solves, and the locked residual posterior. Saved intermediate arrays and predictions are comparison references; physical targets are not used for prediction.

The small learned checkpoints and required research modules are included. Foundation-model weights are downloaded separately at the recorded commit and checked against the supplied SHA-256 manifest. This is an inference example, not a reproduction of upstream training or all 3,060 episodes. The numerical residual control shares the learned appearance prior; it is not a VLM-free pipeline.

## Environment and execution

The recorded execution uses Python 3.11, Windows, an NVIDIA RTX 5080 (16 GB), CUDA PyTorch, and the versions in `requirements-recorded.txt`. Use a fresh environment if desired; only execution in the recorded existing environment has been tested. The runner requires CUDA and does not implement a CPU fallback. Install the matching CUDA PyTorch/torchvision distribution before the remaining packages. Package versions and floating-point behavior on another platform may differ.

```text
python scripts/download_base_model.py --model-dir /absolute/path/qwen3-vl-2b
python scripts/run_access_rgb_bundle.py --model-dir /absolute/path/qwen3-vl-2b
```

The downloader needs Internet access and approximately 4.3 GB for the base weights. The inference runner works offline. An existing model directory can be supplied instead; every recorded model file is hash-checked. Outputs are `reproduced/PLAN.json`, `reproduced/RESULT.json`, and `reproduced/PREDICTIONS.npz`. The result must report `status: passed`.

The runner verifies package hashes and rejects Python file reads outside the extracted bundle, the explicit base-model directory, and installed Python runtime directories (with system metadata exceptions). This tests independence from the original research source/data checkout, not execution in a newly installed environment. The original first-image prior head was evaluated in batches of 256; the eight-case replay preserves this CPU linear-layer shape using zero padding, with no cross-row operation. Tolerances are fixed in the plan before computation, not fitted to errors.

## Provenance and limits

`MANIFEST.json` identifies included research source, synthetic RGB inputs, checkpoints, and comparison arrays. Source modules are preserved from the research implementation; the standalone runner is the supported entry point. Other scripts may retain historical workspace paths and are supplied for implementation inspection, not as standalone commands. The official base-model repository is `Qwen/Qwen3-VL-2B-Instruct`, commit `89644892e4d85e24eaac8bacfd4f463576704203`; its files and hashes are listed in `sources/model_download_manifest.json`. Consult the upstream repository for the model's terms; foundation weights are not redistributed here. No independent hardware/environment validation or new scientific evaluation is implied by this numerical replay.
