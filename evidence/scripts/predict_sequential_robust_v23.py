"""Regression-check V22, then seal every V23 prediction before truth opens."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import scripts.predict_com_physics_v16 as frozen
import scripts.train_com_physics_internal_readout_v12 as base
from canonical_physics_metrics_v7 import score_physics
from sequential_robust_readout_v22 import REGIMES, load_checkpoint, predict


DATA = ROOT / "data/canonical_confirmation_v23"
INPUT = ROOT / "cache/canonical_confirmation_v23/input/inputs.npz"
STREAM = ROOT / "cache/canonical_confirmation_v23/vlm_stream_v1"
LOCK = ROOT / "results/com_observation_v22/candidate_lock_v1/LOCK.json"
CHECKPOINT = ROOT / "results/com_observation_v22/final_development_fit_v1/CHECKPOINT.npz"
FIT_RECEIPT = ROOT / "results/com_observation_v22/final_development_fit_v1/FIT_RECEIPT.json"
DEVELOPMENT = ROOT / "cache/com_observation_v17/development_v1/development.npz"
REGRESSION = ROOT / "results/com_observation_v23/prediction_regression_v1"
OUT = ROOT / "results/com_observation_v23/frozen_predictions_v1"
ANCHOR = "huber_1.345_rgb"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def npwrite(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **value)


def fit_arrays(fits):
    return {
        "generalized_mean": fits[REGIMES[0]]["mean"],
        "generalized_covariance": fits[REGIMES[0]]["covariance"],
        "setup_mean": fits[REGIMES[1]]["mean"],
        "setup_covariance": fits[REGIMES[1]]["covariance"],
    }


def development_regression():
    if REGRESSION.exists():
        raise FileExistsError("Preserve the first V22 prediction regression")
    values = load(DEVELOPMENT)
    checkpoint = load_checkpoint(CHECKPOINT)
    anchor = {regime: {
        "mean": values[f"{ANCHOR}__{regime}__mean"],
        "covariance": values[f"{ANCHOR}__{regime}__covariance"],
    } for regime in REGIMES}
    observed = {}
    invariants = {}
    for name, include_vlm in (("numeric", False), ("vlm", True)):
        result = predict(
            checkpoint, values["features"], values["active"],
            values["protocol"], anchor, include_vlm=include_vlm,
        )
        rows = []
        for regime in REGIMES:
            metric = score_physics(
                result[regime]["mean"], result[regime]["covariance"], values["truth"],
                values["active"], values["family_ids"], values["protocol"],
            )["metrics"]
            active = metric["active_subspace_gaussian"]["family_macro"]
            rows.append({
                "action": metric["action"]["family_macro_mae_m"],
                "nll": active["joint_nll_per_dimension"],
                "crps": active["mean_marginal_crps"],
                "coverage": active["joint90_coverage"],
            })
            inactive = ~values["active"]
            invariants[f"{name}_{regime}_inactive_exact"] = bool(np.array_equal(
                result[regime]["mean"][inactive], anchor[regime]["mean"][inactive]
            ))
            invariants[f"{name}_{regime}_covariance_psd"] = bool(all(
                np.linalg.eigvalsh(row).min() > 0 for row in result[regime]["covariance"]
            ))
        observed[name] = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
    bounce = values["protocol"].astype(str) == "bounce"
    guarded = predict(checkpoint, values["features"], values["active"], values["protocol"], anchor)
    invariants["generalized_bounce_exact_anchor"] = bool(np.array_equal(
        guarded["generalized_translation"]["mean"][bounce],
        anchor["generalized_translation"]["mean"][bounce],
    ))
    expected = read(FIT_RECEIPT)["scores_on_fit_data_descriptive_only"]
    checks = {
        name: {key: bool(np.isclose(observed[name][key], expected[name][key], rtol=0, atol=1e-14))
               for key in observed[name]}
        for name in observed
    }
    status = "passed" if all(all(row.values()) for row in checks.values()) and all(invariants.values()) else "failed"
    REGRESSION.mkdir(parents=True)
    receipt = {
        "status": status, "observed": observed, "expected": expected,
        "score_checks": checks, "invariants": invariants,
        "development_episodes": len(values["ids"]),
        "checkpoint_sha256": sha(CHECKPOINT),
        "model_source_sha256": sha(ROOT / "sequential_robust_readout_v22.py"),
        "predictor_source_sha256": sha(Path(__file__)),
        "confirmation_targets_opened": False,
    }
    write_new(REGRESSION / "REGRESSION.json", receipt)
    print(json.dumps(receipt, indent=2))
    if status != "passed":
        raise RuntimeError("V22 development replay differs from the locked fit")


def predict_v23():
    if OUT.exists():
        raise FileExistsError("Preserve existing V23 predictions")
    regression = read(REGRESSION / "REGRESSION.json")
    if regression["status"] != "passed" or regression["predictor_source_sha256"] != sha(Path(__file__)):
        raise ValueError("A matching passed V22 prediction regression is required")
    lock = read(LOCK)
    if sha(CHECKPOINT) != lock["candidate"]["checkpoint_sha256"]:
        raise ValueError("V22 checkpoint changed")
    if sha(ROOT / "sequential_robust_readout_v22.py") != lock["source_sha256"]["sequential_robust_readout_v22.py"]:
        raise ValueError("V22 implementation changed")
    observation_seal = read(DATA / "OBSERVATION_SEAL.json")
    input_seal = read(INPUT.parent / "INPUT_SEAL.json")
    stream_seal = read(STREAM / "MANIFEST.json")
    if observation_seal["candidate_lock_sha256"] != sha(LOCK):
        raise ValueError("V23 observations are not linked to the V22 lock")
    if input_seal["inputs_sha256"] != sha(INPUT):
        raise ValueError("V23 numerical inputs changed")
    for name, digest in stream_seal["outputs"].items():
        if sha(STREAM / name) != digest:
            raise ValueError("V23 streamed evidence changed: " + name)
    index = read(DATA / "INPUT_INDEX.json")
    episodes = index["episodes"]
    values = load(INPUT)
    pooled = load(STREAM / "pooled_evidence.npz")
    qwen = load(STREAM / "qwen_warp_joint.npz")
    static = load(STREAM / "qwen_warp_joint_static.npz")
    for payload in (pooled, qwen, static):
        if payload["ids"].tolist() != values["ids"].tolist():
            raise ValueError("V23 evidence order changed")
    features, numeric_dim = base.raw_features(values, pooled, qwen, static)
    checkpoint = load_checkpoint(CHECKPOINT)
    if numeric_dim != int(checkpoint["numeric_dim"]):
        raise ValueError("V22 feature schema changed")
    batch = base.tensor_batch(values)
    protocols = values["protocol"].astype(str)
    OUT.mkdir(parents=True)
    methods = [
        "unit_rgb", "huber_1.345_rgb", "cauchy_2.385_rgb",
        "legacy_qwen_warp_joint", "rgb_warp_joint", "v22_numeric", "v22_vlm",
    ]
    plan = {
        "status": "fixed before any V23 prediction",
        "methods": methods,
        "baseline_rule": lock["new_confirmation_design"]["strong_numeric_definition"],
        "same_observations": 1080, "same_frames_per_observation": 8,
        "same_physical_solver_budget": True,
        "candidate_lock_sha256": sha(LOCK),
        "checkpoint_sha256": sha(CHECKPOINT),
        "observation_seal_sha256": sha(DATA / "OBSERVATION_SEAL.json"),
        "input_seal_sha256": sha(INPUT.parent / "INPUT_SEAL.json"),
        "stream_manifest_sha256": sha(STREAM / "MANIFEST.json"),
        "regression_sha256": sha(REGRESSION / "REGRESSION.json"),
        "source_sha256": sha(Path(__file__)),
        "targets_opened": False, "training_or_fit": False,
    }
    write_new(OUT / "PLAN.json", plan)
    write_new(OUT / "RUN_STARTED.json", {"plan_sha256": sha(OUT / "PLAN.json")})
    records = {}
    started = time.perf_counter()

    def save_method(name, fits, **extra):
        path = OUT / "methods" / f"{name}.npz"
        npwrite(path, {
            "ids": values["ids"], "family_ids": values["family_ids"],
            "protocol": values["protocol"], **fit_arrays(fits), **extra,
        })
        records[name] = {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path)}
        print(json.dumps({"sealed_method": name, "elapsed_s": time.perf_counter() - started}), flush=True)

    unit = base.solve_both(batch, batch["xz"])
    save_method("unit_rgb", unit)
    robust_fits = {}
    for name, kind in (("huber_1.345_rgb", "huber"), ("cauchy_2.385_rgb", "cauchy")):
        fits, weights = frozen.fixed_robust_fits(batch, kind)
        robust_fits[name] = fits
        save_method(name, fits, robust_weight=weights)
    legacy = base.solve_both(batch, torch.from_numpy(qwen["xz"]).double())
    save_method("legacy_qwen_warp_joint", legacy)
    rgb_pixels, rgb_checkpoint = frozen.rgb_temporal_pixels(episodes, values)
    rgb_xz = frozen.project_inverse(rgb_pixels, episodes)
    save_method("rgb_warp_joint", base.solve_both(batch, torch.from_numpy(rgb_xz).double()), pixels=rgb_pixels, xz=rgb_xz)
    anchor = {regime: {
        "mean": robust_fits[ANCHOR][regime]["mean"],
        "covariance": robust_fits[ANCHOR][regime]["covariance"],
    } for regime in REGIMES}
    feature_array = features.numpy()
    for name, include_vlm in (("v22_numeric", False), ("v22_vlm", True)):
        outputs = predict(checkpoint, feature_array, values["active"], protocols, anchor, include_vlm=include_vlm)
        save_method(name, outputs,
                    generalized_delta=outputs[REGIMES[0]]["delta"],
                    setup_delta=outputs[REGIMES[1]]["delta"])
    seal = {
        "status": "sealed all V23 predictions before truth",
        "plan_sha256": sha(OUT / "PLAN.json"), "outputs": records,
        "method_count": len(records), "methods_complete": list(records) == methods,
        "rgb_control_checkpoint_sha256": rgb_checkpoint,
        "v22_checkpoint_sha256": sha(CHECKPOINT),
        "targets_opened": False, "labels_opened": False,
        "training_or_fit": False, "elapsed_s": time.perf_counter() - started,
    }
    write_new(OUT / "PREDICTION_SEAL.json", seal)
    print(json.dumps(seal, indent=2))


if __name__ == "__main__":
    os.environ.update(PYTHONDONTWRITEBYTECODE="1")
    frozen.install_target_guard()
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("development_regression", "predict_v23"))
    args = parser.parse_args()
    {"development_regression": development_regression, "predict_v23": predict_v23}[args.stage]()
