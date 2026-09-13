"""Seal all frozen V14 confirmation and matched control predictions before truth."""
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

import scripts.evaluate_canonical_reliability_transfer_v12 as robust
import scripts.train_com_physics_internal_readout_v12 as base
import scripts.train_com_physics_query_readout_v14 as v14
from com_physics_query_readout_v14 import PhysicsQueryReadoutV14
from reliability_update_v2 import robust_weights


DATA = ROOT / "data/canonical_confirmation_v16"
INPUT = ROOT / "cache/canonical_confirmation_v16/input/inputs.npz"
STREAM = ROOT / "cache/canonical_confirmation_v16/vlm_stream_v1"
LOCK = ROOT / "results/com_observation_v14/candidate_lock_v1/LOCK.json"
OUT = ROOT / "results/com_observation_v16/frozen_predictions_v1"
REGRESSION = ROOT / "results/com_observation_v16/prediction_regression_v1"
SEED_RUNS = {
    17: ROOT / "results/com_observation_v14/query_readout_development_seed17",
    43: ROOT / "results/com_observation_v14/query_readout_replication_seed43",
    101: ROOT / "results/com_observation_v14/query_readout_replication_seed101",
}
REGIMES = ("generalized_translation", "setup_translation")


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


def install_target_guard():
    def guard(event, args):
        if event != "open" or not args or isinstance(args[0], int):
            return
        path = Path(args[0])
        parts = {part.lower() for part in path.parts}
        leaf = path.name.lower()
        if "evaluator_only" in parts or "sealed_targets" in parts or "target_traces" in parts:
            raise PermissionError("Prediction stage cannot open evaluator-only truth")
        if ("target" in leaf or "label" in leaf or "quality" in leaf) and path.suffix.lower() in {
            ".json", ".jsonl", ".npz", ".npy", ".pt"
        }:
            raise PermissionError("Prediction stage cannot open supervised artifacts")

    sys.addaudithook(guard)


def fit_arrays(fits):
    return {
        "generalized_mean": fits[REGIMES[0]]["mean"],
        "generalized_covariance": fits[REGIMES[0]]["covariance"],
        "setup_mean": fits[REGIMES[1]]["mean"],
        "setup_covariance": fits[REGIMES[1]]["covariance"],
    }


def fixed_robust_fits(batch, kind):
    translated, _, innovation = robust.initial_feature_state(batch, batch["xz"])
    weights = robust_weights(innovation, translated["valid"], kind=kind).double()
    adjusted = dict(translated, sigma_xz=robust.weighted_sigma(translated, weights))
    with torch.no_grad():
        general_fit = base.refine(adjusted)
    general = {key: general_fit[key].numpy() for key in base.CORE}
    setup = {key: value.copy() for key, value in general.items()}
    bounce = torch.nonzero(adjusted["active"][:, 2], as_tuple=False).flatten()
    with torch.no_grad():
        special = base.fit_setup_bounce(
            adjusted["xz"][bounce, :, 1],
            adjusted["sigma_xz"][bounce, :, 1],
            adjusted["times"][bounce],
            adjusted["prior_mean"][bounce],
            adjusted["prior_scale"][bounce],
            adjusted["valid"][bounce],
            torch.full((len(bounce),), base.ALPHA, dtype=torch.float64),
        )
    indices = bounce.numpy()
    for key in ("mean", "covariance", "nuisance", "weights", "forward_calls_per_observation"):
        setup[key][indices] = special[key].numpy()
    setup["numerical_log_guard_evaluations_per_observation"][indices] = 0
    return {REGIMES[0]: general, REGIMES[1]: setup}, weights.numpy()


def frozen_readout_fits(seed, arm, features, batch, protocols):
    run = SEED_RUNS[seed]
    comparison = read(run / "COMPARISON.json")
    summary = comparison[arm]
    checkpoint = run / arm / "checkpoint.pt"
    if sha(checkpoint) != summary["checkpoint_sha256"]:
        raise ValueError("Readout checkpoint changed")
    normalization = np.load(run / "FEATURE_NORMALIZATION.npz", allow_pickle=False)
    mean = torch.from_numpy(normalization["mean"]).float()
    scale = torch.from_numpy(normalization["scale"]).float()
    numeric_dim = int(normalization["numeric_dim"])
    selected = base.selected_features((features - mean) / scale, numeric_dim, arm)
    if not summary["adopted"]:
        return base.solve_both(batch, batch["xz"]), np.zeros((len(selected), 8), np.float32), np.ones((len(selected), 8), np.float32)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = PhysicsQueryReadoutV14(saved["input_dim"])
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    with torch.no_grad():
        observed, adjusted, output = v14.model_observation(model, selected, batch, protocols)
        fits = base.solve_both(adjusted, observed)
    return fits, output["correction_m"].numpy(), output["frame_weight"].numpy()


def rgb_temporal_pixels(episodes, values):
    from PIL import Image
    from com_observation_decoder_v8 import COMObservationDecoderV8

    checkpoint = ROOT / "results/com_observation_v9/decoder/continuation/rgb_warp_joint/epoch10.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state["condition"] != "rgb_warp_joint" or state["epoch"] != 10:
        raise ValueError("Wrong frozen RGB temporal endpoint")
    model = COMObservationDecoderV8("rgb", "warp").cuda().eval()
    model.load_state_dict(state["model"], strict=True)
    fields = []
    with torch.inference_mode():
        for start in range(0, len(episodes), 8):
            selected = episodes[start : start + 8]
            rgb_rows = []
            for episode in selected:
                frames = []
                for name, digest in zip(episode["frame_paths"], episode["frame_sha256"]):
                    path = ROOT / name
                    if sha(path) != digest:
                        raise ValueError("RGB frame changed")
                    with Image.open(path) as image:
                        frames.append(torch.from_numpy(np.asarray(image.convert("RGB")).copy()).permute(2, 0, 1).float() / 255)
                rgb_rows.append(torch.stack(frames))
            count = len(selected)
            rgb = torch.stack(rgb_rows).cuda()
            bbox = torch.tensor(
                [episode["initial_cue"]["bbox_xyxy"] for episode in selected],
                dtype=torch.float32,
                device="cuda",
            )
            anchor = torch.from_numpy(values["pixels"][start : start + count]).double().cuda()
            times = torch.from_numpy(values["times"][start : start + count]).float().cuda()
            fields.append(model(rgb, None, bbox, anchor, times)["pixels"].cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(fields), sha(checkpoint)


def project_inverse(pixels, episodes):
    from main3d_numeric import world_xz

    return np.asarray(
        [[world_xz(point, episode["camera"]) for point in row] for episode, row in zip(episodes, pixels)],
        dtype=np.float64,
    )


def old900_regression():
    if REGRESSION.exists():
        raise FileExistsError("Preserve existing V14 prediction regression")
    REGRESSION.mkdir(parents=True)
    values = base.load(base.INPUT)
    pooled = base.load(base.POOLED)
    qwen = base.load(base.QWEN)
    static = base.load(base.STATIC)
    features, _ = base.raw_features(values, pooled, qwen, static)
    batch = base.tensor_batch(values)
    protocols = v14.protocol_tensor(values)
    validation = np.flatnonzero(values["split"] == "validation")
    local = base.subset(batch, validation)
    fits, correction, weight = frozen_readout_fits(
        17, "vlm_internal", features[validation], local, protocols[validation]
    )
    reference_path = ROOT / "results/com_observation_v14/three_seed_replication/validation_predictions_seed17_vlm_internal.npz"
    with np.load(reference_path, allow_pickle=False) as reference:
        checks = {
            "ids": np.array_equal(values["ids"][validation], reference["ids"]),
            "correction": np.array_equal(correction, reference["correction_m"]),
            "weight": np.array_equal(weight, reference["frame_weight"]),
            "generalized_mean": np.array_equal(fits[REGIMES[0]]["mean"], reference["generalized_mean"]),
            "generalized_covariance": np.array_equal(fits[REGIMES[0]]["covariance"], reference["generalized_covariance"]),
            "setup_mean": np.array_equal(fits[REGIMES[1]]["mean"], reference["setup_mean"]),
            "setup_covariance": np.array_equal(fits[REGIMES[1]]["covariance"], reference["setup_covariance"]),
        }
    receipt = {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "episodes": len(validation),
        "source_sha256": sha(Path(__file__)),
        "checkpoint_sha256": sha(SEED_RUNS[17] / "vlm_internal/checkpoint.pt"),
        "reference_sha256": sha(reference_path),
        "targets_opened": False,
    }
    write_new(REGRESSION / "REGRESSION.json", receipt)
    if receipt["status"] != "passed":
        raise RuntimeError("Frozen V14 prediction replay failed")
    print(json.dumps(receipt, indent=2))


def predict_v16():
    if OUT.exists():
        raise FileExistsError("Preserve existing V16 predictions")
    regression = read(REGRESSION / "REGRESSION.json")
    if regression["status"] != "passed" or regression["source_sha256"] != sha(Path(__file__)):
        raise ValueError("Matching old900 prediction regression is required")
    lock = read(LOCK)
    observation_seal = read(DATA / "OBSERVATION_SEAL.json")
    input_seal = read(INPUT.parent / "INPUT_SEAL.json")
    stream_seal = read(STREAM / "MANIFEST.json")
    if observation_seal["candidate_lock_sha256"] != sha(LOCK):
        raise ValueError("V16 observations are not linked to the current candidate lock")
    if input_seal["inputs_sha256"] != sha(INPUT):
        raise ValueError("V16 input bytes changed")
    for name, digest in stream_seal["outputs"].items():
        if sha(STREAM / name) != digest:
            raise ValueError("Streamed VLM input changed")
    index = read(DATA / "INPUT_INDEX.json")
    episodes = index["episodes"]
    values = load(INPUT)
    pooled = load(STREAM / "pooled_evidence.npz")
    qwen = load(STREAM / "qwen_warp_joint.npz")
    static = load(STREAM / "qwen_warp_joint_static.npz")
    for payload in (pooled, qwen, static):
        if payload["ids"].tolist() != values["ids"].tolist():
            raise ValueError("V16 evidence order changed")
    features, _ = base.raw_features(values, pooled, qwen, static)
    batch = base.tensor_batch(values)
    protocols = v14.protocol_tensor(values)
    OUT.mkdir(parents=True)
    addendum = {
        "status": "fixed before V16 predictions and truth opening",
        "reason": "The approved redesign requires a non-VLM learned visual control and a numeric learned control in addition to the primary locked baselines",
        "controls": ["frozen V9 rgb_warp_joint", "V14 numeric-channel-only seeds 17,43,101"],
        "effect_on_candidate_or_primary_gate": "none",
        "selection_or_tuning": False,
        "candidate_lock_sha256": sha(LOCK),
        "targets_opened": False,
    }
    write_new(OUT / "MECHANISM_CONTROL_ADDENDUM.json", addendum)
    methods = [
        "unit_rgb", "huber_1.345_rgb", "cauchy_2.385_rgb", "legacy_qwen_warp_joint",
        "rgb_warp_joint", "numeric_seed17", "numeric_seed43", "numeric_seed101",
        "v14_seed17", "v14_seed43", "v14_seed101",
    ]
    plan = {
        "status": "fixed before any V16 method prediction",
        "methods": methods,
        "candidate_reporting": lock["candidate"]["checkpoint_reporting"],
        "baseline_rule": lock["new_confirmation_design"]["strong_numeric_definition"],
        "same_observations": 1080,
        "same_frames_per_observation": 8,
        "same_physical_solver_budget": True,
        "candidate_lock_sha256": sha(LOCK),
        "observation_seal_sha256": sha(DATA / "OBSERVATION_SEAL.json"),
        "input_seal_sha256": sha(INPUT.parent / "INPUT_SEAL.json"),
        "stream_manifest_sha256": sha(STREAM / "MANIFEST.json"),
        "source_sha256": sha(Path(__file__)),
        "mechanism_addendum_sha256": sha(OUT / "MECHANISM_CONTROL_ADDENDUM.json"),
        "targets_opened": False,
        "training_or_fit": False,
    }
    write_new(OUT / "PLAN.json", plan)
    write_new(OUT / "RUN_STARTED.json", {"plan_sha256": sha(OUT / "PLAN.json")})
    records = {}
    started = time.perf_counter()

    def save_method(name, fits, **extra):
        path = OUT / "methods" / f"{name}.npz"
        npwrite(
            path,
            {
                "ids": values["ids"],
                "family_ids": values["family_ids"],
                "protocol": values["protocol"],
                **fit_arrays(fits),
                **extra,
            },
        )
        records[name] = {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path)}
        print(json.dumps({"sealed_method": name, "elapsed_s": time.perf_counter() - started}), flush=True)

    unit = base.solve_both(batch, batch["xz"])
    save_method("unit_rgb", unit)
    for name, kind in (("huber_1.345_rgb", "huber"), ("cauchy_2.385_rgb", "cauchy")):
        fits, weights = fixed_robust_fits(batch, kind)
        save_method(name, fits, robust_weight=weights)
    legacy = base.solve_both(batch, torch.from_numpy(qwen["xz"]).double())
    save_method("legacy_qwen_warp_joint", legacy)
    rgb_pixels, rgb_checkpoint = rgb_temporal_pixels(episodes, values)
    rgb_xz = project_inverse(rgb_pixels, episodes)
    rgb_fits = base.solve_both(batch, torch.from_numpy(rgb_xz).double())
    save_method("rgb_warp_joint", rgb_fits, pixels=rgb_pixels, xz=rgb_xz)
    for seed in SEED_RUNS:
        fits, correction, weight = frozen_readout_fits(seed, "numeric", features, batch, protocols)
        save_method(f"numeric_seed{seed}", fits, correction_m=correction, frame_weight=weight)
    for seed in SEED_RUNS:
        fits, correction, weight = frozen_readout_fits(seed, "vlm_internal", features, batch, protocols)
        save_method(f"v14_seed{seed}", fits, correction_m=correction, frame_weight=weight)
    seal = {
        "status": "sealed all V16 predictions before truth",
        "plan_sha256": sha(OUT / "PLAN.json"),
        "outputs": records,
        "method_count": len(records),
        "methods_complete": list(records) == methods,
        "rgb_control_checkpoint_sha256": rgb_checkpoint,
        "v14_checkpoint_sha256": lock["candidate"]["checkpoints"],
        "targets_opened": False,
        "labels_opened": False,
        "training_or_fit": False,
        "elapsed_s": time.perf_counter() - started,
    }
    write_new(OUT / "PREDICTION_SEAL.json", seal)
    print(json.dumps(seal, indent=2))


if __name__ == "__main__":
    os.environ.update(PYTHONDONTWRITEBYTECODE="1")
    install_target_guard()
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["old900_regression", "predict_v16"])
    arguments = parser.parse_args()
    {"old900_regression": old900_regression, "predict_v16": predict_v16}[arguments.stage]()
