"""Post-confirmation transfer diagnostic for frozen Stage-K frame reliability.

The fresh1080 confirmation truth has already been opened by the preregistered
V11 scorer.  Therefore every output from this program is explicitly
exploratory and cannot replace that confirmation result.  No parameter is fit
here: the numeric reliability checkpoint and its TRAIN-only normalization are
reused byte-for-byte.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from canonical_physics_metrics_v7 import score_physics
from com_confirmation_pipeline_v11 import action_error_cube, natural_truth
from reliability_canonical_v7 import refine
from reliability_experiment_v2 import make_features
from reliability_setup_bounce_v7_v2 import fit_setup_bounce
from reliability_translation_init_v7 import translation_consistent_batch
from reliability_update_v2 import (
    ReliabilityWeightNet,
    robust_weights,
    weighted_gaussian_update,
)


DATA = ROOT / "data/canonical_confirmation_v11"
INPUT = ROOT / "cache/canonical_confirmation_v11/input"
CONFIRMATION = ROOT / "results/com_observation_v11/canonical_confirmation_v1"
STAGE_K = ROOT / "results/reliability_v2_development/stage_k_scaled_unrolled_numeric_seed17"
OUT = ROOT / "results/com_observation_v12/reliability_transfer_v1"
GENERAL_ALPHA = 64.0
SETUP_ALPHA = 64.0
METHODS = ("unit_rgb", "huber_rgb", "cauchy_rgb", "stagek_rgb")
REGIMES = ("generalized_translation", "setup_translation")
CORE = (
    "mean",
    "covariance",
    "nuisance",
    "weights",
    "forward_calls_per_observation",
    "numerical_log_guard_evaluations_per_observation",
)


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def load_npz(path: Path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def source_hashes():
    names = (
        "scripts/evaluate_canonical_reliability_transfer_v12.py",
        "reliability_update_v2.py",
        "reliability_experiment_v2.py",
        "reliability_canonical_v7.py",
        "reliability_translation_init_v7.py",
        "reliability_setup_bounce_v7_v2.py",
        "canonical_physics_metrics_v7.py",
        "com_confirmation_pipeline_v11.py",
    )
    return {name: sha(ROOT / name) for name in names}


def tensor_batch(values):
    keys = (
        "prior_mean",
        "prior_scale",
        "linearization_mean",
        "J",
        "Jn",
        "residual",
        "times",
        "xz",
        "sigma_xz",
        "force",
        "force_switch",
        "valid",
        "active",
        "nuisance",
    )
    return {
        key: torch.as_tensor(
            values[key], dtype=torch.bool if key in ("valid", "active") else torch.float64
        )
        for key in keys
    }


def stagek_model():
    checkpoint_path = STAGE_K / "checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["feature_dim"] != 42 or checkpoint["selected_epoch"] != 20:
        raise ValueError("Unexpected frozen Stage-K checkpoint")
    model = ReliabilityWeightNet(42, initial_weight=checkpoint["initial_weight"]).float().eval()
    model.load_state_dict(checkpoint["state"], strict=True)
    mean = checkpoint["feature_mean"].float()
    scale = checkpoint["feature_scale"].float()
    if mean.shape != (42,) or scale.shape != (42,) or not bool((scale > 0).all()):
        raise ValueError("Invalid frozen Stage-K feature normalization")
    return model, mean, scale, checkpoint_path


def initial_feature_state(batch, observed_xz):
    local, _ = translation_consistent_batch(batch, observed_xz)
    # The sealed RGB observation is the exact linearization input used to make
    # these J/Jn/residual arrays.  Reuse those arrays so the frozen Stage-K
    # feature order, including its four nuisance columns, stays unchanged.
    if not torch.equal(local["xz"], batch["xz"]):
        raise ValueError("This transfer diagnostic is restricted to sealed RGB observations")
    feature_batch = local
    if feature_batch["Jn"].shape[-1] != 4:
        raise ValueError("Unexpected sealed nuisance dimension")
    unit = weighted_gaussian_update(
        feature_batch["prior_mean"],
        feature_batch["prior_scale"],
        feature_batch["J"],
        feature_batch["residual"],
        feature_batch["Jn"],
        feature_batch["valid"].double(),
        feature_batch["valid"],
        feature_batch["active"],
        feature_batch["linearization_mean"],
    )
    features, innovation = make_features(feature_batch, unit)
    if features.shape != (len(observed_xz), 8, 42):
        raise ValueError("Frozen Stage-K feature shape changed")
    return local, features, innovation


def observation_weights(method, batch, observed_xz, model, feature_mean, feature_scale):
    local, features, innovation = initial_feature_state(batch, observed_xz)
    valid = local["valid"]
    if method == "unit_rgb":
        weights = valid.double()
    elif method == "huber_rgb":
        weights = robust_weights(innovation, valid, kind="huber").double()
    elif method == "cauchy_rgb":
        weights = robust_weights(innovation, valid, kind="cauchy").double()
    elif method == "stagek_rgb":
        with torch.no_grad():
            weights = model((features - feature_mean) / feature_scale, valid).double()
    else:
        raise ValueError(method)
    if weights.shape != valid.shape or not bool(torch.isfinite(weights).all()):
        raise ValueError("Invalid transferred frame weights")
    if bool((weights[valid] <= 0).any() or (weights[valid] > 1).any()):
        raise ValueError("Transferred frame weights outside (0,1]")
    return local, weights


def weighted_sigma(local, weights):
    sigma = local["sigma_xz"].clone()
    axis = local["active"][:, 2].long()
    row = torch.arange(len(sigma))[:, None]
    frame = torch.arange(sigma.shape[1])[None]
    sigma[row, frame, axis[:, None]] /= weights.clamp_min(torch.finfo(torch.float64).tiny).sqrt()
    if not bool(torch.isfinite(sigma).all() and (sigma > 0).all()):
        raise ValueError("Invalid reliability-adjusted observation scale")
    return sigma


def predict(method, batch, observed_xz, model, feature_mean, feature_scale):
    local, raw_weights = observation_weights(
        method, batch, observed_xz, model, feature_mean, feature_scale
    )
    local = dict(local, sigma_xz=weighted_sigma(local, raw_weights))
    local["alpha"] = torch.full((len(observed_xz),), GENERAL_ALPHA, dtype=torch.float64)
    with torch.no_grad():
        general_fit = refine(local)
    general = {key: general_fit[key].numpy() for key in CORE}
    bounce = torch.nonzero(local["active"][:, 2], as_tuple=False).flatten()
    setup = {key: value.copy() for key, value in general.items()}
    with torch.no_grad():
        bounce_fit = fit_setup_bounce(
            observed_xz[bounce, :, 1],
            local["sigma_xz"][bounce, :, 1],
            local["times"][bounce],
            local["prior_mean"][bounce],
            local["prior_scale"][bounce],
            local["valid"][bounce],
            torch.full((len(bounce),), SETUP_ALPHA, dtype=torch.float64),
        )
    for key in ("mean", "covariance", "nuisance", "weights", "forward_calls_per_observation"):
        setup[key][bounce.numpy()] = bounce_fit[key].numpy()
    setup["numerical_log_guard_evaluations_per_observation"][bounce.numpy()] = 0
    return {
        REGIMES[0]: general,
        REGIMES[1]: setup,
        "raw_frame_weights": raw_weights.numpy(),
    }


def family_macro(errors, family_ids):
    families = sorted(set(family_ids.tolist()))
    values = [float(errors[family_ids == family].mean()) for family in families]
    return float(np.mean(values)), dict(zip(families, values))


def main():
    if OUT.exists():
        raise FileExistsError("Preserve the existing transfer diagnostic")
    OUT.mkdir(parents=True)
    started = time.perf_counter()
    inputs = load_npz(INPUT / "inputs.npz")
    index = read(DATA / "INPUT_INDEX.json")
    episodes = index["episodes"]
    if [episode["id"] for episode in episodes] != inputs["ids"].tolist():
        raise ValueError("Fresh input order changed")
    model, feature_mean, feature_scale, checkpoint_path = stagek_model()
    plan = {
        "status": "declared after V11 confirmation truth was opened",
        "scope": "exploratory no-fit transfer diagnostic; not confirmation",
        "methods": list(METHODS),
        "regimes": list(REGIMES),
        "source_sha256": source_hashes(),
        "input_seal_sha256": sha(INPUT / "INPUT_SEAL.json"),
        "observation_seal_sha256": sha(DATA / "OBSERVATION_SEAL.json"),
        "stagek_checkpoint_sha256": sha(checkpoint_path),
        "stagek_checkpoint_plan_sha256": read(STAGE_K / "summary.json")["plan_sha256"],
        "v11_evaluation_sha256": sha(CONFIRMATION / "EVALUATION.json"),
        "new_training_updates": 0,
        "new_fit_or_calibration": 0,
        "truth_already_opened": True,
        "confirmation": False,
        "goal_complete": False,
    }
    write_new(OUT / "PLAN.json", plan)
    batch = tensor_batch(inputs)
    observed_xz = batch["xz"]
    predictions = {}
    diagnostics = {}
    for method in METHODS:
        before = time.perf_counter()
        result = predict(method, batch, observed_xz, model, feature_mean, feature_scale)
        predictions[method] = {regime: result[regime] for regime in REGIMES}
        weights = result["raw_frame_weights"]
        diagnostics[method] = {
            "valid_weight_mean": float(weights[inputs["valid"]].mean()),
            "valid_weight_min": float(weights[inputs["valid"]].min()),
            "valid_weight_max": float(weights[inputs["valid"]].max()),
            "elapsed_s": time.perf_counter() - before,
        }
    targets = [
        json.loads(line)
        for line in (DATA / "evaluator_only/physical_targets.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    truth = natural_truth(targets, inputs["ids"])
    scores = {regime: {} for regime in REGIMES}
    error_arrays = {}
    for regime in REGIMES:
        for method in METHODS:
            value = predictions[method][regime]
            scored = score_physics(
                value["mean"],
                value["covariance"],
                truth,
                inputs["active"],
                inputs["family_ids"],
                inputs["protocol"],
            )
            cube = action_error_cube(scored["per_episode"], episodes)
            errors = scored["per_episode"]["action_error_m"]
            macro, by_family = family_macro(errors, inputs["family_ids"])
            scores[regime][method] = {
                "family_macro_action_mae_m": macro,
                "by_protocol_action_mae_m": {
                    protocol: float(errors[inputs["protocol"] == protocol].mean())
                    for protocol in sorted(set(inputs["protocol"].tolist()))
                },
                "by_family_action_mae_m": by_family,
                "metric_payload": scored["metrics"],
                "cube_shape": list(cube.shape),
            }
            error_arrays[f"{regime}__{method}"] = errors
        baseline = scores[regime]["unit_rgb"]["family_macro_action_mae_m"]
        for method in METHODS:
            value = scores[regime][method]["family_macro_action_mae_m"]
            scores[regime][method]["relative_improvement_over_unit_rgb"] = 1 - value / baseline
    np.savez_compressed(
        OUT / "ACTION_ERRORS.npz",
        ids=inputs["ids"],
        family_ids=inputs["family_ids"],
        protocol=inputs["protocol"],
        **error_arrays,
    )
    old = read(CONFIRMATION / "EVALUATION.json")
    for regime in REGIMES:
        expected = old["regimes"][regime]["rgb_centroid"]["action"]["family_macro_mae_m"]
        actual = scores[regime]["unit_rgb"]["family_macro_action_mae_m"]
        if actual != expected:
            raise ValueError(f"Unit RGB replay differs for {regime}: {actual} != {expected}")
    result = {
        "status": "completed exploratory frozen reliability transfer",
        "plan_sha256": sha(OUT / "PLAN.json"),
        "scores": scores,
        "diagnostics": diagnostics,
        "action_errors_sha256": sha(OUT / "ACTION_ERRORS.npz"),
        "elapsed_s": time.perf_counter() - started,
        "new_training_updates": 0,
        "new_fit_or_calibration": 0,
        "v11_unit_rgb_replay_exact": True,
        "confirmation": False,
        "goal_complete": False,
    }
    write_new(OUT / "RESULT.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "elapsed_s": result["elapsed_s"],
                "improvement": {
                    regime: {
                        method: scores[regime][method]["relative_improvement_over_unit_rgb"]
                        for method in METHODS
                    }
                    for regime in REGIMES
                },
            },
            allow_nan=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
