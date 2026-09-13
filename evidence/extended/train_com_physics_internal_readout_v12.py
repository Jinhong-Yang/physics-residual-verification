"""Train matched numeric and frozen-VLM temporal observation readouts on old900.

Only the 48-family TRAIN partition updates parameters.  The 12-family
validation partition selects an epoch under a fixed active-subspace NLL guard.
The fresh1080 confirmation cohort is neither opened nor referenced here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from canonical_physics_metrics_v7 import score_physics
from com_physics_internal_readout_v12 import PhysicsInternalReadoutV12, apply_active_correction
from reliability_canonical_v7 import analytic_actions, refine
from reliability_experiment_v2 import gaussian_scores
from reliability_setup_bounce_v7_v2 import fit_setup_bounce
from reliability_translation_init_v7 import translation_consistent_batch


INPUT = ROOT / "cache/canonical_physics_v7/input/inputs.npz"
POOLED = ROOT / "cache/com_observation_v12/vlm_pooled_v1/pooled_evidence.npz"
POOLED_MANIFEST = POOLED.parent / "MANIFEST.json"
QWEN = ROOT / "results/com_observation_v9/evaluation_v1/generalized_translation/qwen_warp_joint_physics.npz"
STATIC = ROOT / "results/com_observation_v9/evaluation_v1/generalized_translation/qwen_warp_joint_static_physics.npz"
TRAIN_COM = ROOT / "results/com_observation_v7/labels/train_com_labels.npz"
TARGETS = ROOT / "data/main3d_v1/targets.jsonl"
OUT = ROOT / "results/com_observation_v12/internal_readout_development_seed17_v2"
ARMS = ("numeric", "vlm_internal")
SEED = 17
EPOCHS = 20
BATCH_SIZE = 48
ALPHA = 64.0
CORE = (
    "mean", "covariance", "nuisance", "weights", "forward_calls_per_observation",
    "numerical_log_guard_evaluations_per_observation",
)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def subset(batch, indices):
    return {key: value[indices] for key, value in batch.items()}


def tensor_batch(values):
    keys = (
        "prior_mean", "prior_scale", "linearization_mean", "J", "Jn", "residual", "times",
        "xz", "sigma_xz", "force", "force_switch", "valid", "active", "nuisance",
    )
    result = {
        key: torch.as_tensor(values[key], dtype=torch.bool if key in ("valid", "active") else torch.float64)
        for key in keys
    }
    result["alpha"] = torch.full((len(values["ids"]),), ALPHA, dtype=torch.float64)
    return result


def active_values(value, active):
    axis = active[:, 2].long()
    row = torch.arange(len(value), device=value.device)[:, None]
    frame = torch.arange(value.shape[1], device=value.device)[None]
    return value[row, frame, axis[:, None]]


def raw_features(values, pooled, qwen, static):
    xz = torch.from_numpy(values["xz"]).double()
    active = torch.from_numpy(values["active"])
    axis_position = active_values(xz, active)
    times = torch.from_numpy(values["times"]).double()
    delta_t = torch.cat((torch.zeros_like(times[:, :1]), torch.diff(times, dim=1)), 1)
    delta_x = torch.cat((torch.zeros_like(axis_position[:, :1]), torch.diff(axis_position, dim=1)), 1)
    velocity = torch.where(delta_t > 0, delta_x / delta_t.clamp_min(1e-12), torch.zeros_like(delta_x))
    delta_v = torch.cat((torch.zeros_like(velocity[:, :1]), torch.diff(velocity, dim=1)), 1)
    sigma = active_values(torch.from_numpy(values["sigma_xz"]).double(), active)
    force = torch.from_numpy(values["force"]).double()
    switch = torch.from_numpy(values["force_switch"]).double()
    prior = torch.from_numpy(values["prior_mean"]).double()
    scale = torch.from_numpy(values["prior_scale"]).double().log()
    linear = (torch.from_numpy(values["linearization_mean"]).double() - prior) / torch.from_numpy(values["prior_scale"]).double()
    frame_count = axis_position.shape[1]
    global_numeric = torch.cat((force, switch[:, None], active.double(), prior, scale, linear), 1)
    numeric = torch.cat((
        axis_position[..., None],
        (axis_position - axis_position[:, :1])[..., None],
        velocity[..., None],
        delta_v[..., None],
        sigma.log()[..., None],
        times[..., None],
        delta_t[..., None],
        (times - switch[:, None])[..., None],
        (times >= switch[:, None]).double()[..., None],
        global_numeric[:, None].expand(-1, frame_count, -1),
    ), 2)
    qwen_xz = torch.from_numpy(qwen["xz"]).double()
    static_xz = torch.from_numpy(static["xz"]).double()
    dynamic = active_values(qwen_xz - static_xz, active) / sigma
    static_shift = active_values(static_xz - xz, active) / sigma
    visual = torch.cat((
        torch.from_numpy(pooled["pooled"]).double(),
        dynamic[..., None],
        static_shift[..., None],
    ), 2)
    features = torch.cat((numeric, visual), 2)
    if not bool(torch.isfinite(features).all()):
        raise ValueError("Nonfinite readout features")
    return features.float(), numeric.shape[-1]


def train_com_truth(values):
    labels = load(TRAIN_COM)
    lookup = {identifier: index for index, identifier in enumerate(labels["ids"].tolist())}
    truth = torch.zeros(len(values["ids"]), 8, 2, dtype=torch.float64)
    available = torch.zeros(len(values["ids"]), dtype=torch.bool)
    for index, identifier in enumerate(values["ids"].tolist()):
        if identifier in lookup:
            source = labels["world_COM"][lookup[identifier]]
            truth[index, :, 0] = torch.from_numpy(source[:, 0])
            truth[index, :, 1] = torch.from_numpy(source[:, 2])
            available[index] = True
    if int(available.sum()) != 720:
        raise ValueError("Wrong TRAIN COM label count")
    return truth, available


def physical_truth(rows, ids):
    lookup = {row["id"]: row for row in rows}
    if len(lookup) != len(rows) or any(identifier not in lookup for identifier in ids.tolist()):
        raise ValueError("Physical target identity join failed")
    result = []
    for identifier in ids.tolist():
        row = lookup[identifier]
        mass = float(row["mass_kg"])
        friction = float(row["dynamic_friction"])
        restitution = float(row["restitution"])
        if not (mass > 0 and friction > 0 and 0 < restitution < 1):
            raise ValueError("Physical target outside transformed coordinate domain")
        result.append((
            np.log(mass),
            np.log(friction),
            np.log(restitution / (1 - restitution)),
        ))
    value = np.asarray(result, dtype=np.float64)
    if value.shape != (len(ids), 3) or not np.isfinite(value).all():
        raise ValueError("Invalid transformed physical truth")
    return value


def candidate_xz(model, feature_rows, local_batch):
    correction = model(feature_rows, local_batch["valid"]).double()
    return apply_active_correction(local_batch["xz"], local_batch["active"], correction), correction


def solve_general(local_batch, observed_xz):
    translated, _ = translation_consistent_batch(local_batch, observed_xz)
    return refine(translated)


def solve_both(local_batch, observed_xz):
    translated, _ = translation_consistent_batch(local_batch, observed_xz)
    with torch.no_grad():
        general_fit = refine(translated)
    general = {key: general_fit[key].numpy() for key in CORE}
    setup = {key: value.copy() for key, value in general.items()}
    bounce = torch.nonzero(translated["active"][:, 2], as_tuple=False).flatten()
    with torch.no_grad():
        special = fit_setup_bounce(
            observed_xz[bounce, :, 1], translated["sigma_xz"][bounce, :, 1], translated["times"][bounce],
            translated["prior_mean"][bounce], translated["prior_scale"][bounce], translated["valid"][bounce],
            torch.full((len(bounce),), ALPHA, dtype=torch.float64),
        )
    ix = bounce.numpy()
    for key in ("mean", "covariance", "nuisance", "weights", "forward_calls_per_observation"):
        setup[key][ix] = special[key].numpy()
    setup["numerical_log_guard_evaluations_per_observation"][ix] = 0
    return {"generalized_translation": general, "setup_translation": setup}


def evaluate(model, features, batch, target, values, indices):
    model.eval()
    local = subset(batch, indices)
    with torch.no_grad():
        observed, correction = candidate_xz(model, features[indices], local)
        fits = solve_both(local, observed)
    result = {"correction_abs_mean_m": float(correction.abs().mean()), "regimes": {}}
    for regime, fit in fits.items():
        scored = score_physics(
            fit["mean"], fit["covariance"], target[indices].numpy(),
            values["active"][indices], values["family_ids"][indices], values["protocol"][indices],
        )["metrics"]
        result["regimes"][regime] = {
            "action_mae_m": scored["action"]["family_macro_mae_m"],
            "by_protocol_action_mae_m": {
                protocol: item["family_macro_mae_m"]
                for protocol, item in scored["action"]["by_protocol"].items()
            },
            "active_nll_per_dimension": scored["active_subspace_gaussian"]["family_macro"]["joint_nll_per_dimension"],
            "active_crps": scored["active_subspace_gaussian"]["family_macro"]["mean_marginal_crps"],
            "active_joint90_coverage": scored["active_subspace_gaussian"]["family_macro"]["joint90_coverage"],
        }
    result["selection_action_mae_m"] = float(np.mean([
        result["regimes"][regime]["action_mae_m"] for regime in result["regimes"]
    ]))
    result["selection_active_nll"] = float(np.mean([
        result["regimes"][regime]["active_nll_per_dimension"] for regime in result["regimes"]
    ]))
    return result


def selected_features(normalized, numeric_dim, arm):
    output = normalized.clone()
    if arm == "numeric":
        output[:, :, numeric_dim:] = 0
    elif arm != "vlm_internal":
        raise ValueError(arm)
    return output


def main():
    if OUT.exists():
        raise FileExistsError("Preserve the existing V12 development run")
    OUT.mkdir(parents=True)
    torch.set_num_threads(2)
    values = load(INPUT)
    pooled = load(POOLED)
    qwen = load(QWEN)
    static = load(STATIC)
    for payload, name in ((pooled, "pooled"), (qwen, "qwen"), (static, "static")):
        if payload["ids"].tolist() != values["ids"].tolist():
            raise ValueError(f"{name} identity order differs")
    if read(POOLED_MANIFEST)["output"]["sha256"] != sha(POOLED):
        raise ValueError("Pooled evidence seal changed")
    features, numeric_dim = raw_features(values, pooled, qwen, static)
    train = np.flatnonzero(values["split"] == "train")
    validation = np.flatnonzero(values["split"] == "validation")
    if len(train) != 720 or len(validation) != 180:
        raise ValueError("Wrong old900 family split")
    train_values = features[train].reshape(-1, features.shape[-1])
    feature_mean = train_values.mean(0)
    feature_scale = train_values.std(0, unbiased=False).clamp_min(1e-6)
    normalized = (features - feature_mean) / feature_scale
    batch = tensor_batch(values)
    target_rows = [json.loads(line) for line in TARGETS.read_text(encoding="utf-8").splitlines()]
    target = torch.from_numpy(physical_truth(target_rows, values["ids"])).double()
    com_truth, com_available = train_com_truth(values)
    sources = {
        "scripts/train_com_physics_internal_readout_v12.py": sha(Path(__file__)),
        "com_physics_internal_readout_v12.py": sha(ROOT / "com_physics_internal_readout_v12.py"),
        "tests/test_com_physics_internal_readout_v12.py": sha(ROOT / "tests/test_com_physics_internal_readout_v12.py"),
        "reliability_canonical_v7.py": sha(ROOT / "reliability_canonical_v7.py"),
        "reliability_translation_init_v7.py": sha(ROOT / "reliability_translation_init_v7.py"),
        "reliability_setup_bounce_v7_v2.py": sha(ROOT / "reliability_setup_bounce_v7_v2.py"),
        "canonical_physics_metrics_v7.py": sha(ROOT / "canonical_physics_metrics_v7.py"),
    }
    plan = {
        "status": "fixed before V12 training",
        "arms": list(ARMS),
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": {"name": "AdamW", "lr": 0.0003, "weight_decay": 0.0001, "gradient_clip": 1.0},
        "architecture": "matched bidirectional GRU64; zero-output RGB replay; bounded active-world correction",
        "input_dim": features.shape[-1],
        "numeric_dim": numeric_dim,
        "vlm_internal_dim": features.shape[-1] - numeric_dim,
        "correction_limit_m": 0.02,
        "training_loss": "new-action MSE/(0.01m)^2 + 0.2 centered-COM SmoothL1/(0.0025m) + 0.001 correction L2",
        "selection": "lowest mean of generalized/setup validation action MAE over epochs0..20",
        "selection_guard": "mean active-subspace NLL no more than epoch0 + 0.05",
        "train_families": 48,
        "validation_families": 12,
        "train_observations": 720,
        "validation_observations": 180,
        "shared_prior": True,
        "same_parameter_count": True,
        "numeric_arm_policy": "all normalized frozen-VLM channels set to exact zero",
        "vlm_arm_policy": "same numeric channels plus frozen pooled32, Qwen dynamic and static residual channels",
        "fresh1080_opened": False,
        "source_sha256": sources,
        "input_sha256": sha(INPUT),
        "pooled_manifest_sha256": sha(POOLED_MANIFEST),
        "qwen_prediction_sha256": sha(QWEN),
        "static_prediction_sha256": sha(STATIC),
        "train_com_sha256": sha(TRAIN_COM),
        "targets_sha256": sha(TARGETS),
        "confirmation": False,
        "goal_complete": False,
    }
    write(OUT / "PLAN.json", plan)
    np.savez_compressed(
        OUT / "FEATURE_NORMALIZATION.npz",
        mean=feature_mean.numpy(), scale=feature_scale.numpy(), numeric_dim=np.asarray(numeric_dim),
    )
    histories = {}
    summaries = {}
    for arm in ARMS:
        arm_started = time.perf_counter()
        torch.manual_seed(SEED)
        model = PhysicsInternalReadoutV12(features.shape[-1])
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        arm_features = selected_features(normalized, numeric_dim, arm)
        baseline = evaluate(model, arm_features, batch, target, values, validation)
        nll_limit = baseline["selection_active_nll"] + 0.05
        best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        best_epoch = 0
        best_score = baseline["selection_action_mae_m"]
        history = [{"epoch": 0, "updates": 0, "validation": baseline, "eligible": True, "new_best": True}]
        generator = torch.Generator().manual_seed(SEED + 1009)
        updates = 0
        smoke = []
        try:
            for epoch in range(1, EPOCHS + 1):
                model.train()
                totals = {"loss": 0.0, "action": 0.0, "centered_com": 0.0, "correction": 0.0}
                steps = 0
                for offsets in torch.randperm(len(train), generator=generator).split(BATCH_SIZE):
                    indices = train[offsets.numpy()]
                    local = subset(batch, indices)
                    if not bool(com_available[indices].all()):
                        raise ValueError("Non-TRAIN COM label entered training")
                    optimizer.zero_grad(set_to_none=True)
                    observed, correction = candidate_xz(model, arm_features[indices], local)
                    fit = solve_general(local, observed)
                    predicted_action = analytic_actions(fit["mean"], local["active"], training=True)
                    target_action = analytic_actions(target[indices], local["active"], training=True)
                    action_loss = ((predicted_action - target_action) / 0.01).square().mean()
                    predicted_active = active_values(observed, local["active"])
                    target_active = active_values(com_truth[indices], local["active"])
                    predicted_centered = predicted_active - predicted_active.mean(1, keepdim=True)
                    target_centered = target_active - target_active.mean(1, keepdim=True)
                    com_loss = F.smooth_l1_loss(
                        (predicted_centered - target_centered) / 0.0025,
                        torch.zeros_like(predicted_centered), beta=1.0,
                    )
                    correction_loss = (correction / 0.02).square().mean()
                    loss = action_loss + 0.2 * com_loss + 0.001 * correction_loss
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite V12 training loss")
                    loss.backward()
                    gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if not torch.isfinite(gradient):
                        raise ValueError("Nonfinite V12 gradient")
                    optimizer.step()
                    updates += 1
                    steps += 1
                    for key, item in (("loss", loss), ("action", action_loss), ("centered_com", com_loss), ("correction", correction_loss)):
                        totals[key] += float(item.detach())
                    if updates <= 3:
                        smoke.append({
                            "update": updates,
                            "loss": float(loss.detach()),
                            "gradient_norm_before_clip": float(gradient),
                            "finite": True,
                        })
                validation_score = evaluate(model, arm_features, batch, target, values, validation)
                eligible = validation_score["selection_active_nll"] <= nll_limit
                selected = eligible and validation_score["selection_action_mae_m"] < best_score
                if selected:
                    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
                    best_epoch = epoch
                    best_score = validation_score["selection_action_mae_m"]
                row = {
                    "epoch": epoch,
                    "updates": updates,
                    "training": {key: value / steps for key, value in totals.items()},
                    "validation": validation_score,
                    "eligible": eligible,
                    "new_best": selected,
                    "elapsed_s": time.perf_counter() - arm_started,
                }
                history.append(row)
                write(OUT / arm / "HISTORY.json", history)
                print(json.dumps({
                    "arm": arm,
                    "epoch": epoch,
                    "action_mm": validation_score["selection_action_mae_m"] * 1000,
                    "nll": validation_score["selection_active_nll"],
                    "eligible": eligible,
                    "new_best": selected,
                }), flush=True)
            model.load_state_dict(best_state)
            selected_score = evaluate(model, arm_features, batch, target, values, validation)
            all_indices = np.arange(len(values["ids"]))
            model.eval()
            with torch.no_grad():
                all_observed, all_correction = candidate_xz(model, arm_features, batch)
                all_fits = solve_both(batch, all_observed)
            checkpoint = {
                "model": model.state_dict(),
                "arm": arm,
                "seed": SEED,
                "selected_epoch": best_epoch,
                "input_dim": features.shape[-1],
                "numeric_dim": numeric_dim,
                "plan_sha256": sha(OUT / "PLAN.json"),
                "normalization_sha256": sha(OUT / "FEATURE_NORMALIZATION.npz"),
            }
            torch.save(checkpoint, OUT / arm / "checkpoint.pt")
            np.savez_compressed(
                OUT / arm / "predictions.npz",
                ids=values["ids"],
                correction_m=all_correction.numpy(),
                xz=all_observed.numpy(),
                generalized_mean=all_fits["generalized_translation"]["mean"],
                generalized_covariance=all_fits["generalized_translation"]["covariance"],
                setup_mean=all_fits["setup_translation"]["mean"],
                setup_covariance=all_fits["setup_translation"]["covariance"],
            )
            summary = {
                "status": "completed",
                "arm": arm,
                "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
                "selected_epoch": best_epoch,
                "updates": updates,
                "baseline": baseline,
                "selected_validation": selected_score,
                "relative_improvement_over_rgb": 1 - selected_score["selection_action_mae_m"] / baseline["selection_action_mae_m"],
                "checkpoint_sha256": sha(OUT / arm / "checkpoint.pt"),
                "predictions_sha256": sha(OUT / arm / "predictions.npz"),
                "smoke": smoke,
                "elapsed_s": time.perf_counter() - arm_started,
                "fresh1080_opened": False,
                "confirmation": False,
                "goal_complete": False,
            }
            write(OUT / arm / "SUMMARY.json", summary)
            histories[arm] = history
            summaries[arm] = summary
        except BaseException:
            write(OUT / arm / "FAILURE.json", {
                "status": "failed",
                "arm": arm,
                "updates": updates,
                "traceback": traceback.format_exc(),
                "partial_preserved": True,
            })
            raise
    comparison = {
        "status": "completed matched V12 development comparison",
        "plan_sha256": sha(OUT / "PLAN.json"),
        "numeric": summaries["numeric"],
        "vlm_internal": summaries["vlm_internal"],
        "vlm_added_value_relative": 1 - summaries["vlm_internal"]["selected_validation"]["selection_action_mae_m"] / summaries["numeric"]["selected_validation"]["selection_action_mae_m"],
        "same_parameter_count": summaries["numeric"]["trainable_parameters"] == summaries["vlm_internal"]["trainable_parameters"],
        "base_vlm_unchanged_by_construction": True,
        "fresh1080_opened": False,
        "confirmation": False,
        "goal_complete": False,
    }
    write(OUT / "COMPARISON.json", comparison)
    print(json.dumps({
        "status": comparison["status"],
        "numeric_improvement": summaries["numeric"]["relative_improvement_over_rgb"],
        "vlm_improvement": summaries["vlm_internal"]["relative_improvement_over_rgb"],
        "vlm_added_value": comparison["vlm_added_value_relative"],
    }), flush=True)


if __name__ == "__main__":
    main()
