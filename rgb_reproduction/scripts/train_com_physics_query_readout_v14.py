"""Matched development of protocol-query residual/reliability readouts on old900."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import scripts.train_com_physics_internal_readout_v12 as base
from canonical_physics_metrics_v7 import score_physics
from com_physics_query_readout_v14 import PhysicsQueryReadoutV14, apply_observation_update
from reliability_canonical_v7 import analytic_actions
from reliability_setup_bounce_v7_v2 import fit_setup_bounce
from reliability_translation_init_v7 import translation_consistent_batch


OUT = ROOT / "results/com_observation_v14/query_readout_development_seed17"
ARMS = ("numeric", "vlm_internal")
SEED = 17
EPOCHS = 30
BATCH_SIZE = 48
PROTOCOL_TO_INDEX = {"multiple_forces": 0, "unforced_slide": 1, "bounce": 2}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def protocol_tensor(values):
    try:
        result = [PROTOCOL_TO_INDEX[value] for value in values["protocol"].tolist()]
    except KeyError as error:
        raise ValueError(f"Unknown protocol {error.args[0]}") from error
    return torch.tensor(result, dtype=torch.long)


def model_observation(model, feature_rows, local, protocol_rows):
    output = model(feature_rows, local["valid"], protocol_rows)
    observed, sigma = apply_observation_update(
        local["xz"], local["sigma_xz"], local["active"],
        output["correction_m"].double(), output["frame_weight"].double(),
    )
    return observed, dict(local, sigma_xz=sigma), output


def active_marginal_nll(mean, covariance, target, active):
    variance = covariance.diagonal(dim1=-2, dim2=-1)
    term = 0.5 * ((target - mean).square() / variance + variance.log() + math.log(2 * math.pi))
    count = active.sum(-1).clamp_min(1)
    return ((term * active).sum(-1) / count).mean()


def differentiable_fits(adjusted, observed):
    translated, _ = translation_consistent_batch(adjusted, observed)
    general = base.refine(translated)
    setup_mean = general["mean"].clone()
    setup_covariance = general["covariance"].clone()
    bounce = torch.nonzero(translated["active"][:, 2], as_tuple=False).flatten()
    if len(bounce):
        special = fit_setup_bounce(
            observed[bounce, :, 1], translated["sigma_xz"][bounce, :, 1],
            translated["times"][bounce], translated["prior_mean"][bounce],
            translated["prior_scale"][bounce], translated["valid"][bounce],
            torch.full((len(bounce),), base.ALPHA, dtype=torch.float64),
        )
        setup_mean[bounce, 2] = special["mean"][:, 2]
        setup_covariance[bounce, 2, 2] = special["covariance"][:, 2, 2]
    return general, {"mean": setup_mean, "covariance": setup_covariance}


def evaluate(model, features, batch, target, values, protocols, indices, unit=False):
    model.eval()
    local = base.subset(batch, indices)
    with torch.no_grad():
        if unit:
            observed = local["xz"]
            adjusted = local
            correction = torch.zeros_like(local["times"])
            weight = torch.ones_like(local["times"])
        else:
            observed, adjusted, output = model_observation(
                model, features[indices], local, protocols[indices]
            )
            correction = output["correction_m"]
            weight = output["frame_weight"]
        fits = base.solve_both(adjusted, observed)
    result = {
        "correction_abs_mean_m": float(correction.abs().mean()),
        "frame_weight_mean": float(weight.mean()),
        "frame_weight_min": float(weight.min()),
        "frame_weight_max": float(weight.max()),
        "regimes": {},
    }
    for regime, fit in fits.items():
        metrics = score_physics(
            fit["mean"], fit["covariance"], target[indices].numpy(),
            values["active"][indices], values["family_ids"][indices],
            values["protocol"][indices],
        )["metrics"]
        result["regimes"][regime] = {
            "action_mae_m": metrics["action"]["family_macro_mae_m"],
            "by_protocol_action_mae_m": {
                protocol: item["family_macro_mae_m"]
                for protocol, item in metrics["action"]["by_protocol"].items()
            },
            "active_nll_per_dimension": metrics["active_subspace_gaussian"]["family_macro"]["joint_nll_per_dimension"],
            "active_crps": metrics["active_subspace_gaussian"]["family_macro"]["mean_marginal_crps"],
            "active_joint90_coverage": metrics["active_subspace_gaussian"]["family_macro"]["joint90_coverage"],
        }
    result["selection_action_mae_m"] = float(np.mean([
        value["action_mae_m"] for value in result["regimes"].values()
    ]))
    result["selection_active_nll"] = float(np.mean([
        value["active_nll_per_dimension"] for value in result["regimes"].values()
    ]))
    return result


def main():
    if OUT.exists():
        raise FileExistsError("Preserve the existing V14 run")
    OUT.mkdir(parents=True)
    torch.set_num_threads(2)
    values = base.load(base.INPUT)
    pooled = base.load(base.POOLED)
    qwen = base.load(base.QWEN)
    static = base.load(base.STATIC)
    for payload, name in ((pooled, "pooled"), (qwen, "qwen"), (static, "static")):
        if payload["ids"].tolist() != values["ids"].tolist():
            raise ValueError(f"{name} identity order differs")
    features, numeric_dim = base.raw_features(values, pooled, qwen, static)
    train = np.flatnonzero(values["split"] == "train")
    validation = np.flatnonzero(values["split"] == "validation")
    if len(train) != 720 or len(validation) != 180:
        raise ValueError("Wrong old900 split")
    train_flat = features[train].reshape(-1, features.shape[-1])
    feature_mean = train_flat.mean(0)
    feature_scale = train_flat.std(0, unbiased=False).clamp_min(1e-6)
    normalized = (features - feature_mean) / feature_scale
    batch = base.tensor_batch(values)
    protocols = protocol_tensor(values)
    target_rows = [json.loads(line) for line in base.TARGETS.read_text(encoding="utf-8").splitlines()]
    target = torch.from_numpy(base.physical_truth(target_rows, values["ids"])).double()
    com_truth, com_available = base.train_com_truth(values)
    plan = {
        "status": "fixed before V14 training",
        "arms": list(ARMS),
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "architecture": "three protocol queries cross-attend all eight frame slots; selected query plus local evidence emits bounded observation residual and reliability",
        "input_dim": features.shape[-1],
        "numeric_dim": numeric_dim,
        "vlm_internal_dim": features.shape[-1] - numeric_dim,
        "protocol_query_count": 3,
        "correction_limit_m": 0.02,
        "frame_weight_range": [0.05, 1.0],
        "initial_frame_weight": 0.90,
        "optimizer": {
            "name": "AdamW", "main_lr": 0.0003, "reliability_lr": 0.001,
            "weight_decay": 0.0001, "gradient_clip": 1.0,
        },
        "training_loss": "mean generalized/setup new-action absolute error /0.01m + 0.02 centered-COM SmoothL1/0.0025m + 0.05 mean generalized/setup active marginal NLL + 0.0005 correction L2 + 0.0001 reliability-to-one L2",
        "selection": "lowest mean generalized/setup validation action MAE among epochs0..30",
        "selection_guard": "mean active-subspace NLL no more than exact unit-RGB baseline +0.05",
        "numeric_arm_policy": "normalized frozen-VLM channels set to exact zero",
        "vlm_arm_policy": "same numeric channels plus frozen pooled32 and Qwen residual channels",
        "same_parameter_count": True,
        "same_update_and_solver_budget": True,
        "train_families": 48,
        "validation_families": 12,
        "fresh1080_opened": False,
        "source_sha256": {
            "scripts/train_com_physics_query_readout_v14.py": sha(Path(__file__)),
            "scripts/train_com_physics_internal_readout_v12.py": sha(ROOT / "scripts/train_com_physics_internal_readout_v12.py"),
            "com_physics_query_readout_v14.py": sha(ROOT / "com_physics_query_readout_v14.py"),
            "tests/test_com_physics_query_readout_v14.py": sha(ROOT / "tests/test_com_physics_query_readout_v14.py"),
            "reliability_canonical_v7.py": sha(ROOT / "reliability_canonical_v7.py"),
        },
        "input_sha256": sha(base.INPUT),
        "pooled_manifest_sha256": sha(base.POOLED_MANIFEST),
        "train_com_sha256": sha(base.TRAIN_COM),
        "targets_sha256": sha(base.TARGETS),
        "confirmation": False,
        "goal_complete": False,
    }
    write(OUT / "PLAN.json", plan)
    np.savez_compressed(
        OUT / "FEATURE_NORMALIZATION.npz", mean=feature_mean.numpy(),
        scale=feature_scale.numpy(), numeric_dim=np.asarray(numeric_dim),
    )
    summaries = {}
    for arm in ARMS:
        started = time.perf_counter()
        torch.manual_seed(SEED)
        model = PhysicsQueryReadoutV14(features.shape[-1])
        reliability_parameters = list(model.reliability_parameters())
        reliability_ids = {id(parameter) for parameter in reliability_parameters}
        main_parameters = [parameter for parameter in model.parameters() if id(parameter) not in reliability_ids]
        optimizer = torch.optim.AdamW([
            {"params": main_parameters, "lr": 3e-4},
            {"params": reliability_parameters, "lr": 1e-3},
        ], weight_decay=1e-4)
        arm_features = base.selected_features(normalized, numeric_dim, arm)
        unit = evaluate(model, arm_features, batch, target, values, protocols, validation, unit=True)
        initial = evaluate(model, arm_features, batch, target, values, protocols, validation)
        nll_limit = unit["selection_active_nll"] + 0.05
        best_state = None
        best_epoch = None
        best_score = unit["selection_action_mae_m"]
        if initial["selection_active_nll"] <= nll_limit and initial["selection_action_mae_m"] < best_score:
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            best_epoch = 0
            best_score = initial["selection_action_mae_m"]
        history = [{
            "epoch": 0, "updates": 0, "validation": initial,
            "eligible": initial["selection_active_nll"] <= nll_limit,
            "new_best": best_epoch == 0,
        }]
        generator = torch.Generator().manual_seed(SEED + 3011)
        updates = 0
        smoke = []
        try:
            for epoch in range(1, EPOCHS + 1):
                model.train()
                totals = {key: 0.0 for key in (
                    "loss", "action", "centered_com", "nll", "correction", "reliability"
                )}
                steps = 0
                for offsets in torch.randperm(len(train), generator=generator).split(BATCH_SIZE):
                    indices = train[offsets.numpy()]
                    local = base.subset(batch, indices)
                    if not bool(com_available[indices].all()):
                        raise ValueError("Non-TRAIN COM label entered training")
                    optimizer.zero_grad(set_to_none=True)
                    observed, adjusted, output = model_observation(
                        model, arm_features[indices], local, protocols[indices]
                    )
                    general, setup = differentiable_fits(adjusted, observed)
                    target_action = analytic_actions(target[indices], local["active"], training=True)
                    general_action = analytic_actions(general["mean"], local["active"], training=True)
                    setup_action = analytic_actions(setup["mean"], local["active"], training=True)
                    action_loss = 0.5 * (
                        (general_action - target_action).abs().mean()
                        + (setup_action - target_action).abs().mean()
                    ) / 0.01
                    predicted_active = base.active_values(observed, local["active"])
                    target_active = base.active_values(com_truth[indices], local["active"])
                    com_loss = F.smooth_l1_loss(
                        ((predicted_active - predicted_active.mean(1, keepdim=True))
                         - (target_active - target_active.mean(1, keepdim=True))) / 0.0025,
                        torch.zeros_like(predicted_active), beta=1.0,
                    )
                    nll = 0.5 * (
                        active_marginal_nll(
                            general["mean"], general["covariance"], target[indices], local["active"]
                        )
                        + active_marginal_nll(
                            setup["mean"], setup["covariance"], target[indices], local["active"]
                        )
                    )
                    correction_loss = (output["correction_m"] / 0.02).square().mean()
                    reliability_loss = (1 - output["frame_weight"]).square().mean()
                    loss = (
                        action_loss + 0.02 * com_loss + 0.05 * nll
                        + 0.0005 * correction_loss + 0.0001 * reliability_loss
                    )
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite V14 loss")
                    loss.backward()
                    gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if not torch.isfinite(gradient):
                        raise ValueError("Nonfinite V14 gradient")
                    optimizer.step()
                    updates += 1
                    steps += 1
                    for key, value in (
                        ("loss", loss), ("action", action_loss), ("centered_com", com_loss),
                        ("nll", nll), ("correction", correction_loss),
                        ("reliability", reliability_loss),
                    ):
                        totals[key] += float(value.detach())
                    if updates <= 3:
                        smoke.append({
                            "update": updates, "loss": float(loss.detach()),
                            "gradient_norm_before_clip": float(gradient), "finite": True,
                        })
                validation_score = evaluate(
                    model, arm_features, batch, target, values, protocols, validation
                )
                eligible = validation_score["selection_active_nll"] <= nll_limit
                selected = eligible and validation_score["selection_action_mae_m"] < best_score
                if selected:
                    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
                    best_epoch = epoch
                    best_score = validation_score["selection_action_mae_m"]
                history.append({
                    "epoch": epoch, "updates": updates,
                    "training": {key: value / steps for key, value in totals.items()},
                    "validation": validation_score, "eligible": eligible,
                    "new_best": selected, "elapsed_s": time.perf_counter() - started,
                })
                write(OUT / arm / "HISTORY.json", history)
                print(json.dumps({
                    "arm": arm, "epoch": epoch,
                    "action_mm": validation_score["selection_action_mae_m"] * 1000,
                    "nll": validation_score["selection_active_nll"],
                    "weight": validation_score["frame_weight_mean"],
                    "eligible": eligible, "new_best": selected,
                }), flush=True)
            if best_state is None:
                selected_score = unit
                adopted = False
            else:
                model.load_state_dict(best_state)
                selected_score = evaluate(
                    model, arm_features, batch, target, values, protocols, validation
                )
                adopted = True
            checkpoint_path = OUT / arm / "checkpoint.pt"
            torch.save({
                "model": model.state_dict(), "arm": arm, "seed": SEED,
                "selected_epoch": best_epoch, "adopted": adopted,
                "input_dim": features.shape[-1], "numeric_dim": numeric_dim,
                "plan_sha256": sha(OUT / "PLAN.json"),
                "normalization_sha256": sha(OUT / "FEATURE_NORMALIZATION.npz"),
            }, checkpoint_path)
            summary = {
                "status": "completed", "arm": arm,
                "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
                "selected_epoch": best_epoch, "adopted": adopted, "updates": updates,
                "unit_rgb": unit, "initial": initial, "selected_validation": selected_score,
                "relative_improvement_over_unit_rgb": 1 - selected_score["selection_action_mae_m"] / unit["selection_action_mae_m"],
                "checkpoint_sha256": sha(checkpoint_path), "smoke": smoke,
                "elapsed_s": time.perf_counter() - started, "fresh1080_opened": False,
                "confirmation": False, "goal_complete": False,
            }
            write(OUT / arm / "SUMMARY.json", summary)
            summaries[arm] = summary
        except BaseException:
            write(OUT / arm / "FAILURE.json", {
                "status": "failed", "arm": arm, "updates": updates,
                "traceback": traceback.format_exc(), "partial_preserved": True,
            })
            raise
    comparison = {
        "status": "completed matched V14 development comparison",
        "plan_sha256": sha(OUT / "PLAN.json"),
        "numeric": summaries["numeric"], "vlm_internal": summaries["vlm_internal"],
        "vlm_added_value_relative": 1 - summaries["vlm_internal"]["selected_validation"]["selection_action_mae_m"] / summaries["numeric"]["selected_validation"]["selection_action_mae_m"],
        "same_parameter_count": summaries["numeric"]["trainable_parameters"] == summaries["vlm_internal"]["trainable_parameters"],
        "base_vlm_weights_unchanged": True, "fresh1080_opened": False,
        "confirmation": False, "goal_complete": False,
    }
    write(OUT / "COMPARISON.json", comparison)
    print(json.dumps({
        "status": comparison["status"],
        "numeric_improvement": summaries["numeric"]["relative_improvement_over_unit_rgb"],
        "vlm_improvement": summaries["vlm_internal"]["relative_improvement_over_unit_rgb"],
        "vlm_added_value": comparison["vlm_added_value_relative"],
    }), flush=True)


if __name__ == "__main__":
    main()
