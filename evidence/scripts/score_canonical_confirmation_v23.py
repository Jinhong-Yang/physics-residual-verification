"""One-shot truth opening and evaluation of prediction-sealed V23."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.special import logit


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from canonical_physics_metrics_v7 import PROTOCOLS, SUBSPACES
from scripts.score_canonical_confirmation_v16 import score_one, public_summary


PLAN = ROOT / "results/com_observation_v23/confirmation_v1/METRIC_PLAN.json"
OUT = ROOT / "results/com_observation_v23/confirmation_v1"
PREDICTION_SEAL = ROOT / "results/com_observation_v23/frozen_predictions_v1/PREDICTION_SEAL.json"
TARGET_SEAL = ROOT / "data/canonical_confirmation_v23/evaluator_only/GENERATION_LABEL_SEAL.json"
TARGETS = ROOT / "data/canonical_confirmation_v23/evaluator_only/physical_targets.jsonl"
INPUT_INDEX = ROOT / "data/canonical_confirmation_v23/INPUT_INDEX.json"
REGIMES = ("generalized_translation", "setup_translation")
NUMERIC = ("unit_rgb", "huber_1.345_rgb", "cauchy_2.385_rgb")
CANDIDATE = "v22_vlm"
ABLATION = "v22_numeric"
LEGACY = "legacy_qwen_warp_joint"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def action_family_array(scored, families):
    return np.mean(np.stack([
        np.asarray([scored[regime]["by_family_action_mae_m"][name] for name in families])
        for regime in REGIMES
    ]), axis=0)


def main():
    if (OUT / "RESULT.json").exists():
        raise FileExistsError("Preserve the first V23 confirmation result")
    plan = read(PLAN)
    if plan["status"] != "locked after prediction sealing and before truth opening":
        raise ValueError("V23 metric plan is not locked")
    if sha(PREDICTION_SEAL) != plan["prediction_seal_sha256"] or sha(TARGET_SEAL) != plan["target_seal_sha256"]:
        raise ValueError("Prediction or target seal changed")
    for name, digest in plan["source_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("Scoring source changed: " + name)
    prediction_seal = read(PREDICTION_SEAL)
    target_seal = read(TARGET_SEAL)
    if prediction_seal["targets_opened"] or not prediction_seal["methods_complete"]:
        raise ValueError("Predictions were not cleanly sealed")

    # This is the evaluator's first and only target-byte read.
    target_bytes = TARGETS.read_bytes()
    if sha_bytes(target_bytes) != plan["target_sha256"] or sha_bytes(target_bytes) != target_seal["files"]["physical_targets.jsonl"]:
        raise ValueError("Generation target seal mismatch")
    target_rows = [json.loads(line) for line in target_bytes.decode("utf-8").splitlines()]
    rows = read(INPUT_INDEX)["episodes"]
    ids = np.asarray([row["id"] for row in rows])
    family = np.asarray([row["family"] for row in rows])
    protocol = np.asarray([row["protocol"] for row in rows])
    target_by_id = {row["id"]: row for row in target_rows}
    if len(target_by_id) != len(ids) or set(target_by_id) != set(ids.tolist()):
        raise ValueError("Target identity set mismatch")
    natural = np.asarray([
        [target_by_id[name]["mass_kg"], target_by_id[name]["dynamic_friction"], target_by_id[name]["restitution"]]
        for name in ids
    ], dtype=np.float64)
    if np.any(natural[:, :2] <= 0) or np.any((natural[:, 2] <= 0) | (natural[:, 2] >= 1)):
        raise ValueError("Truth is outside the declared transform domain")
    truth = np.column_stack([np.log(natural[:, 0]), np.log(natural[:, 1]), logit(natural[:, 2])])
    active = np.zeros((len(ids), 3), dtype=bool)
    for index, name in enumerate(protocol):
        active[index, list(SUBSPACES[name])] = True

    scored, summaries, method_hashes = {}, {}, {}
    for method, receipt in prediction_seal["outputs"].items():
        path = ROOT / receipt["path"]
        if sha(path) != receipt["sha256"] or receipt["sha256"] != plan["method_sha256"][method]:
            raise ValueError("Prediction changed: " + method)
        prediction = load(path)
        if not np.array_equal(prediction["ids"], ids):
            raise ValueError("ID order mismatch: " + method)
        if not np.array_equal(prediction["family_ids"], family) or not np.array_equal(prediction["protocol"], protocol):
            raise ValueError("Grouping mismatch: " + method)
        scored[method] = score_one(prediction, truth, active)
        summaries[method] = public_summary(scored[method])
        method_hashes[method] = receipt["sha256"]

    families = sorted(set(family.tolist()))
    errors = {name: action_family_array(value, families) for name, value in scored.items()}
    point_strongest = min(NUMERIC, key=lambda name: summaries[name]["mean_two_regimes"]["action_mae_m"])
    rng = np.random.default_rng(plan["bootstrap"]["seed"])
    draws = rng.integers(0, len(families), size=(plan["bootstrap"]["draws"], len(families)))
    candidate_draw = errors[CANDIDATE][draws].mean(1)
    ablation_draw = errors[ABLATION][draws].mean(1)
    legacy_draw = errors[LEGACY][draws].mean(1)
    strongest_draw = np.stack([errors[name][draws].mean(1) for name in NUMERIC], axis=1).min(1)
    point_candidate = float(errors[CANDIDATE].mean())
    point_ablation = float(errors[ABLATION].mean())
    point_numeric = float(errors[point_strongest].mean())
    point_legacy = float(errors[LEGACY].mean())

    primary = {
        "candidate_action_mae_m": point_candidate,
        "strongest_numeric_method_point": point_strongest,
        "strongest_numeric_action_mae_m": point_numeric,
        "relative_improvement": float(1 - point_candidate / point_numeric),
        "candidate_minus_drawwise_strongest_numeric_one_sided95_upper_m": float(np.quantile(candidate_draw - strongest_draw, 0.95)),
    }
    primary["improvement_at_least_10_percent"] = primary["relative_improvement"] >= 0.10
    primary["paired_upper_below_zero"] = primary["candidate_minus_drawwise_strongest_numeric_one_sided95_upper_m"] < 0
    primary["passed"] = primary["improvement_at_least_10_percent"] and primary["paired_upper_below_zero"]

    incremental = {
        "candidate_action_mae_m": point_candidate,
        "matched_numeric_action_mae_m": point_ablation,
        "relative_improvement": float(1 - point_candidate / point_ablation),
        "candidate_point_below_matched_numeric": point_candidate < point_ablation,
        "candidate_minus_matched_numeric_one_sided95_upper_m": float(np.quantile(candidate_draw - ablation_draw, 0.95)),
        "candidate_minus_matched_numeric_two_sided95_interval_m": np.quantile(candidate_draw - ablation_draw, [0.025, 0.975]).tolist(),
    }
    incremental["paired_upper_below_zero"] = incremental["candidate_minus_matched_numeric_one_sided95_upper_m"] < 0
    incremental["passed"] = incremental["candidate_point_below_matched_numeric"] and incremental["paired_upper_below_zero"]

    legacy = {
        "candidate_action_mae_m": point_candidate,
        "legacy_action_mae_m": point_legacy,
        "candidate_below_legacy_point": point_candidate < point_legacy,
        "candidate_over_legacy_ratio_one_sided95_upper": float(np.quantile(candidate_draw / legacy_draw, 0.95)),
    }
    legacy["ratio_upper_below_1_05"] = legacy["candidate_over_legacy_ratio_one_sided95_upper"] < 1.05
    legacy["passed"] = legacy["candidate_below_legacy_point"] and legacy["ratio_upper_below_1_05"]

    cells = []
    for regime in REGIMES:
        for name in PROTOCOLS:
            numeric_value = summaries[point_strongest]["by_regime_protocol_action_mae_m"][regime][name]
            legacy_value = summaries[LEGACY]["by_regime_protocol_action_mae_m"][regime][name]
            reference_method = point_strongest if numeric_value <= legacy_value else LEGACY
            reference = min(numeric_value, legacy_value)
            value = summaries[CANDIDATE]["by_regime_protocol_action_mae_m"][regime][name]
            relative = float(value / reference - 1) if reference > 0 else (0.0 if value == 0 else math.inf)
            absolute = float(value - reference)
            cells.append({
                "regime": regime, "protocol": name,
                "candidate_action_mae_m": value, "reference_method": reference_method,
                "reference_action_mae_m": reference, "relative_degradation": relative,
                "absolute_degradation_m": absolute,
                "relative_within_10_percent": relative <= 0.10,
                "absolute_within_0_10_mm": absolute <= 0.00010,
                "passed": relative <= 0.10 and absolute <= 0.00010,
            })
    protocol_harm = {"cells": cells, "passed": all(row["passed"] for row in cells)}

    aggregate = {name: value["mean_two_regimes"] for name, value in summaries.items()}
    references = (point_strongest, LEGACY)
    nll_reference = min(references, key=lambda name: aggregate[name]["active_nll_per_dimension"])
    crps_reference = min(references, key=lambda name: aggregate[name]["active_crps"])
    candidate_metric = aggregate[CANDIDATE]
    proper = {
        "candidate": candidate_metric,
        "nll_reference_method": nll_reference,
        "nll_reference": aggregate[nll_reference]["active_nll_per_dimension"],
        "nll_delta": candidate_metric["active_nll_per_dimension"] - aggregate[nll_reference]["active_nll_per_dimension"],
        "nll_worsening_at_most_0_10_per_dimension": candidate_metric["active_nll_per_dimension"] - aggregate[nll_reference]["active_nll_per_dimension"] <= 0.10,
        "crps_reference_method": crps_reference,
        "crps_reference": aggregate[crps_reference]["active_crps"],
        "crps_ratio": candidate_metric["active_crps"] / aggregate[crps_reference]["active_crps"],
        "crps_worsening_at_most_10_percent": candidate_metric["active_crps"] <= 1.10 * aggregate[crps_reference]["active_crps"],
        "joint90_in_0_85_to_0_98": 0.85 <= candidate_metric["active_joint90_coverage"] <= 0.98,
        "width_reference_method": crps_reference,
        "joint90_axis_width_ratio": candidate_metric["active_mean_joint90_axis_width"] / aggregate[crps_reference]["active_mean_joint90_axis_width"],
        "joint90_axis_width_at_most_10_percent_wider": candidate_metric["active_mean_joint90_axis_width"] <= 1.10 * aggregate[crps_reference]["active_mean_joint90_axis_width"],
    }
    proper["passed"] = all((
        proper["nll_worsening_at_most_0_10_per_dimension"], proper["crps_worsening_at_most_10_percent"],
        proper["joint90_in_0_85_to_0_98"], proper["joint90_axis_width_at_most_10_percent_wider"],
    ))
    gates = {
        "primary": primary, "vlm_incremental": incremental, "legacy": legacy,
        "protocol_harm": protocol_harm, "proper_scores_and_uncertainty": proper,
    }
    confirmation = all(section["passed"] for section in gates.values())
    np.savez_compressed(
        OUT / "FAMILY_ACTION_ERRORS.npz", family_names=np.asarray(families), bootstrap_draw=draws,
        **{f"{name}__family_action_error": value for name, value in errors.items()},
    )
    result = {
        "status": "completed first truth-open V23 clean confirmation evaluation",
        "metric_plan_sha256": sha(PLAN),
        "truth_opened_after_all_predictions_sealed": True,
        "target_file_reads_by_evaluator": 1,
        "training_tuning_or_exclusion_after_truth": False,
        "observations": len(ids), "independent_geometry_families": len(families),
        "scores": summaries, "gates": gates,
        "mechanism_controls": {"v22_numeric": summaries[ABLATION], "rgb_learned": summaries["rgb_warp_joint"]},
        "all_preregistered_gates_passed": confirmation,
        "confirmation": confirmation, "independent_engine_confirmation": False,
        "goal_complete": False, "method_sha256": method_hashes,
        "target_sha256": sha_bytes(target_bytes),
        "family_action_errors_sha256": sha(OUT / "FAMILY_ACTION_ERRORS.npz"),
    }
    write(OUT / "RESULT.json", result)
    write(OUT / "TRUTH_OPEN_RECEIPT.json", {
        "status": "truth opened once by the preregistered evaluator after all predictions and metrics were sealed",
        "metric_plan_sha256": sha(PLAN), "prediction_seal_sha256": sha(PREDICTION_SEAL),
        "target_sha256": sha_bytes(target_bytes), "result_sha256": sha(OUT / "RESULT.json"),
        "post_truth_fit_tune_select_or_exclude": False,
    })
    print(json.dumps({"confirmation": confirmation, "gates": gates}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
