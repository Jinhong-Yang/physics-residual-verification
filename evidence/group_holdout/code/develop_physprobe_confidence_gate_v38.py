"""Learn a nested-crossfit confidence gate for the V35 VLM vector residual."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from develop_physprobe_vector_readout_v35 import (
    FOLDS, fit, predict, metrics, make_views, fold_assignment,
)


FEATURES = ROOT / "cache/physprobe_push_v31/frozen_features_v1/FEATURES.npz"
INPUTS = ROOT / "data/physprobe_push_v31/observations_v1/inputs.npz"
TARGETS = ROOT / "results/com_observation_v35/physprobe_vector_development_targets_v1/TARGETS.npz"
V35 = ROOT / "results/com_observation_v35/physprobe_vector_readout_development_v1"
OUT = ROOT / "results/com_observation_v38/physprobe_confidence_gate_development_v1"
GATE_LAMBDAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
INNER_FOLDS = 4


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_npz(path: Path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def clipped_update(model, x, limit):
    update = predict(model, x)
    norm = np.linalg.norm(update, axis=1)
    return update * np.minimum(1.0, limit / np.maximum(norm, 1e-12))[:, None]


def quality_base(feature, inputs, rows):
    tracked = inputs["pixels"][rows].astype(np.float64)
    qwen = feature["qwen_pixels"][rows].astype(np.float64)
    static = feature["qwen_static_pixels"][rows].astype(np.float64)
    pooled = feature["pooled_evidence"][rows].astype(np.float64)
    size = inputs["object_size_px"][rows].astype(np.float64)
    vector_norm = lambda value: np.linalg.norm(value, axis=2)
    return np.column_stack(
        (
            np.linalg.norm(tracked[:, -1] - tracked[:, 0], axis=1),
            np.linalg.norm(qwen[:, -1] - qwen[:, 0], axis=1),
            vector_norm(qwen - tracked).mean(1),
            vector_norm(qwen - tracked).max(1),
            vector_norm(qwen - static).mean(1),
            vector_norm(static - tracked).mean(1),
            pooled.std(1).mean(1),
            np.min(size, axis=1) / np.maximum(size[:, 0], 1e-6),
            size[:, -1] / np.maximum(size[:, 0], 1e-6),
        )
    )


def gate_features(base, numeric_prediction, update):
    update_norm = np.linalg.norm(update, axis=1)
    numeric_norm = np.linalg.norm(numeric_prediction, axis=1)
    alignment = np.sum(numeric_prediction * update, axis=1) / np.maximum(numeric_norm * update_norm, 1e-12)
    return np.column_stack((base, numeric_norm, update_norm, alignment))


def optimal_gate(numeric_prediction, update, target):
    numerator = np.sum((target - numeric_prediction) * update, axis=1)
    denominator = np.sum(update ** 2, axis=1).clip(1e-12)
    return np.clip(numerator / denominator, 0.0, 1.0)[:, None]


def fit_components(numeric_x, visual_x, y, train, test, numeric_record, visual_record):
    numeric_model = fit(numeric_x[train], y[train], numeric_record["lambda"], numeric_record["robust_huber"])
    numeric_train = predict(numeric_model, numeric_x[train])
    numeric_test = predict(numeric_model, numeric_x[test])
    visual_model = fit(visual_x[train], y[train] - numeric_train, visual_record["lambda"], False)
    limit = max(float(np.std(np.linalg.norm(y[train], axis=1))), 1e-6)
    return numeric_test, clipped_update(visual_model, visual_x[test], limit)


def main():
    if OUT.exists():
        raise FileExistsError("Preserve the first V38 confidence-gate run")
    v35 = read_json(V35 / "RESULT.json")
    if v35["development_gate"]["passed"] or v35["confirmation_targets_opened"]:
        raise ValueError("V38 follows the development-only V35 failure")
    feature, inputs, target = load_npz(FEATURES), load_npz(INPUTS), load_npz(TARGETS)
    rows = np.flatnonzero(feature["roles"].astype(str) == "development")
    y = target["displacement_xy_m"].astype(np.float64)
    if feature["ids"][rows].tolist() != target["ids"].tolist():
        raise ValueError("Development identities changed")
    view = make_views(feature, inputs, rows)
    numeric_record = v35["selected_numeric_result"]
    visual_record = v35["selected_result"]
    numeric_x = view[numeric_record["view"]]
    visual_x = view[visual_record["visual_view"]]
    base = quality_base(feature, inputs, rows)
    assignment = fold_assignment(len(y))
    results = {}
    best = None
    for gate_lambda in GATE_LAMBDAS:
        output = np.empty_like(y)
        numeric_oof = np.empty_like(y)
        raw_oof = np.empty_like(y)
        gate_oof = np.empty(len(y))
        fold_records = []
        for outer in range(FOLDS):
            outer_train = np.flatnonzero(assignment != outer)
            outer_test = np.flatnonzero(assignment == outer)
            # Build unbiased gate targets within the outer-training cohort.
            order = np.random.default_rng(20260938 + outer).permutation(len(outer_train))
            inner_assignment = np.empty(len(outer_train), dtype=int)
            inner_assignment[order] = np.arange(len(outer_train)) % INNER_FOLDS
            inner_numeric = np.empty((len(outer_train), 2))
            inner_update = np.empty((len(outer_train), 2))
            for inner in range(INNER_FOLDS):
                inner_train = outer_train[inner_assignment != inner]
                inner_test_local = np.flatnonzero(inner_assignment == inner)
                inner_test = outer_train[inner_test_local]
                n_pred, update = fit_components(numeric_x, visual_x, y, inner_train, inner_test,
                                                numeric_record, visual_record)
                inner_numeric[inner_test_local] = n_pred
                inner_update[inner_test_local] = update
            gate_target = optimal_gate(inner_numeric, inner_update, y[outer_train])
            gate_x_train = gate_features(base[outer_train], inner_numeric, inner_update)
            gate_model = fit(gate_x_train, gate_target, gate_lambda, False)

            numeric_test, update_test = fit_components(numeric_x, visual_x, y, outer_train, outer_test,
                                                       numeric_record, visual_record)
            gate_x_test = gate_features(base[outer_test], numeric_test, update_test)
            gate = np.clip(predict(gate_model, gate_x_test).reshape(-1), 0.0, 1.0)
            prediction = numeric_test + gate[:, None] * update_test
            output[outer_test] = prediction
            numeric_oof[outer_test] = numeric_test
            raw_oof[outer_test] = numeric_test + update_test
            gate_oof[outer_test] = gate
            fold_records.append({"fold": outer, "mean_gate": float(gate.mean()),
                                 "numeric_mae_m": metrics(numeric_test, y[outer_test])["vector_mae_m"],
                                 "raw_vlm_mae_m": metrics(numeric_test + update_test, y[outer_test])["vector_mae_m"],
                                 "gated_vlm_mae_m": metrics(prediction, y[outer_test])["vector_mae_m"]})
        record = {"gate_lambda": gate_lambda, "gate_features": base.shape[1] + 3,
                  "mean_gate": float(gate_oof.mean()), "zero_gate_fraction": float(np.mean(gate_oof == 0)),
                  "one_gate_fraction": float(np.mean(gate_oof == 1)),
                  "fold_wins": int(sum(row["gated_vlm_mae_m"] < row["numeric_mae_m"] for row in fold_records)),
                  "folds": fold_records, **metrics(output, y)}
        results[f"gate_lambda{gate_lambda:g}"] = record
        criterion = (-record["vector_mae_m"], -record["vector_rmse_m"], record["fold_wins"], -gate_lambda)
        if best is None or criterion > best[0]:
            best = (criterion, f"gate_lambda{gate_lambda:g}", output, numeric_oof, raw_oof, gate_oof)
    selected_name, selected_prediction, numeric_oof, raw_oof, gate_oof = best[1:]
    selected = results[selected_name]
    numeric_mae = metrics(numeric_oof, y)["vector_mae_m"]
    improvement = 1 - selected["vector_mae_m"] / numeric_mae
    passed = improvement >= 0.10 and selected["fold_wins"] >= 4

    # Final component models use all development rows. Gate training remains
    # cross-fitted so its targets never come from in-sample component predictions.
    final_numeric = fit(numeric_x, y, numeric_record["lambda"], numeric_record["robust_huber"])
    numeric_fit = predict(final_numeric, numeric_x)
    final_visual = fit(visual_x, y - numeric_fit, visual_record["lambda"], False)
    limit = max(float(np.std(np.linalg.norm(y, axis=1))), 1e-6)
    final_update = clipped_update(final_visual, visual_x, limit)
    gate_target = optimal_gate(numeric_oof, raw_oof - numeric_oof, y)
    final_gate_x = gate_features(base, numeric_oof, raw_oof - numeric_oof)
    final_gate = fit(final_gate_x, gate_target, selected["gate_lambda"], False)
    errors = np.sort(np.linalg.norm(selected_prediction - y, axis=1))
    rank = min(len(errors) - 1, int(np.ceil((len(errors) + 1) * 0.9)) - 1)
    conformal_q90 = float(errors[rank])
    OUT.mkdir(parents=True)
    checkpoint = OUT / "CHECKPOINT.npz"
    with checkpoint.open("xb") as stream:
        np.savez_compressed(stream, schema_version=np.asarray("physprobe_confidence_gate_v38_v1"),
                            numeric_view=np.asarray(numeric_record["view"]), visual_view=np.asarray(visual_record["visual_view"]),
                            visual_clip_norm_m=np.asarray(limit), conformal_q90_m=np.asarray(conformal_q90),
                            **{f"numeric_{key}": value for key, value in final_numeric.items()},
                            **{f"visual_{key}": value for key, value in final_visual.items()},
                            **{f"gate_{key}": value for key, value in final_gate.items()})
    np.savez_compressed(OUT / "OOF_PREDICTIONS.npz", ids=target["ids"], target=y, numeric=numeric_oof,
                        raw_vlm=raw_oof, gated_vlm=selected_prediction, gate=gate_oof, fold=assignment)
    report = {"status": "completed nested-crossfit VLM confidence gate development", "episodes": len(y),
              "component_source": v35["selected_sequential"], "selected_gate": selected_name,
              "selected_result": selected, "numeric_replay": metrics(numeric_oof, y),
              "raw_vlm_replay": metrics(raw_oof, y),
              "relative_vector_mae_improvement_vs_strongest_numeric": float(improvement),
              "development_gate": {"required_relative_improvement": 0.10, "required_fold_wins": 4, "passed": bool(passed)},
              "conformal_q90_m": conformal_q90, "all_results": results,
              "checkpoint_sha256": sha256(checkpoint), "oof_predictions_sha256": sha256(OUT / "OOF_PREDICTIONS.npz"),
              "v35_result_sha256": sha256(V35 / "RESULT.json"), "confirmation_targets_opened": False,
              "source_sha256": sha256(Path(__file__)), "goal_complete": False}
    (OUT / "RESULT.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("selected_gate", "selected_result", "numeric_replay",
                      "raw_vlm_replay", "relative_vector_mae_improvement_vs_strongest_numeric",
                      "development_gate", "checkpoint_sha256")}, indent=2))


if __name__ == "__main__":
    main()
