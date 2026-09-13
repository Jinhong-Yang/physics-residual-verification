"""Develop a sequential numeric+VLM readout for 2-D future displacement."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "cache/physprobe_push_v31/frozen_features_v1/FEATURES.npz"
INPUTS = ROOT / "data/physprobe_push_v31/observations_v1/inputs.npz"
TARGET_DIR = ROOT / "results/com_observation_v35/physprobe_vector_development_targets_v1"
TARGETS = TARGET_DIR / "TARGETS.npz"
OUT = ROOT / "results/com_observation_v35/physprobe_vector_readout_development_v1"
FOLDS = 5
FOLD_SEED = 20260925
LAMBDAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
SHRINKAGE = (0.1, 0.25, 0.5, 1.0)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_npz(path: Path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def fit(x, y, regularization, robust=False):
    mean_x, scale_x = x.mean(0), x.std(0).clip(1e-8)
    xn = (x - mean_x) / scale_x
    design = np.column_stack((np.ones(len(x)), xn))
    penalty = np.diag([0.0] + [regularization] * x.shape[1])
    weights = np.ones(len(x))
    coef = None
    for _ in range(50 if robust else 1):
        root = np.sqrt(weights)[:, None]
        coef_new = np.linalg.solve((design * root).T @ (design * root) + penalty,
                                   (design * root).T @ (y * root))
        if not robust:
            coef = coef_new
            break
        residual_norm = np.linalg.norm(y - design @ coef_new, axis=1)
        center = np.median(residual_norm)
        scale = max(1.4826 * np.median(np.abs(residual_norm - center)), 1e-6)
        ratio = residual_norm / (1.345 * scale)
        updated = np.ones_like(ratio)
        mask = ratio > 1
        updated[mask] = 1 / ratio[mask]
        coef = coef_new
        if np.max(np.abs(updated - weights)) < 1e-8:
            break
        weights = updated
    return {"mean_x": mean_x, "scale_x": scale_x, "intercept": coef[0], "weight": coef[1:],
            "regularization": np.asarray(regularization), "robust": np.asarray(robust)}


def predict(model, x):
    return model["intercept"] + (x - model["mean_x"]) / model["scale_x"] @ model["weight"]


def metrics(prediction, target):
    distance = np.linalg.norm(prediction - target, axis=1)
    return {"vector_mae_m": float(distance.mean()), "vector_rmse_m": float(np.sqrt(np.mean(distance ** 2))),
            "median_error_m": float(np.median(distance)), "max_error_m": float(distance.max())}


def make_views(feature, inputs, rows):
    pixels = inputs["pixels"][rows].astype(np.float64)
    relative = pixels - pixels[:, :1]
    size = inputs["object_size_px"][rows].astype(np.float64)
    t = np.arange(8) * 0.1
    design = np.column_stack((np.ones(8), t, 0.5 * t * t))
    coeff_x = np.asarray([np.linalg.lstsq(design, row[:, 0], rcond=None)[0] for row in relative])
    coeff_y = np.asarray([np.linalg.lstsq(design, row[:, 1], rcond=None)[0] for row in relative])
    summary = np.column_stack((relative[:, -1], relative[:, -1] - relative[:, -2],
                               coeff_x[:, 1:], coeff_y[:, 1:], pixels[:, 0], size.mean(1)))
    pooled = feature["pooled_evidence"][rows].astype(np.float64)
    motion = feature["visual_features"][rows, 32:].astype(np.float64)
    time = np.linspace(-1, 1, 8)
    stats = np.concatenate((pooled.mean(1), pooled.std(1),
                            pooled[:, 4:].mean(1) - pooled[:, :4].mean(1),
                            np.einsum("t,ntd->nd", time / np.sum(time ** 2), pooled), motion), axis=1)
    return {
        "numeric_summary": summary,
        "numeric_relative": np.concatenate((relative.reshape(len(rows), -1), size), axis=1),
        "numeric_absolute_relative": np.concatenate((pixels.reshape(len(rows), -1), relative.reshape(len(rows), -1), size), axis=1),
        "visual_mean_motion": feature["visual_features"][rows].astype(np.float64),
        "visual_temporal_stats": stats,
        "visual_all_slots_motion": np.concatenate((pooled.reshape(len(rows), -1), motion), axis=1),
    }


def fold_assignment(count):
    order = np.random.default_rng(FOLD_SEED).permutation(count)
    output = np.empty(count, dtype=int)
    output[order] = np.arange(count) % FOLDS
    return output


def main():
    if OUT.exists():
        raise FileExistsError("Preserve the first V35 vector development run")
    feature, inputs, target = load_npz(FEATURES), load_npz(INPUTS), load_npz(TARGETS)
    rows = np.flatnonzero(feature["roles"].astype(str) == "development")
    if feature["ids"][rows].tolist() != inputs["ids"][rows].tolist() or feature["ids"][rows].tolist() != target["ids"].tolist():
        raise ValueError("Development identities changed")
    y = target["displacement_xy_m"].astype(np.float64)
    views = make_views(feature, inputs, rows)
    assignment = fold_assignment(len(y))
    numeric_results = {}
    best_numeric = None
    for view_name in ("numeric_summary", "numeric_relative", "numeric_absolute_relative"):
        for robust in (False, True):
            for regularization in LAMBDAS:
                prediction = np.empty_like(y)
                fold_mae = []
                for fold_index in range(FOLDS):
                    train, test = assignment != fold_index, assignment == fold_index
                    model = fit(views[view_name][train], y[train], regularization, robust)
                    prediction[test] = predict(model, views[view_name][test])
                    fold_mae.append(float(np.mean(np.linalg.norm(prediction[test] - y[test], axis=1))))
                record = {"view": view_name, "robust_huber": robust, "lambda": regularization,
                          "dimension": views[view_name].shape[1], "fold_mae_m": fold_mae,
                          **metrics(prediction, y)}
                key = f"{view_name}__{'huber' if robust else 'ridge'}__lambda{regularization:g}"
                numeric_results[key] = record
                criterion = (-record["vector_mae_m"], -record["vector_rmse_m"], -record["dimension"], -regularization)
                if best_numeric is None or criterion > best_numeric[0]:
                    best_numeric = (criterion, key, prediction)
    numeric_name, numeric_prediction = best_numeric[1], best_numeric[2]
    numeric_record = numeric_results[numeric_name]
    numeric_x = views[numeric_record["view"]]
    numeric_spec = {"lambda": numeric_record["lambda"], "robust": numeric_record["robust_huber"]}

    sequential_results = {}
    best = None
    for visual_name in ("visual_mean_motion", "visual_temporal_stats", "visual_all_slots_motion"):
        visual_x = views[visual_name]
        for regularization in LAMBDAS:
            for shrinkage in SHRINKAGE:
                prediction = np.empty_like(y)
                component = np.empty_like(y)
                fold_records = []
                for fold_index in range(FOLDS):
                    train, test = assignment != fold_index, assignment == fold_index
                    numeric_model = fit(numeric_x[train], y[train], numeric_spec["lambda"], numeric_spec["robust"])
                    numeric_train = predict(numeric_model, numeric_x[train])
                    numeric_test = predict(numeric_model, numeric_x[test])
                    visual_model = fit(visual_x[train], y[train] - numeric_train, regularization, False)
                    update = predict(visual_model, visual_x[test])
                    limit = max(float(np.std(np.linalg.norm(y[train], axis=1))), 1e-6)
                    norm = np.linalg.norm(update, axis=1)
                    update *= np.minimum(1.0, limit / np.maximum(norm, 1e-12))[:, None]
                    update *= shrinkage
                    prediction[test] = numeric_test + update
                    component[test] = update
                    fold_records.append({"fold": fold_index,
                                         "numeric_mae_m": float(np.mean(np.linalg.norm(numeric_test - y[test], axis=1))),
                                         "sequential_mae_m": float(np.mean(np.linalg.norm(prediction[test] - y[test], axis=1)))})
                record = {"visual_view": visual_name, "visual_dimension": visual_x.shape[1],
                          "lambda": regularization, "shrinkage": shrinkage,
                          "mean_visual_update_norm_m": float(np.mean(np.linalg.norm(component, axis=1))),
                          "fold_wins": int(sum(row["sequential_mae_m"] < row["numeric_mae_m"] for row in fold_records)),
                          "folds": fold_records, **metrics(prediction, y)}
                key = f"{visual_name}__lambda{regularization:g}__shrink{shrinkage:g}"
                sequential_results[key] = record
                criterion = (-record["vector_mae_m"], -record["vector_rmse_m"], record["fold_wins"],
                             -record["visual_dimension"], -regularization, -shrinkage)
                if best is None or criterion > best[0]:
                    best = (criterion, key, prediction, component)
    selected_name, selected_prediction, component = best[1], best[2], best[3]
    selected = sequential_results[selected_name]
    improvement = 1 - selected["vector_mae_m"] / numeric_record["vector_mae_m"]
    passed = improvement >= 0.10 and selected["fold_wins"] >= 4

    final_numeric = fit(numeric_x, y, numeric_spec["lambda"], numeric_spec["robust"])
    residual = y - predict(final_numeric, numeric_x)
    final_visual_x = views[selected["visual_view"]]
    final_visual = fit(final_visual_x, residual, selected["lambda"], False)
    oof_error = np.sort(np.linalg.norm(selected_prediction - y, axis=1))
    rank = min(len(oof_error) - 1, int(np.ceil((len(oof_error) + 1) * 0.9)) - 1)
    conformal_q90 = float(oof_error[rank])
    OUT.mkdir(parents=True)
    checkpoint = OUT / "CHECKPOINT.npz"
    with checkpoint.open("xb") as stream:
        np.savez_compressed(stream, schema_version=np.asarray("physprobe_vector_readout_v35_v1"),
                            numeric_view=np.asarray(numeric_record["view"]), visual_view=np.asarray(selected["visual_view"]),
                            visual_shrinkage=np.asarray(selected["shrinkage"]), target_training_norm_std=np.asarray(np.std(np.linalg.norm(y, axis=1))),
                            conformal_q90_m=np.asarray(conformal_q90),
                            **{f"numeric_{key}": value for key, value in final_numeric.items()},
                            **{f"visual_{key}": value for key, value in final_visual.items()})
    np.savez_compressed(OUT / "OOF_PREDICTIONS.npz", ids=target["ids"], target=y,
                        numeric=numeric_prediction, sequential=selected_prediction, visual_component=component, fold=assignment)
    report = {
        "status": "completed development-only PhysProbe vector readout",
        "episodes": len(y), "selected_numeric": numeric_name, "selected_numeric_result": numeric_record,
        "selected_sequential": selected_name, "selected_result": selected,
        "relative_vector_mae_improvement_vs_strongest_numeric": float(improvement),
        "development_gate": {"required_relative_improvement": 0.10, "required_fold_wins": 4, "passed": bool(passed)},
        "conformal_q90_m": conformal_q90, "all_numeric_results": numeric_results,
        "all_sequential_results": sequential_results, "checkpoint_sha256": sha256(checkpoint),
        "oof_predictions_sha256": sha256(OUT / "OOF_PREDICTIONS.npz"),
        "target_seal_sha256": sha256(TARGET_DIR / "TARGET_SEAL.json"),
        "confirmation_targets_opened": False, "source_sha256": sha256(Path(__file__)), "goal_complete": False,
    }
    (OUT / "RESULT.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("selected_numeric", "selected_numeric_result", "selected_sequential",
                      "selected_result", "relative_vector_mae_improvement_vs_strongest_numeric", "development_gate",
                      "checkpoint_sha256")}, indent=2))


if __name__ == "__main__":
    main()
