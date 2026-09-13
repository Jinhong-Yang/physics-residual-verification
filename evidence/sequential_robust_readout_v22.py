"""Deterministic sequential robust posterior readout selected in V22 development."""
from __future__ import annotations

from pathlib import Path
import numpy as np


PROTOCOLS = ("multiple_forces", "unforced_slide", "bounce")
SUBSPACES = {"multiple_forces": (0, 1), "unforced_slide": (1,), "bounce": (2,)}
REGIMES = ("generalized_translation", "setup_translation")
NUMERIC_LAMBDA = 1.0
NUMERIC_LIMIT_STD = 2.0
VISUAL_LAMBDA = 10.0
VISUAL_LIMIT_STD = 1.0
VISUAL_SHRINKAGE = 0.25
DISAGREEMENT_SCALE = 0.10


def fit_ridge(x, y, regularization):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mean_x, scale_x, mean_y = x.mean(0), x.std(0).clip(1e-6), y.mean(0)
    normalized = (x - mean_x) / scale_x
    weight = np.linalg.solve(
        normalized.T @ normalized + regularization * np.eye(normalized.shape[1]),
        normalized.T @ (y - mean_y),
    )
    return {"mean_x": mean_x, "scale_x": scale_x, "mean_y": mean_y, "weight": weight}


def ridge_predict(model, x):
    return (np.asarray(x, dtype=np.float64) - model["mean_x"]) / model["scale_x"] @ model["weight"] + model["mean_y"]


def feature_views(features, numeric_dim):
    features = np.asarray(features)
    if features.ndim != 3 or features.shape[1] != 8 or not 0 < numeric_dim < features.shape[2]:
        raise ValueError("Unexpected V22 feature schema")
    numeric = features[:, :, :numeric_dim].reshape(len(features), -1).astype(np.float64)
    visual = features[:, :, numeric_dim:].astype(np.float64)
    if visual.shape[2] != 34:
        raise ValueError("V22 expects pooled32 plus two Qwen observation channels")
    visual_view = np.concatenate((visual[:, :, :32].mean(1), visual[:, :, -2:].reshape(len(features), -1)), 1)
    return numeric, visual_view


def _key(branch, regime, protocol, field):
    return f"{branch}__{regime}__{protocol}__{field}"


def store_model(payload, branch, regime, protocol, model):
    for field, value in model.items():
        payload[_key(branch, regime, protocol, field)] = value


def load_model(payload, branch, regime, protocol):
    return {field: payload[_key(branch, regime, protocol, field)] for field in ("mean_x", "scale_x", "mean_y", "weight")}


def fit_checkpoint(features, numeric_dim, truth, active, protocol, anchor):
    numeric_x, visual_x = feature_views(features, numeric_dim)
    truth = np.asarray(truth, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    protocol = np.asarray(protocol).astype(str)
    payload = {
        "schema_version": np.asarray("sequential_robust_readout_v22_v1"),
        "numeric_dim": np.asarray(numeric_dim),
        "numeric_lambda": np.asarray(NUMERIC_LAMBDA),
        "numeric_limit_std": np.asarray(NUMERIC_LIMIT_STD),
        "visual_lambda": np.asarray(VISUAL_LAMBDA),
        "visual_limit_std": np.asarray(VISUAL_LIMIT_STD),
        "visual_shrinkage": np.asarray(VISUAL_SHRINKAGE),
        "disagreement_scale": np.asarray(DISAGREEMENT_SCALE),
    }
    for regime in REGIMES:
        mean = np.asarray(anchor[regime]["mean"], dtype=np.float64)
        covariance = np.asarray(anchor[regime]["covariance"], dtype=np.float64)
        std = np.sqrt(np.diagonal(covariance, axis1=1, axis2=2))
        target_std = (truth - mean) / std
        numeric_std = np.zeros_like(mean)
        for name, columns in SUBSPACES.items():
            rows = np.flatnonzero(protocol == name)
            model = fit_ridge(numeric_x[rows], target_std[rows][:, columns], NUMERIC_LAMBDA)
            store_model(payload, "numeric", regime, name, model)
            numeric_std[np.ix_(rows, columns)] = np.clip(ridge_predict(model, numeric_x[rows]), -NUMERIC_LIMIT_STD, NUMERIC_LIMIT_STD)
        numeric_mean = mean + numeric_std * std * active
        residual_std = (truth - numeric_mean) / std
        for name, columns in SUBSPACES.items():
            rows = np.flatnonzero(protocol == name)
            model = fit_ridge(visual_x[rows], residual_std[rows][:, columns], VISUAL_LAMBDA)
            store_model(payload, "visual", regime, name, model)
    return payload


def predict(checkpoint, features, active, protocol, anchor, include_vlm=True):
    numeric_dim = int(np.asarray(checkpoint["numeric_dim"]))
    if float(np.asarray(checkpoint["disagreement_scale"])) != DISAGREEMENT_SCALE:
        raise ValueError("V22 disagreement scale changed")
    numeric_x, visual_x = feature_views(features, numeric_dim)
    active = np.asarray(active, dtype=bool)
    protocol = np.asarray(protocol).astype(str)
    outputs = {}
    for regime in REGIMES:
        mean = np.asarray(anchor[regime]["mean"], dtype=np.float64)
        covariance = np.asarray(anchor[regime]["covariance"], dtype=np.float64)
        if mean.shape != (len(features), 3) or covariance.shape != (len(features), 3, 3):
            raise ValueError("Anchor shape mismatch")
        std = np.sqrt(np.diagonal(covariance, axis1=1, axis2=2))
        numeric_std = np.zeros_like(mean)
        visual_std = np.zeros_like(mean)
        for name, columns in SUBSPACES.items():
            rows = np.flatnonzero(protocol == name)
            numeric_model = load_model(checkpoint, "numeric", regime, name)
            numeric_std[np.ix_(rows, columns)] = np.clip(
                ridge_predict(numeric_model, numeric_x[rows]), -NUMERIC_LIMIT_STD, NUMERIC_LIMIT_STD
            )
            if include_vlm:
                visual_model = load_model(checkpoint, "visual", regime, name)
                visual_std[np.ix_(rows, columns)] = np.clip(
                    ridge_predict(visual_model, visual_x[rows]), -VISUAL_LIMIT_STD, VISUAL_LIMIT_STD
                )
        delta = (numeric_std + VISUAL_SHRINKAGE * visual_std) * std * active
        if regime == "generalized_translation":
            delta[protocol == "bounce"] = 0
        updated_mean = mean + delta
        updated_covariance = covariance + np.asarray([np.diag((DISAGREEMENT_SCALE * row) ** 2) for row in delta])
        np.linalg.cholesky(updated_covariance)
        outputs[regime] = {
            "mean": updated_mean, "covariance": updated_covariance,
            "delta": delta, "numeric_std": numeric_std, "visual_std": visual_std,
        }
    return outputs


def load_checkpoint(path):
    with np.load(Path(path), allow_pickle=False) as archive:
        result = {key: archive[key].copy() for key in archive.files}
    if str(result["schema_version"]) != "sequential_robust_readout_v22_v1":
        raise ValueError("Wrong V22 checkpoint schema")
    return result
