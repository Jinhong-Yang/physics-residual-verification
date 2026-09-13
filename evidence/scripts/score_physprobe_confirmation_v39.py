"""Open untouched PhysProbe confirmation trajectories and score sealed V39 outputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as parquet


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/physprobe_push_v31"
OBS = DATA / "observations_v1"
OUT = ROOT / "results/com_observation_v39/physprobe_independent_confirmation_v1"
SEAL = OUT / "PREDICTION_SEAL.json"
METRIC = OUT / "METRIC_PLAN.json"
METHODS = ("constant_prior", "numeric", "raw_vlm", "gated_vlm")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def isotropic_nll(error, radial_q90):
    sigma = radial_q90 / np.sqrt(-2 * np.log(0.1))
    return np.log(2 * np.pi * sigma ** 2) + np.sum(error ** 2, axis=1) / (2 * sigma ** 2)


def main():
    result_path = OUT / "RESULT.json"
    if result_path.exists():
        raise FileExistsError("Preserve the first V39 confirmation result")
    seal, metric = read_json(SEAL), read_json(METRIC)
    if sha256(Path(__file__)) != metric["scorer_sha256"] or sha256(SEAL) != metric["prediction_seal_sha256"]:
        raise ValueError("V39 metric lock or prediction seal changed")
    prediction_path = ROOT / seal["prediction_path"]
    if sha256(prediction_path) != metric["prediction_sha256"]:
        raise ValueError("V39 prediction bytes changed")
    with np.load(prediction_path, allow_pickle=False) as archive:
        prediction = {key: archive[key].copy() for key in archive.files}
    download = read_json(DATA / "DOWNLOAD_SEAL.json")
    episodes = [row for row in read_json(OBS / "INPUT_INDEX.json")["episodes"] if row["role"] == "confirmation"]
    if prediction["ids"].tolist() != [row["id"] for row in episodes]:
        raise ValueError("Confirmation identity/order changed")
    file_index = {(row["role"], row["episode_index"], row["kind"]): row for row in download["files"]}
    target = []
    source_hashes = []
    for row in episodes:
        receipt = file_index[("confirmation", row["episode_index"], "parquet")]
        path = ROOT / receipt["path"]
        if sha256(path) != receipt["sha256"]:
            raise ValueError("Public confirmation trajectory changed")
        table = parquet.read_table(path, columns=["physics_gt.object_position"])
        position = np.asarray(table["physics_gt.object_position"].to_pylist(), dtype=np.float64)
        target.append(position[row["target_index"], :2] - position[row["onset_index"], :2])
        source_hashes.append(receipt["sha256"])
    target = np.asarray(target)
    errors = {method: np.linalg.norm(prediction[method] - target, axis=1) for method in METHODS}
    method_results = {method: {"vector_mae_m": float(value.mean()),
                               "vector_rmse_m": float(np.sqrt(np.mean(value ** 2))),
                               "median_error_m": float(np.median(value)),
                               "max_error_m": float(value.max())}
                      for method, value in errors.items()}
    rng = np.random.default_rng(metric["bootstrap"]["seed"])
    draws = rng.integers(0, len(target), size=(metric["bootstrap"]["draws"], len(target)))
    delta = errors["gated_vlm"] - errors["numeric"]
    boot_delta = delta[draws].mean(1)
    prior_relative = 1 - errors["gated_vlm"].mean() / errors["constant_prior"].mean()
    boot_prior_relative = 1 - errors["gated_vlm"][draws].mean(1) / errors["constant_prior"][draws].mean(1)
    numeric_q90 = float(prediction["numeric_radial_q90_m"])
    gated_q90 = float(prediction["gated_radial_q90_m"])
    numeric_coverage = float(np.mean(errors["numeric"] <= numeric_q90))
    gated_coverage = float(np.mean(errors["gated_vlm"] <= gated_q90))
    nll_numeric = isotropic_nll(prediction["numeric"] - target, numeric_q90)
    nll_gated = isotropic_nll(prediction["gated_vlm"] - target, gated_q90)
    primary = {
        "mean_paired_error_delta_gated_minus_numeric_m": float(delta.mean()),
        "one_sided_95pct_bootstrap_upper_m": float(np.quantile(boot_delta, 0.95)),
        "two_sided_95pct_bootstrap_interval_m": np.quantile(boot_delta, [0.025, 0.975]).tolist(),
        "relative_mae_improvement_vs_numeric": float(1 - errors["gated_vlm"].mean() / errors["numeric"].mean()),
        "strict_directional_support": bool(np.quantile(boot_delta, 0.95) < 0),
    }
    calibration = {
        "numeric_radial_q90_m": numeric_q90, "numeric_coverage": numeric_coverage,
        "gated_radial_q90_m": gated_q90, "gated_coverage": gated_coverage,
        "candidate_coverage_in_locked_range": bool(metric["calibration"]["coverage_min"] <= gated_coverage <= metric["calibration"]["coverage_max"]),
        "numeric_mean_isotropic_nll": float(nll_numeric.mean()),
        "gated_mean_isotropic_nll": float(nll_gated.mean()),
        "gated_nll_better": bool(nll_gated.mean() < nll_numeric.mean()),
    }
    support = primary["strict_directional_support"] and calibration["candidate_coverage_in_locked_range"]
    result = {
        "status": "completed untouched public independent-engine confirmation",
        "dataset": "PhysProbe Push / Isaac Sim 4.5 and Isaac Lab 2.2.1",
        "episodes": len(target),
        "target": "planar object displacement vector from visual motion onset to +1.8 s",
        "methods": method_results,
        "primary_vlm_increment": primary,
        "candidate_vs_constant_prior": {
            "relative_mae_improvement": float(prior_relative),
            "bootstrap95_interval": np.quantile(boot_prior_relative, [0.025, 0.975]).tolist(),
        },
        "calibration": calibration,
        "mean_confidence_gate": float(prediction["gate"].mean()),
        "independent_engine_support": bool(support),
        "claim_scope": metric["claim_scope"],
        "v38_original_10pct_development_gate_passed": seal["v38_original_10pct_development_gate_passed"],
        "prediction_seal_sha256": sha256(SEAL),
        "metric_plan_sha256": sha256(METRIC),
        "confirmation_source_digest_sha256": hashlib.sha256("".join(source_hashes).encode()).hexdigest(),
        "confirmation_target_reads": len(target),
        "goal_complete": False,
    }
    np.savez_compressed(OUT / "CONFIRMATION_ERRORS.npz", ids=prediction["ids"], target=target,
                        bootstrap_draw=draws, **{f"error_{key}": value for key, value in errors.items()})
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
