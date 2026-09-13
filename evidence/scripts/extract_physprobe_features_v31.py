"""Extract frozen Qwen/V22 evidence for sealed PhysProbe RGB observations."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import stream_com_internal_features_v16 as stream


DATA = ROOT / "data/physprobe_push_v31"
OBS = DATA / "observations_v1"
OUT = ROOT / "cache/physprobe_push_v31/frozen_features_v1"
MODEL = ROOT / "models/qwen3-vl-2b/model.safetensors"
DECODER = ROOT / "results/com_observation_v9/decoder/continuation/qwen_warp_joint/epoch10.pt"


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_npz(path: Path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def install_label_guard():
    def guard(event, args):
        if event != "open" or not args or isinstance(args[0], int):
            return
        path = Path(args[0])
        if path.suffix.lower() == ".parquet" or "evaluator_only" in {part.lower() for part in path.parts}:
            raise PermissionError("V31 feature extractor cannot access labels or trajectories")

    sys.addaudithook(guard)


def main():
    if OUT.exists():
        raise FileExistsError("Preserve the first PhysProbe frozen features")
    seal = read_json(OBS / "OBSERVATION_SEAL.json")
    if seal["labels_or_trajectories_read"]:
        raise ValueError("Observation seal is not label-free")
    if seal["input_index_sha256"] != sha256(OBS / "INPUT_INDEX.json"):
        raise ValueError("Observation index changed")
    if seal["inputs_sha256"] != sha256(OBS / "inputs.npz"):
        raise ValueError("Observation arrays changed")
    episodes = read_json(OBS / "INPUT_INDEX.json")["episodes"]
    inputs = load_npz(OBS / "inputs.npz")
    if inputs["ids"].tolist() != [row["id"] for row in episodes]:
        raise ValueError("Observation row order changed")

    OUT.mkdir(parents=True)
    plan = {
        "status": "fixed frozen feature extraction before any selected labels or trajectories",
        "observation_seal_sha256": sha256(OBS / "OBSERVATION_SEAL.json"),
        "episodes": len(episodes),
        "frames": len(episodes) * 8,
        "base_model_sha256": sha256(MODEL),
        "decoder_checkpoint_sha256": sha256(DECODER),
        "training": False,
        "labels_or_trajectories_read": False,
        "source_sha256": sha256(Path(__file__)),
    }
    (OUT / "PLAN.json").write_text(json.dumps(plan, indent=2, allow_nan=False), encoding="utf-8")
    result = stream.run_rows(episodes, inputs, stream.load_models(), progress=True)
    axes = inputs["pixels"][:, -1] - inputs["pixels"][:, 0]
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    dynamic = np.einsum("nti,ni->nt", result["pixels"] - result["static_pixels"], axes)
    static = np.einsum("nti,ni->nt", result["static_pixels"] - inputs["pixels"], axes)
    visual = np.concatenate((result["pooled"].mean(1), dynamic, static), axis=1).astype(np.float32)
    numeric = np.concatenate(
        (inputs["tracked_along"], inputs["tracked_perpendicular"], inputs["object_size_px"]), axis=1
    ).astype(np.float64)
    feature_path = OUT / "FEATURES.npz"
    with feature_path.open("xb") as handle:
        np.savez_compressed(
            handle,
            ids=inputs["ids"],
            episode_indices=inputs["episode_indices"],
            roles=inputs["roles"],
            numeric_features=numeric,
            visual_features=visual,
            pooled_evidence=result["pooled"],
            qwen_pixels=result["pixels"],
            qwen_static_pixels=result["static_pixels"],
        )
    manifest = {
        "status": "sealed frozen PhysProbe Qwen/V22 evidence",
        "plan_sha256": sha256(OUT / "PLAN.json"),
        "features_path": feature_path.relative_to(ROOT).as_posix(),
        "features_sha256": sha256(feature_path),
        "episodes": len(episodes),
        "frames": len(episodes) * 8,
        "numeric_dim": numeric.shape[1],
        "visual_dim": visual.shape[1],
        "vision_forwards": len(episodes),
        "decoder_forwards": len(episodes),
        "training": False,
        "labels_or_trajectories_read": False,
        "elapsed_s": result["elapsed_s"],
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      TOKENIZERS_PARALLELISM="false", PYTHONDONTWRITEBYTECODE="1")
    install_label_guard()
    main()
