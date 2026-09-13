"""Stream frozen Qwen vision features into the V9 head and retain only V14 inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODEL = ROOT / "models/qwen3-vl-2b"
CHECKPOINT = ROOT / "results/com_observation_v9/decoder/continuation/qwen_warp_joint/epoch10.pt"
DATA_V16 = ROOT / "data/canonical_confirmation_v16"
INPUT_V16 = ROOT / "cache/canonical_confirmation_v16/input"
OUT = ROOT / "cache/canonical_confirmation_v16/vlm_stream_v1"
REGRESSION = ROOT / "results/com_observation_v16/streaming_regression_v3"


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
            raise PermissionError("V16 predictor cannot open evaluator-only data")
        if ("target" in leaf or "label" in leaf or "quality" in leaf) and path.suffix.lower() in {
            ".json", ".jsonl", ".npz", ".npy", ".pt"
        }:
            raise PermissionError("V16 predictor cannot open supervised artifacts")

    sys.addaudithook(guard)


def verify_v16_inputs():
    observation_seal = read(DATA_V16 / "OBSERVATION_SEAL.json")
    input_seal = read(INPUT_V16 / "INPUT_SEAL.json")
    if observation_seal["status"] != "sealed" or input_seal["status"] != "sealed":
        raise ValueError("V16 inputs are not sealed")
    if input_seal["inputs_sha256"] != sha(INPUT_V16 / "inputs.npz"):
        raise ValueError("V16 numerical inputs changed")
    index = read(DATA_V16 / "INPUT_INDEX.json")
    values = load(INPUT_V16 / "inputs.npz")
    episodes = index["episodes"]
    if values["ids"].tolist() != [row["id"] for row in episodes]:
        raise ValueError("V16 input order changed")
    if len(episodes) != 1080:
        raise ValueError("Expected the complete V16 cohort")
    return episodes, values, observation_seal, input_seal


def load_models():
    import torch
    from transformers import AutoImageProcessor
    from transformers.vision_utils import get_vision_position_ids
    from scripts.extract_spatial_vlm_observations_v3 import load_vision_only
    from com_observation_decoder_v8 import COMObservationDecoderV8

    processor = AutoImageProcessor.from_pretrained(MODEL, local_files_only=True)
    position = get_vision_position_ids(torch.tensor([[1, 32, 32]], dtype=torch.int64), 2).cpu()
    order = torch.argsort(position[:, 0] * 32 + position[:, 1])
    raster = position[order]
    if not torch.equal(raster[:, 0] * 32 + raster[:, 1], torch.arange(1024)):
        raise ValueError("Vision raster order changed")
    vision = load_vision_only()
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    if state["condition"] != "qwen_warp_joint" or state["epoch"] != 10:
        raise ValueError("Wrong frozen V9 endpoint")
    decoder = COMObservationDecoderV8("qwen", "warp").cuda().eval()
    decoder.load_state_dict(state["model"], strict=True)
    decoder.requires_grad_(False)
    return torch, processor, order, vision, decoder


def run_rows(episodes, values, models, progress=False, decoder_batch_size=8):
    torch, processor, order, vision, decoder = models
    from PIL import Image

    static_calls = []
    dynamic_calls = []

    def capture_static(module, inputs, output):
        static_calls.append(inputs[0].detach().cpu())

    def capture_dynamic(module, inputs, output):
        dynamic_calls.append(inputs[0].detach().cpu())

    static_hook = decoder.static_head.register_forward_hook(capture_static)
    dynamic_hook = decoder.dynamic_head.register_forward_hook(capture_dynamic)
    pooled = []
    pixels = []
    static_pixels = []
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for batch_start in range(0, len(episodes), decoder_batch_size):
                batch_episodes = episodes[batch_start : batch_start + decoder_batch_size]
                batch_premerge = []
                batch_rgb = []
                batch_bbox = []
                for episode in batch_episodes:
                    images = []
                    for name, digest in zip(episode["frame_paths"], episode["frame_sha256"]):
                        path = ROOT / name
                        if sha(path) != digest:
                            raise ValueError("RGB frame changed")
                        with Image.open(path) as image:
                            images.append(np.asarray(image.convert("RGB")).copy())
                    payload = processor(
                        images=images,
                        size={"shortest_edge": 512**2, "longest_edge": 512**2},
                        return_tensors="pt",
                    )
                    if payload["image_grid_thw"].tolist() != [[1, 32, 32]] * 8:
                        raise ValueError("Vision grid changed")
                    dense = vision(
                        payload["pixel_values"].to("cuda", dtype=torch.bfloat16),
                        grid_thw=payload["image_grid_thw"].to("cuda"),
                        return_dict=True,
                    ).last_hidden_state
                    if dense.shape != (8192, 1024):
                        raise ValueError("Unexpected streamed vision shape")
                    batch_premerge.append(
                        dense.reshape(8, 1024, 1024)[:, order.to("cuda")].contiguous().cpu()
                    )
                    batch_rgb.append(
                        torch.stack(
                            [torch.from_numpy(image).permute(2, 0, 1).float() / 255 for image in images]
                        )
                    )
                    batch_bbox.append(episode["initial_cue"]["bbox_xyxy"])
                count = len(batch_episodes)
                premerge = torch.stack(batch_premerge).cuda()
                rgb = torch.stack(batch_rgb).cuda()
                bbox = torch.tensor(batch_bbox, dtype=torch.float32, device="cuda")
                anchor = torch.from_numpy(values["pixels"][batch_start : batch_start + count]).double().cuda()
                times = torch.from_numpy(values["times"][batch_start : batch_start + count]).float().cuda()
                static_calls.clear()
                dynamic_calls.clear()
                result = decoder(rgb, premerge, bbox, anchor, times, return_diagnostics=True)
                if len(static_calls) != 1 or len(dynamic_calls) != 7:
                    raise ValueError("Frozen head call topology changed")
                evidence = torch.cat((static_calls[0][:, None], torch.stack(dynamic_calls, 1)), 1)
                if evidence.shape != (count, 8, 32):
                    raise ValueError("Unexpected V14 evidence shape")
                pooled.append(evidence.numpy())
                pixels.append(result["pixels"].cpu().numpy())
                static_pixels.append(
                    (anchor + result["static_bias"].double()[:, None]).cpu().numpy()
                )
                completed = batch_start + count
                if progress and (completed % 40 == 0 or batch_start == 0 or completed == len(episodes)):
                    print(
                        json.dumps(
                            {"episodes": completed, "total": len(episodes), "elapsed_s": time.perf_counter() - started}
                        ),
                        flush=True,
                    )
    finally:
        static_hook.remove()
        dynamic_hook.remove()
    return {
        "pooled": np.concatenate(pooled).astype(np.float32, copy=False),
        "pixels": np.concatenate(pixels),
        "static_pixels": np.concatenate(static_pixels),
        "elapsed_s": time.perf_counter() - started,
    }


def regression_v11():
    if REGRESSION.exists():
        raise FileExistsError("Preserve existing streaming regression")
    REGRESSION.mkdir(parents=True)
    data = ROOT / "data/canonical_confirmation_v11/INPUT_INDEX.json"
    inputs = ROOT / "cache/canonical_confirmation_v11/input/inputs.npz"
    reference = ROOT / "results/com_observation_v11/canonical_confirmation_v1/heads/qwen_warp_joint.npz"
    static_reference = ROOT / "results/com_observation_v11/canonical_confirmation_v1/heads/qwen_warp_joint_static_pixels.npz"
    episodes = read(data)["episodes"][:8]
    all_values = load(inputs)
    values = {key: value[:8] for key, value in all_values.items()}
    models = load_models()
    result = run_rows(episodes, values, models)
    expected = load(reference)["pixels"][:8]
    expected_static = load(static_reference)["pixels"][:8]
    receipt = {
        "status": "passed" if np.array_equal(result["pixels"], expected) and np.array_equal(result["static_pixels"], expected_static) else "failed",
        "episodes": 8,
        "streamed_qwen_pixels_bitexact": bool(np.array_equal(result["pixels"], expected)),
        "streamed_static_pixels_bitexact": bool(np.array_equal(result["static_pixels"], expected_static)),
        "qwen_max_abs_difference_pixels": float(np.max(np.abs(result["pixels"] - expected))),
        "static_max_abs_difference_pixels": float(np.max(np.abs(result["static_pixels"] - expected_static))),
        "checkpoint_sha256": sha(CHECKPOINT),
        "reference_sha256": sha(reference),
        "static_reference_sha256": sha(static_reference),
        "source_sha256": sha(Path(__file__)),
        "targets_opened": False,
        "elapsed_s": result["elapsed_s"],
    }
    write_new(REGRESSION / "REGRESSION.json", receipt)
    if receipt["status"] != "passed":
        raise RuntimeError("Streaming path differs from saved V11 predictions")
    print(json.dumps(receipt, indent=2))


def extract_v16():
    if OUT.exists():
        raise FileExistsError("Preserve existing V16 streamed features")
    regression = read(REGRESSION / "REGRESSION.json")
    if regression["status"] != "passed" or regression["source_sha256"] != sha(Path(__file__)):
        raise ValueError("Matching passed streaming regression is required")
    episodes, values, observation_seal, input_seal = verify_v16_inputs()
    OUT.mkdir(parents=True)
    plan = {
        "status": "fixed before V16 streamed Qwen inference",
        "candidate_lock_sha256": observation_seal["candidate_lock_sha256"],
        "observation_seal_sha256": sha(DATA_V16 / "OBSERVATION_SEAL.json"),
        "input_seal_sha256": sha(INPUT_V16 / "INPUT_SEAL.json"),
        "input_sha256": input_seal["inputs_sha256"],
        "checkpoint_sha256": sha(CHECKPOINT),
        "model_sha256": sha(MODEL / "model.safetensors"),
        "source_sha256": {
            "scripts/stream_com_internal_features_v16.py": sha(Path(__file__)),
            "scripts/extract_spatial_vlm_observations_v3.py": sha(ROOT / "scripts/extract_spatial_vlm_observations_v3.py"),
            "com_observation_decoder_v8.py": sha(ROOT / "com_observation_decoder_v8.py"),
        },
        "regression_sha256": sha(REGRESSION / "REGRESSION.json"),
        "episodes": 1080,
        "frames": 8640,
        "training": False,
        "targets_opened": False,
    }
    write_new(OUT / "PLAN.json", plan)
    models = load_models()
    result = run_rows(episodes, values, models, progress=True)
    ids = values["ids"]
    families = values["family_ids"]
    split = values["split"]
    qwen_xz = project_inverse(result["pixels"], episodes)
    static_xz = project_inverse(result["static_pixels"], episodes)
    npwrite(
        OUT / "pooled_evidence.npz",
        {"ids": ids, "family_ids": families, "split": split, "pooled": result["pooled"]},
    )
    npwrite(OUT / "qwen_warp_joint.npz", {"ids": ids, "pixels": result["pixels"], "xz": qwen_xz})
    npwrite(
        OUT / "qwen_warp_joint_static.npz",
        {"ids": ids, "pixels": result["static_pixels"], "xz": static_xz},
    )
    manifest = {
        "status": "sealed streamed frozen Qwen and V14 evidence",
        "plan_sha256": sha(OUT / "PLAN.json"),
        "outputs": {
            name: sha(OUT / name)
            for name in ("pooled_evidence.npz", "qwen_warp_joint.npz", "qwen_warp_joint_static.npz")
        },
        "episodes": 1080,
        "frames": 8640,
        "feature_dim": 32,
        "vision_forwards": 1080,
        "decoder_forwards": 1080,
        "static_additional_forward": False,
        "checkpoint_sha256": sha(CHECKPOINT),
        "base_qwen_checkpoint_sha256": read(
            ROOT / "results/com_observation_v14/candidate_lock_v1/LOCK.json"
        )["frozen_vlm_checkpoint_sha256"],
        "targets_opened": False,
        "labels_opened": False,
        "training": False,
        "intermediate_premerge_saved": False,
        "elapsed_s": result["elapsed_s"],
    }
    write_new(OUT / "MANIFEST.json", manifest)
    print(json.dumps(manifest, indent=2))


def project_inverse(pixels, episodes):
    from main3d_numeric import world_xz

    return np.asarray(
        [[world_xz(point, episode["camera"]) for point in row] for episode, row in zip(episodes, pixels)],
        dtype=np.float64,
    )


if __name__ == "__main__":
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        PYTHONDONTWRITEBYTECODE="1",
    )
    install_target_guard()
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["regression_v11", "extract_v16"])
    arguments = parser.parse_args()
    {"regression_v11": regression_v11, "extract_v16": extract_v16}[arguments.stage]()
