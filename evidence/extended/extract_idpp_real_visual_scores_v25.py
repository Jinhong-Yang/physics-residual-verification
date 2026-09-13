"""Extract frozen V22 scores using the fixed V25 RGB tracking protocol."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import stream_com_internal_features_v16 as stream
from scripts.extract_idpp_real_visual_scores_v24 import overlay_geometry, video_frames
from sequential_robust_readout_v22 import load_checkpoint, load_model, ridge_predict, VISUAL_SHRINKAGE


OUT = ROOT / "results/com_observation_v25/idpp_real_friction_test3_v1"
TARGET = ROOT / "sources/idpp_data/friction_absolute_test3/7.18_friction_test3_rgb_seg_videos_152frames_abs_test/bounce_analysis_results.json"
CHECKPOINT = ROOT / "results/com_observation_v22/final_development_fit_v1/CHECKPOINT.npz"


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def install_target_guard():
    forbidden = TARGET.resolve()
    def guard(event, args):
        if event == "open" and args and not isinstance(args[0], int) and Path(args[0]).resolve() == forbidden:
            raise PermissionError("V25 extractor cannot open IDPP physical labels")
    sys.addaudithook(guard)


def track_centers(rgb_frames, cue_frame):
    _, ellipse = overlay_geometry(rgb_frames[0], cue_frame)
    x0, y0, x1, y1 = ellipse
    width, height = x1 - x0, y1 - y0
    box = (int(x0 + 0.25 * width), int(y0 + 0.20 * height), max(2, int(0.50 * width)), max(2, int(0.60 * height)))
    tracker = cv2.TrackerMIL_create()
    tracker.init(cv2.cvtColor(rgb_frames[0], cv2.COLOR_RGB2BGR), box)
    boxes, centers = [], []
    for frame_index, image in enumerate(rgb_frames):
        if frame_index:
            success, box = tracker.update(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            if not success:
                raise ValueError("TrackerMIL failed")
        boxes.append(np.asarray([box[0], box[1], box[0] + box[2], box[1] + box[3]], dtype=np.float64))
        centers.append(np.asarray([box[0] + box[2] / 2, box[1] + box[3] / 2], dtype=np.float64))
    return np.asarray(centers), np.asarray(boxes)


def main():
    plan = read(OUT / "PLAN.json")
    if (OUT / "PREDICTION_SEAL.json").exists():
        raise FileExistsError("Preserve the first V25 external scores")
    for name, digest in plan["source_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("V25 source changed: " + name)
    if plan["checkpoint_sha256"] != sha(CHECKPOINT):
        raise ValueError("V22 checkpoint changed")
    tracking_indices = plan["frames"]["tracking_indices"]
    model_indices = plan["frames"]["model_indices"]
    lookup = {value: index for index, value in enumerate(tracking_indices)}
    frame_dir = OUT / "observation_frames"
    episodes, anchors, axes = [], [], []
    started = time.perf_counter()
    for record in plan["records"]:
        rgb_path, cue_path = ROOT / record["rgb_path"], ROOT / record["cue_path"]
        if sha(rgb_path) != record["rgb_sha256"] or sha(cue_path) != record["cue_sha256"]:
            raise ValueError("Public video changed: " + record["id"])
        all_rgb = video_frames(rgb_path, tracking_indices)
        cue0 = video_frames(cue_path, [0])[0]
        centers, boxes = track_centers(all_rgb, cue0)
        selected_rgb = [all_rgb[lookup[index]] for index in model_indices]
        selected_centers = centers[[lookup[index] for index in model_indices]]
        selected_boxes = boxes[[lookup[index] for index in model_indices]]
        displacement = selected_centers[-1] - selected_centers[0]
        if np.linalg.norm(displacement) < 2:
            raise ValueError("Tracked motion is too small: " + record["id"])
        axis = displacement / np.linalg.norm(displacement)
        paths, hashes = [], []
        for index, image_array in enumerate(selected_rgb):
            image = Image.fromarray(image_array).resize((280, 280), Image.Resampling.BILINEAR)
            path = frame_dir / record["id"] / f"{index}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                image.save(handle, format="PNG")
            paths.append(path.relative_to(ROOT).as_posix())
            hashes.append(sha(path))
        scale = np.asarray([280 / selected_rgb[0].shape[1], 280 / selected_rgb[0].shape[0]])
        scaled_centers = selected_centers * scale
        scaled_boxes = selected_boxes * np.tile(scale, 2)
        scaled_axis = displacement * scale
        scaled_axis /= np.linalg.norm(scaled_axis)
        anchors.append(scaled_centers)
        axes.append(scaled_axis)
        episodes.append({
            "id": record["id"], "frame_paths": paths, "frame_sha256": hashes,
            "initial_cue": {"bbox_xyxy": scaled_boxes[0].tolist()},
        })
    anchors = np.asarray(anchors, dtype=np.float64)
    axes = np.asarray(axes, dtype=np.float64)
    values = {"pixels": anchors, "times": np.tile(np.asarray(plan["frames"]["timestamps_s"]), (len(episodes), 1))}
    write_new(OUT / "OBSERVATION_SEAL.json", {
        "status": "sealed label-free IDPP RGB observations with deterministic tracking",
        "plan_sha256": sha(OUT / "PLAN.json"), "episodes": len(episodes), "frames": len(episodes) * 8,
        "frame_files_sha256": hashlib.sha256("".join(value for row in episodes for value in row["frame_sha256"]).encode()).hexdigest(),
        "labels_opened": False, "tracking_failures": 0, "excluded_after_observation": 0,
    })
    result = stream.run_rows(episodes, values, stream.load_models(), progress=True)
    dynamic = np.einsum("nti,ni->nt", result["pixels"] - result["static_pixels"], axes)
    static = np.einsum("nti,ni->nt", result["static_pixels"] - anchors, axes)
    motion = np.stack((dynamic, static), axis=2).reshape(len(episodes), 16)
    visual = np.concatenate((result["pooled"].mean(1), motion), axis=1).astype(np.float64)
    checkpoint = load_checkpoint(CHECKPOINT)
    model = load_model(checkpoint, "visual", "generalized_translation", "unforced_slide")
    other = load_model(checkpoint, "visual", "setup_translation", "unforced_slide")
    if not all(np.array_equal(model[key], other[key]) for key in model):
        raise ValueError("V22 slide branches differ")
    pooled_only, motion_only = visual.copy(), visual.copy()
    pooled_only[:, 32:] = model["mean_x"][32:]
    motion_only[:, :32] = model["mean_x"][:32]
    t = np.asarray(plan["frames"]["timestamps_s"])
    position = np.einsum("nti,ni->nt", anchors - anchors[:, :1], axes)
    design = np.column_stack((np.ones_like(t), t, 0.5 * t * t))
    acceleration = np.asarray([np.linalg.lstsq(design, row, rcond=None)[0][2] for row in position])
    score_path = OUT / "VISUAL_SCORES.npz"
    with score_path.open("xb") as handle:
        np.savez_compressed(
            handle,
            ids=np.asarray([row["id"] for row in plan["records"]]),
            family_ids=np.asarray([row["family"] for row in plan["records"]]),
            objects=np.asarray([row["object"] for row in plan["records"]]),
            surfaces=np.asarray([row["surface"] for row in plan["records"]]),
            cameras=np.asarray([row["camera"] for row in plan["records"]]),
            axes=axes, cue_centers=anchors,
            v22_visual_score=VISUAL_SHRINKAGE * ridge_predict(model, visual).reshape(-1),
            v22_pooled_only_score=VISUAL_SHRINKAGE * ridge_predict(model, pooled_only).reshape(-1),
            v22_motion_only_score=VISUAL_SHRINKAGE * ridge_predict(model, motion_only).reshape(-1),
            kinematic_deceleration_score=-acceleration,
        )
    write_new(OUT / "PREDICTION_SEAL.json", {
        "status": "sealed frozen V22 external scores before IDPP label evaluation",
        "plan_sha256": sha(OUT / "PLAN.json"), "observation_seal_sha256": sha(OUT / "OBSERVATION_SEAL.json"),
        "score_path": score_path.relative_to(ROOT).as_posix(), "score_sha256": sha(score_path),
        "checkpoint_sha256": sha(CHECKPOINT), "episodes": len(episodes),
        "labels_opened": False, "training_tuning_or_calibration": False,
        "elapsed_s": time.perf_counter() - started,
    })
    print(json.dumps(read(OUT / "PREDICTION_SEAL.json"), indent=2))


if __name__ == "__main__":
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false", PYTHONDONTWRITEBYTECODE="1")
    install_target_guard()
    stream.install_target_guard()
    main()
