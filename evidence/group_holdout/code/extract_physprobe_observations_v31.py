"""Create label-free eight-frame PhysProbe observations from the sealed videos."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/physprobe_push_v31"
PLAN = DATA / "PLAN.json"
DOWNLOAD = DATA / "DOWNLOAD_SEAL.json"
OUT = DATA / "observations_v1"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def install_label_guard():
    def guard(event, args):
        if event != "open" or not args or isinstance(args[0], int):
            return
        path = Path(args[0])
        if path.suffix.lower() == ".parquet" or "evaluator_only" in {part.lower() for part in path.parts}:
            raise PermissionError("V31 observation extractor cannot access labels or trajectories")

    sys.addaudithook(guard)


def red_component(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 80, 80), (12, 255, 255))
    mask |= cv2.inRange(hsv, (170, 80, 80), (179, 255, 255))
    count, _, stats, centers = cv2.connectedComponentsWithStats(mask)
    if count <= 1:
        raise ValueError("no red component")
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, width, height, area = stats[index]
    if area < 25:
        raise ValueError("red component too small")
    return centers[index], np.asarray([x, y, x + width, y + height], dtype=np.float64), int(area)


def read_video(path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened() or abs(capture.get(cv2.CAP_PROP_FPS) - 50.0) > 1e-3:
        raise ValueError("video/fps mismatch")
    frames = []
    while True:
        success, frame = capture.read()
        if not success:
            break
        frames.append(frame)
    capture.release()
    if len(frames) < 120:
        raise ValueError("video is too short")
    return frames


def locate(frames, offsets, target_offset):
    centers, boxes, areas = [], [], []
    for frame in frames:
        center, box, area = red_component(frame)
        centers.append(center)
        boxes.append(box)
        areas.append(area)
    centers = np.asarray(centers)
    boxes = np.asarray(boxes)
    baseline = np.median(centers[:25], axis=0)
    displacement = np.linalg.norm(centers - baseline, axis=1)
    onset = next((i for i in range(20, len(frames) - 5)
                  if np.all(displacement[i:i + 5] > 2.0)), None)
    if onset is None or onset + min(offsets) < 0 or onset + target_offset >= len(frames):
        raise ValueError("valid onset/target horizon unavailable")
    indices = np.asarray([onset + value for value in offsets], dtype=int)
    selected = centers[indices]
    direction = selected[-1] - selected[0]
    if np.linalg.norm(direction) < 2.0:
        raise ValueError("insufficient observed red-object displacement")
    return onset, indices, centers, boxes, np.asarray(areas), direction / np.linalg.norm(direction)


def main():
    if OUT.exists():
        raise FileExistsError("Preserve the first V31 observation extraction")
    plan, download = read_json(PLAN), read_json(DOWNLOAD)
    if download["plan_sha256"] != sha256(PLAN) or download["target_metadata_parsed"]:
        raise ValueError("V31 download seal is invalid")
    file_index = {(row["role"], row["episode_index"], row["kind"]): row for row in download["files"]}
    offsets = plan["observation_protocol"]["frame_offsets_from_onset"]
    target_offset = plan["observation_protocol"]["future_target_offset_from_onset"]
    frame_root = OUT / "frames"
    episodes, pixels, times, along, perpendicular, sizes = [], [], [], [], [], []
    excluded = []
    for count, record in enumerate(plan["records"], 1):
        key = (record["role"], record["episode_index"], "video")
        source = ROOT / file_index[key]["path"]
        if sha256(source) != file_index[key]["sha256"]:
            raise ValueError("Downloaded public video changed")
        try:
            frames = read_video(source)
            onset, indices, centers, boxes, areas, axis = locate(frames, offsets, target_offset)
        except ValueError as error:
            excluded.append({"role": record["role"], "episode_index": record["episode_index"],
                             "reason": str(error)})
            continue
        scale = np.asarray([280 / frames[0].shape[1], 280 / frames[0].shape[0]])
        selected_centers = centers[indices] * scale
        selected_boxes = boxes[indices] * np.tile(scale, 2)
        scaled_axis = axis * scale
        scaled_axis /= np.linalg.norm(scaled_axis)
        centered = selected_centers - selected_centers[0]
        normal = np.asarray([-scaled_axis[1], scaled_axis[0]])
        episode_name = f"episode_{record['episode_index']:06d}"
        paths, hashes = [], []
        for ordinal, frame_index in enumerate(indices):
            image = Image.fromarray(cv2.cvtColor(frames[int(frame_index)], cv2.COLOR_BGR2RGB))
            image = image.resize((280, 280), Image.Resampling.BILINEAR)
            path = frame_root / record["role"] / episode_name / f"{ordinal}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                image.save(stream, format="PNG")
            paths.append(path.relative_to(ROOT).as_posix())
            hashes.append(sha256(path))
        episodes.append(
            {
                "id": episode_name,
                "episode_index": record["episode_index"],
                "role": record["role"],
                "onset_index": onset,
                "observation_indices": indices.tolist(),
                "target_index": onset + target_offset,
                "frame_paths": paths,
                "frame_sha256": hashes,
                "initial_cue": {"bbox_xyxy": selected_boxes[0].tolist()},
            }
        )
        pixels.append(selected_centers)
        times.append(np.asarray(plan["observation_protocol"]["frame_times_s"]))
        along.append(centered @ scaled_axis)
        perpendicular.append(centered @ normal)
        widths = selected_boxes[:, 2] - selected_boxes[:, 0]
        heights = selected_boxes[:, 3] - selected_boxes[:, 1]
        sizes.append(np.sqrt(np.maximum(widths * heights, 1.0)))
        if count == 1 or count % 32 == 0 or count == len(plan["records"]):
            print(json.dumps({"processed": count, "total": len(plan["records"]),
                              "eligible": len(episodes), "excluded": len(excluded)}), flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    index_path = OUT / "INPUT_INDEX.json"
    index_path.write_text(json.dumps({"episodes": episodes, "excluded": excluded}, indent=2, allow_nan=False), encoding="utf-8")
    input_path = OUT / "inputs.npz"
    with input_path.open("xb") as stream:
        np.savez_compressed(
            stream,
            ids=np.asarray([row["id"] for row in episodes]),
            episode_indices=np.asarray([row["episode_index"] for row in episodes]),
            roles=np.asarray([row["role"] for row in episodes]),
            pixels=np.asarray(pixels),
            times=np.asarray(times),
            tracked_along=np.asarray(along),
            tracked_perpendicular=np.asarray(perpendicular),
            object_size_px=np.asarray(sizes),
        )
    role_counts = {role: sum(row["role"] == role for row in episodes) for role in plan["roles"]}
    seal = {
        "status": "sealed label-free PhysProbe observations",
        "plan_sha256": sha256(PLAN),
        "download_seal_sha256": sha256(DOWNLOAD),
        "input_index_sha256": sha256(index_path),
        "inputs_sha256": sha256(input_path),
        "role_counts": role_counts,
        "eligible": len(episodes),
        "excluded": len(excluded),
        "labels_or_trajectories_read": False,
        "source_sha256": sha256(Path(__file__)),
    }
    (OUT / "OBSERVATION_SEAL.json").write_text(json.dumps(seal, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(seal, indent=2))


if __name__ == "__main__":
    install_label_guard()
    main()
