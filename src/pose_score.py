from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.common import ensure_parent, frame_diagonal, iter_sampled_frames, preload_onnx_cuda_libs, read_labels


@dataclass
class PoseScoreConfig:
    pipeline: str = "body"
    rtmlib_mode: str = "balanced"
    target_fps: float = 8.0
    resize_long_side: int = 720
    max_frames: int | None = None
    device: str = "cuda"
    keypoint_confidence: float = 0.25
    min_visible_joints: int = 4
    association_iou: float = 0.1
    aggregation: str = "mean_top2"


@dataclass
class PersonPose:
    keypoints: np.ndarray
    scores: np.ndarray
    box: np.ndarray


def bbox_from_keypoints(keypoints: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
    valid = scores >= threshold
    if not np.any(valid):
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    xy = keypoints[valid, :2]
    x1, y1 = np.min(xy, axis=0)
    x2, y2 = np.max(xy, axis=0)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter = max(inter_x2 - inter_x1, 0.0) * max(inter_y2 - inter_y1, 0.0)
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def resolve_onnx_device(device: str) -> str:
    if device.lower() != "cuda":
        return device
    preload_onnx_cuda_libs()
    try:
        import onnxruntime as ort  # noqa: PLC0415

        if "CUDAExecutionProvider" in ort.get_available_providers():
            return "cuda"
    except Exception as exc:
        print(f"warning: onnxruntime GPU init failed ({exc}), using CPU for rtmlib", file=sys.stderr)
        return "cpu"
    print("warning: onnxruntime CUDA provider unavailable, using CPU for rtmlib", file=sys.stderr)
    return "cpu"


class RtmPoseBackend:
    def __init__(self, config: PoseScoreConfig) -> None:
        self.config = config
        try:
            from rtmlib import Body  # type: ignore  # noqa: PLC0415
        except Exception as exc:
            raise RuntimeError("rtmlib is required for pose scoring. Install requirements.txt first.") from exc

        device = resolve_onnx_device(config.device)
        kwargs: dict[str, Any] = {
            "mode": config.rtmlib_mode,
            "to_openpose": False,
            "backend": "onnxruntime",
            "device": device,
        }
        if config.pipeline == "rtmo":
            # Body switches to one-stage RTMO when the pose model name contains "rtmo".
            kwargs["pose"] = Body.RTMO_MODE[config.rtmlib_mode]["pose"]
            kwargs["pose_input_size"] = Body.RTMO_MODE[config.rtmlib_mode]["pose_input_size"]

        self.model = Body(**kwargs)

    def infer(self, frame_bgr: np.ndarray) -> list[PersonPose]:
        result = self.model(frame_bgr)
        if isinstance(result, tuple):
            keypoints, scores = result[:2]
        elif isinstance(result, dict):
            keypoints = result.get("keypoints", result.get("keypoint"))
            scores = result.get("scores", result.get("keypoint_scores"))
        else:
            raise RuntimeError(f"Unsupported rtmlib output type: {type(result)!r}")

        keypoints = np.asarray(keypoints, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        if keypoints.size == 0:
            return []
        if keypoints.ndim == 2:
            keypoints = keypoints[None]
        if scores.ndim == 1:
            scores = scores[None]

        people = []
        for kpts, conf in zip(keypoints, scores):
            if int(np.sum(conf >= self.config.keypoint_confidence)) < self.config.min_visible_joints:
                continue
            box = bbox_from_keypoints(kpts, conf, self.config.keypoint_confidence)
            people.append(PersonPose(kpts[:, :2], conf, box))
        return people


def match_people(prev: list[PersonPose], curr: list[PersonPose], min_iou: float) -> list[tuple[int, int]]:
    pairs = []
    used_prev: set[int] = set()
    used_curr: set[int] = set()
    candidates = []
    for i, p in enumerate(prev):
        for j, c in enumerate(curr):
            candidates.append((box_iou(p.box, c.box), i, j))
    for iou, i, j in sorted(candidates, reverse=True):
        if iou < min_iou or i in used_prev or j in used_curr:
            continue
        used_prev.add(i)
        used_curr.add(j)
        pairs.append((i, j))
    return pairs


def person_motion(prev: PersonPose, curr: PersonPose, config: PoseScoreConfig, fallback_scale: float) -> float | None:
    conf = np.minimum(prev.scores, curr.scores)
    valid = conf >= config.keypoint_confidence
    if int(np.sum(valid)) < config.min_visible_joints:
        return None
    disp = np.linalg.norm(curr.keypoints[valid] - prev.keypoints[valid], axis=1)
    weights = conf[valid]
    box = np.maximum(prev.box, curr.box)
    scale = float(np.hypot(box[2] - box[0], box[3] - box[1]))
    scale = max(scale, fallback_scale * 0.05, 1.0)
    return float(np.average(disp / scale, weights=weights))


def aggregate_pair_scores(scores: list[float], mode: str) -> float:
    if not scores:
        return 0.0
    arr = np.asarray(scores, dtype=np.float32)
    if mode == "max":
        return float(np.max(arr))
    if mode == "mean":
        return float(np.mean(arr))
    if mode == "mean_top2":
        return float(np.mean(np.sort(arr)[-2:]))
    raise ValueError(f"Unknown aggregation mode: {mode}")


def score_video(
    path: str | Path,
    config: PoseScoreConfig | None = None,
    backend: RtmPoseBackend | None = None,
    return_tracks: bool = False,
) -> dict[str, Any]:
    cfg = config or PoseScoreConfig()
    pose_backend = backend or RtmPoseBackend(cfg)
    frames = list(
        iter_sampled_frames(
            path,
            target_fps=cfg.target_fps,
            max_frames=cfg.max_frames,
            resize_long_side=cfg.resize_long_side,
        )
    )
    if not frames:
        return {"pose_motion": 0.0, "pose_pairs": 0.0, "pose_people_mean": 0.0}

    all_people = [pose_backend.infer(frame) for frame in frames]
    pair_scores = []
    people_counts = []
    fallback_scale = frame_diagonal(frames[0])
    for prev, curr in zip(all_people[:-1], all_people[1:]):
        people_counts.append(len(curr))
        motions = []
        for i, j in match_people(prev, curr, cfg.association_iou):
            motion = person_motion(prev[i], curr[j], cfg, fallback_scale)
            if motion is not None:
                motions.append(motion)
        pair_scores.append(aggregate_pair_scores(motions, cfg.aggregation))

    boxes = [[[float(x) for x in person.box] for person in people] for people in all_people]
    out: dict[str, Any] = {
        "pose_motion": float(np.mean(pair_scores)) if pair_scores else 0.0,
        "pose_pairs": float(len(pair_scores)),
        "pose_people_mean": float(np.mean(people_counts)) if people_counts else float(len(all_people[0])),
    }
    if return_tracks:
        out["boxes"] = boxes
    return out


def score_labels(labels_csv: str | Path, output_csv: str | Path, boxes_json: str | Path | None, config: PoseScoreConfig) -> pd.DataFrame:
    labels = read_labels(labels_csv)
    backend = RtmPoseBackend(config)
    rows = []
    boxes_by_path = {}
    for row in tqdm(labels.to_dict("records"), desc="pose"):
        scores = score_video(row["path"], config=config, backend=backend, return_tracks=boxes_json is not None)
        boxes = scores.pop("boxes", None)
        if boxes is not None:
            boxes_by_path[row["path"]] = boxes
            boxes_by_path[str(Path(row["path"]).resolve())] = boxes
        rows.append({**row, **scores})

    out = pd.DataFrame(rows)
    ensure_parent(output_csv)
    out.to_csv(output_csv, index=False)
    if boxes_json:
        ensure_parent(boxes_json)
        Path(boxes_json).write_text(json.dumps(boxes_by_path), encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute human-keypoint motion magnitude scores.")
    parser.add_argument("--labels", default="data/labels.csv")
    parser.add_argument("--output", default="results/pose_scores.csv")
    parser.add_argument("--boxes-json", default="results/pose_boxes.json")
    parser.add_argument("--target-fps", type=float, default=8.0)
    parser.add_argument("--resize-long-side", type=int, default=720)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pipeline", default="body", choices=["body", "rtmo"], help="body=YOLOX+RTMPose (default), rtmo=one-stage RTMO")
    parser.add_argument("--rtmlib-mode", default="balanced", choices=["lightweight", "balanced", "performance"])
    parser.add_argument("--keypoint-confidence", type=float, default=0.25)
    parser.add_argument("--association-iou", type=float, default=0.1)
    parser.add_argument("--aggregation", default="mean_top2", choices=["mean", "max", "mean_top2"])
    args = parser.parse_args()
    cfg = PoseScoreConfig(
        pipeline=args.pipeline,
        rtmlib_mode=args.rtmlib_mode,
        target_fps=args.target_fps,
        resize_long_side=args.resize_long_side,
        max_frames=args.max_frames,
        device=args.device,
        keypoint_confidence=args.keypoint_confidence,
        association_iou=args.association_iou,
        aggregation=args.aggregation,
    )
    score_labels(args.labels, args.output, args.boxes_json, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
