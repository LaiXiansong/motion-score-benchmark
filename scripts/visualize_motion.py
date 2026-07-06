#!/usr/bin/env python3
"""Visualize per-frame motion scores for flow, pose, and learned methods."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import ensure_parent, frame_diagonal, minmax01, resize_by_long_side, top_percent_mean, video_meta
from src.flow_score import FlowScoreConfig, build_backend, mask_from_boxes
from src.learned_score import LearnedScoreConfig, load_motion_model, tier_probability_score
from src.pose_score import (
    PoseScoreConfig,
    RtmPoseBackend,
    aggregate_pair_scores,
    match_people,
    person_motion,
)

COCO_SKELETON = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)


@dataclass
class FrameSample:
    time_sec: float
    frame_bgr: np.ndarray


@dataclass
class TimelinePoint:
    time_sec: float
    score: float
    frame_bgr: np.ndarray
    extra: dict[str, Any]


def load_sampled_frames(
    path: str | Path,
    target_fps: float = 8.0,
    resize_long_side: int = 720,
    max_frames: int | None = None,
) -> list[FrameSample]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or target_fps or 1.0)
    stride = max(int(round(src_fps / target_fps)), 1) if target_fps > 0 else 1
    frame_idx = 0
    yielded = 0
    samples: list[FrameSample] = []

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if frame_idx % stride == 0:
            if resize_long_side:
                frame_bgr = resize_by_long_side(frame_bgr, resize_long_side)
            samples.append(FrameSample(time_sec=frame_idx / src_fps, frame_bgr=frame_bgr.copy()))
            yielded += 1
            if max_frames is not None and yielded >= max_frames:
                break
        frame_idx += 1

    cap.release()
    if not samples:
        raise ValueError(f"No frames read from video: {path}")
    return samples


def flow_to_color(flow: np.ndarray) -> np.ndarray:
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros(flow.shape[:2] + (3,), dtype=np.uint8)
    hsv[..., 0] = (ang * 180.0 / np.pi / 2.0).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def blend_flow_overlay(frame_bgr: np.ndarray, flow_color: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    return cv2.addWeighted(frame_bgr, 1.0 - alpha, flow_color, alpha, 0.0)


def draw_pose(frame_bgr: np.ndarray, people: list[Any], config: PoseScoreConfig) -> np.ndarray:
    out = frame_bgr.copy()
    palette = [
        (255, 128, 0),
        (0, 255, 128),
        (128, 0, 255),
        (0, 128, 255),
    ]
    for person_idx, person in enumerate(people):
        color = palette[person_idx % len(palette)]
        kpts = person.keypoints
        scores = person.scores
        for i, j in COCO_SKELETON:
            if scores[i] < config.keypoint_confidence or scores[j] < config.keypoint_confidence:
                continue
            p1 = tuple(np.round(kpts[i]).astype(int))
            p2 = tuple(np.round(kpts[j]).astype(int))
            cv2.line(out, p1, p2, color, 2, cv2.LINE_AA)
        for idx, (x, y) in enumerate(kpts):
            if scores[idx] < config.keypoint_confidence:
                continue
            cv2.circle(out, (int(round(x)), int(round(y))), 4, color, -1, cv2.LINE_AA)
    return out


def put_text_block(frame: np.ndarray, lines: list[str], origin: tuple[int, int] = (16, 28)) -> np.ndarray:
    out = frame.copy()
    x, y = origin
    for line in lines:
        cv2.putText(out, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(out, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 1, cv2.LINE_AA)
        y += 28
    return out


def stack_horizontal(left: np.ndarray, right: np.ndarray, gap: int = 8) -> np.ndarray:
    h = max(left.shape[0], right.shape[0])
    if left.shape[0] != h:
        left = cv2.resize(left, (int(left.shape[1] * h / left.shape[0]), h))
    if right.shape[0] != h:
        right = cv2.resize(right, (int(right.shape[1] * h / right.shape[0]), h))
    divider = np.full((h, gap, 3), 32, dtype=np.uint8)
    return np.concatenate([left, divider, right], axis=1)


def preprocess_learned_frames(frames_rgb: list[np.ndarray], image_size: int) -> torch.Tensor:
    processed = []
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    for frame in frames_rgb:
        resized = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
        arr = resized.astype(np.float32) / 255.0
        arr = (arr - mean) / std
        processed.append(arr.transpose(2, 0, 1))
    return torch.from_numpy(np.stack(processed)).float()


def compute_flow_timeline(
    samples: list[FrameSample],
    flow_cfg: FlowScoreConfig,
    person_boxes: list[list[list[float]]] | None = None,
) -> list[TimelinePoint]:
    backend = build_backend(flow_cfg)
    timeline: list[TimelinePoint] = []
    timeline.append(
        TimelinePoint(
            time_sec=samples[0].time_sec,
            score=float("nan"),
            frame_bgr=samples[0].frame_bgr,
            extra={"flow_color": np.zeros_like(samples[0].frame_bgr)},
        )
    )

    for i in range(1, len(samples)):
        prev = samples[i - 1].frame_bgr
        curr = samples[i].frame_bgr
        flow = backend.estimate(prev, curr)
        mag = np.linalg.norm(flow, axis=-1) / max(frame_diagonal(curr), 1.0)
        score = top_percent_mean(mag, flow_cfg.top_percent)

        boxes = person_boxes[i - 1] if person_boxes and i - 1 < len(person_boxes) else None
        mask = mask_from_boxes(mag.shape, boxes)
        if mask is not None and np.any(mask):
            score = top_percent_mean(mag[mask], flow_cfg.top_percent)

        flow_color = flow_to_color(flow)
        timeline.append(
            TimelinePoint(
                time_sec=samples[i].time_sec,
                score=float(score),
                frame_bgr=curr,
                extra={"flow_color": flow_color, "flow_overlay": blend_flow_overlay(curr, flow_color)},
            )
        )
    return timeline


def compute_pose_timeline(samples: list[FrameSample], pose_cfg: PoseScoreConfig) -> list[TimelinePoint]:
    backend = RtmPoseBackend(pose_cfg)
    all_people = [backend.infer(sample.frame_bgr) for sample in tqdm(samples, desc="pose infer", leave=False)]
    fallback_scale = frame_diagonal(samples[0].frame_bgr)
    timeline: list[TimelinePoint] = []

    timeline.append(
        TimelinePoint(
            time_sec=samples[0].time_sec,
            score=float("nan"),
            frame_bgr=draw_pose(samples[0].frame_bgr, all_people[0], pose_cfg),
            extra={"people_count": len(all_people[0])},
        )
    )

    for i in range(1, len(samples)):
        prev_people = all_people[i - 1]
        curr_people = all_people[i]
        motions = []
        for pi, pj in match_people(prev_people, curr_people, pose_cfg.association_iou):
            motion = person_motion(prev_people[pi], curr_people[pj], pose_cfg, fallback_scale)
            if motion is not None:
                motions.append(motion)
        score = aggregate_pair_scores(motions, pose_cfg.aggregation)
        timeline.append(
            TimelinePoint(
                time_sec=samples[i].time_sec,
                score=float(score),
                frame_bgr=draw_pose(samples[i].frame_bgr, curr_people, pose_cfg),
                extra={"people_count": len(curr_people), "tracked_motions": len(motions)},
            )
        )
    return timeline


def compute_learned_timeline(
    samples: list[FrameSample],
    learned_cfg: LearnedScoreConfig,
    stride: int = 1,
) -> list[TimelinePoint]:
    model = load_motion_model(learned_cfg)
    device = learned_cfg.device
    timeline: list[TimelinePoint] = []
    indices = list(range(0, len(samples), max(stride, 1)))
    last_score = float("nan")

    for idx in tqdm(indices, desc="learned windows", leave=False):
        end = idx + 1
        window = samples[:end]
        pick = np.linspace(0, len(window) - 1, learned_cfg.num_frames).round().astype(int)
        frames_rgb = [window[i].frame_bgr[..., ::-1] for i in pick]
        batch = preprocess_learned_frames(frames_rgb, learned_cfg.image_size).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(batch)
            prob = torch.softmax(out["logits"], dim=-1)[0].detach().cpu().numpy()
            score = tier_probability_score(prob)
            motion = float(torch.clamp(out["motion"][0], 0.0, 1.0).detach().cpu())
        last_score = score
        timeline.append(
            TimelinePoint(
                time_sec=samples[idx].time_sec,
                score=float(score),
                frame_bgr=samples[idx].frame_bgr,
                extra={
                    "learned_motion": motion,
                    "tier_pred": int(np.argmax(prob)) + 1,
                    "prob_low": float(prob[0]),
                    "prob_medium": float(prob[1]),
                    "prob_high": float(prob[2]),
                },
            )
        )

    if stride > 1:
        dense: list[TimelinePoint] = []
        sparse_idx = 0
        for i, sample in enumerate(samples):
            while sparse_idx + 1 < len(timeline) and timeline[sparse_idx + 1].time_sec <= sample.time_sec:
                sparse_idx += 1
            score = timeline[sparse_idx].score if np.isfinite(timeline[sparse_idx].score) else last_score
            dense.append(
                TimelinePoint(
                    time_sec=sample.time_sec,
                    score=float(score),
                    frame_bgr=sample.frame_bgr,
                    extra=timeline[sparse_idx].extra,
                )
            )
        return dense

    if len(timeline) < len(samples):
        filled = [
            TimelinePoint(
                time_sec=samples[0].time_sec,
                score=float("nan"),
                frame_bgr=samples[0].frame_bgr,
                extra={},
            )
        ]
        by_time = {point.time_sec: point for point in timeline}
        for sample in samples[1:]:
            point = by_time.get(sample.time_sec)
            if point is None:
                point = TimelinePoint(time_sec=sample.time_sec, score=last_score, frame_bgr=sample.frame_bgr, extra={})
            filled.append(point)
        return filled
    return timeline


def render_flow_frame(point: TimelinePoint) -> np.ndarray:
    original = point.frame_bgr
    flow_color = point.extra.get("flow_color", np.zeros_like(original))
    overlay = point.extra.get("flow_overlay", blend_flow_overlay(original, flow_color))
    left = put_text_block(original, ["Original"])
    right = put_text_block(overlay, ["Optical Flow Overlay"])
    panel = stack_horizontal(left, right)
    score_text = "N/A" if not np.isfinite(point.score) else f"{point.score:.4f}"
    return put_text_block(panel, [f"Flow score (top-5%): {score_text}", f"t = {point.time_sec:.2f}s"], origin=(16, panel.shape[0] - 56))


def render_pose_frame(point: TimelinePoint) -> np.ndarray:
    lines = [
        f"Pose score: {'N/A' if not np.isfinite(point.score) else f'{point.score:.4f}'}",
        f"People: {point.extra.get('people_count', 0)}",
        f"t = {point.time_sec:.2f}s",
    ]
    return put_text_block(point.frame_bgr, lines)


def render_learned_frame(point: TimelinePoint) -> np.ndarray:
    tier_pred = point.extra.get("tier_pred")
    motion = point.extra.get("learned_motion")
    motion_text = f"{motion:.4f}" if isinstance(motion, (int, float)) and np.isfinite(motion) else "N/A"
    lines = [
        f"Learned score: {'N/A' if not np.isfinite(point.score) else f'{point.score:.4f}'}",
        f"Regression: {motion_text}",
        f"Tier pred: {tier_pred}" if tier_pred is not None else "Tier pred: N/A",
        f"t = {point.time_sec:.2f}s",
    ]
    return put_text_block(point.frame_bgr, lines)


def play_timeline(
    timeline: list[TimelinePoint],
    render_fn,
    window_name: str,
    display: bool,
    save_video: Path | None,
    wait_ms: int,
) -> None:
    writer = None
    if save_video is not None:
        ensure_parent(save_video)

    for point in timeline:
        frame = render_fn(point)
        if save_video is not None:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(save_video),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    8.0,
                    (w, h),
                )
            writer.write(frame)
        if display:
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(wait_ms) & 0xFF
            if key in (27, ord("q")):
                break

    if writer is not None:
        writer.release()
    if display:
        cv2.destroyWindow(window_name)


def smooth_scores(values: list[float] | np.ndarray, window: int = 5) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or window <= 1:
        return arr
    window = min(int(window), arr.size)
    if window <= 1:
        return arr
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(arr, kernel, mode="same")


def plot_timelines(
    series: dict[str, list[TimelinePoint]],
    output_path: Path,
    title: str,
    smooth_window: int = 5,
) -> None:
    ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=140)
    colors = {"flow": "#1f77b4", "pose": "#ff7f0e", "learned": "#2ca02c"}

    for name, timeline in series.items():
        times = [p.time_sec for p in timeline if np.isfinite(p.score)]
        scores = [p.score for p in timeline if np.isfinite(p.score)]
        if not times:
            continue
        smoothed = smooth_scores(scores, window=smooth_window)
        norm_scores = minmax01(smoothed)
        ax.plot(
            times,
            norm_scores,
            label=name,
            color=colors.get(name, None),
            linewidth=2,
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized motion score (0-1)")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize motion scores over time for one video.")
    parser.add_argument("--video", required=True, help="Path to input video.")
    parser.add_argument("--method", default="all", choices=["flow", "pose", "learned", "all"])
    parser.add_argument("--output-dir", default="results/visualizations")
    parser.add_argument("--target-fps", type=float, default=8.0)
    parser.add_argument("--resize-long-side", type=int, default=720)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--display", action="store_true", help="Show OpenCV windows while rendering.")
    parser.add_argument("--save-video", action="store_true", help="Write annotated preview MP4 files.")
    parser.add_argument("--wait-ms", type=int, default=80, help="Delay between frames when displaying.")
    parser.add_argument("--flow-backend", default="auto", choices=["auto", "sea-raft", "searaft", "farneback"])
    parser.add_argument("--sea-raft-root", default=None)
    parser.add_argument("--sea-raft-checkpoint", default=None)
    parser.add_argument("--sea-raft-cfg", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pose-pipeline", default="body", choices=["body", "rtmo"])
    parser.add_argument("--rtmlib-mode", default="balanced", choices=["lightweight", "balanced", "performance"])
    parser.add_argument("--boxes-json", default=None, help="Optional pose boxes JSON for foreground flow masking.")
    parser.add_argument("--checkpoint", default="checkpoints/learned_motion.pt")
    parser.add_argument("--model-name", default="MCG-NJU/videomae-base-finetuned-kinetics")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learned-stride", type=int, default=2, help="Run learned inference every N sampled frames.")
    parser.add_argument("--smooth-window", type=int, default=5, help="Moving-average window for timeline plot (1=off).")
    return parser.parse_args()


def load_boxes_for_video(boxes_json: str | None, video_path: str) -> list[list[list[float]]] | None:
    if not boxes_json:
        return None
    import json

    payload = json.loads(Path(boxes_json).read_text(encoding="utf-8"))
    boxes = payload.get(video_path) or payload.get(str(Path(video_path).resolve()))
    return boxes


def video_output_dir(video_path: Path, output_dir: Path) -> Path:
    return output_dir / video_path.stem


def main() -> int:
    args = parse_args()
    video_path = str(Path(args.video).resolve())
    meta = video_meta(video_path)
    stem = Path(video_path).stem
    out_dir = Path(args.output_dir)
    video_out_dir = video_output_dir(Path(video_path), out_dir)
    video_out_dir.mkdir(parents=True, exist_ok=True)

    samples = load_sampled_frames(
        video_path,
        target_fps=args.target_fps,
        resize_long_side=args.resize_long_side,
        max_frames=args.max_frames,
    )

    methods = ["flow", "pose", "learned"] if args.method == "all" else [args.method]
    timelines: dict[str, list[TimelinePoint]] = {}

    if "flow" in methods:
        flow_cfg = FlowScoreConfig(
            backend=args.flow_backend,
            target_fps=args.target_fps,
            resize_long_side=args.resize_long_side,
            sea_raft_root=args.sea_raft_root,
            sea_raft_checkpoint=args.sea_raft_checkpoint,
            sea_raft_cfg=args.sea_raft_cfg,
            device=args.device,
        )
        boxes = load_boxes_for_video(args.boxes_json, video_path)
        timelines["flow"] = compute_flow_timeline(samples, flow_cfg, person_boxes=boxes)

    if "pose" in methods:
        pose_cfg = PoseScoreConfig(
            pipeline=args.pose_pipeline,
            rtmlib_mode=args.rtmlib_mode,
            target_fps=args.target_fps,
            resize_long_side=args.resize_long_side,
            device=args.device,
        )
        timelines["pose"] = compute_pose_timeline(samples, pose_cfg)

    if "learned" in methods:
        learned_cfg = LearnedScoreConfig(
            model_name=args.model_name,
            checkpoint=args.checkpoint,
            num_frames=args.num_frames,
            image_size=args.image_size,
            device=args.device,
        )
        timelines["learned"] = compute_learned_timeline(samples, learned_cfg, stride=args.learned_stride)

    plot_path = video_out_dir / f"{stem}_timeline.png"
    plot_timelines(timelines, plot_path, title=f"Motion score vs time — {stem}", smooth_window=args.smooth_window)

    renderers = {
        "flow": render_flow_frame,
        "pose": render_pose_frame,
        "learned": render_learned_frame,
    }

    for name, timeline in timelines.items():
        save_video = video_out_dir / f"{stem}_{name}_preview.mp4" if args.save_video else None
        play_timeline(
            timeline,
            renderers[name],
            window_name=f"motion-{name}",
            display=args.display,
            save_video=save_video,
            wait_ms=args.wait_ms,
        )
        csv_path = video_out_dir / f"{stem}_{name}_timeline.csv"
        rows = []
        for point in timeline:
            row = {"time_sec": point.time_sec, "score": point.score}
            row.update({f"extra_{k}": v for k, v in point.extra.items() if isinstance(v, (int, float, str))})
            rows.append(row)
        if rows:
            import pandas as pd

            pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"Saved timeline plot: {plot_path}")
    for name in timelines:
        print(f"Saved {name} timeline CSV: {video_out_dir / f'{stem}_{name}_timeline.csv'}")
        if args.save_video:
            print(f"Saved {name} preview video: {video_out_dir / f'{stem}_{name}_preview.mp4'}")
    print(f"Video: {video_path} ({meta.duration:.1f}s, sampled {len(samples)} frames @ {args.target_fps} fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
