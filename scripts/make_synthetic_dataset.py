#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def draw_clip(path: Path, tier: int, frames: int, fps: int, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video: {path}")

    amplitudes = {1: 3, 2: 28, 3: 70}
    amp = amplitudes[tier]
    center_y = size // 2
    for i in range(frames):
        frame = np.zeros((size, size, 3), dtype=np.uint8)
        progress = i / max(frames - 1, 1)
        x = int(size * 0.15 + progress * amp)
        if tier == 3:
            y = int(center_y + np.sin(progress * np.pi * 4) * 30)
        elif tier == 2:
            y = int(center_y + np.sin(progress * np.pi * 2) * 14)
        else:
            y = center_y
        cv2.rectangle(frame, (x, y - 35), (x + 35, y + 35), (230, 230, 230), -1)
        cv2.circle(frame, (x + 18, y - 48), 14, (210, 210, 210), -1)
        writer.write(frame)
    writer.release()


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a tiny synthetic dataset for smoke tests.")
    parser.add_argument("--output-dir", default="data/synthetic")
    parser.add_argument("--labels", default="data/synthetic_labels.csv")
    parser.add_argument("--clips-per-tier", type=int, default=3)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--size", type=int, default=160)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    labels_path = Path(args.labels)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    class_names = {1: "synthetic low motion", 2: "synthetic medium motion", 3: "synthetic high motion"}
    for tier in (1, 2, 3):
        for idx in range(args.clips_per_tier):
            path = output_dir / f"tier{tier}" / f"clip_{idx:03d}.mp4"
            draw_clip(path, tier, args.frames, args.fps, args.size)
            rows.append(
                {
                    "path": str(path),
                    "youtube_id": f"synthetic_{tier}_{idx}",
                    "class": class_names[tier],
                    "tier": tier,
                    "time_start": 0,
                    "time_end": args.frames / args.fps,
                    "human_rating": tier * 2 - 1,
                }
            )

    with labels_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} synthetic clips to {output_dir}")
    print(f"wrote labels to {labels_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
