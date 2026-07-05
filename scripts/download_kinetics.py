#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import load_config  # noqa: E402


YOUTUBE_URL = "https://www.youtube.com/watch?v={video_id}"


def normalize_label(label: str) -> str:
    return re.sub(r"\s+", " ", str(label).strip().lower())


def class_to_tier(config: dict) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for tier_id, tier_cfg in config["tiers"].items():
        for cls in tier_cfg["classes"]:
            mapping[normalize_label(cls)] = int(tier_id)
    return mapping


def load_annotations(path_or_url: str) -> pd.DataFrame:
    df = pd.read_csv(path_or_url)
    rename = {}
    for col in df.columns:
        key = col.strip().lower()
        if key in {"youtube_id", "youtubeid", "ytid"}:
            rename[col] = "youtube_id"
        elif key in {"time_start", "start", "start_time"}:
            rename[col] = "time_start"
        elif key in {"time_end", "end", "end_time"}:
            rename[col] = "time_end"
        elif key in {"label", "class", "class_name"}:
            rename[col] = "label"
    df = df.rename(columns=rename)
    required = {"youtube_id", "time_start", "time_end", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Annotation CSV missing columns: {sorted(missing)}")
    df["label_norm"] = df["label"].map(normalize_label)
    return df


def select_subset(df: pd.DataFrame, config: dict, clips_per_class: int, seed: int) -> pd.DataFrame:
    tier_by_class = class_to_tier(config)
    wanted = set(tier_by_class)
    selected = df[df["label_norm"].isin(wanted)].copy()
    if selected.empty:
        raise ValueError("No configured classes were found in the annotation CSV.")

    rng = random.Random(seed)
    rows = []
    for cls in sorted(wanted):
        cls_df = selected[selected["label_norm"] == cls].copy()
        if cls_df.empty:
            print(f"warning: class not found in annotations: {cls}", file=sys.stderr)
            continue
        indices = list(cls_df.index)
        rng.shuffle(indices)
        rows.append(cls_df.loc[indices[:clips_per_class]])

    out = pd.concat(rows, ignore_index=True)
    out["tier"] = out["label_norm"].map(tier_by_class)
    return out


def run(cmd: list[str], dry_run: bool = False) -> bool:
    if dry_run:
        print(" ".join(cmd))
        return True
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    if proc.returncode != 0:
        print(proc.stdout[-2000:], file=sys.stderr)
        return False
    return True


def proxy_url() -> str | None:
    return os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY") or os.environ.get("http_proxy") or os.environ.get("HTTP_PROXY")


def download_source(video_id: str, tmp_dir: Path, dry_run: bool) -> Path | None:
    out_tmpl = tmp_dir / f"{video_id}.%(ext)s"
    cmd = [
        "yt-dlp",
        "-f",
        "bv*[height<=720]+ba/b[height<=720]/b",
        "--merge-output-format",
        "mp4",
        "--no-playlist",
        "-o",
        str(out_tmpl),
        YOUTUBE_URL.format(video_id=video_id),
    ]
    proxy = proxy_url()
    if proxy:
        cmd[1:1] = ["--proxy", proxy]
    if not run(cmd, dry_run=dry_run):
        return None
    if dry_run:
        return tmp_dir / f"{video_id}.mp4"
    matches = sorted(tmp_dir.glob(f"{video_id}.*"))
    return matches[0] if matches else None


def trim_clip(
    source: Path,
    dest: Path,
    start: float,
    end: float,
    target_fps: float,
    max_seconds: float,
    dry_run: bool,
) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    duration = max(min(float(end) - float(start), max_seconds), 0.1)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{float(start):.3f}",
        "-i",
        str(source),
        "-t",
        f"{duration:.3f}",
        "-vf",
        f"fps={target_fps},scale='min(720,iw)':-2",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        str(dest),
    ]
    return run(cmd, dry_run=dry_run)


def safe_class_dir(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", normalize_label(label)).strip("_")


def build_labels(rows: Iterable[dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "path",
        "youtube_id",
        "class",
        "tier",
        "time_start",
        "time_end",
        "human_rating",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and trim a curated Kinetics-700 subset.")
    parser.add_argument("--annotations", required=True, help="Kinetics annotation CSV path or URL.")
    parser.add_argument("--config", default="configs/classes.yaml")
    parser.add_argument("--output-dir", default="data/clips")
    parser.add_argument("--labels", default="data/labels.csv")
    parser.add_argument("--clips-per-class", type=int, default=None)
    parser.add_argument("--target-fps", type=float, default=None)
    parser.add_argument("--max-clip-seconds", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    download_cfg = config.get("download", {})
    clips_per_class = args.clips_per_class or int(download_cfg.get("clips_per_class", 20))
    target_fps = args.target_fps or float(download_cfg.get("target_fps", 8))
    max_seconds = args.max_clip_seconds or float(download_cfg.get("max_clip_seconds", 10))

    annotations = load_annotations(args.annotations)
    subset = select_subset(annotations, config, clips_per_class, args.seed)

    output_dir = Path(args.output_dir)
    labels_path = Path(args.labels)
    label_rows = []

    with tempfile.TemporaryDirectory(prefix="kinetics_download_") as tmp:
        tmp_dir = Path(tmp)
        for row in tqdm(subset.to_dict("records"), desc="clips"):
            video_id = str(row["youtube_id"])
            label = str(row["label"])
            tier = int(row["tier"])
            start = float(row["time_start"])
            end = float(row["time_end"])
            rel = Path(f"tier{tier}") / safe_class_dir(label) / f"{video_id}_{int(start)}_{int(end)}.mp4"
            dest = output_dir / rel

            if not dest.exists():
                source = download_source(video_id, tmp_dir, args.dry_run)
                if source is None:
                    continue
                ok = trim_clip(source, dest, start, end, target_fps, max_seconds, args.dry_run)
                if not ok:
                    continue

            label_rows.append(
                {
                    "path": str(dest),
                    "youtube_id": video_id,
                    "class": label,
                    "tier": tier,
                    "time_start": start,
                    "time_end": end,
                    "human_rating": "",
                }
            )

    build_labels(label_rows, labels_path)
    print(f"wrote {len(label_rows)} clips to {labels_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
