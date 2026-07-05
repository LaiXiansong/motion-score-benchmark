from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

import cv2
import numpy as np
import pandas as pd
import yaml


@dataclass
class VideoMeta:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int
    duration: float


def preload_onnx_cuda_libs() -> bool:
    """Load CUDA runtime libs required by onnxruntime-gpu inside WSL/PyTorch venv."""
    import ctypes

    site = Path(sys.prefix) / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    candidates = [
        site / "nvidia/cu13/lib/libcudart.so.13",
        site / "nvidia/cuda_runtime/lib/libcudart.so.12",
    ]
    for lib in candidates:
        if lib.exists():
            ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
            return True
    return False


def load_config(path: str | Path = "configs/classes.yaml") -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_parent(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def video_meta(path: str | Path) -> VideoMeta:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    duration = frame_count / fps if fps > 0 else 0.0
    return VideoMeta(str(path), fps, frame_count, width, height, duration)


def iter_sampled_frames(
    path: str | Path,
    target_fps: float = 8.0,
    max_frames: int | None = None,
    resize_long_side: int | None = None,
) -> Iterator[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or target_fps or 1.0)
    stride = max(int(round(src_fps / target_fps)), 1) if target_fps > 0 else 1
    frame_idx = 0
    yielded = 0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if frame_idx % stride == 0:
            if resize_long_side:
                frame_bgr = resize_by_long_side(frame_bgr, resize_long_side)
            yield frame_bgr
            yielded += 1
            if max_frames is not None and yielded >= max_frames:
                break
        frame_idx += 1

    cap.release()


def read_frames_array(
    path: str | Path,
    target_fps: float = 8.0,
    max_frames: int | None = None,
    resize_long_side: int | None = None,
    rgb: bool = True,
) -> np.ndarray:
    frames = list(iter_sampled_frames(path, target_fps, max_frames, resize_long_side))
    if not frames:
        raise ValueError(f"No frames read from video: {path}")
    arr = np.stack(frames)
    if rgb:
        arr = arr[..., ::-1]
    return arr


def resize_by_long_side(frame: np.ndarray, long_side: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = long_side / max(h, w)
    if scale >= 1.0:
        return frame
    return cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)


def frame_diagonal(frame: np.ndarray) -> float:
    h, w = frame.shape[:2]
    return float(np.hypot(h, w))


def top_percent_mean(values: np.ndarray, percent: float = 5.0) -> float:
    flat = values.reshape(-1)
    if flat.size == 0:
        return 0.0
    k = max(int(round(flat.size * percent / 100.0)), 1)
    top = np.partition(flat, flat.size - k)[-k:]
    return float(np.mean(top))


def minmax01(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return arr
    lo = np.nanmin(arr)
    hi = np.nanmax(arr)
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - lo) / (hi - lo)


def write_json(path: str | Path, data: dict) -> None:
    out = ensure_parent(path)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def dataclass_to_dict(obj: object) -> dict:
    return asdict(obj)


def read_labels(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"path", "class", "tier"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"labels CSV missing columns: {sorted(missing)}")
    return df
