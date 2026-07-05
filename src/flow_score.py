from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.common import ensure_parent, frame_diagonal, iter_sampled_frames, read_labels, top_percent_mean


@dataclass
class FlowScoreConfig:
    backend: str = "auto"
    target_fps: float = 8.0
    resize_long_side: int = 720
    top_percent: float = 5.0
    max_frames: int | None = None
    sea_raft_root: str | None = None
    sea_raft_checkpoint: str | None = None
    sea_raft_cfg: str | None = None
    device: str = "cuda"


def setup_sea_raft_paths(root: str | Path) -> Path:
    root_path = Path(root).resolve()
    core_path = root_path / "core"
    if not core_path.is_dir():
        raise RuntimeError(f"SEA-RAFT core directory not found: {core_path}")
    for path in (root_path, core_path):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    return root_path


class FlowBackend:
    def estimate(self, frame1_bgr: np.ndarray, frame2_bgr: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class FarnebackBackend(FlowBackend):
    """CPU fallback for smoke tests; use SEA-RAFT for final numbers."""

    def estimate(self, frame1_bgr: np.ndarray, frame2_bgr: np.ndarray) -> np.ndarray:
        prev = cv2.cvtColor(frame1_bgr, cv2.COLOR_BGR2GRAY)
        nxt = cv2.cvtColor(frame2_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.calcOpticalFlowFarneback(
            prev,
            nxt,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )


class SeaRaftBackend(FlowBackend):
    """Adapter for princeton-vl/SEA-RAFT using the upstream config + checkpoint loader."""

    def __init__(
        self,
        root: str | None,
        checkpoint: str | None,
        device: str = "cuda",
        cfg_path: str | None = None,
    ) -> None:
        root = root or os.environ.get("SEA_RAFT_ROOT")
        if not root:
            raise RuntimeError("SEA-RAFT root is required. Set SEA_RAFT_ROOT or pass --sea-raft-root.")
        self.root = setup_sea_raft_paths(root)

        self.checkpoint = checkpoint or os.environ.get("SEA_RAFT_CHECKPOINT")
        if not self.checkpoint:
            raise RuntimeError("SEA-RAFT checkpoint is required. Set SEA_RAFT_CHECKPOINT or pass --sea-raft-checkpoint.")
        self.checkpoint = str(Path(self.checkpoint).resolve())
        if not Path(self.checkpoint).is_file():
            raise FileNotFoundError(f"SEA-RAFT checkpoint not found: {self.checkpoint}")

        cfg_path = cfg_path or os.environ.get("SEA_RAFT_CFG")
        if not cfg_path:
            cfg_path = str(self.root / "config/eval/sintel-M.json")
        self.cfg_path = str(Path(cfg_path).resolve())
        if not Path(self.cfg_path).is_file():
            raise FileNotFoundError(f"SEA-RAFT config not found: {self.cfg_path}")

        import torch  # noqa: PLC0415
        import torch.nn.functional as F  # noqa: PLC0415
        from config.parser import json_to_args  # type: ignore  # noqa: PLC0415
        from raft import RAFT  # type: ignore  # noqa: PLC0415
        from utils.utils import InputPadder, load_ckpt  # type: ignore  # noqa: PLC0415

        self.torch = torch
        self.F = F
        self.InputPadder = InputPadder
        self.device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
        self.args = json_to_args(self.cfg_path)
        self.model = RAFT(self.args)
        load_ckpt(self.model, self.checkpoint)
        self.model.eval().to(self.device)

    def _to_tensor(self, frame_bgr: np.ndarray) -> Any:
        rgb = frame_bgr[..., ::-1].astype(np.float32)
        tensor = self.torch.from_numpy(rgb).permute(2, 0, 1).float()[None]
        return tensor.to(self.device)

    def estimate(self, frame1_bgr: np.ndarray, frame2_bgr: np.ndarray) -> np.ndarray:
        image1 = self._to_tensor(frame1_bgr)
        image2 = self._to_tensor(frame2_bgr)
        padder = self.InputPadder(image1.shape, mode="sintel")
        image1, image2 = padder.pad(image1, image2)
        with self.torch.no_grad():
            output = self.model(image1, image2, iters=self.args.iters, test_mode=True)
            flow = output["flow"][-1]
            flow = padder.unpad(flow)
        return flow[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32, copy=False)


def build_backend(config: FlowScoreConfig) -> FlowBackend:
    backend = config.backend.lower()
    if backend == "farneback":
        return FarnebackBackend()
    if backend in {"sea-raft", "searaft"}:
        return SeaRaftBackend(
            config.sea_raft_root or os.environ.get("SEA_RAFT_ROOT"),
            config.sea_raft_checkpoint,
            config.device,
            config.sea_raft_cfg,
        )
    if backend == "auto":
        try:
            return SeaRaftBackend(
                config.sea_raft_root or os.environ.get("SEA_RAFT_ROOT"),
                config.sea_raft_checkpoint,
                config.device,
                config.sea_raft_cfg,
            )
        except Exception as exc:
            print(f"warning: SEA-RAFT unavailable, using Farneback fallback: {exc}", file=sys.stderr)
            return FarnebackBackend()
    raise ValueError(f"Unknown flow backend: {config.backend}")


def mask_from_boxes(shape: tuple[int, int], boxes: list[list[float]] | None) -> np.ndarray | None:
    if not boxes:
        return None
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    for box in boxes:
        if len(box) < 4:
            continue
        x1, y1, x2, y2 = box[:4]
        x1i, y1i = max(int(x1), 0), max(int(y1), 0)
        x2i, y2i = min(int(x2), w - 1), min(int(y2), h - 1)
        if x2i > x1i and y2i > y1i:
            mask[y1i : y2i + 1, x1i : x2i + 1] = True
    return mask


def load_person_boxes(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def score_video(
    path: str | Path,
    config: FlowScoreConfig | None = None,
    backend: FlowBackend | None = None,
    person_boxes: list[list[list[float]]] | None = None,
) -> dict[str, float]:
    cfg = config or FlowScoreConfig()
    flow_backend = backend or build_backend(cfg)
    frames = list(
        iter_sampled_frames(
            path,
            target_fps=cfg.target_fps,
            max_frames=cfg.max_frames,
            resize_long_side=cfg.resize_long_side,
        )
    )
    if len(frames) < 2:
        return {"flow_global": 0.0, "flow_foreground": np.nan, "flow_pairs": 0}

    global_scores = []
    foreground_scores = []
    for i, (a, b) in enumerate(zip(frames[:-1], frames[1:])):
        flow = flow_backend.estimate(a, b)
        mag = np.linalg.norm(flow, axis=-1) / max(frame_diagonal(a), 1.0)
        global_scores.append(top_percent_mean(mag, cfg.top_percent))

        boxes = person_boxes[i] if person_boxes and i < len(person_boxes) else None
        mask = mask_from_boxes(mag.shape, boxes)
        if mask is not None and np.any(mask):
            foreground_scores.append(top_percent_mean(mag[mask], cfg.top_percent))

    return {
        "flow_global": float(np.mean(global_scores)) if global_scores else 0.0,
        "flow_foreground": float(np.mean(foreground_scores)) if foreground_scores else np.nan,
        "flow_pairs": float(len(global_scores)),
    }


def score_labels(
    labels_csv: str | Path,
    output_csv: str | Path,
    config: FlowScoreConfig,
    boxes_json: str | Path | None = None,
) -> pd.DataFrame:
    labels = read_labels(labels_csv)
    boxes_by_path = load_person_boxes(boxes_json)
    backend = build_backend(config)
    rows = []
    for row in tqdm(labels.to_dict("records"), desc="flow"):
        path = row["path"]
        boxes = boxes_by_path.get(path) or boxes_by_path.get(str(Path(path).resolve()))
        scores = score_video(path, config=config, backend=backend, person_boxes=boxes)
        rows.append({**row, **scores})
    out = pd.DataFrame(rows)
    ensure_parent(output_csv)
    out.to_csv(output_csv, index=False)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute optical-flow motion magnitude scores.")
    parser.add_argument("--labels", default="data/labels.csv")
    parser.add_argument("--output", default="results/flow_scores.csv")
    parser.add_argument("--backend", default="auto", choices=["auto", "sea-raft", "searaft", "farneback"])
    parser.add_argument("--target-fps", type=float, default=8.0)
    parser.add_argument("--resize-long-side", type=int, default=720)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--sea-raft-root", default=None)
    parser.add_argument("--sea-raft-checkpoint", default=None)
    parser.add_argument(
        "--sea-raft-cfg",
        default=None,
        help="SEA-RAFT eval config JSON, e.g. external/SEA-RAFT/config/eval/sintel-M.json",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--boxes-json", default=None, help="Optional pose output JSON for foreground flow masks.")
    args = parser.parse_args()
    cfg = FlowScoreConfig(
        backend=args.backend,
        target_fps=args.target_fps,
        resize_long_side=args.resize_long_side,
        max_frames=args.max_frames,
        sea_raft_root=args.sea_raft_root,
        sea_raft_checkpoint=args.sea_raft_checkpoint,
        sea_raft_cfg=args.sea_raft_cfg,
        device=args.device,
    )
    score_labels(args.labels, args.output, cfg, boxes_json=args.boxes_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
