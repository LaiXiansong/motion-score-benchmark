from __future__ import annotations

import os
from pathlib import Path

DEFAULT_HF_HOME = Path.home() / ".cache" / "huggingface"


def _hf_path_usable(path: Path) -> bool:
    try:
        if path.exists():
            next(path.iterdir(), None)
        else:
            path.mkdir(parents=True, exist_ok=True)
        return True
    except OSError:
        return False


def ensure_hf_cache() -> None:
    """Use a local WSL cache when HF_HOME points at an unavailable mount."""
    hf_home = os.environ.get("HF_HOME")
    if hf_home and _hf_path_usable(Path(hf_home)):
        return
    fallback = DEFAULT_HF_HOME
    fallback.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(fallback)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(fallback / "hub")


ensure_hf_cache()

import argparse
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoConfig, AutoModel

from src.common import ensure_parent, minmax01, read_frames_array, read_labels


DEFAULT_MODEL = "MCG-NJU/videomae-base-finetuned-kinetics"
TIER_VALUES = np.array([0.0, 0.5, 1.0], dtype=np.float32)


@dataclass
class LearnedScoreConfig:
    model_name: str = DEFAULT_MODEL
    checkpoint: str | None = None
    num_frames: int = 16
    image_size: int = 224
    device: str = "cuda"


def sample_uniform_frames(path: str | Path, num_frames: int, image_size: int) -> torch.Tensor:
    frames = read_frames_array(path, target_fps=8.0, max_frames=None, resize_long_side=image_size, rgb=True)
    idx = np.linspace(0, len(frames) - 1, num_frames).round().astype(int)
    frames = frames[idx]
    processed = []
    for frame in frames:
        resized = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
        arr = resized.astype(np.float32) / 255.0
        arr = (arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        processed.append(arr.transpose(2, 0, 1))
    return torch.from_numpy(np.stack(processed)).float()


def tier_probability_score(prob: np.ndarray) -> float:
    return float(np.dot(prob.astype(np.float32), TIER_VALUES))


class MotionHead(nn.Module):
    def __init__(self, model_name: str = DEFAULT_MODEL, dropout: float = 0.1) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = getattr(self.backbone.config, "hidden_size", None) or AutoConfig.from_pretrained(model_name).hidden_size
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.regressor = nn.Linear(hidden // 2, 1)
        self.classifier = nn.Linear(hidden // 2, 3)

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.backbone(pixel_values=pixel_values)
        if hasattr(outputs, "last_hidden_state"):
            pooled = outputs.last_hidden_state[:, 0]
        elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs[0][:, 0]
        features = self.head(pooled)
        return {
            "motion": self.regressor(features).squeeze(-1),
            "logits": self.classifier(features),
        }


def load_motion_model(config: LearnedScoreConfig) -> MotionHead:
    payload = torch.load(config.checkpoint, map_location="cpu") if config.checkpoint else None
    model_name = payload.get("model_name", config.model_name) if isinstance(payload, dict) else config.model_name
    model = MotionHead(model_name)
    if isinstance(payload, dict):
        model.load_state_dict(payload["state_dict"])
    model.to(config.device)
    model.eval()
    return model


def score_video(path: str | Path, model: MotionHead, config: LearnedScoreConfig) -> dict[str, float]:
    frames = sample_uniform_frames(path, config.num_frames, config.image_size).unsqueeze(0).to(config.device)
    with torch.no_grad():
        out = model(frames)
        prob = torch.softmax(out["logits"], dim=-1)[0].detach().cpu().numpy()
        tier_pred = int(np.argmax(prob)) + 1
        motion_raw = float(out["motion"][0].detach().cpu())
        motion = float(np.clip(motion_raw, 0.0, 1.0))
        tier_prob_score = tier_probability_score(prob)
    return {
        "learned_score": tier_prob_score,
        "learned_tier_prob": tier_prob_score,
        "learned_motion": motion,
        "learned_motion_raw": motion_raw,
        "learned_tier_pred": float(tier_pred),
        "learned_prob_low": float(prob[0]),
        "learned_prob_medium": float(prob[1]),
        "learned_prob_high": float(prob[2]),
    }


def score_labels(labels_csv: str | Path, output_csv: str | Path, config: LearnedScoreConfig) -> pd.DataFrame:
    if not config.checkpoint:
        raise ValueError("--checkpoint is required for learned inference.")
    labels = read_labels(labels_csv)
    model = load_motion_model(config)
    rows = []
    for row in tqdm(labels.to_dict("records"), desc="learned"):
        rows.append({**row, **score_video(row["path"], model, config)})
    out = pd.DataFrame(rows)
    ensure_parent(output_csv)
    out.to_csv(output_csv, index=False)
    return out


def _normalize_column(df: pd.DataFrame, col: str) -> np.ndarray:
    vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
    if np.isfinite(vals).sum() < 2:
        return np.zeros(len(df), dtype=np.float32)
    return minmax01(np.nan_to_num(vals, nan=np.nanmedian(vals))).astype(np.float32)


def build_distilled_target(df: pd.DataFrame, distill_target: str = "pose") -> np.ndarray:
    distill_target = distill_target.lower()
    target = np.full(len(df), np.nan, dtype=np.float32)

    human = pd.to_numeric(df.get("human_rating"), errors="coerce") if "human_rating" in df else None
    if human is not None and human.notna().any():
        human_vals = human.astype(float).to_numpy()
        valid = np.isfinite(human_vals)
        if valid.any():
            lo = np.nanmin(human_vals[valid])
            hi = np.nanmax(human_vals[valid])
            human_norm = np.full(len(df), np.nan, dtype=np.float32)
            human_norm[valid] = (human_vals[valid] - lo) / max(hi - lo, 1e-6)
            if distill_target == "human":
                return human_norm.astype(np.float32)
            target = np.where(valid, human_norm, target)

    if distill_target == "pose" and "pose_motion" in df.columns:
        target = np.where(np.isfinite(target), target, _normalize_column(df, "pose_motion"))
    elif distill_target == "flow":
        flow_col = "flow_foreground" if "flow_foreground" in df.columns else "flow_global"
        if flow_col in df.columns:
            target = np.where(np.isfinite(target), target, _normalize_column(df, flow_col))
    elif distill_target == "fused":
        parts = []
        for col in ("flow_foreground", "flow_global", "pose_motion"):
            if col in df.columns:
                parts.append(_normalize_column(df, col))
        if parts:
            fused = np.mean(np.stack(parts), axis=0)
            target = np.where(np.isfinite(target), target, fused)
    elif distill_target == "tier":
        tier = pd.to_numeric(df["tier"], errors="coerce").to_numpy(dtype=np.float32)
        target = np.where(np.isfinite(target), target, (tier - 1.0) / 2.0)

    if not np.isfinite(target).all():
        if "pose_motion" in df.columns:
            target = np.where(np.isfinite(target), target, _normalize_column(df, "pose_motion"))
        else:
            tier = pd.to_numeric(df["tier"], errors="coerce").to_numpy(dtype=np.float32)
            target = np.where(np.isfinite(target), target, (tier - 1.0) / 2.0)
    return target.astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run learned motion scorer inference.")
    parser.add_argument("--labels", default="data/labels.csv")
    parser.add_argument("--output", default="results/learned_scores.csv")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    cfg = LearnedScoreConfig(
        model_name=args.model_name,
        checkpoint=args.checkpoint,
        num_frames=args.num_frames,
        image_size=args.image_size,
        device=args.device,
    )
    score_labels(args.labels, args.output, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
