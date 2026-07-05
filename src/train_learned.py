from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.common import ensure_parent, read_labels
from src.learned_score import (
    DEFAULT_MODEL,
    MotionHead,
    TIER_VALUES,
    build_distilled_target,
    sample_uniform_frames,
)


class MotionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, targets: np.ndarray, num_frames: int, image_size: int) -> None:
        self.df = df.reset_index(drop=True)
        self.targets = targets.astype(np.float32)
        self.num_frames = num_frames
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        frames = sample_uniform_frames(row["path"], self.num_frames, self.image_size)
        tier = int(row["tier"]) - 1
        return {
            "pixel_values": frames,
            "tier": torch.tensor(tier, dtype=torch.long),
            "motion": torch.tensor(float(self.targets[idx]), dtype=torch.float32),
        }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_epoch(
    model: MotionHead,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    lambda_reg: float,
) -> dict[str, float]:
    model.train()
    ce = nn.CrossEntropyLoss()
    huber = nn.SmoothL1Loss()
    losses = []
    for batch in tqdm(loader, desc="train", leave=False):
        pixel_values = batch["pixel_values"].to(device)
        tier = batch["tier"].to(device)
        motion = batch["motion"].to(device)
        out = model(pixel_values)
        loss_cls = ce(out["logits"], tier)
        motion_pred = torch.clamp(out["motion"], 0.0, 1.0)
        loss_reg = huber(motion_pred, motion)
        loss = loss_cls + lambda_reg * loss_reg
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append([float(loss.detach().cpu()), float(loss_cls.detach().cpu()), float(loss_reg.detach().cpu())])
    arr = np.asarray(losses)
    return {"loss": float(arr[:, 0].mean()), "loss_cls": float(arr[:, 1].mean()), "loss_reg": float(arr[:, 2].mean())}


@torch.no_grad()
def validate(model: MotionHead, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    preds = []
    tiers = []
    motions = []
    motion_preds = []
    tier_prob_scores = []
    for batch in tqdm(loader, desc="val", leave=False):
        pixel_values = batch["pixel_values"].to(device)
        out = model(pixel_values)
        prob = torch.softmax(out["logits"], dim=-1).detach().cpu().numpy()
        tier_prob_scores.extend((prob @ TIER_VALUES).tolist())
        preds.extend(torch.argmax(out["logits"], dim=-1).cpu().tolist())
        tiers.extend(batch["tier"].cpu().tolist())
        motions.extend(batch["motion"].cpu().tolist())
        motion_preds.extend(torch.clamp(out["motion"], 0.0, 1.0).cpu().tolist())
    preds_arr = np.asarray(preds)
    tiers_arr = np.asarray(tiers)
    motion_arr = np.asarray(motions, dtype=np.float32)
    motion_pred_arr = np.asarray(motion_preds, dtype=np.float32)
    tier_prob_arr = np.asarray(tier_prob_scores, dtype=np.float32)
    acc = float(np.mean(preds_arr == tiers_arr)) if len(tiers_arr) else 0.0
    mae = float(np.mean(np.abs(motion_arr - motion_pred_arr))) if len(motion_arr) else 0.0
    spearman_reg = float("nan")
    spearman_tier_prob = float("nan")
    if len(motion_arr) >= 3 and np.unique(motion_arr).size > 1 and np.unique(motion_pred_arr).size > 1:
        spearman_reg = float(spearmanr(motion_arr, motion_pred_arr).statistic)
    tier_continuous = tiers_arr.astype(np.float32) / 2.0
    if len(tier_prob_arr) >= 3 and np.unique(tier_prob_arr).size > 1:
        spearman_tier_prob = float(spearmanr(tier_continuous, tier_prob_arr).statistic)
    return {
        "tier_acc": acc,
        "motion_mae": mae,
        "spearman_reg": spearman_reg,
        "spearman_tier_prob": spearman_tier_prob,
    }


def selection_score(metrics: dict[str, float], selection_metric: str) -> float:
    if selection_metric == "tier_acc":
        return metrics["tier_acc"]
    if selection_metric == "spearman_reg":
        return metrics.get("spearman_reg", float("nan"))
    if selection_metric == "spearman_tier_prob":
        return metrics.get("spearman_tier_prob", float("nan"))
    if selection_metric == "combined":
        spearman = metrics.get("spearman_tier_prob", float("nan"))
        if not np.isfinite(spearman):
            spearman = 0.0
        return metrics["tier_acc"] + 0.25 * spearman - 0.25 * metrics["motion_mae"]
    raise ValueError(f"Unknown selection metric: {selection_metric}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune VideoMAE motion magnitude head.")
    parser.add_argument("--labels", default="results/fused_training_scores.csv")
    parser.add_argument("--output", default="checkpoints/learned_motion.pt")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lambda-reg", type=float, default=2.0)
    parser.add_argument(
        "--distill-target",
        default="pose",
        choices=["human", "pose", "flow", "fused", "tier"],
        help="Regression target for the motion head.",
    )
    parser.add_argument(
        "--selection-metric",
        default="combined",
        choices=["tier_acc", "spearman_reg", "spearman_tier_prob", "combined"],
        help="Validation metric used to save the best checkpoint.",
    )
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    lr = args.lr
    if lr is None:
        lr = 3e-4 if args.freeze_backbone else 1e-5

    seed_everything(args.seed)
    df = read_labels(args.labels)
    targets = build_distilled_target(df, distill_target=args.distill_target)
    train_df, val_df, train_targets, val_targets = train_test_split(
        df,
        targets,
        test_size=args.val_size,
        stratify=df["tier"] if df["tier"].nunique() > 1 else None,
        random_state=args.seed,
    )

    train_ds = MotionDataset(train_df, train_targets, args.num_frames, args.image_size)
    val_ds = MotionDataset(val_df, val_targets, args.num_frames, args.image_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = MotionHead(args.model_name).to(args.device)
    if args.freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=args.weight_decay)

    history = []
    best_score = float("-inf")
    output = ensure_parent(args.output)
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, args.device, args.lambda_reg)
        val_metrics = validate(model, val_loader, args.device)
        score = selection_score(val_metrics, args.selection_metric)
        metrics = {
            "epoch": epoch,
            **train_metrics,
            **val_metrics,
            "selection_score": score,
            "distill_target": args.distill_target,
            "freeze_backbone": args.freeze_backbone,
        }
        history.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False))
        if score >= best_score:
            best_score = score
            torch.save(
                {
                    "model_name": args.model_name,
                    "state_dict": model.state_dict(),
                    "num_frames": args.num_frames,
                    "image_size": args.image_size,
                    "distill_target": args.distill_target,
                    "freeze_backbone": args.freeze_backbone,
                    "selection_metric": args.selection_metric,
                    "history": history,
                },
                output,
            )

    Path(str(output) + ".history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
