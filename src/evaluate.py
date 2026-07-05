from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import accuracy_score, confusion_matrix

from src.common import ensure_parent, minmax01, read_labels


METHOD_COLUMNS = {
    "flow": ["flow_foreground", "flow_global"],
    "pose": ["pose_motion"],
    "learned": ["learned_score", "learned_tier_prob", "learned_motion"],
}


def merge_scores(labels_csv: str | Path, score_csvs: list[str]) -> pd.DataFrame:
    merged = read_labels(labels_csv)
    key_cols = ["path", "class", "tier"]
    if "human_rating" in merged.columns:
        key_cols.append("human_rating")
    for csv_path in score_csvs:
        if not csv_path:
            continue
        score_df = pd.read_csv(csv_path)
        score_cols = [c for c in score_df.columns if c not in {"class", "tier", "human_rating", "youtube_id", "time_start", "time_end"}]
        keep = ["path"] + [c for c in score_cols if c != "path"]
        merged = merged.merge(score_df[keep], on="path", how="left")
    return merged


def choose_method_columns(df: pd.DataFrame) -> dict[str, str]:
    chosen = {}
    for method, candidates in METHOD_COLUMNS.items():
        for col in candidates:
            if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any():
                chosen[method] = col
                break
    return chosen


def normalize_methods(df: pd.DataFrame, chosen: dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    for method, col in chosen.items():
        vals = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=np.float64)
        fill = np.nanmedian(vals) if np.isfinite(vals).any() else 0.0
        out[f"{method}_norm"] = minmax01(np.nan_to_num(vals, nan=fill))
    norm_cols = [f"{m}_norm" for m in chosen]
    if norm_cols:
        out["fused_motion"] = out[norm_cols].mean(axis=1)
        out["difficulty"] = pd.cut(
            out["fused_motion"],
            bins=[-1e-9, 1.0 / 3.0, 2.0 / 3.0, 1.0 + 1e-9],
            labels=["easy", "medium", "hard"],
        ).astype(str)
    return out


def corr_or_nan(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.unique(x[valid]).size < 2 or np.unique(y[valid]).size < 2:
        return float("nan"), float("nan")
    return float(spearmanr(x[valid], y[valid]).statistic), float(kendalltau(x[valid], y[valid]).statistic)


def compute_metrics(df: pd.DataFrame, chosen: dict[str, str]) -> dict:
    metrics: dict[str, object] = {"methods": chosen}
    tier = pd.to_numeric(df["tier"], errors="coerce").to_numpy(dtype=np.float64)
    human = pd.to_numeric(df.get("human_rating"), errors="coerce").to_numpy(dtype=np.float64) if "human_rating" in df else None

    method_metrics = {}
    for method in chosen:
        score = df[f"{method}_norm"].to_numpy(dtype=np.float64)
        spearman_tier, kendall_tier = corr_or_nan(score, tier)
        item = {"spearman_tier": spearman_tier, "kendall_tier": kendall_tier}
        if human is not None and np.isfinite(human).sum() >= 3:
            spearman_human, kendall_human = corr_or_nan(score, human)
            item.update({"spearman_human": spearman_human, "kendall_human": kendall_human})
        method_metrics[method] = item
    metrics["method_metrics"] = method_metrics

    if "fused_motion" in df:
        fused = df["fused_motion"].to_numpy(dtype=np.float64)
        spearman_tier, kendall_tier = corr_or_nan(fused, tier)
        metrics["fused"] = {"spearman_tier": spearman_tier, "kendall_tier": kendall_tier}
        pred_tier = np.digitize(fused, bins=[1.0 / 3.0, 2.0 / 3.0]) + 1
        valid = np.isfinite(tier)
        metrics["tier_accuracy"] = float(accuracy_score(tier[valid].astype(int), pred_tier[valid].astype(int))) if valid.any() else float("nan")
        metrics["confusion_matrix"] = confusion_matrix(
            tier[valid].astype(int),
            pred_tier[valid].astype(int),
            labels=[1, 2, 3],
        ).tolist()
        if human is not None and np.isfinite(human).sum() >= 3:
            spearman_human, kendall_human = corr_or_nan(fused, human)
            metrics["fused"].update({"spearman_human": spearman_human, "kendall_human": kendall_human})

    norm_cols = [f"{m}_norm" for m in chosen]
    if len(norm_cols) >= 2:
        metrics["inter_method_correlation"] = df[norm_cols].corr(method="spearman").to_dict()

    if "learned_tier_pred" in df.columns:
        pred = pd.to_numeric(df["learned_tier_pred"], errors="coerce").to_numpy(dtype=np.float64)
        valid = np.isfinite(tier) & np.isfinite(pred)
        if valid.any():
            metrics["learned_tier_accuracy"] = float(
                accuracy_score(tier[valid].astype(int), pred[valid].astype(int))
            )
            metrics["learned_tier_confusion_matrix"] = confusion_matrix(
                tier[valid].astype(int),
                pred[valid].astype(int),
                labels=[1, 2, 3],
            ).tolist()

    learned_aux = {}
    for col in ("learned_motion", "learned_tier_prob"):
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
            if np.isfinite(vals).sum() >= 3:
                learned_aux[col] = {
                    "spearman_tier": corr_or_nan(vals, tier)[0],
                    "kendall_tier": corr_or_nan(vals, tier)[1],
                }
    if learned_aux:
        metrics["learned_aux_metrics"] = learned_aux
    return metrics


def tier_boxplot_data(df: pd.DataFrame, score_col: str) -> list[np.ndarray]:
    return [df[df["tier"] == tier][score_col].dropna().to_numpy(dtype=np.float64) for tier in [1, 2, 3]]


def save_tier_boxplot(
    data: list[np.ndarray],
    ylabel: str,
    title: str,
    out_path: Path,
    figsize: tuple[float, float] = (8, 5),
) -> None:
    plt.figure(figsize=figsize)
    plt.boxplot(data, tick_labels=["low", "medium", "high"])
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(-0.02, 1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_outputs(df: pd.DataFrame, chosen: dict[str, str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    norm_cols = [f"{m}_norm" for m in chosen]
    if not norm_cols:
        return

    tier_labels = ["low", "medium", "high"]
    plot_specs: list[tuple[str, str, str, str]] = []
    for method in chosen:
        plot_specs.append(
            (
                method,
                f"{method}_norm",
                f"{method.title()} normalized motion",
                f"Motion score by tier ({method})",
            )
        )
    if "fused_motion" in df.columns:
        plot_specs.append(
            (
                "fused",
                "fused_motion",
                "Fused normalized motion",
                "Motion score by tier (fused)",
            )
        )

    series_data: dict[str, list[np.ndarray]] = {}
    for key, col, ylabel, title in plot_specs:
        data = tier_boxplot_data(df, col)
        series_data[key] = data
        save_tier_boxplot(data, ylabel, title, out_dir / f"tier_boxplot_{key}.png")

    # Backward-compatible alias for the fused plot.
    if "fused" in series_data:
        save_tier_boxplot(
            series_data["fused"],
            "Fused normalized motion",
            "Motion score by tier",
            out_dir / "tier_boxplot.png",
        )

    learned_aux_specs = [
        ("learned_motion", "learned_motion", "Learned regression (raw)", "Regression head by tier"),
        ("learned_tier_prob", "learned_tier_prob", "Learned tier probability", "Tier-probability score by tier"),
    ]
    for key, col, ylabel, title in learned_aux_specs:
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any():
            data = tier_boxplot_data(df, col)
            save_tier_boxplot(data, ylabel, title, out_dir / f"tier_boxplot_{key}.png", figsize=(7, 5))

    n = len(plot_specs)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (key, col, ylabel, title) in zip(axes, plot_specs):
        data = series_data[key]
        ax.boxplot(data, tick_labels=tier_labels)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.02, 1.02)
        medians = [float(np.median(d)) if len(d) else float("nan") for d in data]
        ax.text(
            0.02,
            0.98,
            f"med: {medians[0]:.2f} / {medians[1]:.2f} / {medians[2]:.2f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
        )
    fig.suptitle("Tier separation comparison", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "tier_boxplot_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    if "human_rating" in df and pd.to_numeric(df["human_rating"], errors="coerce").notna().any():
        human = pd.to_numeric(df["human_rating"], errors="coerce")
        plt.figure(figsize=(6, 5))
        plt.scatter(human, df["fused_motion"], alpha=0.75)
        plt.xlabel("Human rating")
        plt.ylabel("Fused normalized motion")
        plt.title("Human rating agreement")
        plt.tight_layout()
        plt.savefig(out_dir / "human_scatter.png", dpi=160)
        plt.close()

    corr = df[norm_cols].corr(method="spearman")
    plt.figure(figsize=(5, 4))
    plt.imshow(corr, vmin=-1, vmax=1)
    plt.xticks(range(len(norm_cols)), norm_cols, rotation=30, ha="right")
    plt.yticks(range(len(norm_cols)), norm_cols)
    plt.colorbar(label="Spearman")
    plt.title("Inter-method agreement")
    plt.tight_layout()
    plt.savefig(out_dir / "method_correlation.png", dpi=160)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate motion magnitude benchmark outputs.")
    parser.add_argument("--labels", default="data/labels.csv")
    parser.add_argument("--score-csv", action="append", default=[], help="Score CSV to merge; pass multiple times.")
    parser.add_argument("--output", default="results/scores.csv")
    parser.add_argument("--metrics", default="results/metrics.json")
    parser.add_argument("--plots-dir", default="results/plots")
    args = parser.parse_args()

    df = merge_scores(args.labels, args.score_csv)
    chosen = choose_method_columns(df)
    df = normalize_methods(df, chosen)

    ensure_parent(args.output)
    df.to_csv(args.output, index=False)

    metrics = compute_metrics(df, chosen)
    ensure_parent(args.metrics)
    Path(args.metrics).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_outputs(df, chosen, Path(args.plots_dir))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
