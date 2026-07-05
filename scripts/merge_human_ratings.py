#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import ensure_parent, read_labels  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge completed human ratings into labels.csv.")
    parser.add_argument("--labels", default="data/labels.csv")
    parser.add_argument("--ratings", default="data/human_rating_sheet.csv")
    parser.add_argument("--output", default="data/labels_with_ratings.csv")
    args = parser.parse_args()

    labels = read_labels(args.labels).copy()
    ratings = pd.read_csv(args.ratings)
    if "path" not in ratings.columns or "human_rating" not in ratings.columns:
        raise ValueError("ratings CSV must contain path and human_rating columns")

    ratings = ratings[["path", "human_rating"]].copy()
    ratings["human_rating"] = pd.to_numeric(ratings["human_rating"], errors="coerce")
    merged = labels.drop(columns=["human_rating"], errors="ignore").merge(ratings, on="path", how="left")
    ensure_parent(args.output)
    merged.to_csv(args.output, index=False)
    print(f"wrote merged labels to {args.output}")
    print(f"ratings present for {merged['human_rating'].notna().sum()} / {len(merged)} clips")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
