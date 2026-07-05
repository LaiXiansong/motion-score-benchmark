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
    parser = argparse.ArgumentParser(description="Create a stratified human-rating sheet.")
    parser.add_argument("--labels", default="data/labels.csv")
    parser.add_argument("--output", default="data/human_rating_sheet.csv")
    parser.add_argument("--clips", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = read_labels(args.labels).copy()
    per_tier = max(args.clips // max(df["tier"].nunique(), 1), 1)
    sampled_parts = []
    for _, group in df.groupby("tier", sort=True):
        n = min(len(group), per_tier)
        sampled_parts.append(group.sample(n=n, random_state=args.seed))
    sampled = pd.concat(sampled_parts, ignore_index=True)

    if len(sampled) < args.clips:
        remaining = df[~df["path"].isin(sampled["path"])]
        extra = remaining.sample(n=min(len(remaining), args.clips - len(sampled)), random_state=args.seed)
        sampled = pd.concat([sampled, extra], ignore_index=True)

    sampled["human_rating"] = ""
    sampled["notes"] = ""
    sampled = sampled[["path", "class", "tier", "human_rating", "notes"]]
    ensure_parent(args.output)
    sampled.to_csv(args.output, index=False)
    print(f"wrote {len(sampled)} rows to {args.output}")
    print("Fill human_rating with integers 1-5, then merge this CSV back into data/labels.csv or pass it to evaluation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
