#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], dry_run: bool = False) -> None:
    print("+ " + " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the motion benchmark pipeline.")
    parser.add_argument("--labels", default="data/labels.csv")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--flow-backend", default="auto", choices=["auto", "sea-raft", "searaft", "farneback"])
    parser.add_argument("--sea-raft-root", default=None)
    parser.add_argument("--sea-raft-checkpoint", default=None)
    parser.add_argument("--learned-checkpoint", default=None)
    parser.add_argument("--skip-pose", action="store_true")
    parser.add_argument("--skip-flow", action="store_true")
    parser.add_argument("--skip-learned", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    results = Path(args.results_dir)
    pose_csv = results / "pose_scores.csv"
    boxes_json = results / "pose_boxes.json"
    flow_csv = results / "flow_scores.csv"
    learned_csv = results / "learned_scores.csv"

    score_csvs = []
    if not args.skip_pose:
        run(
            [
                sys.executable,
                "-m",
                "src.pose_score",
                "--labels",
                args.labels,
                "--output",
                str(pose_csv),
                "--boxes-json",
                str(boxes_json),
                "--device",
                args.device,
            ],
            args.dry_run,
        )
        score_csvs.append(str(pose_csv))
    elif pose_csv.exists():
        score_csvs.append(str(pose_csv))

    if not args.skip_flow:
        cmd = [
            sys.executable,
            "-m",
            "src.flow_score",
            "--labels",
            args.labels,
            "--output",
            str(flow_csv),
            "--backend",
            args.flow_backend,
            "--device",
            args.device,
        ]
        if boxes_json.exists() or not args.skip_pose:
            cmd += ["--boxes-json", str(boxes_json)]
        if args.sea_raft_root:
            cmd += ["--sea-raft-root", args.sea_raft_root]
        if args.sea_raft_checkpoint:
            cmd += ["--sea-raft-checkpoint", args.sea_raft_checkpoint]
        run(cmd, args.dry_run)
        score_csvs.append(str(flow_csv))
    elif flow_csv.exists():
        score_csvs.append(str(flow_csv))

    if not args.skip_learned and args.learned_checkpoint:
        run(
            [
                sys.executable,
                "-m",
                "src.learned_score",
                "--labels",
                args.labels,
                "--checkpoint",
                args.learned_checkpoint,
                "--output",
                str(learned_csv),
                "--device",
                args.device,
            ],
            args.dry_run,
        )
        score_csvs.append(str(learned_csv))
    elif learned_csv.exists():
        score_csvs.append(str(learned_csv))

    eval_cmd = [
        sys.executable,
        "-m",
        "src.evaluate",
        "--labels",
        args.labels,
        "--output",
        str(results / "scores.csv"),
        "--metrics",
        str(results / "metrics.json"),
        "--plots-dir",
        str(results / "plots"),
    ]
    for csv_path in score_csvs:
        eval_cmd += ["--score-csv", csv_path]
    run(eval_cmd, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
