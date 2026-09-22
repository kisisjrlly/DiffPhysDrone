#!/usr/bin/env python3
"""Summarize CSV output from run_simple_active_sensing_closed_loop.py."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_csv", type=Path)
    args = parser.parse_args()
    with args.summary_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("summary CSV is empty")

    groups = defaultdict(list)
    for row in rows:
        groups[row["method"]].append(row)
    print("method,episodes,success_rate,collision_rate,commit_rate,mean_final_goal_dist")
    for method in sorted(groups):
        group = groups[method]
        n = len(group)
        mean = sum(float(r["final_goal_dist"]) for r in group) / n
        print(
            f"{method},{n},"
            f"{sum(float(r['success']) for r in group) / n:.4f},"
            f"{sum(float(r['collided']) for r in group) / n:.4f},"
            f"{sum(float(r['committed']) for r in group) / n:.4f},"
            f"{mean:.4f}"
        )


if __name__ == "__main__":
    main()
