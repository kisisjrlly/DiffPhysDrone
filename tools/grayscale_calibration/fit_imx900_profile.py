#!/usr/bin/env python3
"""Fit a conservative profile from a calibration summary JSON.

The generated profile remains ``calibrated: false`` until hardware gates and
holdout validation have been reviewed by an operator.
"""
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    profile = dict(summary.get("profile", summary))
    profile["calibrated"] = False
    profile["calibration_status"] = "pending_hardware_gate"
    profile["source_summary"] = str(args.summary.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, indent=2) + "\n")


if __name__ == "__main__":
    main()
