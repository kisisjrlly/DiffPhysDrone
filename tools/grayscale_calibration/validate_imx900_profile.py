#!/usr/bin/env python3
"""Validate profile provenance without claiming that hardware is calibrated."""
import argparse
import json
from pathlib import Path


REQUIRED = ("exposure", "gain", "noise", "motion_blur", "actuator")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text())
    missing = [key for key in REQUIRED if key not in profile]
    if missing:
        raise SystemExit("missing profile sections: " + ", ".join(missing))
    if profile.get("calibrated") is not True:
        print("PASS: schema valid; calibrated flag remains false/pending")
    else:
        print("PASS: schema valid; calibrated=true requires separate hardware-gate evidence")


if __name__ == "__main__":
    main()
