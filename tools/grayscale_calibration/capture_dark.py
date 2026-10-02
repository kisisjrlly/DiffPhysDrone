#!/usr/bin/env python3
"""Capture dark-frame metadata and pixel statistics from a camera adapter."""
import argparse
import json
from pathlib import Path

import numpy as np

from tools.realflight.imx900_camera import Imx900Camera, append_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="JSONL metadata output")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--exposure-us", type=float, required=True)
    parser.add_argument("--gain", type=float, required=True)
    parser.add_argument("--device", help="Reserved for target-specific adapter")
    args = parser.parse_args()
    raise SystemExit(
        "No generic capture backend is selected. Construct Imx900Camera with the "
        "target Jetson backend and call this routine from the deployment node."
    )


if __name__ == "__main__":
    main()
