#!/usr/bin/env python3
"""Entry point reserved for controlled motion-blur capture on Jetson."""
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.parse_args()
    raise SystemExit("hardware motion platform is required; no blur data were generated")


if __name__ == "__main__":
    main()
