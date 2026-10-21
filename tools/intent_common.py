"""Configuration helpers shared by the intent benchmark's audit tools."""

import shlex
from pathlib import Path

from config import build_parser, parse_scenarios, validate_args


def load_intent_args(config="configs/gray_joint_intent.args", **overrides):
    args = build_parser().parse_args(shlex.split(Path(config).read_text(), comments=True))
    for name, value in overrides.items():
        setattr(args, name, value)
    args.scenarios = parse_scenarios(args.scenarios)
    validate_args(args)
    return args
