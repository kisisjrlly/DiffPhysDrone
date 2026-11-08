"""The hard stops must be enforced by the tool, not by discipline.

The plan registers two gates before any arm may run: the frozen base must reach
the goal, and the interaction probe must establish that exposure and motion are
coupled. Both were previously advisory - ``stage_train`` checked only the
protocol lock, so ``--stage train`` could be invoked directly and silently skip
them. These tests pin the enforcement down.

They read artifacts rather than a flag on purpose: the only way to proceed is to
produce evidence that passes, which is the property we want.
"""

import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "run_residual_joint_slit",
    os.path.join(ROOT, "tools", "run_residual_joint_slit.py"),
)
orch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(orch)


def _write_gate(out_dir, *, passed, successes=0, episodes=32):
    payload = {
        "episodes": episodes,
        "successes": successes,
        "fraction": successes / episodes,
        "passed": passed,
    }
    with open(os.path.join(out_dir, "baseline_gate.json"), "w") as handle:
        json.dump(payload, handle)


def _write_probe(out_dir, *, verdict, blocks_training, meaning="test fixture"):
    directory = os.path.join(out_dir, "interaction_probe")
    os.makedirs(directory, exist_ok=True)
    payload = {"classification": {"verdict": verdict,
                                  "blocks_training": blocks_training,
                                  "meaning": meaning}}
    with open(os.path.join(directory, "exposure_motion_probe.json"), "w") as handle:
        json.dump(payload, handle)


def test_missing_baseline_gate_blocks(tmp_path):
    with pytest.raises(SystemExit) as error:
        orch.require_gates(str(tmp_path))
    assert "no baseline gate" in str(error.value)


def test_failed_baseline_gate_blocks(tmp_path):
    _write_gate(tmp_path, passed=False, successes=0)
    _write_probe(tmp_path, verdict="R1_INTERACTION_EXISTS", blocks_training=False)

    with pytest.raises(SystemExit) as error:
        orch.require_gates(str(tmp_path))
    message = str(error.value)
    assert "baseline gate FAILED" in message
    assert "0/32" in message
    # The message must say why this is fatal rather than an inconvenience.
    assert "measuring the base's failure" in message


def test_missing_probe_blocks_even_when_gate_passes(tmp_path):
    _write_gate(tmp_path, passed=True, successes=20)

    with pytest.raises(SystemExit) as error:
        orch.require_gates(str(tmp_path))
    assert "no interaction probe" in str(error.value)


def test_probe_that_blocks_training_blocks(tmp_path):
    _write_gate(tmp_path, passed=True, successes=20)
    _write_probe(tmp_path, verdict="R4_GRID_TRUNCATED", blocks_training=True,
                 meaning="the grid does not bracket the optimum")

    with pytest.raises(SystemExit) as error:
        orch.require_gates(str(tmp_path))
    message = str(error.value)
    assert "R4_GRID_TRUNCATED" in message
    assert "does not bracket" in message


def test_both_gates_passing_allows_training(tmp_path):
    _write_gate(tmp_path, passed=True, successes=20)
    _write_probe(tmp_path, verdict="R1_INTERACTION_EXISTS", blocks_training=False)

    gate, probe = orch.require_gates(str(tmp_path))

    assert gate["passed"] is True
    assert probe["verdict"] == "R1_INTERACTION_EXISTS"


def test_probe_defaults_to_blocking_when_field_absent(tmp_path):
    """A probe file without ``blocks_training`` must not be read as permission.

    Fail closed: an older artifact that predates the field cannot authorise a
    training run.
    """
    _write_gate(tmp_path, passed=True, successes=20)
    directory = os.path.join(tmp_path, "interaction_probe")
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "exposure_motion_probe.json"), "w") as handle:
        json.dump({"classification": {"verdict": "R3_LONG_EXPOSURE_WORSE"}}, handle)

    with pytest.raises(SystemExit) as error:
        orch.require_gates(str(tmp_path))
    assert "did not establish" in str(error.value)
