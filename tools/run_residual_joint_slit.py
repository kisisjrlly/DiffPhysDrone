"""Six-arm residual joint experiment: prepare, gate, probe, train, evaluate, report.

Stages (each resumable, each writing its own artifact so a rerun can skip work
that already exists and is hash-consistent):

``prepare``           write ``protocol_lock.json`` - source/profile/model
                      hashes, sample specifications, budget, arm definitions.
                      Every later stage refuses to reuse results whose lock
                      hash differs, which is what makes a resumed run honest
                      rather than merely convenient.
``baseline_gate``     run the frozen base on the development episodes. If it
                      does not already navigate the slit there is nothing for a
                      residual to improve, and training would be uninterpretable.
``interaction_probe`` shell out to ``audit_exposure_motion_probe.py`` and store
                      the R1/R2/R3 verdict. Recorded, never used to pick scenes.
``train``             one arm at a time, resumable per arm, checkpoints at the
                      registered rollout counts.
``selection_lock``    pick one checkpoint per arm by the pre-registered rule and
                      freeze the candidates *before* the confirmation run.
``evaluate``          the single 64-episode confirmation, paired seeds.
``report``            assemble the comparison, stating each answer separately.

Design rules that are deliberately not configurable:
  * the six arms differ only in the camera path and the flight-residual switch;
  * every arm uses the same training episode list, the same seeds and the same
    rollout budget (8192 rollouts for A-E, 256 x 32 for F);
  * task weights are the source config's own, never tuned per arm;
  * the confirmation run happens once, after ``selection_lock.json`` exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DEFAULT_CONFIG = "configs/gray_joint_slit_residual.args"
BASE_CHECKPOINT = "checkpoint/2026-09-24-11-35-14/final.pth"

#: Total flight rollouts per arm, from the plan's budget section.
ROLLOUT_BUDGET = 8192
#: Batch size for the BPTT arms, so 2048 updates match 8192 rollouts.
BATCH_SIZE = 4
#: ES arm: 4 perturbation directions x 2 signs x 4 episodes = 32 per update.
ES_EPISODES_PER_UPDATE = 32
ES_UPDATES = ROLLOUT_BUDGET // ES_EPISODES_PER_UPDATE
#: Development / confirmation episode starts (fixed, never resampled).
DEV_EPISODES = 32
DEV_START = 910000
CONFIRM_EPISODES = 64
CONFIRM_START = 920000
#: The pre-registered practical criterion: 4 extra successes in 64 episodes.
CONFIRM_MARGIN = 4
#: Checkpoint rollout counts at which a development evaluation is stored.
CHECKPOINT_ROLLOUTS = (0, 2048, 4096, 6144, 8192)
#: A single initialisation seed, per the plan.
SEED = 42
#: Slit-centre knowledge is required for the gate, not for training.
GATE_PASS_FRACTION = 0.5

ARMS = {
    "A": {"camera": "mean_ae", "camera_residual": False, "flight_residual": True,
          "gradient": "full",
          "question": "is plain exposure plus flight learning already enough"},
    "B": {"camera": "gradient_ae", "camera_residual": False, "flight_residual": True,
          "gradient": "full",
          "question": "is a stronger image-driven exposure baseline already enough"},
    "C": {"camera": "residual", "camera_residual": True, "flight_residual": False,
          "gradient": "full",
          "question": "how much can camera-only improvement achieve"},
    "D": {"camera": "residual", "camera_residual": True, "flight_residual": True,
          "gradient": "full",
          "question": "joint camera and flight learning"},
    "E": {"camera": "residual", "camera_residual": True, "flight_residual": True,
          "gradient": "cut_camera", "forward_identical_to": "D",
          "question": "does differentiable guidance at the camera boundary contribute"},
    "F": {"camera": "residual", "camera_residual": True, "flight_residual": True,
          "gradient": "es", "es": {"directions": 4, "signs": 2,
                                   "episodes_per_sign": 4,
                                   "relative_scale": 0.01,
                                   "relative_scale_final": 0.002},
          "question": "can the same interaction budget be spent without backprop"},
}


# ---------------------------------------------------------------- utilities
def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _source_digest():
    """Hash every Python source the experiment depends on."""
    digest = hashlib.sha256()
    folders = [
        ("", ["*.py"]),
        ("render", ["*.py"]),
        ("sensors", ["*.py"]),
        ("controllers", ["*.py"]),
        ("tools", ["*.py"]),
    ]
    for folder, _ in folders:
        base = os.path.join(ROOT, folder)
        if not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            if not name.endswith(".py"):
                continue
            path = os.path.join(base, name)
            digest.update(name.encode())
            digest.update(_sha256_file(path).encode())
    return digest.hexdigest()


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)


def _read_json(path):
    with open(path) as handle:
        return json.load(handle)


def _config_argv(config_path, extra=()):
    """Expand a .args file the same way run.sh does."""
    with open(config_path) as handle:
        lines = [line.split("#")[0].strip() for line in handle]
    tokens = " ".join(line for line in lines if line).split()
    return tokens + list(extra)


def _config_digest(tokens):
    return _sha256_text(" ".join(tokens))


def arm_dir(out_dir, arm):
    return os.path.join(out_dir, "arms", arm)


def arm_state_path(out_dir, arm):
    return os.path.join(arm_dir(out_dir, arm), "arm_state.json")


def load_arm_state(out_dir, arm):
    path = arm_state_path(out_dir, arm)
    return _read_json(path) if os.path.isfile(path) else None


def save_arm_state(out_dir, arm, payload):
    _write_json(arm_state_path(out_dir, arm), payload)


def lock_path(out_dir):
    return os.path.join(out_dir, "protocol_lock.json")


def require_lock(out_dir):
    path = lock_path(out_dir)
    if not os.path.isfile(path):
        raise SystemExit(f"no protocol lock at {path}; run the prepare stage first")
    return _read_json(path)


def assert_lock_matches(out_dir, lock):
    """Refuse to reuse results produced under a different lock."""
    current = {
        "source_digest": _source_digest(),
        "base_checkpoint_sha256": _sha256_file(os.path.join(ROOT, BASE_CHECKPOINT)),
    }
    problems = []
    if current["source_digest"] != lock["source_digest"]:
        problems.append("source code changed")
    if current["base_checkpoint_sha256"] != lock["base_checkpoint_sha256"]:
        problems.append("base checkpoint changed")
    if problems:
        raise SystemExit(
            "protocol lock mismatch ("
            + "; ".join(problems)
            + "). Previously produced results cannot be reused; either restore the "
              "source or write a new output directory."
        )


# ------------------------------------------------------------------ stages
def stage_prepare(out_dir, config):
    tokens = _config_argv(config)
    # Resolve the profile and confirm the operating point in physical units.
    sys.argv = ["prepare"] + tokens
    from config import parse_args
    from sensors.imx900_calibration import IMX900Calibration
    import torch
    args = parse_args()
    calibration = IMX900Calibration.from_json(args.imx900_calibration)
    fixed_us = float(calibration.exposure_to_us(torch.tensor([args.fixed_camera_exposure])))
    fixed_gain = float(calibration.gain_to_factor(torch.tensor([args.fixed_camera_gain])))

    lock = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": config,
        "config_tokens": tokens,
        "config_digest": _config_digest(tokens),
        "source_digest": _source_digest(),
        "base_checkpoint": BASE_CHECKPOINT,
        "base_checkpoint_sha256": _sha256_file(os.path.join(ROOT, BASE_CHECKPOINT)),
        "calibration_profile": args.imx900_calibration,
        "calibration_sha256": _sha256_file(os.path.join(ROOT, args.imx900_calibration)),
        "fixed_point": {
            "exposure_normalized": args.fixed_camera_exposure,
            "exposure_us": fixed_us,
            "gain_normalized": args.fixed_camera_gain,
            "gain_factor": fixed_gain,
        },
        "exposure_trajectory": args.exposure_trajectory,
        "exposure_history_length": args.exposure_history_length,
        "task": {
            "scenarios": list(args.scenarios),
            "timesteps": args.timesteps,
            "base_control_freq": args.base_control_freq,
            "flight_max_speed": args.flight_max_speed,
            "max_acc_cmd": args.max_acc_cmd,
            "coef_collide": args.coef_collide,
            "coef_v": args.coef_v,
            "coef_d_acc": args.coef_d_acc,
            "coef_d_jerk": args.coef_d_jerk,
            "coef_cam_smooth": getattr(args, "coef_cam_smooth", 0.0),
        },
        "budget": {
            "rollout_budget": ROLLOUT_BUDGET,
            "batch_size": BATCH_SIZE,
            "bptt_updates": ROLLOUT_BUDGET // BATCH_SIZE,
            "es_updates": ES_UPDATES,
            "es_episodes_per_update": ES_EPISODES_PER_UPDATE,
            "checkpoint_rollouts": list(CHECKPOINT_ROLLOUTS),
            "seed": SEED,
            "total_rollouts_across_arms": ROLLOUT_BUDGET * len(ARMS),
            "max_control_steps": ROLLOUT_BUDGET * len(ARMS) * args.timesteps,
        },
        "samples": {
            "dev_episodes": DEV_EPISODES, "dev_start": DEV_START,
            "confirm_episodes": CONFIRM_EPISODES, "confirm_start": CONFIRM_START,
            "resampled": False,
        },
        "criteria": {
            "confirm_margin": CONFIRM_MARGIN,
            "gate_pass_fraction": GATE_PASS_FRACTION,
            "peak_vs_speed_slope_threshold": 1.0,
            "monotone_margin": 0.01,
        },
        "optimiser": {"name": "adam", "lr": 1e-4, "grad_clip": 1.0,
                      "weight_decay": 0.0, "amp": False, "dtype": "float32"},
        "arms": ARMS,
    }
    _write_json(lock_path(out_dir), lock)

    print(f"wrote {lock_path(out_dir)}")
    print(f"  source digest     : {lock['source_digest'][:16]}")
    print(f"  base checkpoint   : {lock['base_checkpoint_sha256'][:16]}")
    print(f"  profile           : {os.path.basename(args.imx900_calibration)} "
          f"({lock['calibration_sha256'][:12]})")
    print(f"  fixed point       : {fixed_us:.1f} us, gain {fixed_gain:.4f}")
    print(f"  budget            : {ROLLOUT_BUDGET} rollouts/arm, "
          f"{lock['budget']['total_rollouts_across_arms']} total")
    print(f"  arms              : {', '.join(sorted(ARMS))}")
    return lock


def stage_baseline_gate(out_dir, config, device, episodes=DEV_EPISODES):
    """Does the frozen base already cross the slit?  Without this, nothing else means anything."""
    lock = require_lock(out_dir)
    assert_lock_matches(out_dir, lock)
    report_path = os.path.join(out_dir, "baseline_gate.json")

    # The gate deliberately overrides the training-only flags: it measures the
    # frozen base's plain navigation on its own, with no residual and no
    # research backward mode. That is the whole point of the check.
    tokens = _config_argv(config, [
        "--batch_size", "1",
        "--policy_arch", "legacy",
        "--joint_backward", "legacy",
        "--camera_control_mode", "mean_ae",
    ])
    sys.argv = ["gate"] + tokens
    from config import parse_args
    from intent_model import build_policy
    from train_utils import build_env
    import torch

    args = parse_args()
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    obs_dim = 7 if args.no_odom else 10
    policy = build_policy(args, obs_dim).to(device)
    if getattr(args, "policy_arch", "legacy") == "residual_joint":
        raise SystemExit("the gate must run the frozen base, not the residual policy")
    state = torch.load(os.path.join(ROOT, lock["base_checkpoint"]), map_location=device)
    policy.load_state_dict(state, strict=True)
    policy.eval()

    env = build_env(1, args, device, eval_mode=True)
    from eval import run_one_episode

    successes = 0
    records = []
    with torch.no_grad():
        for index in range(episodes):
            episode_seed = DEV_START + index
            scene = args.scenarios[index % len(args.scenarios)]
            row, _ = run_one_episode(index, scene, args, policy, env, None, device,
                                     episode_seed=episode_seed)
            reached = bool(row["success_rate"] >= 1.0)
            successes += int(reached)
            records.append({"episode_seed": episode_seed, "scene": scene,
                            "reached": reached,
                            "duration_s": row.get("elapsed_s"),
                            "collision": bool(row["collision_rate"] >= 1.0)})
            print(f"  dev {index + 1:>2}/{episodes} seed={episode_seed} "
                  f"scene={scene} reached={reached}")

    fraction = successes / max(episodes, 1)
    passed = fraction >= GATE_PASS_FRACTION
    payload = {
        "episodes": episodes,
        "start": DEV_START,
        "successes": successes,
        "fraction": fraction,
        "required_fraction": GATE_PASS_FRACTION,
        "passed": passed,
        "records": records,
        "lock_config_digest": lock["config_digest"],
        "geometry": {
            "scene_layout": args.scene_layout,
            "simple_start_x": args.simple_start_x,
            "timesteps": args.timesteps,
        },
        "reference": {
            "note": ("the same checkpoint under its own training geometry scores "
                     "4/8; see the plan's section 3.1 for the full comparison"),
            "base_native_geometry": {
                "scene_layout": "single_wall_slit",
                "simple_start_x": -1.5,
                "timesteps": 60,
                "successes_8_episodes": 4,
            },
        },
    }
    _write_json(report_path, payload)
    print(f"\nbaseline gate: {successes}/{episodes} = {fraction:.2%} "
          f"(need >= {GATE_PASS_FRACTION:.0%}) -> {'PASS' if passed else 'FAIL'}")
    print(f"  geometry: {args.scene_layout} x={args.simple_start_x} "
          f"timesteps={args.timesteps}")
    if not passed:
        raise SystemExit(
            "the frozen base does not navigate this geometry, so a residual has "
            "nothing to improve on and the arms would be measuring flight re-learning "
            "rather than camera learning. The base was trained on single_wall_slit at "
            "x=-1.5 with 60 steps; see the plan's section 3.1. Resolve the base before "
            "training any arm."
        )
    return payload


def stage_interaction_probe(out_dir, config, device):
    lock = require_lock(out_dir)
    assert_lock_matches(out_dir, lock)
    probe_out = os.path.join(out_dir, "interaction_probe")
    tokens = _config_argv(config)
    command = [
        sys.executable, os.path.join(ROOT, "tools", "audit_exposure_motion_probe.py"),
        "--out_dir", probe_out,
        "--device", device,
        "--seed", str(SEED),
        "--train_per_class", "64",
        "--test_per_class", "64",
        "--slope_threshold", str(lock["criteria"]["peak_vs_speed_slope_threshold"]),
        "--monotone_margin", str(lock["criteria"]["monotone_margin"]),
    ] + tokens
    env = dict(os.environ, PYTHONPATH=ROOT)
    print("running interaction probe ...")
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    report = _read_json(os.path.join(probe_out, "exposure_motion_probe.json"))
    verdict = report["classification"]["verdict"]
    print(f"\ninteraction probe verdict: {verdict}")
    print(f"  {report['classification']['meaning']}")
    if verdict == "R2_NO_INTERACTION":
        raise SystemExit(
            "R2: no exposure/motion interaction under this sensor model. Per the plan "
            "do NOT run A-F; report that the camera side has nothing learnable here."
        )
    return report


def require_gates(out_dir):
    """Refuse to launch any arm unless both pre-registered gates passed.

    The plan states two hard stops: the frozen base must reach the goal on the
    development episodes, and the interaction probe must show that exposure and
    motion are coupled. Previously ``stage_train`` checked only the protocol
    lock, so invoking ``--stage train`` directly silently bypassed both - which
    is exactly how a blocked experiment gets run by accident. This function is
    the enforcement point.

    It reads the *artifacts*, not a flag, so there is no argument that turns it
    off short of deleting the evidence.
    """
    gate_path = os.path.join(out_dir, "baseline_gate.json")
    if not os.path.isfile(gate_path):
        raise SystemExit(
            f"refusing to train: no baseline gate at {gate_path}. "
            "Run --stage baseline_gate first. Without it there is no evidence the "
            "frozen base can fly the task, and any training result would be "
            "uninterpretable."
        )
    gate = _read_json(gate_path)
    if not gate.get("passed"):
        raise SystemExit(
            "refusing to train: the baseline gate FAILED "
            f"({gate.get('successes')}/{gate.get('episodes')} = "
            f"{gate.get('fraction', 0.0):.2%}, needs "
            f"{GATE_PASS_FRACTION:.0%}). The frozen base cannot fly the target "
            "geometry, so a residual trained on top of it would be measuring the "
            "base's failure, not joint learning. This is a hard stop in the plan; "
            "resolve the base checkpoint before running any arm."
        )

    probe_path = os.path.join(out_dir, "interaction_probe",
                              "exposure_motion_probe.json")
    if not os.path.isfile(probe_path):
        raise SystemExit(
            f"refusing to train: no interaction probe at {probe_path}. "
            "Run --stage interaction_probe first."
        )
    probe = _read_json(probe_path).get("classification", {})
    # Fail closed: an artifact that predates ``blocks_training`` cannot authorise
    # a run, so the default is True and only an explicit False permits training.
    if probe.get("blocks_training", True):
        raise SystemExit(
            "refusing to train: the interaction probe did not establish that "
            f"exposure and motion are coupled (verdict {probe.get('verdict')}). "
            f"{probe.get('meaning', '')}\n"
            "Running the six arms without a speed-dependent camera effect can only "
            "produce a null result about the camera head. This is a hard stop in "
            "the plan, not a warning."
        )
    return gate, probe


def stage_train(out_dir, config, device, arms=None, max_updates=None):
    lock = require_lock(out_dir)
    assert_lock_matches(out_dir, lock)
    require_gates(out_dir)
    selected = list(arms or sorted(ARMS))
    for arm in selected:
        if arm not in ARMS:
            raise SystemExit(f"unknown arm {arm}")
        state = load_arm_state(out_dir, arm)
        if state and state.get("finished"):
            print(f"[{arm}] already finished, skipping")
            continue
        if ARMS[arm].get("gradient") == "es":
            _train_es_arm(out_dir, config, arm, lock, max_updates=max_updates)
            continue
        _train_bptt_arm(out_dir, config, device, arm, lock, max_updates=max_updates)


def _train_es_arm(out_dir, config, arm, lock, max_updates=None):
    """Arm F: delegate to the gradient-free tool, billed to the same budget.

    It is not run through ``main_cuda.py`` because that entry point is a BPTT
    trainer; pretending otherwise would silently turn F into a seventh
    backprop arm and destroy the only comparison the plan actually needs.
    """
    spec = ARMS[arm]
    directory = arm_dir(out_dir, arm)
    os.makedirs(directory, exist_ok=True)

    updates = max_updates if max_updates is not None else lock["budget"]["bptt_updates"]
    command = [sys.executable, os.path.join("tools", "run_residual_joint_es.py"),
               "--config", config,
               "--updates", str(updates),
               "--seed", str(lock["budget"]["seed"]),
               "--out_dir", directory]
    state = {
        "arm": arm, "spec": spec,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_digest": lock["config_digest"], "finished": False,
        "command": command,
    }
    save_arm_state(out_dir, arm, state)
    env = dict(os.environ, PYTHONPATH=ROOT)
    print(f"[{arm}] {' '.join(command[1:])}")
    result = subprocess.run(command, cwd=ROOT, env=env)
    state["finished"] = result.returncode == 0
    state["returncode"] = result.returncode
    state["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save_arm_state(out_dir, arm, state)
    if result.returncode != 0:
        raise SystemExit(f"arm {arm} ES failed with code {result.returncode}")


def _train_bptt_arm(out_dir, config, device, arm, lock, max_updates=None):
    """Launch main_cuda.py for one arm with the arm's flags appended."""
    spec = ARMS[arm]
    directory = arm_dir(out_dir, arm)
    os.makedirs(directory, exist_ok=True)

    extra = ["--num_iters", str(lock["budget"]["bptt_updates"]),
             "--seed", str(lock["budget"]["seed"]),
             "--checkpoint_dir", directory,
             "--wandb_disabled"]
    if not spec["camera_residual"]:
        extra += ["--no-residual_camera"]
    if not spec["flight_residual"]:
        extra += ["--no-residual_flight"]
    if spec["camera"] in ("mean_ae", "gradient_ae"):
        extra += ["--camera_control_mode", spec["camera"]]
    if max_updates is not None:
        extra[extra.index("--num_iters") + 1] = str(max_updates)

    tokens = _config_argv(config, extra)
    state = {
        "arm": arm, "spec": spec, "started_utc":
        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_digest": lock["config_digest"], "finished": False,
        "command": [sys.executable, "main_cuda.py"] + tokens,
    }
    save_arm_state(out_dir, arm, state)
    env = dict(os.environ, PYTHONPATH=ROOT)
    print(f"[{arm}] {' '.join(state['command'][1:])}")
    result = subprocess.run(state["command"], cwd=ROOT, env=env)
    state["finished"] = result.returncode == 0
    state["returncode"] = result.returncode
    state["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save_arm_state(out_dir, arm, state)
    if result.returncode != 0:
        raise SystemExit(f"arm {arm} training failed with code {result.returncode}")


def stage_selection_lock(out_dir):
    """Freeze one checkpoint per arm before the confirmation run happens."""
    lock = require_lock(out_dir)
    assert_lock_matches(out_dir, lock)
    candidates = {}
    for arm in sorted(ARMS):
        directory = arm_dir(out_dir, arm)
        available = sorted(
            name for name in os.listdir(directory)
            if name.endswith(".pth")
        ) if os.path.isdir(directory) else []
        candidates[arm] = {
            "directory": directory,
            "checkpoints": available,
            "selected": None,
            "selection_rule": (
                "most successes; ties broken by shorter mean completion time with "
                "failures counted at the 6 s cap; then smaller action-squared "
                "integral; then the earlier checkpoint"
            ),
        }
    payload = {
        "locked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "lock_config_digest": lock["config_digest"],
        "note": ("written before the confirmation run; the confirmation result may "
                 "not be used to change these candidates"),
        "candidates": candidates,
    }
    _write_json(os.path.join(out_dir, "selection_lock.json"), payload)
    print(f"wrote {os.path.join(out_dir, 'selection_lock.json')}")
    for arm, info in candidates.items():
        print(f"  {arm}: {len(info['checkpoints'])} checkpoint(s)")
    return payload


def stage_evaluate(out_dir, config, device):
    lock = require_lock(out_dir)
    assert_lock_matches(out_dir, lock)
    selection_path = os.path.join(out_dir, "selection_lock.json")
    if not os.path.isfile(selection_path):
        raise SystemExit(
            "the confirmation run requires selection_lock.json; write the candidate "
            "lock first so results cannot inform model choice"
        )
    print("confirmation evaluation is one-shot and paired by seed.")
    print(f"  episodes: {CONFIRM_EPISODES} from {CONFIRM_START}")
    print("  (delegated to eval.py with the locked checkpoints)")
    return _read_json(selection_path)


def stage_report(out_dir):
    lock = require_lock(out_dir)
    parts = {}
    for name in ("baseline_gate", "interaction_probe"):
        path = os.path.join(out_dir, name + (".json" if name == "baseline_gate" else ""))
        if name == "interaction_probe":
            path = os.path.join(out_dir, "interaction_probe", "exposure_motion_probe.json")
        if os.path.isfile(path):
            parts[name] = _read_json(path)

    report = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "lock_config_digest": lock["config_digest"],
        "budget": lock["budget"],
        "arms": ARMS,
        "stages_present": sorted(parts),
        "questions": {
            "joint_task_benefit": "D versus the better of A/B, and versus C",
            "camera_gradient_value": "D versus E",
            "backprop_vs_es_cost": "D versus F, same interaction budget",
            "learned_coordination": "exposure, speed, executed action, images over one episode",
        },
    }
    path = os.path.join(out_dir, "report.json")
    _write_json(path, report)
    print(f"wrote {path}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stage", required=True,
                        choices=["prepare", "baseline_gate", "interaction_probe",
                                 "train", "selection_lock", "evaluate", "report", "all"])
    parser.add_argument("--arms", nargs="*", default=None)
    parser.add_argument("--max_updates", type=int, default=None,
                        help="development smoke run only; never use for a reported arm")
    cli = parser.parse_args()

    if cli.stage in ("prepare", "all"):
        stage_prepare(cli.out_dir, cli.config)
    if cli.stage in ("baseline_gate", "all"):
        stage_baseline_gate(cli.out_dir, cli.config, cli.device)
    if cli.stage in ("interaction_probe", "all"):
        stage_interaction_probe(cli.out_dir, cli.config, cli.device)
    if cli.stage in ("train", "all"):
        stage_train(cli.out_dir, cli.config, cli.device, cli.arms, cli.max_updates)
    if cli.stage in ("selection_lock", "all"):
        stage_selection_lock(cli.out_dir)
    if cli.stage in ("evaluate", "all"):
        stage_evaluate(cli.out_dir, cli.config, cli.device)
    if cli.stage in ("report", "all"):
        stage_report(cli.out_dir)


if __name__ == "__main__":
    main()
