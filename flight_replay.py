"""Retain a common actor's post-measurement flight skills during fine tuning.

Replay uses the existing common flight dataset. Targets are the INITIAL actor's
own flight outputs, not new privileged labels. Initial measurement actions stay
unlabelled. This is an optional training regularizer, never an inference module.
"""

import copy
import hashlib
import json
import time
from pathlib import Path

import torch


def flight_sequence(model, data, indices, device):
    current = data["frames"][indices].to(device=device, dtype=torch.float32)
    previous = torch.cat([current[:, :1], current[:, :-1]], 1)
    images = torch.cat([current, previous], 2)
    state = data["state"][indices].to(device)
    batch, steps = current.shape[:2]
    encoded = model.encode_observation(state.flatten(0, 1), images.flatten(0, 1)).reshape(batch, steps, -1)
    hx, actions = None, []
    for step in range(steps):
        action, hx = model.flight_from_encoded(encoded[:, step], hx)
        actions.append(action)
    return torch.stack(actions, 1)


def masked_action_anchor(actions, targets, valid):
    squared = (actions - targets).square().mean(-1)
    return (squared * valid).sum() / valid.sum().clamp_min(1)


class FlightReplayAnchor:
    @staticmethod
    def validate_configuration(stored, args):
        for key, value in stored["dataset_configuration"].items():
            if key.startswith("imitation_"):
                continue
            actual = hashlib.sha256(Path(args.imx900_calibration).read_bytes()).hexdigest() if key == "calibration_sha256" else getattr(args, key)
            if actual != value:
                raise ValueError(f"replay dataset configuration mismatch: {key}")

    def __init__(self, model, args, device, directory):
        start = time.perf_counter()
        path = Path(args.flight_replay_dataset).resolve()
        stored = torch.load(path, map_location="cpu")
        self.validate_configuration(stored, args)
        self.data = stored["data"]
        self.device = device
        self.batch = args.flight_replay_batch
        self.generator = torch.Generator().manual_seed(args.seed + 9381)
        probe = int(stored["metadata"].get("unlabelled_initial_probe_steps", 0))
        if args.pole_layout == "parallax_pair" and (probe < 24 or self.data["valid"][:, :probe].any()):
            raise ValueError("parallax replay must not supervise initial measurement actions")
        # Precompute once, then release the temporary reference actor.
        reference = copy.deepcopy(model).eval().requires_grad_(False)
        targets = []
        with torch.no_grad():
            for start_idx in range(0, len(self.data["frames"]), 8):
                ids = torch.arange(start_idx, min(start_idx + 8, len(self.data["frames"])))
                targets.append(flight_sequence(reference, self.data, ids, device).cpu())
        self.targets = torch.cat(targets)
        del reference
        self.sequence_samples = 0
        self.valid_samples = 0
        self.manifest = {
            "dataset": str(path), "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "targets_sha256": hashlib.sha256(self.targets.numpy().tobytes()).hexdigest(),
            "reference_checkpoint_sha256": hashlib.sha256(Path(args.resume).read_bytes()).hexdigest(),
            "target_source": "initial common actor's own flight outputs on common replay observations",
            "unlabelled_initial_steps": probe, "camera_labels": False,
            "extra_deployed_parameters": 0, "weight": args.flight_replay_weight,
            "batch": self.batch, "seed": args.seed + 9381,
            "precompute_wall_time_s": time.perf_counter() - start,
            "independent_of_rollout_rng": True,
        }
        self.directory = Path(directory)
        self.save()

    def loss(self, model):
        ids = torch.randint(len(self.targets), (self.batch,), generator=self.generator)
        actions = flight_sequence(model, self.data, ids, self.device)
        valid = self.data["valid"][ids].to(self.device)
        value = masked_action_anchor(actions, self.targets[ids].to(self.device), valid)
        self.sequence_samples += self.batch
        self.valid_samples += int(valid.sum())
        return value

    def save(self):
        value = dict(self.manifest, replay_sequence_samples=self.sequence_samples,
                     replay_valid_action_samples=self.valid_samples)
        (self.directory / "flight_replay_manifest.json").write_text(json.dumps(value, indent=2) + "\n")
