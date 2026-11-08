"""Shared-memory flight policy with action-conditioned camera control.

The intent is a bounded acceleration actually sent to the controller and its
short kinematic velocity prediction, not an unconstrained hidden message.
Camera parameters have no direct route back into the flight head.
"""

import torch
from torch import nn

from model import Model


class IntentModel(nn.Module):
    def __init__(self, dim_obs=10, *, gray_nn_width=48, gray_nn_height=36,
                 intent_enabled=True, intent_horizon_s=0.2, max_acc_cmd=2.5,
                 nominal_speed=2.0):
        super().__init__()
        self.gray_nn_width, self.gray_nn_height = int(gray_nn_width), int(gray_nn_height)
        self.intent_enabled = bool(intent_enabled)
        self.intent_horizon_s = float(intent_horizon_s)
        self.max_acc_cmd, self.nominal_speed = float(max_acc_cmd), float(nominal_speed)
        self.camera_action_dim = 2
        self.uses_camera_timeline = True
        self.stem = nn.Sequential(
            nn.Conv2d(2, 24, 3, stride=2, padding=1), nn.LeakyReLU(0.05),
            nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.LeakyReLU(0.05),
            nn.AdaptiveAvgPool2d((3, 4)), nn.Flatten(),
            nn.Linear(48 * 3 * 4, 128), nn.LayerNorm(128),
        )
        self.state_proj = nn.Linear(dim_obs, 128)
        self.gru = nn.GRUCell(128, 128)
        self.hx_norm = nn.LayerNorm(128)
        self.fc = nn.Linear(128, 3)
        self.cam_pre = nn.Sequential(nn.Linear(128 + 6 + 2 + 6 + 6, 96), nn.LeakyReLU(0.05))
        self.fc_cam = nn.Linear(96, 2)
        with torch.no_grad():
            self.fc.weight.mul_(0.01)
            self.fc.bias.zero_()
            self.fc_cam.weight.mul_(0.01)
            self.fc_cam.bias.copy_(torch.logit(torch.tensor([0.08, 0.05])))

    preprocess_gray_input = Model.preprocess_gray_input

    @staticmethod
    def _is_camera_parameter(name):
        return name.startswith(("cam_pre.", "fc_cam."))

    def reset(self):
        pass

    def initialize_camera_output(self, exposure, gain):
        with torch.no_grad():
            target = self.fc_cam.bias.new_tensor([exposure, gain]).clamp(1e-4, 1 - 1e-4)
            self.fc_cam.bias.copy_(torch.logit(target))

    def freeze_camera_for_flight_only(self):
        names = []
        for name, param in self.named_parameters():
            if self._is_camera_parameter(name):
                param.requires_grad_(False)
                names.append(name)
        return names

    def forward(self, v, hx=None, *, gray_obs=None, add_noise=False, cam_hx=None,
                camera_state=None, camera_motion_state=None, intent_override=None,
                camera_timeline=None, action_noise=None, camera_logit_noise=None,
                return_camera_logits=False, intent_motion_predictor=None):
        encoded = self.encode_observation(v, gray_obs, add_noise=add_noise)
        action, hx = self.flight_from_encoded(encoded, hx)
        if action_noise is not None:
            if action_noise.shape != action.shape:
                raise ValueError("action_noise must have shape [batch,3]")
            action = (action + action_noise).clamp(-self.max_acc_cmd, self.max_acc_cmd)
        if camera_state is None:
            camera_state = v.new_zeros((v.shape[0], 2))
        if camera_motion_state is None:
            camera_motion_state = v.new_zeros((v.shape[0], 6))
        if camera_timeline is None:
            camera_timeline = v.new_zeros((v.shape[0], 6))
        future_velocity = camera_motion_state[:, :3] + action * self.intent_horizon_s / self.nominal_speed
        intent = torch.cat([action / self.max_acc_cmd, future_velocity], -1)
        if intent_motion_predictor is not None:
            intent = intent_motion_predictor(action)
        if intent_override is not None:
            if intent_override.shape != intent.shape:
                raise ValueError("intent_override must have shape [batch,6]")
            intent = intent_override
        if not self.intent_enabled:
            intent = torch.zeros_like(intent)
        cam_input = torch.cat([hx, intent, camera_state, camera_motion_state, camera_timeline], -1)
        mean_logits = self.fc_cam(self.cam_pre(cam_input))
        camera_logits = mean_logits
        if camera_logit_noise is not None:
            if camera_logit_noise.shape != camera_logits.shape:
                raise ValueError("camera_logit_noise must have shape [batch,2]")
            camera_logits = camera_logits + camera_logit_noise
        camera = torch.sigmoid(camera_logits)
        if return_camera_logits:
            return action, camera, hx, None, camera_logits, mean_logits
        return action, camera, hx, None

    def encode_observation(self, state, gray_obs, *, add_noise=False):
        """Batched encoding also used by disclosed flight-only imitation."""
        image = self.preprocess_gray_input(gray_obs, add_noise=add_noise)
        return torch.tanh(self.stem(image) + self.state_proj(state))

    def flight_from_encoded(self, encoded, hx=None):
        hx = self.hx_norm(self.gru(encoded, hx))
        return self.max_acc_cmd * torch.tanh(self.fc(hx)), hx


def build_policy(args, obs_dim=None):
    obs_dim = (7 if args.no_odom else 10) if obs_dim is None else obs_dim
    if getattr(args, "policy_arch", "legacy") == "residual_joint":
        return build_residual_joint_policy(args, obs_dim)
    if getattr(args, "policy_arch", "legacy") == "intent_shared":
        return IntentModel(
            obs_dim, gray_nn_width=args.gray_nn_width, gray_nn_height=args.gray_nn_height,
            intent_enabled=args.camera_use_flight_intent,
            intent_horizon_s=args.intent_horizon_s, max_acc_cmd=args.max_acc_cmd,
            nominal_speed=args.flight_max_speed,
        )
    return Model(obs_dim, 3, include_camera_state_in_obs=args.include_camera_state_in_obs,
                 gray_nn_width=args.gray_nn_width, gray_nn_height=args.gray_nn_height)


def build_residual_joint_policy(args, obs_dim=None):
    """Frozen legacy base + jointly-learned flight/camera residuals.

    Imported lazily so that runs using the plain legacy policy do not pay for
    ``residual_joint_model``'s extra imports.
    """
    import json
    import os

    from residual_joint_model import ResidualJointPolicy, load_legacy_base

    if getattr(args, "residual_flight", True) is False and getattr(args, "residual_camera", True) is False:
        raise ValueError("residual_joint needs at least one enabled residual head")
    checkpoint = getattr(args, "residual_base_checkpoint", None)
    if not checkpoint:
        raise ValueError("--residual_base_checkpoint must be set for policy_arch=residual_joint")
    if not os.path.isfile(checkpoint):
        raise ValueError(f"--residual_base_checkpoint not found: {checkpoint}")
    base_args_path = getattr(args, "residual_base_args", None) or os.path.join(
        os.path.dirname(checkpoint), "args.json"
    )
    if not os.path.isfile(base_args_path):
        raise ValueError(f"residual base args not found: {base_args_path}")
    with open(base_args_path) as handle:
        base_args = json.load(handle)

    # The base must be reconstructed with *its own* observation/vision
    # configuration, not the current run's - the checkpoint only loads strictly
    # when the widths match.
    state_dict = torch.load(checkpoint, map_location="cpu")
    state_dict = {k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
    base = load_legacy_base(
        state_dict,
        include_camera_state_in_obs=bool(base_args.get("include_camera_state_in_obs", False)),
        gray_nn_width=int(base_args.get("gray_nn_width", 48)),
        gray_nn_height=int(base_args.get("gray_nn_height", 36)),
        no_odom=bool(base_args.get("no_odom", False)),
    )
    policy = ResidualJointPolicy(
        base,
        include_camera_state_in_obs=bool(base_args.get("include_camera_state_in_obs", False)),
        no_odom=bool(base_args.get("no_odom", False)),
        gray_nn_width=policy_gray_width(args),
        gray_nn_height=policy_gray_height(args),
        hidden=int(getattr(args, "residual_hidden", 64)),
        flight_residual=bool(getattr(args, "residual_flight", True)),
        camera_residual=bool(getattr(args, "residual_camera", True)),
    )
    policy.base_source = checkpoint
    policy.base_digest = None  # filled below for the protocol lock
    from residual_joint_model import state_dict_digest
    policy.base_digest = state_dict_digest(state_dict)
    return policy


def policy_gray_width(args):
    return int(getattr(args, "gray_nn_width", 48))


def policy_gray_height(args):
    return int(getattr(args, "gray_nn_height", 36))
