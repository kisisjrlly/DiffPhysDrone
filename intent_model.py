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
                camera_timeline=None):
        x = self.preprocess_gray_input(gray_obs, add_noise=add_noise)
        hx = self.hx_norm(self.gru(torch.tanh(self.stem(x) + self.state_proj(v)), hx))
        action = self.max_acc_cmd * torch.tanh(self.fc(hx))
        if camera_state is None:
            camera_state = v.new_zeros((v.shape[0], 2))
        if camera_motion_state is None:
            camera_motion_state = v.new_zeros((v.shape[0], 6))
        if camera_timeline is None:
            camera_timeline = v.new_zeros((v.shape[0], 6))
        future_velocity = camera_motion_state[:, :3] + action * self.intent_horizon_s / self.nominal_speed
        intent = torch.cat([action / self.max_acc_cmd, future_velocity], -1)
        if intent_override is not None:
            if intent_override.shape != intent.shape:
                raise ValueError("intent_override must have shape [batch,6]")
            intent = intent_override
        if not self.intent_enabled:
            intent = torch.zeros_like(intent)
        cam_input = torch.cat([hx, intent, camera_state, camera_motion_state, camera_timeline], -1)
        camera = torch.sigmoid(self.fc_cam(self.cam_pre(cam_input)))
        return action, camera, hx, None


def build_policy(args, obs_dim=None):
    obs_dim = (7 if args.no_odom else 10) if obs_dim is None else obs_dim
    if getattr(args, "policy_arch", "legacy") == "intent_shared":
        return IntentModel(
            obs_dim, gray_nn_width=args.gray_nn_width, gray_nn_height=args.gray_nn_height,
            intent_enabled=args.camera_use_flight_intent,
            intent_horizon_s=args.intent_horizon_s, max_acc_cmd=args.max_acc_cmd,
            nominal_speed=args.flight_max_speed,
        )
    return Model(obs_dim, 3, include_camera_state_in_obs=args.include_camera_state_in_obs,
                 gray_nn_width=args.gray_nn_width, gray_nn_height=args.gray_nn_height)
