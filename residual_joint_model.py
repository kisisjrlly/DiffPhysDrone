"""Frozen legacy flight policy plus a small jointly-learned residual.

Structure
---------
The base policy is the verified legacy ``Model``, loaded strictly and kept in
evaluation mode with ``requires_grad=False``. Freezing the *parameters* is not
the same as freezing the *behaviour*: the base is still run with gradients
enabled, so

    task loss -> base flight action -> rendered grayscale frame
              -> camera residual / flight residual

remains a single differentiable chain. Never wrap the base in ``no_grad`` and
never detach the images on the way in; that would sever exactly the path this
experiment is about.

Two independent residual heads sit beside it:

* **flight** - a 64-wide GRU over (image feature, the existing motion/target
  observation, pending flight commands). It must not see exposure, gain, camera
  memory or any hidden scene variable, so that a flight improvement cannot be
  attributed to privileged camera information.
* **camera** - a separate 64-wide GRU over (image, motion observation, the
  flight memory, the flight command just proposed, the pending camera queue,
  the exposure/gain actually in force, and the camera-time information). Its
  state never flows back into the flight branch.

Both heads are zero-initialised, so at step 0 the composite policy reproduces
the base flight action and the base camera command exactly. Any measured change
is therefore attributable to learning, not to a lucky initialisation.

Camera mapping
--------------
The plan specifies::

    T = clip(T_AE * exp(log(500) * tanh(r_T)), 0.1, 50)   # milliseconds
    G = clip(G_AE * exp(log(8)   * tanh(r_G)), 1,   8)

``T_AE``/``G_AE`` are the physical values a mean-brightness auto-exposure
controller would ask for. The result is mapped back through the *active
profile* so the request is a normalized camera command, never milliseconds
used directly as ``exposure01``.

Why residuals rather than a fresh policy
----------------------------------------
The base already navigates the slit. Relearning it from scratch would confound
"we can learn to see" with "we can relearn to fly". A zero-initialised residual
can only improve on the base, and when it fails the failure is informative:
the residual was not able to help, rather than the optimiser was lost.
"""

from __future__ import annotations

import hashlib
import math

import torch
import torch.nn as nn

from controllers.physical_auto_exposure import physical_mean_auto_exposure_target
from model import Model

#: Camera residual range: multiplicative factor on the AE exposure suggestion.
#: ``exp(log(500) * tanh(r))`` spans 1/500x to 500x, i.e. the 0.1-50 ms range
#: the plan specifies once applied to a ~1 ms AE suggestion.
CAMERA_EXPOSURE_RANGE = 500.0
#: Gain residual range: multiplicative factor on the AE gain suggestion.
CAMERA_GAIN_RANGE = 8.0
LOG_EXPOSURE_RANGE = math.log(CAMERA_EXPOSURE_RANGE)
LOG_GAIN_RANGE = math.log(CAMERA_GAIN_RANGE)
#: Flight residual bound per axis, in m/s^2.
FLIGHT_RESIDUAL_MAX = 1.1
#: Physical exposure bounds for the camera mapping, in milliseconds.
EXPOSURE_MS_MIN = 0.1
EXPOSURE_MS_MAX = 50.0
#: Physical gain factor bounds for the camera mapping.
GAIN_FACTOR_MIN = 1.0
GAIN_FACTOR_MAX = 8.0


def load_legacy_base(state_dict, *, include_camera_state_in_obs, gray_nn_width,
                     gray_nn_height, no_odom=False):
    """Build the legacy ``Model`` and load ``state_dict`` strictly.

    Strict on purpose: a shape mismatch between the checkpoint and the
    reconstructed architecture means the assumed configuration is wrong, and
    silently loading a partial model would invalidate every comparison.
    """
    model = Model(
        dim_obs=flight_obs_dim(no_odom=no_odom),
        include_camera_state_in_obs=include_camera_state_in_obs,
        gray_nn_width=gray_nn_width,
        gray_nn_height=gray_nn_height,
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def flight_obs_dim(*, no_odom=False):
    """Width of the observation vector *before* the camera block is appended.

    ``Model.__init__`` adds two camera channels itself when
    ``include_camera_state_in_obs`` is set, so this must match what
    ``build_policy`` passes: 7 without odometry, 10 with it. Passing 8 here
    (the full non-camera slice width) made ``v_proj`` 8-wide and broke the
    strict load of the 10-wide checkpoint.
    """
    return 7 if no_odom else 10


def flight_state_slice(*, include_camera_state_in_obs=False, no_odom=False):
    """The part of the observation the flight residual is allowed to see.

    This is the real end of the vector when camera state is carried in the
    observation (``state[:8]``), otherwise the whole thing. It is a *slice
    width*, deliberately different from ``flight_obs_dim``: the latter sizes a
    ``Model`` parameter, this one sizes a feature we concatenate.
    """
    width = flight_obs_dim(no_odom=no_odom)
    if include_camera_state_in_obs:
        return width - 2
    return width


def state_dict_digest(state_dict) -> str:
    """Order-independent digest of a state dict, for the protocol lock."""
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        value = state_dict[key]
        digest.update(key.encode("utf-8"))
        digest.update(bytes(str(tuple(value.shape)), "utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class _ImageEncoder(nn.Module):
    """Small CNN over the same two-frame 48x36 input the base sees."""

    def __init__(self, in_channels=2, width=(8, 16, 32), out_dim=64):
        super().__init__()
        c1, c2, c3 = width
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, padding=1, bias=False),
            nn.LeakyReLU(0.05),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.05),
            nn.Conv2d(c2, c3, 3, padding=1, bias=False),
            nn.LeakyReLU(0.05),
            nn.AdaptiveAvgPool2d((3, 4)),
        )
        self.proj = nn.Linear(c3 * 3 * 4, out_dim)

    def forward(self, image):
        return self.proj(self.net(image).flatten(1))


class ResidualJointPolicy(nn.Module):
    """Legacy base policy with zero-initialised flight and camera residuals.

    Drop-in for ``policy_forward``: it accepts the same keyword arguments the
    rollout already passes, plus ``state_observation`` (the policy's own input
    vector) which the residual GRUs need and which ``policy_forward`` has on
    hand as ``state``.
    """

    uses_camera_timeline = True
    #: ``policy_forward`` should forward the policy's own observation vector so
    #: the residual GRUs do not have to re-derive it.
    needs_state_observation = True

    def __init__(
        self,
        base: Model,
        *,
        include_camera_state_in_obs=False,
        no_odom=False,
        gray_nn_width=48,
        gray_nn_height=36,
        hidden=64,
        flight_residual=True,
        camera_residual=True,
        target_luma=0.35,
    ):
        super().__init__()
        self.base = base
        self.include_camera_state_in_obs = bool(include_camera_state_in_obs)
        self.no_odom = bool(no_odom)
        self.gray_nn_width = int(gray_nn_width)
        self.gray_nn_height = int(gray_nn_height)
        self.hidden = int(hidden)
        self.flight_residual_enabled = bool(flight_residual)
        self.camera_residual_enabled = bool(camera_residual)
        self.target_luma = float(target_luma)
        self._calibration = None
        self._flight_memory = None
        self._camera_memory = None

        # Mirrors the base so the rest of the pipeline (queue-aligned intent,
        # action decoding, camera delay) reads the same attributes.
        self.max_acc_cmd = float(getattr(base, "max_acc_cmd", 2.2))
        self.camera_action_dim = 2
        self.nominal_speed = float(getattr(base, "nominal_speed", 1.0))
        self.intent_horizon_s = float(getattr(base, "intent_horizon_s", 0.5))

        obs_dim = flight_state_slice(
            include_camera_state_in_obs=include_camera_state_in_obs, no_odom=no_odom
        )
        self.obs_dim = obs_dim

        self.image_encoder = _ImageEncoder(out_dim=hidden)

        # Flight memory: image + motion/target observation + pending commands.
        # Deliberately no exposure/gain/camera memory, so a flight gain cannot
        # be read as privileged camera information.
        flight_input = hidden + obs_dim + 3
        self.flight_gru = nn.GRUCell(flight_input, hidden)
        self.flight_norm = nn.LayerNorm(hidden)
        self.flight_head = nn.Linear(hidden, 3)
        nn.init.zeros_(self.flight_head.weight)
        nn.init.zeros_(self.flight_head.bias)

        # Camera memory: image, motion, flight memory, proposed command,
        # pending queue, applied E/G, camera timeline (6).
        camera_input = hidden + obs_dim + hidden + 3 + 2 + 2 + 6
        self.camera_gru = nn.GRUCell(camera_input, hidden)
        self.camera_norm = nn.LayerNorm(hidden)
        self.camera_head = nn.Linear(hidden, 2)
        nn.init.zeros_(self.camera_head.weight)
        nn.init.zeros_(self.camera_head.bias)

    # ------------------------------------------------------------------ setup
    def freeze_base(self):
        self.base.eval()
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def freeze_flight_residual(self):
        self.flight_residual_enabled = False

    def freeze_camera_residual(self):
        self.camera_residual_enabled = False

    def freeze_camera_for_flight_only(self):
        """Flight-only training: freeze the camera residual memory and head.

        ``base.*`` tensors are already frozen. The image encoder is shared by
        both branches, so it stays trainable here - freezing it would also
        freeze the flight residual's only view of the world.
        """
        frozen = []
        for name, parameter in self.named_parameters():
            if name.startswith("base."):
                continue
            if name.startswith("camera_"):
                parameter.requires_grad_(False)
                frozen.append(name)
        return frozen

    def freeze_flight_for_camera_only(self):
        """Camera-only training: freeze the flight residual memory and head."""
        frozen = []
        for name, parameter in self.named_parameters():
            if name.startswith("base."):
                continue
            if name.startswith("flight_"):
                parameter.requires_grad_(False)
                frozen.append(name)
        return frozen

    def initialize_camera_output(self, exposure, gain):
        """The residual already starts at plain AE, so this is a no-op.

        Kept so ``main_cuda`` can call it uniformly; the base's own camera head
        is frozen and never used as the residual's starting point.
        """
        self.reset_camera_parameters_from_seed(0, exposure, gain)

    @staticmethod
    def _is_camera_parameter(name):
        """Classify a parameter for the manifest hash and freeze helpers."""
        if name.startswith("base."):
            return Model._is_camera_parameter(name[len("base."):])
        return name.startswith(("camera_",))

    def trainable_parameter_names(self):
        """Names of parameters that actually receive gradient updates."""
        names = []
        for name, parameter in self.named_parameters():
            if name.startswith("base."):
                continue
            if not parameter.requires_grad:
                continue
            if not self.flight_residual_enabled and name.startswith("flight_"):
                continue
            if not self.camera_residual_enabled and name.startswith("camera_"):
                continue
            names.append(name)
        return names

    def trainable_parameter_count(self):
        wanted = set(self.trainable_parameter_names())
        return sum(p.numel() for n, p in self.named_parameters() if n in wanted)

    def reset(self):
        self._flight_memory = None
        self._camera_memory = None
        self.reset_camera_parameters_from_seed(0, 0.5, 0.5)

    def reset_camera_parameters_from_seed(self, seed, exposure, gain):
        self._camera_bias = (float(exposure), float(gain))

    # --------------------------------------------------------------- forward
    def forward(
        self,
        v,
        hx=None,
        *,
        gray_obs=None,
        add_noise=False,
        cam_hx=None,
        camera_state=None,
        camera_motion_state=None,
        pending_flight_commands=None,
        camera_timeline=None,
        action_noise=None,
        camera_logit_noise=None,
        return_camera_logits=False,
        intent_motion_predictor=None,
        intent_override=None,
        state_observation=None,
    ):
        """Return the same 4- or 6-tuple shape the base policy returns."""
        base_out = self.base(
            v,
            hx,
            gray_obs=gray_obs,
            add_noise=add_noise,
            cam_hx=cam_hx,
            camera_state=camera_state,
            camera_motion_state=camera_motion_state,
        )
        base_action, base_camera, hx_out, cam_hx_out = base_out

        image = self.base.preprocess_gray_input(gray_obs, add_noise=add_noise)
        feature = self.image_encoder(image)
        batch = image.shape[0]
        observation = self._observation_slice(v, state_observation)

        # ---- flight residual --------------------------------------------
        pending = pending_flight_commands
        if pending is None:
            pending = v.new_zeros(batch, 3)
        flight_input = torch.cat([feature, observation, pending], -1)
        flight_memory = self.flight_gru(flight_input, self._flight_state(batch, v))
        flight_memory = self.flight_norm(flight_memory)
        self._flight_memory = flight_memory
        flight_correction = self._flight_correction(flight_memory)

        action = base_action
        if action_noise is not None:
            action = (action + action_noise).clamp(-self.max_acc_cmd, self.max_acc_cmd)
        residual_action = (action + flight_correction).clamp(
            -self.max_acc_cmd, self.max_acc_cmd
        )

        # ---- camera residual --------------------------------------------
        if camera_state is None:
            camera_state = v.new_zeros(batch, 2)
        if camera_motion_state is None:
            camera_motion_state = v.new_zeros(batch, 6)
        if camera_timeline is None:
            camera_timeline = v.new_zeros(batch, 6)
        applied = (camera_state.detach() + 1.0) * 0.5
        ae_suggestion = self._auto_exposure_suggestion(
            gray_obs, applied[:, 0], applied[:, 1]
        )
        camera_input = torch.cat(
            [
                feature,
                observation,
                flight_memory,
                residual_action,
                applied,
                ae_suggestion,
                camera_timeline,
            ],
            -1,
        )
        camera_memory = self.camera_gru(
            camera_input, self._camera_state(batch, v)
        )
        camera_memory = self.camera_norm(camera_memory)
        self._camera_memory = camera_memory
        camera_residual = self._camera_correction(camera_memory)

        mean_logits = self._camera_logits(ae_suggestion, camera_residual)
        camera_logits = mean_logits
        if camera_logit_noise is not None:
            camera_logits = camera_logits + camera_logit_noise
        camera = torch.sigmoid(camera_logits)

        if return_camera_logits:
            return residual_action, camera, hx_out, cam_hx_out, camera_logits, mean_logits
        return residual_action, camera, hx_out, cam_hx_out

    # ------------------------------------------------------------- internals
    def _flight_state(self, batch, reference):
        if self._flight_memory is None or self._flight_memory.shape[0] != batch:
            return reference.new_zeros(batch, self.hidden)
        return self._flight_memory

    def _camera_state(self, batch, reference):
        if self._camera_memory is None or self._camera_memory.shape[0] != batch:
            return reference.new_zeros(batch, self.hidden)
        return self._camera_memory

    def _observation_slice(self, v, state_observation):
        """The motion/target part of the observation, never the camera part."""
        source = state_observation if state_observation is not None else v
        return source[:, : self.obs_dim]

    def _flight_correction(self, memory):
        if not self.flight_residual_enabled:
            return memory.new_zeros(memory.shape[0], 3)
        return FLIGHT_RESIDUAL_MAX * torch.tanh(self.flight_head(memory))

    def _camera_correction(self, memory):
        if not self.camera_residual_enabled:
            return memory.new_zeros(memory.shape[0], 2)
        return torch.tanh(self.camera_head(memory))

    def bind_calibration(self, calibration):
        """Attach the active profile's calibration (required before use)."""
        self._calibration = calibration
        return self

    def _auto_exposure_suggestion(self, gray_obs, exposure, gain):
        """Physical mean-AE suggestion in normalized coordinates.

        Applied to the *last* channel of the two-frame observation, which is
        the current frame. Detached on purpose: the AE suggestion is a
        *reference*, not a differentiable path. Gradients must reach the camera
        head through the rendered image it changed, not through the controller
        that proposed the command.
        """
        if self._calibration is None:
            raise RuntimeError(
                "ResidualJointPolicy needs an IMX900 profile before it can read the "
                "image: call bind_calibration(env.imx900_calibration) first."
            )
        current = gray_obs[:, -1:] if gray_obs.shape[1] > 1 else gray_obs
        return physical_mean_auto_exposure_target(
            current,
            exposure.detach(),
            gain.detach(),
            self._calibration,
            target_luma=self.target_luma,
        )

    def _camera_logits(self, ae_suggestion, residual):
        """Map (AE target, log residual) to logits whose sigmoid is the command.

        The residual multiplies *physical* exposure time and gain factor, so it
        is applied in the profile's own units and mapped back. Adding it to the
        normalized coordinate instead would mean a different factor under each
        profile, which is the bug this avoids.
        """
        profile = self._calibration
        exposure_us = profile.exposure_to_us(ae_suggestion[:, 0])
        gain_factor = profile.gain_to_factor(ae_suggestion[:, 1])

        # tanh(r) in [-1, 1] -> factor in [1/500, 500] (or [1/8, 8] for gain).
        exposure_scale = torch.exp(residual[:, 0] * LOG_EXPOSURE_RANGE)
        gain_scale = torch.exp(residual[:, 1] * LOG_GAIN_RANGE)

        requested_ms = (exposure_us * 1e-3 * exposure_scale).clamp(
            EXPOSURE_MS_MIN, EXPOSURE_MS_MAX
        )
        requested_factor = (gain_factor * gain_scale).clamp(
            GAIN_FACTOR_MIN, GAIN_FACTOR_MAX
        )
        command = torch.stack(
            [
                profile.exposure_normalized(requested_ms * 1e3),
                profile.gain_from_factor(requested_factor),
            ],
            -1,
        ).clamp(1e-4, 1.0 - 1e-4)
        return torch.logit(command)
