"""Motion-history shutter integration for long exposures.

Why this exists
---------------
``render_exposure`` historically reconstructed the whole shutter interval by
extrapolating the *current* velocity and body rate backwards::

    p(-t) = p_end - v_end * t
    R(-t) = R_end @ exp(-omega * t)

That is only defensible while the pose change over the shutter is small. Its
own docstring said "<=8 ms". At 50 ms on a 15 Hz loop the shutter covers 75% of
a frame period, over which the vehicle accelerates and re-orients measurably,
so the extrapolation renders motion that never happened. With an
end-velocity that is already the *post-step* value (``env.run`` updates ``v``
before the next frame is rendered), the error is biased, not random.

This module keeps a short ring buffer of the motion that actually happened and
integrates inside the shutter window:

* translation is the accumulated time-displacement
  ``p(-t) = p_now - Sum_i v_i * (overlap of segment i with [-t, 0])``, which
  is the trapezoidal integral of the sampled velocity. It is exact for
  piecewise-linear motion, reduces to ``-v*t`` for constant velocity (so the
  first frames agree with the legacy path), and unlike interpolating between
  stored positions it still accounts for the displacement across any gap
  between the buffer head and the current pose;
* rotation is interpolated on the rotation group as
  ``R(s) = R0 @ exp(w * log(R0^T R1))`` between the two bracketing attitudes,
  because a body rate is frame dependent and does not average;
* only *past* samples are ever read. The buffer is appended with the pre-step
  pose after each ``env.run``, so a sample can never see a future simulator
  state.

Gradients
---------
Samples keep whatever graph the recorded poses carried. Freezing a policy
parameter (``requires_grad=False``) does not sever the graph through its
*computation*, so the task loss can still reach the camera head and the flight
correction through the poses that the policy produced. Never wrap recording in
``torch.no_grad()``; pass tensors straight through.

First frame
-----------
At ``t = 0`` there is no past. We synthesize a one-entry pre-history by
replaying the motion law backwards from the current pose under the assumption
of zero acceleration and zero body rate. That is the *same* assumption the
legacy extrapolation makes, so the first frame is identical under both paths -
which is the point: the first frame must not differ, only later frames may.
The construction is described by :data:`PRE_HISTORY_METHOD` and hashed into the
protocol lock so a silent change becomes a loud mismatch.
"""

from __future__ import annotations

import hashlib
import math
from typing import Optional, Tuple

import torch

from .pose import body_angular_velocity

#: Identifier for how the synthetic pre-history at t=0 is built. The hash of
#: this string plus its numeric parameters goes into the protocol lock.
PRE_HISTORY_METHOD = (
    "zero-acceleration/zero-body-rate backward replay of the current scene-local "
    "pose over one control period (single synthetic entry)"
)


def pre_history_fingerprint(buffer_length: int) -> str:
    """Stable hash of the synthetic pre-history construction."""
    payload = f"{PRE_HISTORY_METHOD}|buffer={int(buffer_length)}|v1"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ExposureHistory:
    """Ring buffer of recent scene-local poses, queried inside the shutter.

    Parameters
    ----------
    length:
        Number of control periods retained. Must cover the longest supported
        shutter plus one step of margin; the integrator never asks for more
        than the shutter duration, so anything that satisfies
        ``length * min_dt >= max_exposure_s`` is sufficient.
    scene_local:
        ``True`` (default) stores poses after applying ``env.R_scene_T``, which
        is the frame the raytracer works in.
    """

    def __init__(self, length: int = 4, *, scene_local: bool = True):
        length = int(length)
        if length < 2:
            raise ValueError("exposure history must retain at least two samples")
        self.length = length
        self.scene_local = bool(scene_local)
        self._p = []
        self._v = []
        self._R = []
        self._dt = []
        self._primed = False

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self._p.clear()
        self._v.clear()
        self._R.clear()
        self._dt.clear()
        self._primed = False

    @property
    def ready(self) -> bool:
        """True once a usable history exists (priming counts)."""
        return self._primed and len(self._p) >= 1

    def __len__(self) -> int:
        return len(self._p)

    def spanned_seconds(self) -> float:
        """Total duration covered by the retained history."""
        return float(sum(self._dt))

    def append(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor,
        rotation: torch.Tensor,
        dt: float,
    ) -> None:
        """Record one realized pose and the interval since the previous one.

        Append the pose that was current *before* the step described by
        ``dt``. Callers therefore push the pre-step pose with the step
        duration after each ``env.run``, so the newest entry always sits one
        interval behind "now".

        Graph is preserved on purpose: freezing parameters does not sever the
        path through their computation, and the task loss must still reach the
        camera head through these poses.
        """
        if position.ndim != 2 or position.shape[-1] != 3:
            raise ValueError("position must be [B,3]")
        if rotation.shape != (position.shape[0], 3, 3):
            raise ValueError("rotation must be [B,3,3]")
        if velocity.shape != position.shape:
            raise ValueError("velocity must match position")
        dt = float(dt)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        self._p.append(position)
        self._v.append(velocity)
        self._R.append(rotation)
        self._dt.append(dt)
        while len(self._p) > self.length:
            self._p.pop(0)
            self._v.pop(0)
            self._R.pop(0)
            self._dt.pop(0)

    def prime_from_env(self, env) -> None:
        """Seed the synthetic pre-history for the very first frame.

        Uses only the current state plus the configured control period. Called
        after ``env.reset``; nothing that happens later is referenced.

        A single entry is stored, one period behind the current pose under
        zero acceleration and zero body rate. That is byte-for-byte the
        assumption the legacy extrapolation makes, so frame 0 renders the same
        under both paths. The correction uses one velocity record (the
        current one) and interpolates ``now -> entry`` for attitude, so the
        whole backward span is exactly ``v * tau``.
        """
        self.reset()
        dt = float(getattr(env, "last_control_dt", 1.0 / 15.0))
        position = (env.R_scene_T @ env.p[:, :, None]).squeeze(-1)
        velocity = (env.R_scene_T @ env.v[:, :, None]).squeeze(-1)
        rotation = env.R_scene_T @ env.R
        # Body rate over the synthetic step, expressed in the body frame so
        # that ``R(-tau) = R @ exp(-omega * tau)`` matches the legacy path.
        # ``env.last_control_dt`` is the *forward* step just applied, so the
        # synthetic step must be the same length or the rate is scaled by
        # dt/last_control_dt and the single-entry attitude misfits.
        rate = body_angular_velocity(env.R_old, env.R, dt)
        omega = (env.R.transpose(1, 2) @ rate[:, :, None]).squeeze(-1)
        behind_R = rotation @ rotation_exp_safe(-omega * dt)
        behind = position - velocity * dt
        self.append(behind, velocity, behind_R, dt)
        self._primed = True

    def sync_from_env(self, env) -> None:
        """Re-seed from the current env state (cheap, used after resets)."""
        self.reset()
        self.prime_from_env(env)

    # -------------------------------------------------------------- querying
    def matched_angular_rate(
        self,
        position: torch.Tensor,
        rotation: torch.Tensor,
        fallback: torch.Tensor,
    ) -> torch.Tensor:
        """Body rate for the step that ended at the current pose.

        Prefers the recorded last step so the integrator and the history use
        the same number; ``fallback`` (the caller's two-pose estimate) is used
        only when the buffer cannot supply one.
        """
        if len(self._R) < 2:
            return fallback
        dt = self._dt[-1]
        return _body_rate(self._R[-2], self._R[-1], dt)

    def sample(
        self,
        position: torch.Tensor,
        rotation: torch.Tensor,
        angular_rate: torch.Tensor,
        offset: torch.Tensor,
        *,
        camera_rotation: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pose at ``offset`` seconds in the past (``offset <= 0``).

        Returns ``(scene_position, camera_rotation_matrix)``; when
        ``camera_rotation`` is given the second value is already composed with
        it so the caller can hand it straight to the renderer.

        ``position``/``rotation`` must be the *current* pose in the same frame
        the buffer stores. ``position`` anchors the backward integration, so
        any gap between the buffer head and "now" is integrated rather than
        silently dropped.
        """
        if position.ndim != 2 or position.shape[-1] != 3:
            raise ValueError("position must be [B,3]")
        if rotation.shape != (position.shape[0], 3, 3):
            raise ValueError("rotation must be [B,3,3]")
        if len(self._p) < 1:
            # Degenerate: no history yet. Fall back to constant rate from now,
            # which is exactly the legacy behaviour.
            sample_p = position + _scaled(self._v_last(angular_rate), offset)
            sample_R = rotation @ rotation_exp_safe(angular_rate * offset[:, None])
            if camera_rotation is not None:
                sample_R = sample_R @ camera_rotation
            return sample_p, sample_R

        # Timeline anchored at "now". Index 0 is now; index i (>=1) is the
        # stored sample, newest first, each straddling one recorded interval.
        # The buffer holds poses *before* now, so "now" is one interval in
        # front of the newest stored sample.
        #
        # Careful: these names must not shadow the ``position`` argument, and
        # they are passed positionally in this exact order below.
        timeline_p = [position] + list(reversed(self._p))
        timeline_R = [rotation] + list(reversed(self._R))
        timeline_v = [self._v[-1]] + list(reversed(self._v))
        # Intervals, newest first, matching the timeline above: the first is
        # the gap from "now" back to the newest stored sample, the rest are
        # the intervals between stored samples.
        intervals = list(reversed(self._dt))

        need = (-offset).clamp_min(0.0)  # seconds behind now, per batch item
        return self._interpolate(
            timeline_p, timeline_R, timeline_v, need, intervals, camera_rotation
        )

    def _v_last(self, fallback: torch.Tensor) -> torch.Tensor:
        return self._v[-1] if self._v else fallback

    def _interpolate(
        self,
        positions,
        rotations,
        velocities,
        need: torch.Tensor,
        intervals,
        camera_rotation: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Attitude from history, translation from an exact time-displacement.

        Translation is *not* interpolated between stored positions. It is
        obtained by walking backwards from the current pose over the recorded
        velocities::

            p(-t) = p_now - Sum_k v_k * |segment k  intersect  [0, t]|

        where ``v_k`` is the velocity recorded over interval ``k``. Each term
        is credited exactly the piece of ``[-t, 0]`` it was measured on, so
        this is the trapezoidal integral of the sampled velocity and is exact
        for piecewise-linear motion. It also stays correct when the buffer
        head is older than the current pose (a reset or a bulk rollout
        append), which interpolating between stored positions is not: that
        construction ignores the displacement across the head-to-now gap
        entirely. When every sample lies on one constant velocity the
        displacement is exactly ``v * t``, so the legacy extrapolation is
        reproduced in that limit.

        Attitude cannot use the same construction: a body rate is frame
        dependent and rotations do not average, so the rotation is
        interpolated on the rotation group between the two bracketing
        recorded attitudes instead.
        """
        if not intervals:
            raise ValueError("history must record at least one interval")
        batch = positions[0].shape[0]
        idx = torch.arange(batch, device=need.device)

        # ---- translation: cumulative time-displacement from now ----------
        segments = list(intervals)
        boundaries = [0.0]
        for step in segments:
            boundaries.append(boundaries[-1] + float(step))
        # ``boundaries[k]`` is the age of timeline entry ``k``. Velocities are
        # indexed the same way: entry 0 is "now" (its stored velocity is the
        # newest one), entry k is the velocity recorded over the k-th
        # interval. ``v_seg[k]`` therefore owns ``[b[k-1], b[k]]`` for k >= 1.
        b = torch.as_tensor(boundaries, device=need.device, dtype=need.dtype)
        v_stack = torch.stack(velocities, 0)  # [K+1, B, 3]
        tau = need.clamp_min(0.0).unsqueeze(0)  # [1, B]

        sample_p = positions[0]
        for k in range(1, b.numel()):
            prev = b[k - 1]
            cur = b[k]
            # Length of [-tau, 0] that falls inside segment k. Walking into the
            # past, so the displacement is subtracted from the current pose.
            length = torch.clamp(tau - prev, min=0.0) - torch.clamp(tau - cur, min=0.0)
            sample_p = sample_p - v_stack[k] * length.transpose(0, 1)

        # ---- attitude: interpolate between bracketing recorded poses ------
        # ``b`` has one stamp per timeline entry plus the closing boundary, so
        # entry ``k`` sits at ``b[k]`` and ``n`` entries span ``b[0..n-1]``.
        stamps = b[: len(rotations)]
        span = stamps[-1]
        need_rot = need.clamp(max=float(span))
        n = len(rotations)
        lo = torch.searchsorted(stamps, need_rot, right=True).sub(1).clamp(0, n - 2)
        hi = lo + 1
        t_lo = stamps[lo]
        t_hi = stamps[hi]
        denom = (t_hi - t_lo).clamp_min(1e-9)
        w = ((need_rot - t_lo) / denom).clamp(0.0, 1.0)

        R_stack = torch.stack(rotations, 0)
        # ``lo``/``hi`` are per-batch-item rows; pairing them with the batch
        # index selects ``R_stack[rows[b], b]`` for each item ``b``.
        R_lo = R_stack[lo, idx]
        R_hi = R_stack[hi, idx]
        delta = _rotation_log(R_lo.transpose(1, 2) @ R_hi)
        sample_R = R_lo @ rotation_exp_safe(delta * w[:, None])
        if camera_rotation is not None:
            sample_R = sample_R @ camera_rotation
        return sample_p, sample_R


def rotation_exp_safe(vector: torch.Tensor) -> torch.Tensor:
    """Rodrigues exponential with a finite derivative at zero rotation."""
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], -1).reshape(-1, 3, 3)
    angle = vector.norm(dim=-1)
    a = torch.sinc(angle / math.pi)
    b = 0.5 * torch.sinc(angle / (2 * math.pi)).square()
    eye = torch.eye(3, dtype=vector.dtype, device=vector.device).expand_as(skew)
    return eye + a[:, None, None] * skew + b[:, None, None] * (skew @ skew)


def _rotation_log(matrix: torch.Tensor) -> torch.Tensor:
    """Principal-axis rotation vector of a batch of rotation matrices."""
    vector = 0.5 * torch.stack(
        [
            matrix[:, 2, 1] - matrix[:, 1, 2],
            matrix[:, 0, 2] - matrix[:, 2, 0],
            matrix[:, 1, 0] - matrix[:, 0, 1],
        ],
        -1,
    )
    sine = vector.norm(dim=-1)
    cosine = ((matrix.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) * 0.5).clamp(-1.0, 1.0)
    angle = torch.atan2(sine, cosine)
    scale = torch.where(sine > 1e-7, angle / sine.clamp_min(1e-12), torch.ones_like(sine))
    return vector * scale[:, None]


def _body_rate(previous: torch.Tensor, current: torch.Tensor, dt: float) -> torch.Tensor:
    """Body-frame rate from two attitudes; mirrors ``pose.body_angular_velocity``."""
    relative = previous.transpose(1, 2) @ current
    return _rotation_log(relative) / float(dt)


def _scaled(velocity: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    return velocity * offset[:, None]


def attach(env, *, length: int = 4, scene_local: bool = True) -> ExposureHistory:
    """Create a history buffer and bind it to ``env`` (opt-in)."""
    history = ExposureHistory(length=length, scene_local=scene_local)
    env.exposure_history = history
    return history


def record_history_step(env, dt: float) -> None:
    """Append the current pose as the one *before* a ``dt``-long step.

    Call this immediately before ``env.run``. ``env.v`` still holds the
    velocity over the step being entered at that point; after ``run`` it holds
    the post-step value and no longer describes the interval. The recorded
    rotation is ``env.R_old``, the attitude at the start of the step.

    Syncing from ``env.p``/``env.R_old`` keeps the buffer correct after a
    reset: the caller primes once after ``reset``, and the first ``run`` then
    appends real motion on top of the synthetic entry.
    """
    history = getattr(env, "exposure_history", None)
    if history is None:
        return
    position = (env.R_scene_T @ env.p[:, :, None]).squeeze(-1)
    velocity = (env.R_scene_T @ env.v[:, :, None]).squeeze(-1)
    rotation = env.R_scene_T @ env.R_old
    history.append(position, velocity, rotation, dt)
