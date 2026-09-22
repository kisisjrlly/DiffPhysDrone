"""Classical camera-control baselines.

These controllers operate on observed grayscale frames and never use navigation
loss gradients. They share the same normalized exposure/gain action domain and
the same actuator EMA as the learned camera controller.

The gradient controller is a project baseline inspired by image-gradient-based
exposure control; it is not claimed to be a bit-exact reproduction of any
specific published implementation.
"""

import math

import torch


@torch.no_grad()
def mean_auto_exposure_target(
    gray,
    exposure,
    gain,
    *,
    target_luma=0.35,
    exposure_rate=0.35,
    gain_rate=0.10,
):
    """Drive average image brightness toward a fixed target."""
    if gray.ndim != 4:
        raise ValueError("gray must be BCHW")
    luma = gray.float().mean(dim=(1, 2, 3))
    error = float(target_luma) - luma

    e_target = (exposure + float(exposure_rate) * error).clamp(0.0, 1.0)

    # Gain is deliberately slower than exposure. It participates more strongly
    # when exposure is already high in dark scenes, and is reduced when an
    # over-bright image persists.
    under_gate = torch.sigmoid((exposure - 0.65) * 10.0)
    over_gate = torch.sigmoid((0.35 - exposure) * 10.0)
    gain_gate = torch.where(error >= 0.0, under_gate, over_gate)
    g_target = (
        gain + float(gain_rate) * error * (0.25 + 0.75 * gain_gate)
    ).clamp(0.0, 1.0)
    return torch.stack([e_target, g_target], -1)


def _gradient_score(candidate):
    gx = (candidate[..., 1:] - candidate[..., :-1]).abs()
    gy = (candidate[..., 1:, :] - candidate[..., :-1, :]).abs()
    score = gx.mean(dim=(1, 2, 3)) + gy.mean(dim=(1, 2, 3))
    sat = (candidate >= 0.98).float().mean(dim=(1, 2, 3))
    dark = (candidate <= 0.02).float().mean(dim=(1, 2, 3))
    return score - 0.50 * sat - 0.15 * dark


@torch.no_grad()
def gradient_auto_exposure_target(
    gray,
    exposure,
    gain,
    *,
    exposure_step=0.12,
    gain_step=0.06,
):
    """Choose a local exposure direction that preserves image detail.

    Candidate brightness scalings approximate nearby exposure changes. The
    score rewards spatial image gradients while penalizing saturated and fully
    dark pixels. Exposure carries most of the adjustment; gain is only used for
    residual pressure near the exposure bounds.
    """
    if gray.ndim != 4:
        raise ValueError("gray must be BCHW")

    scales = gray.new_tensor([0.50, 0.75, 1.00, 1.50, 2.00])
    candidates = (gray[:, None] * scales.view(1, -1, 1, 1, 1)).clamp(0.0, 1.0)
    B, K = candidates.shape[:2]
    scores = torch.stack(
        [_gradient_score(candidates[:, k]) for k in range(K)],
        dim=1,
    )
    best = scores.argmax(dim=1)
    best_scale = scales[best]
    direction = torch.log2(best_scale)

    raw_e = exposure + float(exposure_step) * direction
    e_target = raw_e.clamp(0.0, 1.0)

    # Only push gain when the desired exposure move runs into a bound.
    low_residual = raw_e.clamp_max(0.0)
    high_residual = (raw_e - 1.0).clamp_min(0.0)
    gain_delta = float(gain_step) * (high_residual + low_residual)
    g_target = (gain + gain_delta).clamp(0.0, 1.0)
    return torch.stack([e_target, g_target], -1)
