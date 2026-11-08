"""Auto-exposure in physical units (microseconds and gain multiples).

Why not reuse ``auto_exposure.py``
----------------------------------
The controllers in ``controllers/auto_exposure.py`` operate on the *normalized*
exposure coordinate ``exposure01``. Their gains (``exposure_rate=0.35`` per unit
of luma error) were tuned when that coordinate spanned 100-8000 us, i.e. a
factor of 80. Under the 50 ms research profile the same coordinate spans a
factor of 500, so one unit of ``exposure01`` is no longer the same number of
f-stops and the controller silently becomes ~6x slower in relative terms.
Multiplying the step by a constant would fix the speed but not the other
coordinate-dependent decisions (the ``0.65``/``0.35`` gates), and it would
make the controller profile-specific.

These controllers instead read and write the normalized coordinates through the
calibration object, so the feedback law is expressed in quantities the sensor
actually has:

* exposure is adjusted in **log exposure time**, because image brightness is
  proportional to exposure time and the operator that makes a fixed relative
  step equal is the logarithm;
* gain is a multiplicative factor of the analog/digital chain, so it is also
  adjusted in log space;
* the brightness error is turned into a log-exposure correction using the
  local slope of the mean response, so a dark frame moves further than an
  almost-correct one.

The feedback law is written so that it means the same thing regardless of the
loaded profile:

* exposure is corrected **multiplicatively in exposure time**, because image
  brightness is proportional to exposure time. A step of ``exposure_rate`` is
  then always "move this fraction of the way to the correct exposure", which is
  what the historical ``0.35`` meant under the 8 ms profile;
* gain is a multiplicative factor of the analog/digital chain, so it is also
  shifted in factor space through the profile's own gain map;
* the step size in *relative* terms is unchanged from the legacy controller.
  What changes is that the absolute meaning of a step no longer depends on
  which profile is loaded, so the same controller behaves the same way on the
  8 ms and 50 ms profiles instead of silently becoming ~6x slower relative to
  the available range.

The old normalized controller is kept for reproducing historical runs.

Both controllers return normalized ``[E, G]`` in the *active* profile's
coordinate system, so callers never have to know which profile is loaded.
"""

import math

import torch

#: Per-step bound on the log-gain correction, in log units (factor 1.5).
LOG_GAIN_STEP = math.log(1.5)


def _profile(args_or_calibration):
    """Accept either the calibration object or an object carrying one."""
    if hasattr(args_or_calibration, "exposure_to_us"):
        return args_or_calibration
    for attribute in ("imx900_calibration", "gray_camera"):
        candidate = getattr(args_or_calibration, attribute, None)
        if candidate is None:
            continue
        if hasattr(candidate, "exposure_to_us"):
            return candidate
        calibration = getattr(candidate, "calibration", None)
        if calibration is not None:
            return calibration
    raise TypeError("could not find an IMX900 calibration on the given object")


@torch.no_grad()
def physical_mean_auto_exposure_target(
    gray,
    exposure,
    gain,
    calibration,
    *,
    target_luma=0.35,
    exposure_rate=0.35,
    gain_rate=0.10,
):
    """Drive mean image brightness toward a target, in microseconds.

    ``exposure``/``gain`` are the normalized coordinates the sensor produced
    the frame with; the returned ``[E, G]`` is normalized in the same profile,
    so the caller can apply it directly.

    The correction is multiplicative in exposure time::

        T' = T * (1 + exposure_rate * (target_luma / luma - 1))

    Brightness is proportional to exposure time, so the bracket is the exact
    fraction by which the current exposure must change for a locally linear
    sensor. ``exposure_rate`` therefore keeps the meaning it had in the legacy
    controller - a fraction of the remaining error per step - while the unit of
    that error no longer depends on the profile's range. That is what makes the
    law transfer between the 8 ms and 50 ms profiles unchanged.
    """
    if gray.ndim != 4:
        raise ValueError("gray must be BCHW")
    profile = _profile(calibration)
    luma = gray.float().mean(dim=(1, 2, 3))

    exposure_us = profile.exposure_to_us(exposure.clamp(0.0, 1.0))
    floor_us = float(profile.exposure_us_min)
    ceiling_us = float(profile.exposure_us_max)

    # Ratio of needed to current brightness; bounded so a nearly black frame
    # (luma -> 0) cannot demand an unbounded exposure jump in one step.
    ratio = (float(target_luma) / luma.clamp_min(1e-3)).clamp(0.25, 4.0)
    relative = (float(exposure_rate) * (ratio - 1.0)).clamp(-0.95, 8.0)
    target_us = (exposure_us * (1.0 + relative)).clamp(floor_us, ceiling_us)

    # Gain: slower than exposure, and only when exposure has run out of room in
    # the direction the image needs. Expressed through the profile's own gain
    # map so linear and logarithmic profiles both behave multiplicatively.
    at_ceiling = (target_us >= ceiling_us - 1e-6) & (ratio > 1.0)
    at_floor = (target_us <= floor_us + 1e-6) & (ratio < 1.0)
    stuck = at_ceiling | at_floor
    unmet = torch.log(ratio).clamp(-0.5, 0.5)
    gain_offset = torch.where(stuck, float(gain_rate) * unmet, torch.zeros_like(unmet))
    target_gain = _shift_gain(profile, gain, gain_offset.clamp(-LOG_GAIN_STEP, LOG_GAIN_STEP))

    return torch.stack(
        [profile.exposure_normalized(target_us), target_gain.clamp(0.0, 1.0)], -1
    )


def _shift_gain(profile, gain, log_offset):
    """Apply a multiplicative log offset to a gain coordinate.

    ``gain_to_factor`` is the physical map; shifting in factor space and
    mapping back keeps the correction multiplicative whether the profile
    declares linear or logarithmic gain.
    """
    factor = profile.gain_to_factor(gain.clamp(0.0, 1.0))
    factor_floor = float(profile.gain_factor_min)
    factor_ceiling = float(profile.gain_factor_max)
    shifted = (factor * torch.exp(log_offset)).clamp(factor_floor, factor_ceiling)
    return profile.gain_from_factor(shifted)


def _gradient_score(candidate):
    gx = (candidate[..., 1:] - candidate[..., :-1]).abs()
    gy = (candidate[..., 1:, :] - candidate[..., :-1, :]).abs()
    score = gx.mean(dim=(1, 2, 3)) + gy.mean(dim=(1, 2, 3))
    sat = (candidate >= 0.98).float().mean(dim=(1, 2, 3))
    dark = (candidate <= 0.02).float().mean(dim=(1, 2, 3))
    return score - 0.50 * sat - 0.15 * dark


@torch.no_grad()
def physical_gradient_auto_exposure_target(
    gray,
    exposure,
    gain,
    calibration,
    *,
    exposure_step_octaves=0.12,
    gain_step_octaves=0.06,
):
    """Gradient/contrast-seeking exposure control, in physical units.

    The scoring is unchanged from the legacy controller (spatial gradient
    magnitude minus saturation and darkness penalties); what changes is the
    meaning of a step. ``exposure_step_octaves`` is now octaves of exposure
    *time*, so the search grid is the same relative grid on any profile.
    """
    if gray.ndim != 4:
        raise ValueError("gray must be BCHW")
    profile = _profile(calibration)

    scales = gray.new_tensor([0.50, 0.75, 1.00, 1.50, 2.00])
    candidates = (gray[:, None] * scales.view(1, -1, 1, 1, 1)).clamp(0.0, 1.0)
    B, K = candidates.shape[:2]
    scores = torch.stack([_gradient_score(candidates[:, k]) for k in range(K)], dim=1)
    best = scores.argmax(dim=1)
    direction = torch.log2(scales[best])

    exposure_us = profile.exposure_to_us(exposure.clamp(0.0, 1.0))
    floor_us = float(profile.exposure_us_min)
    ceiling_us = float(profile.exposure_us_max)
    log2_target = torch.log2(exposure_us.clamp_min(floor_us) / floor_us) + float(exposure_step_octaves) * direction
    target_us = (floor_us * torch.pow(2.0, log2_target)).clamp(floor_us, ceiling_us)

    # Only push gain when the desired exposure move runs into a bound; the
    # residual is measured in octaves of exposure time.
    log2_span = math.log2(ceiling_us / floor_us)
    raw_log2 = torch.log2(exposure_us.clamp_min(floor_us) / floor_us) + float(exposure_step_octaves) * direction
    residual = torch.where(raw_log2 < 0.0, raw_log2, (raw_log2 - log2_span).clamp_min(0.0))
    gain_offset = (residual * float(gain_step_octaves) / max(float(exposure_step_octaves), 1e-6))
    target_gain = _shift_gain(profile, gain, gain_offset.clamp(-0.9, 3.0))

    return torch.stack(
        [profile.exposure_normalized(target_us), target_gain.clamp(0.0, 1.0)], -1
    )
