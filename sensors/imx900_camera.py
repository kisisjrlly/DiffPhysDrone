"""Calibratable differentiable surrogate for e-con e-CAM37M_CUONX / Sony IMX900.

Design provenance:
- sensor/noise decomposition follows the structure used by End2endImaging's
  MonoSensor (shot noise, read noise, black level, quantization);
- online exposure/gain control and task-driven usage are aligned with the
  problem studied by JOCA;
- full differentiable optics (DeepLens) are intentionally out of the runtime
  loop for v1 and may later be distilled into lightweight PSF/vignetting terms.

This is not a transistor-level or firmware-exact digital twin. Its numerical
parameters come from an IMX900Calibration profile and must be replaced by
measured values before real-camera claims.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .imx900_calibration import IMX900Calibration


class IMX900DifferentiableCamera(nn.Module):
    def __init__(
        self,
        calibration: IMX900Calibration,
        *,
        blur_kernel_size: int = 5,
        dark_threshold: float = 0.05,
        saturation_mode: str = "soft",
        soft_clip_beta: float = 12.0,
    ):
        super().__init__()
        self.calibration = calibration
        self.blur_kernel_size = int(blur_kernel_size)
        self.dark_threshold = float(dark_threshold)
        self.saturation_mode = str(saturation_mode)
        self.soft_clip_beta = float(soft_clip_beta)

        if self.blur_kernel_size < 1 or self.blur_kernel_size % 2 == 0:
            raise ValueError("blur_kernel_size must be a positive odd integer")
        if not (0.0 <= self.dark_threshold <= 1.0):
            raise ValueError("dark_threshold must be in [0, 1]")
        if self.saturation_mode not in {"ste", "hard", "soft"}:
            raise ValueError("saturation_mode must be ste, hard, or soft")
        if self.soft_clip_beta <= 0:
            raise ValueError("soft_clip_beta must be > 0")

    @staticmethod
    def _as_image_param(value, image):
        value = torch.as_tensor(value, device=image.device, dtype=image.dtype)
        if value.ndim == 0:
            return value.view(1, 1, 1, 1)
        if value.ndim == 1:
            if value.shape[0] not in {1, image.shape[0]}:
                raise ValueError("1-D camera parameter must have length 1 or batch size")
            return value.view(-1, 1, 1, 1)
        if value.ndim == 2 and value.shape[-1] == 1:
            return value.view(value.shape[0], 1, 1, 1)
        while value.ndim < image.ndim:
            value = value.unsqueeze(-1)
        return value

    def _motion_blur(self, irradiance, exposure_relative, motion):
        if motion is None or self.calibration.blur_scale == 0.0:
            zero = torch.zeros(
                (irradiance.shape[0], 1, 1, 1),
                device=irradiance.device,
                dtype=irradiance.dtype,
            )
            return irradiance, zero

        motion = self._as_image_param(motion, irradiance).abs()
        strength = 1.0 - torch.exp(
            -self.calibration.blur_scale * exposure_relative * motion
        )
        strength = strength.clamp(0.0, 1.0)

        pad = self.blur_kernel_size // 2
        padded = F.pad(irradiance, (pad, pad, pad, pad), mode="replicate")
        blurred = F.avg_pool2d(
            padded,
            kernel_size=self.blur_kernel_size,
            stride=1,
            padding=0,
        )
        return irradiance + strength * (blurred - irradiance), strength

    def _saturate(self, signal_normalized):
        if self.saturation_mode == "hard":
            return signal_normalized.clamp(0.0, 1.0)
        if self.saturation_mode == "ste":
            hard = signal_normalized.clamp(0.0, 1.0)
            return signal_normalized + (hard - signal_normalized).detach()

        beta = self.soft_clip_beta
        upper = signal_normalized - F.softplus(
            signal_normalized - 1.0,
            beta=beta,
        )
        return upper.clamp_min(0.0)

    def _quantize(self, image):
        bits = int(self.calibration.quantization_bits)
        if bits <= 0:
            return image
        levels = float((1 << bits) - 1)
        rounded = torch.round(image * levels) / levels
        return image + (rounded - image).detach()

    def forward(
        self,
        irradiance: torch.Tensor,
        exposure01: torch.Tensor,
        gain01: torch.Tensor,
        *,
        motion: Optional[torch.Tensor] = None,
        shot_noise: Optional[torch.Tensor] = None,
        read_noise: Optional[torch.Tensor] = None,
        enable_noise: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if irradiance.ndim != 4 or irradiance.shape[1] != 1:
            raise ValueError("irradiance must have shape [B,1,H,W]")
        if not torch.is_floating_point(irradiance):
            raise TypeError("irradiance must be floating point")

        exposure01_img = self._as_image_param(exposure01, irradiance)
        gain01_img = self._as_image_param(gain01, irradiance)

        exposure_us = self.calibration.exposure_to_us(exposure01_img)
        exposure_relative = exposure_us / self.calibration.exposure_reference_us
        gain_factor = self.calibration.gain_to_factor(gain01_img)

        optical, blur_strength = self._motion_blur(
            irradiance,
            exposure_relative,
            motion,
        )

        # Charge/signal before gain. signal_scale absorbs the unknown conversion
        # between the renderer's normalized irradiance and the real sensor's
        # electron/DN scale and is fitted from exposure-response calibration.
        charge = (
            optical.clamp_min(0.0)
            * exposure_relative
            * self.calibration.signal_scale
        )
        clean_signal = gain_factor * charge

        # End2endImaging-inspired decomposition, but written as a differentiable
        # surrogate for online exposure/gain optimization. The parameters are
        # fitted from IMX900 PTC/dark-frame measurements rather than copied from
        # another camera.
        shot_std = gain_factor * (
            self.calibration.shot_noise_alpha * torch.sqrt(charge + 1e-6)
            + self.calibration.shot_noise_beta
        )
        read_std = self.calibration.read_noise_std_base * gain_factor.pow(
            self.calibration.read_noise_gain_exponent
        )

        noisy_signal = clean_signal
        if enable_noise:
            if shot_noise is None:
                shot_noise = torch.randn_like(charge)
            else:
                shot_noise = torch.as_tensor(
                    shot_noise,
                    device=charge.device,
                    dtype=charge.dtype,
                ).expand_as(charge)
            if read_noise is None:
                read_noise = torch.randn_like(charge)
            else:
                read_noise = torch.as_tensor(
                    read_noise,
                    device=charge.device,
                    dtype=charge.dtype,
                ).expand_as(charge)
            noisy_signal = noisy_signal + shot_std * shot_noise + read_std * read_noise

        pre_clip = noisy_signal + self.calibration.black_level
        signal_normalized = pre_clip / self.calibration.saturation_level
        image = self._saturate(signal_normalized)
        image = self._quantize(image)

        reduce_dims = tuple(range(1, image.ndim))
        aux = {
            "exposure_us": exposure_us,
            "gain_factor": gain_factor,
            "blur_strength": blur_strength,
            "shot_noise_std_mean": shot_std.mean(dim=reduce_dims),
            "read_noise_std_mean": read_std.expand_as(charge).mean(dim=reduce_dims),
            "saturation_fraction": (
                pre_clip >= self.calibration.saturation_level
            ).to(image.dtype).mean(dim=reduce_dims),
            "dark_fraction": (
                image <= self.dark_threshold
            ).to(image.dtype).mean(dim=reduce_dims),
        }
        return image, aux


def build_from_args(args) -> IMX900DifferentiableCamera:
    calibration = IMX900Calibration.from_json(args.imx900_calibration)
    if bool(getattr(args, "require_calibrated_imx900", False)) and not calibration.calibrated:
        raise ValueError(
            "A calibrated IMX900 profile is required, but the selected profile "
            f"is marked calibrated=false: {args.imx900_calibration}"
        )
    return IMX900DifferentiableCamera(
        calibration,
        blur_kernel_size=int(args.gray_blur_kernel_size),
        dark_threshold=float(args.gray_dark_threshold),
        saturation_mode=str(args.gray_saturation_mode),
        soft_clip_beta=float(args.gray_soft_clip_beta),
    )
