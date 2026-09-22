"""Differentiable monochrome camera model for active exposure/gain control.

This module intentionally models only task-relevant image-formation effects.
It is designed to be calibrated against the real IMX900 camera later; the
default coefficients are provisional simulation values, not an IMX900 digital
twin.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gray_camera_semantics import GrayCameraSemantics, from_args as semantics_from_args


class DifferentiableGrayCamera(nn.Module):
    """A compact differentiable grayscale sensor model.

    Expected irradiance shape is ``[B, 1, H, W]``. Exposure/gain may be scalar,
    ``[B]``, ``[B, 1]``, or already broadcastable to the image.
    """

    def __init__(
        self,
        semantics: Optional[GrayCameraSemantics] = None,
        *,
        shot_noise_scale: float = 0.015,
        read_noise_std: float = 0.005,
        read_noise_gain_scale: float = 0.35,
        black_level: float = 0.0,
        blur_scale: float = 1.0,
        blur_kernel_size: int = 5,
        dark_threshold: float = 0.05,
        saturation_mode: str = "ste",
        soft_clip_beta: float = 6.0,
        quantization_bits: int = 0,
    ):
        super().__init__()
        self.semantics = semantics or GrayCameraSemantics()
        self.shot_noise_scale = float(shot_noise_scale)
        self.read_noise_std = float(read_noise_std)
        self.read_noise_gain_scale = float(read_noise_gain_scale)
        self.black_level = float(black_level)
        self.blur_scale = float(blur_scale)
        self.blur_kernel_size = int(blur_kernel_size)
        self.dark_threshold = float(dark_threshold)
        self.saturation_mode = str(saturation_mode)
        self.soft_clip_beta = float(soft_clip_beta)
        self.quantization_bits = int(quantization_bits)

        if self.shot_noise_scale < 0 or self.read_noise_std < 0:
            raise ValueError("noise scales must be >= 0")
        if self.read_noise_gain_scale < 0:
            raise ValueError("read_noise_gain_scale must be >= 0")
        if self.blur_scale < 0:
            raise ValueError("blur_scale must be >= 0")
        if self.blur_kernel_size < 1 or self.blur_kernel_size % 2 == 0:
            raise ValueError("blur_kernel_size must be a positive odd integer")
        if not (0.0 <= self.dark_threshold <= 1.0):
            raise ValueError("dark_threshold must be in [0, 1]")
        if self.saturation_mode not in {"ste", "hard", "soft"}:
            raise ValueError("saturation_mode must be one of: ste, hard, soft")
        if self.soft_clip_beta <= 0:
            raise ValueError("soft_clip_beta must be > 0")
        if self.quantization_bits < 0:
            raise ValueError("quantization_bits must be >= 0")

    @staticmethod
    def _as_image_param(value: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
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

    def _motion_blur(
        self,
        irradiance: torch.Tensor,
        exposure_relative: torch.Tensor,
        motion: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if motion is None or self.blur_scale == 0.0:
            zero = torch.zeros(
                (irradiance.shape[0], 1, 1, 1),
                device=irradiance.device,
                dtype=irradiance.dtype,
            )
            return irradiance, zero

        motion = self._as_image_param(motion, irradiance).abs()
        strength = 1.0 - torch.exp(
            -self.blur_scale * exposure_relative * motion
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

    def _saturate(self, signal: torch.Tensor) -> torch.Tensor:
        if self.saturation_mode == "hard":
            return signal.clamp(0.0, 1.0)
        if self.saturation_mode == "ste":
            hard = signal.clamp(0.0, 1.0)
            return signal + (hard - signal).detach()

        beta = self.soft_clip_beta
        lo = torch.sigmoid(signal.new_tensor(-0.5 * beta))
        hi = torch.sigmoid(signal.new_tensor(0.5 * beta))
        y = torch.sigmoid(beta * (signal - 0.5))
        return (y - lo) / (hi - lo)

    def _quantize(self, image: torch.Tensor) -> torch.Tensor:
        if self.quantization_bits <= 0:
            return image
        levels = float((1 << self.quantization_bits) - 1)
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
            raise ValueError("irradiance must have shape [B, 1, H, W]")
        if not torch.is_floating_point(irradiance):
            raise TypeError("irradiance must be floating point")

        exposure01_img = self._as_image_param(exposure01, irradiance)
        gain01_img = self._as_image_param(gain01, irradiance)

        exposure_us = self.semantics.exposure_to_us(exposure01_img)
        exposure_relative = exposure_us / self.semantics.exposure_reference_us
        gain_factor = self.semantics.gain_to_factor(gain01_img)

        optical, blur_strength = self._motion_blur(
            irradiance, exposure_relative, motion
        )
        charge = optical.clamp_min(0.0) * exposure_relative

        shot_std = self.shot_noise_scale * torch.sqrt(charge + 1e-6)
        read_std = self.read_noise_std * (
            1.0 + self.read_noise_gain_scale * gain01_img.clamp(0.0, 1.0)
        )

        if enable_noise:
            if shot_noise is None:
                shot_noise = torch.randn_like(charge)
            else:
                shot_noise = torch.as_tensor(
                    shot_noise, device=charge.device, dtype=charge.dtype
                ).expand_as(charge)
            if read_noise is None:
                read_noise = torch.randn_like(charge)
            else:
                read_noise = torch.as_tensor(
                    read_noise, device=charge.device, dtype=charge.dtype
                ).expand_as(charge)
            charge = charge + shot_std * shot_noise + read_std * read_noise

        pre_clip = gain_factor * charge + self.black_level
        image = self._saturate(pre_clip)
        image = self._quantize(image)

        reduce_dims = tuple(range(1, image.ndim))
        aux = {
            "exposure_us": exposure_us,
            "gain_factor": gain_factor,
            "blur_strength": blur_strength,
            "shot_noise_std_mean": shot_std.mean(dim=reduce_dims),
            "read_noise_std_mean": read_std.expand_as(charge).mean(dim=reduce_dims),
            "saturation_fraction": (pre_clip >= 1.0).to(image.dtype).mean(dim=reduce_dims),
            "dark_fraction": (image <= self.dark_threshold).to(image.dtype).mean(dim=reduce_dims),
        }
        return image, aux


def build_from_args(args) -> DifferentiableGrayCamera:
    """Construct the grayscale camera from the repository argparse namespace."""
    return DifferentiableGrayCamera(
        semantics_from_args(args),
        shot_noise_scale=float(args.gray_shot_noise_scale),
        read_noise_std=float(args.gray_read_noise_std),
        read_noise_gain_scale=float(args.gray_read_noise_gain_scale),
        black_level=float(args.gray_black_level),
        blur_scale=float(args.gray_blur_scale),
        blur_kernel_size=int(args.gray_blur_kernel_size),
        dark_threshold=float(args.gray_dark_threshold),
        saturation_mode=str(args.gray_saturation_mode),
        soft_clip_beta=float(args.gray_soft_clip_beta),
        quantization_bits=int(args.gray_quantization_bits),
    )
