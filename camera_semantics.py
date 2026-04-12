"""
Unified camera semantics constants and mappings.

Single source of truth for camera parameter-domain conversions used by
Python training/rendering code.
"""
from dataclasses import dataclass
from typing import Union

import torch

TensorOrFloat = Union[torch.Tensor, float]


@dataclass(frozen=True)
class CameraSemantics:
    # exposure01 -> t_cmd
    exposure_t_min: float = 0.25
    exposure_t_span: float = 2.75

    # t_eff clamp after AE multiplier
    exposure_eff_min: float = 0.15
    exposure_eff_max: float = 4.0

    # iso01 -> iso_gain
    iso_gain_base: float = 1.0
    iso_gain_scale: float = 10.0
    iso_gain_gamma: float = 1.2

    # noise model base
    shot_noise_base: float = 0.03

    def exposure_to_command(self, exposure01: TensorOrFloat) -> TensorOrFloat:
        if isinstance(exposure01, torch.Tensor):
            ex = exposure01.clamp(0.0, 1.0)
            return self.exposure_t_min + self.exposure_t_span * ex
        ex = min(max(float(exposure01), 0.0), 1.0)
        return self.exposure_t_min + self.exposure_t_span * ex

    def exposure_to_effective(self, exposure01: TensorOrFloat) -> TensorOrFloat:
        if isinstance(exposure01, torch.Tensor):
            return self.exposure_to_command(exposure01).clamp(
                self.exposure_eff_min,
                self.exposure_eff_max,
            )
        return min(
            max(float(self.exposure_to_command(exposure01)), self.exposure_eff_min),
            self.exposure_eff_max,
        )

    def exposure_to_time(self, exposure01: TensorOrFloat) -> TensorOrFloat:
        return self.exposure_to_effective(exposure01)

    def iso_to_gain(self, iso01: TensorOrFloat) -> TensorOrFloat:
        if isinstance(iso01, torch.Tensor):
            iv = iso01.clamp(0.0, 1.0)
            return self.iso_gain_base + self.iso_gain_scale * (iv ** self.iso_gain_gamma)
        iv = min(max(float(iso01), 0.0), 1.0)
        return self.iso_gain_base + self.iso_gain_scale * (iv ** self.iso_gain_gamma)


def from_args(args) -> CameraSemantics:
    return CameraSemantics(
        exposure_t_min=float(args.cam_exposure_t_min),
        exposure_t_span=float(args.cam_exposure_t_span),
        exposure_eff_min=float(args.cam_exposure_eff_min),
        exposure_eff_max=float(args.cam_exposure_eff_max),
        iso_gain_base=float(args.cam_iso_gain_base),
        iso_gain_scale=float(args.cam_iso_gain_scale),
        iso_gain_gamma=float(args.cam_iso_gain_gamma),
        shot_noise_base=float(args.cam_shot_noise_base),
    )
