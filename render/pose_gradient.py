"""Local image-warp pose Jacobian with the unchanged hard-rendered forward.

This is a training approximation: fixed reference depth, brightness constancy,
and bilinear image gradients. It captures moving-edge signals but does not
model visibility changes or prove an exact physical rendering derivative.
Depth is internal renderer data, never an actor observation.
"""

import torch
import torch.nn.functional as F

from .ideal_gray import _camera_rays


def image_warp_pose_surrogate(image, depth, position, rotation, fov_x_half_tan):
    """Return exactly ``image`` with an inverse-warp pose derivative.

    ``depth`` is the ray parameter (camera forward-axis depth), not Euclidean
    range. The reference pose, image and depth are detached. There is no
    change to the sampled image, including under a gradient-boundary cut.
    """
    reference_p, reference_r = position.detach(), rotation.detach()
    rays = _camera_rays(rotation, image.shape[-2], image.shape[-1], fov_x_half_tan)
    world = position[:, None, None, :] + depth.detach()[..., None] * rays
    reference_points = torch.einsum("bji,bhwj->bhwi", reference_r,
                                    world - reference_p[:, None, None, :])
    x, y, z = reference_points.unbind(-1)
    fov_y = float(fov_x_half_tan) * image.shape[-2] / image.shape[-1]
    grid = torch.stack([(-y / x.clamp_min(1e-6) + 1e-5) / float(fov_x_half_tan),
                        (-z / x.clamp_min(1e-6) + 1e-5) / fov_y], -1)
    warped = F.grid_sample(image.detach(), grid, mode="bilinear",
                           padding_mode="border", align_corners=False)
    # Subtract before adding, preserving the exact hard-forward pixel values.
    return image.detach() + (warped - warped.detach())
