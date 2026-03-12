import torch
import torch.nn.functional as F


def apply_camera_effects(depth, exposure, iso):
    """
    将可微的相机传感器效应应用到渲染出的纯净深度图上。
    模拟真实相机的曝光、ISO 噪点等物理效应。
    """
    exposure_phys = exposure * 10 + 0.5
    iso_phys = iso * 6400 + 100
    max_range = 2.0 + 1.5 * exposure_phys + 0.001 * iso_phys
    max_range = max_range[:, None, None]
    depth = max_range - F.softplus(max_range - depth, beta=2.0)
    noise_sigma = 0.03 * (1.0 + 2.0 * iso) / (exposure + 0.3)
    depth_dist_scale = depth.detach().clamp(0.3, 20) / 5.0
    depth = depth + torch.randn_like(depth) * noise_sigma[:, None, None] * depth_dist_scale
    return depth


def _safe_normalize(x, dim=-1, eps=1e-6):
    return x / torch.clamp(torch.norm(x, 2, dim=dim, keepdim=True), min=eps)


def _make_separable_gaussian_kernel1d(sigma, device, dtype):
    sigma = max(float(sigma), 1e-3)
    radius = max(1, int(3.0 * sigma + 0.5))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    k = k / torch.clamp(k.sum(), min=1e-12)
    return k


def _separable_gaussian_blur(img, sigma):
    """
    对单通道图像执行可微高斯模糊。
    img: (B, H, W)
    """
    if sigma <= 1e-4:
        return img
    k = _make_separable_gaussian_kernel1d(sigma, img.device, img.dtype)
    r = (k.numel() - 1) // 2
    x = img[:, None]
    kx = k.view(1, 1, 1, -1)
    ky = k.view(1, 1, -1, 1)
    x = F.pad(x, (r, r, 0, 0), mode='reflect')
    x = F.conv2d(x, kx)
    x = F.pad(x, (0, 0, r, r), mode='reflect')
    x = F.conv2d(x, ky)
    return x[:, 0]
