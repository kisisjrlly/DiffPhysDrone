#!/usr/bin/env python3
"""Verify the D455-like fused CUDA sensor against a PyTorch reference."""
import argparse
import math
import os
import sys
import time

import torch
import torch.nn.functional as F


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(REPO_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from autograd_ops import active_sensing_sensor  # noqa: E402
from config import build_parser, parse_diff_sensor_impl, parse_scenarios, validate_args  # noqa: E402
from env_cuda import Env  # noqa: E402


SCENE_TO_ID = {'glare': 0, 'specular': 1, 'dark': 2}
EXPOSURE_T_MIN = 0.25
EXPOSURE_T_SPAN = 2.75
ISO_GAIN_BASE = 1.0
ISO_GAIN_SCALE = 0.8
ISO_GAIN_GAMMA = 0.6
SHOT_NOISE_BASE = 0.01


def iso_to_gain(gain01):
    eps = 1e-4
    eps_gamma = eps ** ISO_GAIN_GAMMA
    denom = max((1.0 + eps) ** ISO_GAIN_GAMMA - eps_gamma, 1e-12)
    shaped = ((gain01.clamp(0.0, 1.0) + eps).pow(ISO_GAIN_GAMMA) - eps_gamma) / denom
    return ISO_GAIN_BASE + ISO_GAIN_SCALE * shaped


def sensor_reference(depth, mask, power, exposure, gain, speed, regime_id, min_valid, max_range):
    raw = depth.clamp(float(min_valid), float(max_range))
    mask = mask.clamp(0.0, 1.0)
    p = power.clamp(0, 1)[:, None, None]
    e01 = exposure.clamp(0, 1)[:, None, None]
    g01 = gain.clamp(0, 1)[:, None, None]
    exposure_t = (EXPOSURE_T_MIN + EXPOSURE_T_SPAN * exposure.clamp(0, 1))[:, None, None]
    gain_scale = iso_to_gain(gain)[:, None, None].clamp_min(1e-6)
    spd = speed.clamp_min(0.0)[:, None, None]

    d4 = raw[:, None]
    d_far = F.max_pool2d(d4, 3, stride=1, padding=1)[:, 0]
    d_near = -F.max_pool2d(-d4, 3, stride=1, padding=1)[:, 0]
    edge = ((d_far - d_near) / (raw + 0.18)).clamp(0.0, 1.0)

    dist = raw / max(float(max_range), 1e-6)
    active_signal = 1.70 * p * exposure_t / (raw.square() + 0.75)
    passive_signal = 0.10 * exposure_t * torch.sqrt(gain_scale)
    signal = active_signal + passive_signal
    ambient_ir = 0.18 + 0.55 * mask
    motion = (spd * exposure_t * 0.075).clamp(0.0, 1.6)
    washout = ambient_ir * exposure_t / (active_signal + 0.20)
    noise_proxy = SHOT_NOISE_BASE * (0.45 + 0.18 * gain_scale) / (signal + 0.08)
    snr = signal / (0.18 + 0.55 * ambient_ir + 0.38 * noise_proxy + 0.45 * motion * (0.20 + edge))
    quality = torch.sigmoid(2.15 * snr - 0.95 * washout - 0.85 * edge - 1.45 * torch.relu(dist - 0.92))
    effect = torch.zeros_like(raw)

    if int(regime_id) == 0:
        overexp = torch.sigmoid((e01 - 0.22) / 0.045)
        gain_sat = torch.sigmoid((g01 - 0.28) / 0.055)
        gain_exposure_sat = torch.sigmoid(((g01 + 0.85 * e01) - 0.52) / 0.070)
        rescue = torch.sigmoid((p - 0.50) / 0.09)
        rescue_window = torch.sigmoid((0.30 - e01) / 0.06)
        joint_sat = torch.sigmoid((p - 0.65) / 0.08) * torch.sigmoid((e01 - 0.32) / 0.06)
        under_power = torch.sigmoid((0.45 - p) / 0.08)
        penalty = mask * (
            0.88 * overexp
            + 0.28 * joint_sat
            + 0.42 * under_power * rescue_window
            + 0.72 * gain_sat
            + 0.44 * gain_exposure_sat
        )
        low_gain_window = torch.sigmoid((0.26 - g01) / 0.06)
        bonus = mask * rescue * rescue_window * low_gain_window * 0.34
        quality = quality - penalty + bonus
        effect = penalty
    elif int(regime_id) == 1:
        power_bloom = torch.sigmoid((p - 0.30) / 0.055) * (0.62 + 0.38 * torch.sigmoid((e01 - 0.22) / 0.07))
        exposure_bloom = torch.sigmoid((e01 - 0.48) / 0.075) * (0.60 + 0.40 * torch.sigmoid((g01 - 0.50) / 0.08))
        gain_bloom = torch.sigmoid((g01 - 0.36) / 0.060) * (0.55 + 0.45 * torch.sigmoid((e01 - 0.28) / 0.07))
        safe = (
            torch.sigmoid((0.42 - p) / 0.070)
            * torch.sigmoid((0.52 - e01) / 0.08)
            * torch.sigmoid((0.42 - g01) / 0.07)
        )
        very_safe = torch.sigmoid((0.24 - p) / 0.055) * torch.sigmoid((0.30 - e01) / 0.07)
        penalty = mask * (1.06 * power_bloom + 0.58 * exposure_bloom + 0.74 * gain_bloom)
        bonus = mask * (0.38 * safe + 0.18 * very_safe)
        quality = quality - penalty + bonus
        effect = penalty
    else:
        exposure_lift = torch.sigmoid((e01 - 0.62) / 0.070)
        gain_lift = torch.sigmoid((g01 - 0.52) / 0.075)
        projector_lift = torch.sigmoid((p - 0.45) / 0.10)
        rescue = (
            exposure_lift * (
                0.10
                + 0.70 * gain_lift
                + 0.20 * projector_lift * gain_lift
            )
        ).clamp(max=1.0)
        need = mask * 0.92
        penalty = need * (1.0 - rescue)
        quality = quality - penalty + mask * rescue * 0.24
        effect = penalty

    quality = quality.clamp(0.0, 1.0)
    valid_prob = torch.sigmoid((quality - 0.42) / 0.055)
    hard_valid = (valid_prob > 0.5).to(raw.dtype)
    valid_st = hard_valid.detach() - valid_prob.detach() + valid_prob
    depth_obs = raw * valid_st
    quality_obs = quality * valid_st
    return depth_obs, quality_obs, quality, valid_prob, hard_valid, effect


def make_synthetic_inputs(batch, height, width, device):
    y = torch.linspace(-1.0, 1.0, width, device=device)
    z = torch.linspace(-1.0, 1.0, height, device=device)
    yy, zz = torch.meshgrid(y, z, indexing='xy')
    yy = yy.unsqueeze(0).expand(batch, -1, -1)
    zz = zz.unsqueeze(0).expand(batch, -1, -1)
    b = torch.arange(batch, device=device, dtype=torch.float32)[:, None, None]
    depth = 2.5 + 0.9 * yy + 0.35 * torch.sin(3.0 * zz + 0.7 * b)
    depth = depth + 0.18 * torch.sin(5.0 * yy + 1.3 * zz)
    depth = depth.clamp(0.45, 5.7).contiguous()
    center = torch.linspace(-0.55, 0.55, batch, device=device)[:, None, None]
    mask = torch.sigmoid((0.30 - (yy - center).abs()) / 0.07)
    mask = (mask * torch.sigmoid((0.72 - zz.abs()) / 0.08)).contiguous()
    speed = torch.linspace(0.05, 1.15, batch, device=device)
    return depth, mask, speed


def deterministic_weights(shape, device):
    n = math.prod(shape)
    return torch.linspace(-0.7, 0.9, n, device=device, dtype=torch.float32).reshape(shape)


def compare_tensor(name, ref, cuda, atol):
    diff = (ref - cuda).abs()
    max_abs = float(diff.max().detach().cpu())
    mean_abs = float(diff.mean().detach().cpu())
    ok = max_abs <= float(atol)
    return ok, f'{name}: max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} tol={atol:.1e}'


def run_case(scene, power_vals, exposure_vals, gain_vals, args):
    device = torch.device(args.device)
    depth, mask, speed = make_synthetic_inputs(len(power_vals), args.height, args.width, device)
    regime_id = SCENE_TO_ID[scene]
    min_valid, max_range = 0.3, 6.0

    p_ref = torch.tensor(power_vals, device=device, dtype=torch.float32, requires_grad=True)
    e_ref = torch.tensor(exposure_vals, device=device, dtype=torch.float32, requires_grad=True)
    g_ref = torch.tensor(gain_vals, device=device, dtype=torch.float32, requires_grad=True)
    p_cuda = p_ref.detach().clone().requires_grad_(True)
    e_cuda = e_ref.detach().clone().requires_grad_(True)
    g_cuda = g_ref.detach().clone().requires_grad_(True)

    ref_out = sensor_reference(depth, mask, p_ref, e_ref, g_ref, speed, regime_id, min_valid, max_range)
    cuda_out = active_sensing_sensor(
        depth, mask, p_cuda, e_cuda, g_cuda, speed, regime_id, min_valid, max_range,
        EXPOSURE_T_MIN, EXPOSURE_T_SPAN, ISO_GAIN_BASE, ISO_GAIN_SCALE, ISO_GAIN_GAMMA, SHOT_NOISE_BASE)

    weights = deterministic_weights(depth.shape, device)
    ref_loss = (
        (ref_out[0] * weights).mean()
        + 0.37 * (ref_out[1] * weights.flip(-1)).mean()
        + 0.11 * ref_out[2].mean()
        + 0.23 * ref_out[3].mean()
        + 0.07 * ref_out[5].mean()
    )
    cuda_loss = (
        (cuda_out[0] * weights).mean()
        + 0.37 * (cuda_out[1] * weights.flip(-1)).mean()
        + 0.11 * cuda_out[2].mean()
        + 0.23 * cuda_out[3].mean()
        + 0.07 * cuda_out[5].mean()
    )
    ref_loss.backward()
    cuda_loss.backward()

    checks = []
    names = ['depth_obs', 'quality_obs', 'quality', 'valid_prob', 'hard_valid', 'effect']
    for name, ref, cud in zip(names, ref_out, cuda_out):
        checks.append(compare_tensor(f'{scene}/{name}', ref, cud, args.forward_atol))
    for name, ref_grad, cuda_grad in [
        ('grad_power', p_ref.grad, p_cuda.grad),
        ('grad_exposure', e_ref.grad, e_cuda.grad),
        ('grad_gain', g_ref.grad, g_cuda.grad),
    ]:
        checks.append(compare_tensor(f'{scene}/{name}', ref_grad, cuda_grad, args.grad_atol))
    return checks


def run_env_smoke(args):
    device = torch.device(args.device)
    parser = build_parser()
    cfg = parser.parse_args([])
    cfg.batch_size = 4
    cfg.depth_height = args.height
    cfg.depth_width = args.width
    cfg.diff_sensor_impl = parse_diff_sensor_impl(['diff_depth=cuda'])
    cfg.scenarios = parse_scenarios(['glare', 'specular', 'dark'])
    cfg.sun_glare_eval_slot = None
    validate_args(cfg)
    env = Env(
        cfg.batch_size,
        int(cfg.depth_width),
        int(cfg.depth_height),
        cfg.grad_decay,
        device,
        eval_mode=True,
        fov_x_half_tan=cfg.fov_x_half_tan,
        cam_angle=cfg.cam_angle,
        cam_power_baseline=cfg.cam_power_baseline,
        camera_control_mode=cfg.camera_control_mode,
        sensor_grad_mode=cfg.sensor_grad_mode,
        fixed_camera_power=cfg.fixed_camera_power,
        fixed_camera_exposure=cfg.fixed_camera_exposure,
        fixed_camera_gain=cfg.fixed_camera_gain,
        fixed_random_power_min=cfg.fixed_random_power_min,
        fixed_random_power_max=cfg.fixed_random_power_max,
        fixed_random_exposure_min=cfg.fixed_random_exposure_min,
        fixed_random_exposure_max=cfg.fixed_random_exposure_max,
        fixed_random_gain_min=cfg.fixed_random_gain_min,
        fixed_random_gain_max=cfg.fixed_random_gain_max,
        cam_exposure_t_min=cfg.cam_exposure_t_min,
        cam_exposure_t_span=cfg.cam_exposure_t_span,
        cam_exposure_eff_min=cfg.cam_exposure_eff_min,
        cam_exposure_eff_max=cfg.cam_exposure_eff_max,
        cam_iso_gain_base=cfg.cam_iso_gain_base,
        cam_iso_gain_scale=cfg.cam_iso_gain_scale,
        cam_iso_gain_gamma=cfg.cam_iso_gain_gamma,
        cam_shot_noise_base=cfg.cam_shot_noise_base,
        depth_min_valid=cfg.depth_min_valid,
        depth_max_range=cfg.depth_max_range,
        scenarios=cfg.scenarios,
        sun_glare_eval_slot=None,
        diff_sensor_impl=cfg.diff_sensor_impl,
    )
    ok = True
    lines = []
    for scene in SCENE_TO_ID:
        env.reset(scene_name=scene)
        p = torch.full((cfg.batch_size,), 0.45, device=device)
        e = torch.full((cfg.batch_size,), 0.30, device=device)
        g = torch.full((cfg.batch_size,), 0.35, device=device)
        depth_obs, quality_obs = env.render_diff_depth(p, e, g)
        finite = torch.isfinite(depth_obs).all() and torch.isfinite(quality_obs).all()
        aux = env.get_last_diff_depth_train_aux()
        valid_prob = aux.get('valid_prob_map')
        finite = finite and valid_prob is not None and torch.isfinite(valid_prob).all()
        ok = ok and bool(finite)
        lines.append(
            f'env/{scene}: finite={bool(finite)} '
            f'depth_mean={float(depth_obs.mean().detach().cpu()):.3f} '
            f'valid_mean={float(valid_prob.mean().detach().cpu()):.3f}'
        )
    return ok, lines


def run_perf(args):
    device = torch.device(args.device)
    depth, mask, speed = make_synthetic_inputs(args.perf_batch, args.height, args.width, device)
    regime_id = SCENE_TO_ID['glare']
    power = torch.full((args.perf_batch,), 0.45, device=device, requires_grad=True)
    exposure = torch.full((args.perf_batch,), 0.30, device=device, requires_grad=True)
    gain = torch.full((args.perf_batch,), 0.35, device=device, requires_grad=True)
    weights = deterministic_weights(depth.shape, device)

    def ref_step():
        p = power.detach().clone().requires_grad_(True)
        e = exposure.detach().clone().requires_grad_(True)
        g = gain.detach().clone().requires_grad_(True)
        out = sensor_reference(depth, mask, p, e, g, speed, regime_id, 0.3, 6.0)
        loss = (out[0] * weights).mean() + 0.1 * out[1].mean()
        loss.backward()

    def cuda_step():
        p = power.detach().clone().requires_grad_(True)
        e = exposure.detach().clone().requires_grad_(True)
        g = gain.detach().clone().requires_grad_(True)
        out = active_sensing_sensor(
            depth, mask, p, e, g, speed, regime_id, 0.3, 6.0,
            EXPOSURE_T_MIN, EXPOSURE_T_SPAN, ISO_GAIN_BASE, ISO_GAIN_SCALE, ISO_GAIN_GAMMA, SHOT_NOISE_BASE)
        loss = (out[0] * weights).mean() + 0.1 * out[1].mean()
        loss.backward()

    for _ in range(10):
        ref_step()
        cuda_step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.perf_iters):
        ref_step()
    torch.cuda.synchronize()
    ref_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(args.perf_iters):
        cuda_step()
    torch.cuda.synchronize()
    cuda_s = time.perf_counter() - t0
    return ref_s, cuda_s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--height', type=int, default=48)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--forward_atol', type=float, default=1e-5)
    parser.add_argument('--grad_atol', type=float, default=5e-4)
    parser.add_argument('--perf', action='store_true')
    parser.add_argument('--perf_batch', type=int, default=64)
    parser.add_argument('--perf_iters', type=int, default=200)
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for fused sensor verification')
    torch.manual_seed(7)

    all_ok = True
    camera_sets = [
        ([0.22, 0.58, 0.78], [0.12, 0.23, 0.50], [0.18, 0.35, 0.70]),
        ([0.35, 0.44, 0.62], [0.16, 0.31, 0.72], [0.28, 0.46, 0.82]),
    ]
    for scene in SCENE_TO_ID:
        for p_vals, e_vals, g_vals in camera_sets:
            checks = run_case(scene, p_vals, e_vals, g_vals, args)
            for ok, msg in checks:
                all_ok = all_ok and ok
                print(('OK  ' if ok else 'BAD ') + msg)

    env_ok, env_lines = run_env_smoke(args)
    all_ok = all_ok and env_ok
    for line in env_lines:
        print(('OK  ' if env_ok else 'BAD ') + line)

    if args.perf:
        ref_s, cuda_s = run_perf(args)
        speedup = ref_s / max(cuda_s, 1e-9)
        print(f'perf: reference={ref_s:.4f}s cuda={cuda_s:.4f}s speedup={speedup:.2f}x')

    if not all_ok:
        raise SystemExit(1)
    print('active sensing CUDA verification passed')


if __name__ == '__main__':
    main()
