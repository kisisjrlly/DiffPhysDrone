#!/usr/bin/env python3
"""Verify the minimal active-sensing fused CUDA sensor against PyTorch reference.

The CUDA op intentionally treats geometry/raw depth as non-differentiable, but
must match the reference forward outputs and gradients w.r.t.
power/exposure/gain.
"""
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
from train_utils import build_env  # noqa: E402
import quadsim_cuda  # noqa: E402


SCENE_TO_ID = {'glare': 0, 'specular': 1, 'dark': 2}


def sensor_reference(depth, mask, power, exposure, gain, regime_id, min_valid, max_range):
    raw = depth.clamp(float(min_valid), float(max_range))
    mask = mask.clamp(0.0, 1.0)
    p = power.clamp(0, 1)[:, None, None]
    e = exposure.clamp(0, 1)[:, None, None]
    g = gain.clamp(0, 1)[:, None, None]

    base = torch.sigmoid((float(max_range) - raw) / 0.9)
    base = base * (0.90 + 0.18 * p + 0.08 * e + 0.06 * g)
    quality = base
    effect = torch.zeros_like(raw)

    if int(regime_id) == 0:
        overexp = torch.sigmoid((e - 0.18) / 0.045)
        rescue = torch.sigmoid((p - 0.56) / 0.08)
        penalty = mask * overexp * (0.95 - 0.55 * rescue)
        bonus = mask * rescue * (1.0 - overexp) * 0.20
        quality = quality - penalty + bonus
        effect = penalty
    elif int(regime_id) == 1:
        wash = torch.sigmoid((p - 0.38) / 0.07) * (
            0.55 + 0.45 * torch.sigmoid((e - 0.25) / 0.08)
        )
        safe = torch.sigmoid((0.42 - p) / 0.10)
        penalty = mask * wash * 0.92
        bonus = mask * safe * 0.25
        quality = quality - penalty + bonus
        effect = penalty
    else:
        need = mask * 0.82
        rescue = (
            torch.sigmoid((e - 0.48) / 0.08) * 0.55
            + torch.sigmoid((g - 0.42) / 0.08) * 0.45
        )
        penalty = need * (1.0 - rescue)
        quality = quality - penalty + mask * rescue * 0.18
        effect = penalty

    d4 = raw[:, None]
    d_far = F.max_pool2d(d4, 3, stride=1, padding=1)[:, 0]
    d_near = -F.max_pool2d(-d4, 3, stride=1, padding=1)[:, 0]
    edge = ((d_far - d_near) / (raw + 0.2)).clamp(0.0, 1.0)
    quality = (quality - 0.12 * edge).clamp(0.0, 1.0)

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
    return depth, mask


def deterministic_weights(shape, device):
    n = math.prod(shape)
    w = torch.linspace(-0.7, 0.9, n, device=device, dtype=torch.float32).reshape(shape)
    return w


def compare_tensor(name, ref, cuda, atol):
    diff = (ref - cuda).abs()
    max_abs = float(diff.max().detach().cpu())
    mean_abs = float(diff.mean().detach().cpu())
    ok = max_abs <= float(atol)
    return ok, f'{name}: max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} tol={atol:.1e}'


def run_case(scene, power_vals, exposure_vals, gain_vals, args):
    device = torch.device(args.device)
    depth, mask = make_synthetic_inputs(len(power_vals), args.height, args.width, device)
    regime_id = SCENE_TO_ID[scene]
    min_valid, max_range = 0.3, 6.0

    p_ref = torch.tensor(power_vals, device=device, dtype=torch.float32, requires_grad=True)
    e_ref = torch.tensor(exposure_vals, device=device, dtype=torch.float32, requires_grad=True)
    g_ref = torch.tensor(gain_vals, device=device, dtype=torch.float32, requires_grad=True)
    p_cuda = p_ref.detach().clone().requires_grad_(True)
    e_cuda = e_ref.detach().clone().requires_grad_(True)
    g_cuda = g_ref.detach().clone().requires_grad_(True)

    ref_out = sensor_reference(depth, mask, p_ref, e_ref, g_ref, regime_id, min_valid, max_range)
    cuda_out = active_sensing_sensor(depth, mask, p_cuda, e_cuda, g_cuda, regime_id, min_valid, max_range)

    weights = deterministic_weights(depth.shape, device)
    # Exercise every differentiable output path. hard_valid is intentionally not
    # included because the reference also treats the hard threshold as non-diff.
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
    env = build_env(cfg.batch_size, cfg, device)
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
    depth, mask = make_synthetic_inputs(args.perf_batch, args.height, args.width, device)
    regime_id = SCENE_TO_ID['glare']
    power = torch.full((args.perf_batch,), 0.45, device=device, requires_grad=True)
    exposure = torch.full((args.perf_batch,), 0.30, device=device, requires_grad=True)
    gain = torch.full((args.perf_batch,), 0.35, device=device, requires_grad=True)
    weights = deterministic_weights(depth.shape, device)

    def ref_step():
        p = power.detach().clone().requires_grad_(True)
        e = exposure.detach().clone().requires_grad_(True)
        g = gain.detach().clone().requires_grad_(True)
        out = sensor_reference(depth, mask, p, e, g, regime_id, 0.3, 6.0)
        loss = (out[0] * weights).mean() + 0.1 * out[1].mean()
        loss.backward()

    def cuda_step():
        p = power.detach().clone().requires_grad_(True)
        e = exposure.detach().clone().requires_grad_(True)
        g = gain.detach().clone().requires_grad_(True)
        out = active_sensing_sensor(depth, mask, p, e, g, regime_id, 0.3, 6.0)
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
    parser.add_argument('--grad_atol', type=float, default=3e-5)
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
        for cam in camera_sets:
            for ok, line in run_case(scene, *cam, args):
                all_ok = all_ok and ok
                print(('PASS ' if ok else 'FAIL ') + line)

    smoke_ok, smoke_lines = run_env_smoke(args)
    all_ok = all_ok and smoke_ok
    for line in smoke_lines:
        print(('PASS ' if smoke_ok else 'FAIL ') + line)

    if args.perf:
        ref_s, cuda_s = run_perf(args)
        speedup = ref_s / max(cuda_s, 1e-9)
        print(f'PERF reference={ref_s:.4f}s cuda={cuda_s:.4f}s speedup={speedup:.2f}x')

    if not all_ok:
        raise SystemExit(1)
    print('All active-sensing CUDA sensor checks passed.')


if __name__ == '__main__':
    main()
