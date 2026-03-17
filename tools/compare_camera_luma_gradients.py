#!/usr/bin/env python3
import argparse
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import torch


@dataclass
class GradPack:
    loss: float
    grads: Dict[str, torch.Tensor]


@dataclass
class PerfPack:
    ms_per_iter: float
    peak_mem_mb: float


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def safe_rel_err(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    num = torch.norm(a - b)
    den = torch.norm(b).clamp_min(eps)
    return float((num / den).detach().cpu())


def safe_cos(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    a1 = a.flatten()
    b1 = b.flatten()
    na = torch.norm(a1)
    nb = torch.norm(b1)
    if float(na.detach().cpu()) < eps or float(nb.detach().cpu()) < eps:
        return float("nan")
    return float(torch.dot(a1, b1).div(na * nb).detach().cpu())


def setup_import_paths(repo_root: str) -> None:
    src_path = os.path.join(repo_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    old_ld = os.environ.get("LD_LIBRARY_PATH", "")
    if torch_lib not in old_ld.split(":"):
        os.environ["LD_LIBRARY_PATH"] = torch_lib + (":" + old_ld if old_ld else "")


def profile_toggles(profile: str) -> Dict[str, Any]:
    p = profile.lower()
    if p == "low":
        return dict(
            camera_preset="low",
            cam_enable_shadow=False,
            cam_enable_specular=True,
            cam_enable_distortion=False,
            cam_enable_flare=False,
            cam_enable_motion_blur=False,
            cam_enable_rolling=False,
        )
    if p == "high":
        return dict(
            camera_preset="high",
            cam_enable_shadow=True,
            cam_enable_specular=True,
            cam_enable_distortion=True,
            cam_enable_flare=False,
            cam_enable_motion_blur=False,
            cam_enable_rolling=False,
        )
    return dict(
        camera_preset="ultra",
        cam_enable_shadow=True,
        cam_enable_specular=True,
        cam_enable_distortion=True,
        cam_enable_flare=True,
        cam_enable_motion_blur=True,
        cam_enable_rolling=True,
    )


def make_env(repo_root: str, impl: str, seed: int, batch_size: int, width: int, height: int, profile: str):
    seed_all(seed)
    setup_import_paths(repo_root)
    from env_cuda import Env

    device = "cuda"
    kwargs = profile_toggles(profile)
    env = Env(
        batch_size=batch_size,
        width=width,
        height=height,
        grad_decay=0.4,
        device=device,
        single=True,
        diff_sensor_impl={"camera_luma": impl, "active_depth": "python"},
        **kwargs,
    )
    return env


def one_grad(env, batch_size: int) -> GradPack:
    device = torch.device("cuda")
    fov = torch.full((batch_size,), 0.53, device=device, requires_grad=True)
    exposure = torch.full((batch_size,), 0.5, device=device, requires_grad=True)
    iso = torch.full((batch_size,), 0.5, device=device, requires_grad=True)

    y = env.render_main_luma_diff(fov, exposure, iso)
    loss = y.mean() + 0.1 * y.square().mean()
    loss.backward()

    return GradPack(
        loss=float(loss.detach().cpu()),
        grads={
            "fov": fov.grad.detach().clone(),
            "exposure": exposure.grad.detach().clone(),
            "iso": iso.grad.detach().clone(),
        },
    )


def bench(env, batch_size: int, iters: int, warmup: int) -> PerfPack:
    device = torch.device("cuda")

    for _ in range(max(warmup, 0)):
        fov = torch.full((batch_size,), 0.53, device=device, requires_grad=True)
        exposure = torch.full((batch_size,), 0.5, device=device, requires_grad=True)
        iso = torch.full((batch_size,), 0.5, device=device, requires_grad=True)
        y = env.render_main_luma_diff(fov, exposure, iso)
        loss = y.mean()
        loss.backward()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    for _ in range(max(iters, 1)):
        fov = torch.full((batch_size,), 0.53, device=device, requires_grad=True)
        exposure = torch.full((batch_size,), 0.5, device=device, requires_grad=True)
        iso = torch.full((batch_size,), 0.5, device=device, requires_grad=True)
        y = env.render_main_luma_diff(fov, exposure, iso)
        loss = y.mean()
        loss.backward()

    end.record()
    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
    return PerfPack(ms_per_iter=total_ms / max(iters, 1), peak_mem_mb=peak_mem_mb)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare camera_luma python/cuda gradients and performance.")
    parser.add_argument("--repo_root", type=str, default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=48)
    parser.add_argument("--profile", type=str, default="high", choices=["low", "high", "ultra"])
    parser.add_argument("--bench_iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This script requires CUDA.")

    print("[info] gradient compare: python reference...")
    env_py = make_env(args.repo_root, "python", args.seed, args.batch_size, args.width, args.height, args.profile)
    g_py = one_grad(env_py, args.batch_size)

    print("[info] gradient compare: cuda implementation...")
    env_cu = make_env(args.repo_root, "cuda", args.seed, args.batch_size, args.width, args.height, args.profile)
    g_cu = one_grad(env_cu, args.batch_size)

    print("\n=== camera_luma gradient compare ===")
    print(f"profile        : {args.profile}")
    print(f"seed           : {args.seed}")
    print(f"loss_python    : {g_py.loss:.8f}")
    print(f"loss_cuda      : {g_cu.loss:.8f}")
    print(f"loss_abs_diff  : {abs(g_cu.loss - g_py.loss):.8f}")
    print("\nparam       | cos_sim    | rel_err    | py_grad_mean | cu_grad_mean | finite(py/cu)")
    print("-" * 88)
    for k in ("fov", "exposure", "iso"):
        p = g_py.grads[k]
        c = g_cu.grads[k]
        print(
            f"{k:<11}| {safe_cos(c, p):>10.6f} | {safe_rel_err(c, p):>10.6f} | "
            f"{float(p.mean().cpu()):>12.6e} | {float(c.mean().cpu()):>12.6e} | "
            f"{bool(torch.isfinite(p).all().item())}/{bool(torch.isfinite(c).all().item())}"
        )

    print("\n[info] performance benchmark: python...")
    env_py_b = make_env(args.repo_root, "python", args.seed + 1, args.batch_size, args.width, args.height, args.profile)
    p_py = bench(env_py_b, args.batch_size, args.bench_iters, args.warmup)

    print("[info] performance benchmark: cuda...")
    env_cu_b = make_env(args.repo_root, "cuda", args.seed + 1, args.batch_size, args.width, args.height, args.profile)
    p_cu = bench(env_cu_b, args.batch_size, args.bench_iters, args.warmup)

    speedup = p_py.ms_per_iter / max(p_cu.ms_per_iter, 1e-9)
    mem_ratio = p_py.peak_mem_mb / max(p_cu.peak_mem_mb, 1e-9)

    print("\n=== camera_luma performance ===")
    print(f"python: {p_py.ms_per_iter:.3f} ms/iter, peak_mem={p_py.peak_mem_mb:.1f} MB")
    print(f"cuda  : {p_cu.ms_per_iter:.3f} ms/iter, peak_mem={p_cu.peak_mem_mb:.1f} MB")
    print(f"speedup (py/cuda)     : {speedup:.2f}x")
    print(f"memory ratio (py/cuda): {mem_ratio:.2f}x")


if __name__ == "__main__":
    main()
