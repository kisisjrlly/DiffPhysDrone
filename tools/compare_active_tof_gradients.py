#!/usr/bin/env python3
import argparse
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch


@dataclass
class GradPack:
    loss: float
    grads: Dict[str, torch.Tensor]


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


def run_one(
    repo_root: str,
    impl: str,
    seed: int,
    batch_size: int,
    imx_w: int,
    imx_h: int,
    tof_w: int,
    tof_h: int,
    loss_mode: str,
) -> GradPack:
    seed_all(seed)
    setup_import_paths(repo_root)

    # 延迟导入，确保路径和环境变量已经就绪
    from env_cuda import Env

    device = torch.device("cuda")
    env = Env(
        batch_size=batch_size,
        width=imx_w,
        height=imx_h,
        grad_decay=0.4,
        device=device,
        single=True,
        tof_width=tof_w,
        tof_height=tof_h,
        diff_sensor_impl={"yuv": "python", "active_tof": impl},
    )

    power = torch.full((batch_size,), 0.5, device=device, requires_grad=True)
    exposure = torch.full((batch_size,), 0.5, device=device, requires_grad=True)
    gain = torch.full((batch_size,), 0.5, device=device, requires_grad=True)

    depth, conf = env.render_active_tof_diff(power, exposure, gain)

    if loss_mode == "conf":
        loss = conf.mean()
    elif loss_mode == "depth":
        loss = depth.mean()
    else:
        loss = depth.mean() + conf.mean()

    loss.backward()

    grads = {
        "power": power.grad.detach().clone(),
        "exposure": exposure.grad.detach().clone(),
        "gain": gain.grad.detach().clone(),
    }

    return GradPack(loss=float(loss.detach().cpu()), grads=grads)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare active ToF gradients between python and cuda implementations.")
    parser.add_argument("--repo_root", type=str, default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--imx_width", type=int, default=320)
    parser.add_argument("--imx_height", type=int, default=240)
    parser.add_argument("--tof_width", type=int, default=64)
    parser.add_argument("--tof_height", type=int, default=48)
    parser.add_argument("--loss_mode", type=str, default="conf", choices=["conf", "depth", "both"],
                        help="conf: 无噪声路径更稳定；both: 与训练更接近")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This script requires CUDA.")

    print("[info] running python reference...")
    py = run_one(
        repo_root=args.repo_root,
        impl="python",
        seed=args.seed,
        batch_size=args.batch_size,
        imx_w=args.imx_width,
        imx_h=args.imx_height,
        tof_w=args.tof_width,
        tof_h=args.tof_height,
        loss_mode=args.loss_mode,
    )

    print("[info] running cuda implementation...")
    cu = run_one(
        repo_root=args.repo_root,
        impl="cuda",
        seed=args.seed,
        batch_size=args.batch_size,
        imx_w=args.imx_width,
        imx_h=args.imx_height,
        tof_w=args.tof_width,
        tof_h=args.tof_height,
        loss_mode=args.loss_mode,
    )

    print("\n=== active_tof grad compare ===")
    print(f"loss_mode      : {args.loss_mode}")
    print(f"seed           : {args.seed}")
    print(f"loss_python    : {py.loss:.8f}")
    print(f"loss_cuda      : {cu.loss:.8f}")
    print(f"loss_abs_diff  : {abs(cu.loss - py.loss):.8f}")

    print("\nparam       | cos_sim    | rel_err    | py_grad_mean | cu_grad_mean | finite(py/cu)")
    print("-" * 88)
    for k in ("power", "exposure", "gain"):
        g_py = py.grads[k]
        g_cu = cu.grads[k]
        cos = safe_cos(g_cu, g_py)
        rel = safe_rel_err(g_cu, g_py)
        py_mean = float(g_py.mean().detach().cpu())
        cu_mean = float(g_cu.mean().detach().cpu())
        py_ok = bool(torch.isfinite(g_py).all().item())
        cu_ok = bool(torch.isfinite(g_cu).all().item())
        print(f"{k:<11}| {cos:>10.6f} | {rel:>10.6f} | {py_mean:>12.6e} | {cu_mean:>12.6e} | {py_ok}/{cu_ok}")

    print("\n[hint] 建议先看 cos_sim 是否接近 1，再看 rel_err；若 loss_mode=both，随机噪声会放大差异。")


if __name__ == "__main__":
    main()
