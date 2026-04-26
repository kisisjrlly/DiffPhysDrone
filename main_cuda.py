"""
DiffPhysDrone — slim entry point.

All heavy logic lives in config.py, trainer.py, train_utils.py, etc.
This file only wires together: arg parsing, object creation, and the training call.
"""
import os
import time
import faulthandler

import torch
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb

# torch.autograd.set_detect_anomaly(True)

from config import parse_args, print_runtime_mode
from model import Model
from rerun_vis import RerunVis
from train_utils import build_env, estimate_optimizer_steps
from trainer import train

faulthandler.enable(all_threads=True)


def _print_cuda_failure_summary(args, device, exc: Exception):
    if device.type != 'cuda' or not torch.cuda.is_available():
        return
    dev_idx = device.index if device.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(dev_idx)
    mib = 1024 ** 2
    allocated = torch.cuda.memory_allocated(dev_idx) / mib
    reserved = torch.cuda.memory_reserved(dev_idx) / mib
    peak_alloc = torch.cuda.max_memory_allocated(dev_idx) / mib
    peak_reserved = torch.cuda.max_memory_reserved(dev_idx) / mib
    print("\n[diag] CUDA failure summary")
    print(f"[diag] exception: {exc}")
    print(f"[diag] device: {props.name}, total={props.total_memory / mib:.0f} MiB")
    print(
        f"[diag] allocated={allocated:.0f} MiB reserved={reserved:.0f} MiB "
        f"peak_allocated={peak_alloc:.0f} MiB peak_reserved={peak_reserved:.0f} MiB"
    )
    print(
        f"[diag] config: batch_size={args.batch_size}, timesteps={args.timesteps}, "
        f"depth_hw={args.depth_height}x{args.depth_width}, amp={args.amp}, "
        f"scenarios={args.scenarios}"
    )
    print(
        "[diag] 当前 diff_depth python 渲染 + full BPTT 会把整段 rollout 的计算图保留在显存里，"
        "在 16GB GPU 上 batch 太大时容易先触发 OOM，随后表现成 cuDNN 初始化失败。"
    )
    print(
        "[diag] 建议优先尝试：1) 降低 --batch_size 到 64~80；"
        "2) 需要更大等效 batch 时再考虑 TBPTT；"
        "3) 如果仍吃紧，再降低深度分辨率。"
    )


def main():
    # ── 1. Parse arguments & validate ────────────────────────────────────
    args = parse_args()

    # ── 2. Device setup ──────────────────────────────────────────────────
    device = torch.device('cuda')
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ── 3. WandB + checkpoint dir ────────────────────────────────────────
    mode_tag = f"cam-{args.camera_control_mode}_grad-{args.sensor_grad_mode}_depth-{args.policy_depth_mode}"
    run_name = f"{mode_tag}_{time.strftime('%Y%m%d_%H%M%S')}"
    ckpt_timestamp = time.strftime('%Y-%m-%d-%H-%M-%S')
    checkpoint_dir = os.path.join('checkpoint', ckpt_timestamp)
    os.makedirs(checkpoint_dir, exist_ok=True)

    wandb.init(
        project="diff-simulation",
        name=run_name,
        config=args,
        settings=wandb.Settings(code_dir="."),
        mode="disabled" if args.wandb_disabled else "online",
    )
    wandb.save("*.py"); wandb.save("src/*.cu"); wandb.save("src/*.cpp")
    wandb.save("src/*.py"); wandb.save("configs/*.args"); wandb.save("*.sh")

    # Configuration banner
    print("\n" + "=" * 30 + " Configuration " + "=" * 30)
    for k, v in vars(args).items():
        print(f"{k:<30}: {v}")
    print(f"{'checkpoint_dir':<30}: {checkpoint_dir}")
    print("=" * 75 + "\n")
    print_runtime_mode(args)

    # ── 4. Create environments ───────────────────────────────────────────
    env_train = build_env(args.batch_size, args, device)
    env_full = env_train
    if (args.tbptt_enable and args.hybrid_full_bptt_every > 0
            and args.hybrid_full_bptt_batch_size > 0
            and args.hybrid_full_bptt_batch_size != args.batch_size):
        env_full = build_env(args.hybrid_full_bptt_batch_size, args, device)
        print(f"[info] 混合调度启用：完整BPTT每 {args.hybrid_full_bptt_every} 轮一次，batch={args.hybrid_full_bptt_batch_size}")
    # ── 5. Create model ──────────────────────────────────────────────────
    obs_dim = 7 if args.no_odom else 10
    model = Model(
        obs_dim, 6,
        include_camera_state_in_obs=args.include_camera_state_in_obs,
        use_policy_intent=args.policy_output_intent,
        intent_dim=9,
        depth_nn_width=args.depth_nn_width,
        depth_nn_height=args.depth_nn_height,
        depth_use_pipeline=args.depth_use_pipeline,
        depth_min_valid=args.depth_min_valid,
        depth_max_range=args.depth_max_range,
    ).to(device)

    use_amp = bool(args.amp and device.type == 'cuda')
    scaler = GradScaler(enabled=use_amp)

    # ── 6. Visualization ─────────────────────────────────────────────────
    vis = RerunVis(
        enabled=(args.vis_enable and args.vis_backend == 'rerun'),
        app_id='DiffPhysDrone-Train',
        spawn=args.vis_spawn,
    )

    # ── 7. Resume checkpoint ─────────────────────────────────────────────
    if args.resume:
        print(f"[info] 从 {args.resume} 恢复训练")
        state_dict = torch.load(args.resume, map_location=device)
        missing, unexpected = model.load_state_dict(state_dict, False)
        if missing:
            print("missing_keys:", missing)
        if unexpected:
            print("unexpected_keys:", unexpected)

    # ── 8. Optimizer & scheduler ─────────────────────────────────────────
    optim = AdamW(model.parameters(), args.lr)
    sched = CosineAnnealingLR(optim, estimate_optimizer_steps(args), args.lr * 0.01)

    # ── 9. Train ─────────────────────────────────────────────────────────
    try:
        train(args, model, env_train, env_full,
              optim, sched, scaler, vis, checkpoint_dir, device)
    except RuntimeError as exc:
        msg = str(exc).lower()
        if device.type == 'cuda' and (
            'out of memory' in msg or 'cudnn_status_not_initialized' in msg
        ):
            _print_cuda_failure_summary(args, device, exc)
        raise


if __name__ == '__main__':
    main()
