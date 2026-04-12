"""
DiffPhysDrone — slim entry point.

All heavy logic lives in config.py, trainer.py, train_utils.py, etc.
This file only wires together: arg parsing, object creation, and the training call.
"""
import os
import time

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


def main():
    # ── 1. Parse arguments & validate ────────────────────────────────────
    args = parse_args()

    # ── 2. Device setup ──────────────────────────────────────────────────
    device = torch.device('cuda')
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ── 3. WandB + checkpoint dir ────────────────────────────────────────
    run_name = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
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
    train(args, model, env_train, env_full,
          optim, sched, scaler, vis, checkpoint_dir, device)


if __name__ == '__main__':
    main()
