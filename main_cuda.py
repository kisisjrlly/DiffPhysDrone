"""DiffPhysDrone grayscale/IMX900 training entry point."""

import faulthandler
import os
import shutil
import time

import torch
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb

from config import parse_args, print_runtime_mode
from model import Model
from rerun_vis import RerunVis
from sensors.imx900_calibration import IMX900Calibration
from train_utils import build_env, estimate_optimizer_steps
from trainer import train

faulthandler.enable(all_threads=True)


def _print_cuda_failure_summary(args, device, exc):
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    dev_idx = device.index if device.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(dev_idx)
    mib = 1024 ** 2
    print("\n[diag] CUDA failure summary")
    print(f"[diag] exception: {exc}")
    print(f"[diag] device: {props.name}, total={props.total_memory / mib:.0f} MiB")
    print(
        f"[diag] allocated={torch.cuda.memory_allocated(dev_idx) / mib:.0f} MiB "
        f"reserved={torch.cuda.memory_reserved(dev_idx) / mib:.0f} MiB "
        f"peak_allocated={torch.cuda.max_memory_allocated(dev_idx) / mib:.0f} MiB "
        f"peak_reserved={torch.cuda.max_memory_reserved(dev_idx) / mib:.0f} MiB"
    )
    print(
        f"[diag] batch={args.batch_size}, timesteps={args.timesteps}, "
        f"gray_hw={args.gray_height}x{args.gray_width}, amp={args.amp}, "
        f"scenarios={args.scenarios}"
    )


def main():
    args = parse_args()
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("training requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    mode_tag = (
        f"gray_cam-{args.camera_control_mode}_"
        f"grad-{args.sensor_grad_mode}_vision-{args.policy_gray_mode}"
    )
    if args.train_camera_only:
        mode_tag += "_cameraonly"
    elif args.train_flight_only:
        mode_tag += "_flightonly"

    run_name = f"{mode_tag}_{time.strftime('%Y%m%d_%H%M%S')}"
    checkpoint_dir = os.path.join("checkpoint", time.strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(checkpoint_dir, exist_ok=True)

    calibration = IMX900Calibration.from_json(args.imx900_calibration)
    calibration_snapshot = os.path.join(
        checkpoint_dir,
        "imx900_calibration.json",
    )
    shutil.copy2(args.imx900_calibration, calibration_snapshot)

    wandb.init(
        project="diff-simulation",
        name=run_name,
        config=args,
        settings=wandb.Settings(code_dir="."),
        mode="disabled" if args.wandb_disabled else "online",
    )
    wandb.save("*.py")
    wandb.save("src/*.cu")
    wandb.save("src/*.cpp")
    wandb.save("configs/*.args")
    wandb.save(args.imx900_calibration)
    wandb.save("*.sh")

    print("\n" + "=" * 30 + " Configuration " + "=" * 30)
    for key, value in vars(args).items():
        print(f"{key:<30}: {value}")
    print(f"{'checkpoint_dir':<30}: {checkpoint_dir}")
    print(f"{'camera_profile':<30}: {calibration.profile_name}")
    print(f"{'camera_profile_calibrated':<30}: {calibration.calibrated}")
    print(f"{'camera_profile_snapshot':<30}: {calibration_snapshot}")
    print("=" * 75 + "\n")
    print_runtime_mode(args)

    env_train = build_env(args.batch_size, args, device)
    obs_dim = 7 if args.no_odom else 10
    model = Model(
        obs_dim,
        3,
        include_camera_state_in_obs=args.include_camera_state_in_obs,
        gray_nn_width=args.gray_nn_width,
        gray_nn_height=args.gray_nn_height,
    ).to(device)

    if args.resume:
        print(f"[info] loading checkpoint: {args.resume}")
        state_dict = torch.load(args.resume, map_location=device)
        model.load_state_dict(state_dict, strict=True)

    if args.train_flight_only:
        frozen = model.freeze_camera_for_flight_only()
        print(
            f"[info] train_flight_only: frozen_tensors={len(frozen)} "
            f"sample={', '.join(frozen[:12])}"
        )
    elif args.train_camera_only:
        model.initialize_camera_visual_from_flight()
        frozen = model.freeze_flight_for_camera_only()
        print(
            f"[info] train_camera_only: frozen_tensors={len(frozen)} "
            f"sample={', '.join(frozen[:12])}"
        )

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable model parameters remain")

    use_amp = bool(args.amp and device.type == "cuda")
    scaler = GradScaler(enabled=use_amp)
    optim = AdamW(trainable, args.lr)
    sched = CosineAnnealingLR(
        optim,
        estimate_optimizer_steps(args),
        eta_min=args.lr * 0.01,
    )

    vis = RerunVis(
        enabled=(args.vis_enable and args.vis_backend == "rerun"),
        app_id="DiffPhysDrone-Gray-Train",
        spawn=args.vis_spawn,
        show_aabb=args.vis_show_aabb,
    )

    try:
        train(
            args,
            model,
            env_train,
            None,
            optim,
            sched,
            scaler,
            vis,
            checkpoint_dir,
            device,
        )
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "out of memory" in msg or "cudnn_status_not_initialized" in msg:
            _print_cuda_failure_summary(args, device, exc)
        raise


if __name__ == "__main__":
    main()
