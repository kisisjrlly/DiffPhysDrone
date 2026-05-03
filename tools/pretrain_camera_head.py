#!/usr/bin/env python3
"""Supervised pretraining for the camera head from a teacher dataset."""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import build_parser, parse_diff_sensor_impl, parse_scenarios, set_global_seed, validate_args  # noqa: E402
from model import Model  # noqa: E402


def _read_args_file(path: Path) -> list[str]:
    tokens: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            tokens.extend(shlex.split(line))
    return tokens


def _build_project_args(config_path: Path, overrides: list[str]):
    parser = build_parser()
    args = parser.parse_args(_read_args_file(config_path) + list(overrides))
    args.diff_sensor_impl = parse_diff_sensor_impl(args.diff_sensor_impl)
    args.scenarios = parse_scenarios(args.scenarios)
    args.wandb_disabled = True
    args.vis_enable = False
    validate_args(args)
    return args


def _make_model(args, device):
    obs_dim = 7 if args.no_odom else 10
    return Model(
        obs_dim,
        3,
        include_camera_state_in_obs=args.include_camera_state_in_obs,
        use_policy_intent=False,
        depth_nn_width=args.depth_nn_width,
        depth_nn_height=args.depth_nn_height,
        depth_use_pipeline=args.depth_use_pipeline,
        depth_min_valid=args.depth_min_valid,
        depth_max_range=args.depth_max_range,
    ).to(device)


def _set_trainable_camera_only(model: Model, *, train_shared_visual_encoder: bool = False):
    for param in model.parameters():
        param.requires_grad_(False)
    trainable_prefixes = (
        "cam_spatial_stem",
        "cam_spatial_proj",
        "cam_img_adapter",
        "cam_img_norm",
        "cam_state_proj",
        "cam_state_norm",
        "cam_motion_proj",
        "cam_motion_norm",
        "cam_pre",
        "cam_gru",
        "cam_hx_norm",
        "fc_cam",
    )
    if train_shared_visual_encoder:
        trainable_prefixes = trainable_prefixes + (
            "spatial_stem",
            "spatial_proj",
            "spatial_attn",
            "attn_proj",
            "img_norm",
        )
    for name, param in model.named_parameters():
        if name.startswith(trainable_prefixes):
            param.requires_grad_(True)
    return [
        name for name, param in model.named_parameters()
        if param.requires_grad
    ]


def _camera_forward_sequence(model, depth, state, camera_state, camera_motion_state):
    B, T = state.shape[:2]
    h = None
    cam_h = None
    preds = []
    for t in range(T):
        _act, cam, h, cam_h = model(
            state[:, t],
            h,
            depth_obs=depth[:, t].float(),
            add_noise=False,
            cam_hx=cam_h,
            camera_state=camera_state[:, t],
            camera_motion_state=camera_motion_state[:, t],
        )
        # Flight hidden state is not trained in this stage.  Detach it so the
        # loss only optimizes the camera branch and its recurrent state.
        h = h.detach()
        preds.append(cam)
    return torch.stack(preds, dim=1)


def _parse_script_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/slit_active_sensing.args")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", default="checkpoint/camera_pretrain/camera_head_pretrained.pth")
    parser.add_argument("--best_out", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--temporal_smooth", type=float, default=0.02)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train_shared_visual_encoder",
        action="store_true",
        help=(
            "Also tune the shared flight depth encoder.  Default keeps flight "
            "features frozen and trains only camera-specific visual modules."
        ),
    )
    return parser.parse_known_args()


def main():
    script_args, project_overrides = _parse_script_args()
    args = _build_project_args(Path(script_args.config), project_overrides)
    set_global_seed(int(script_args.seed), getattr(args, "deterministic", False))
    device = torch.device(script_args.device)

    data = torch.load(script_args.dataset, map_location="cpu")
    required = ["depth_obs", "state", "camera_state", "camera_motion_state", "teacher_camera"]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"dataset missing keys: {missing}")
    ds = TensorDataset(
        data["depth_obs"],
        data["state"],
        data["camera_state"],
        data["camera_motion_state"],
        data["teacher_camera"],
    )
    num_items = len(ds)
    if num_items < 2:
        raise ValueError("dataset is too small")
    val_fraction = min(max(float(script_args.val_fraction), 0.0), 0.9)
    val_size = max(1, int(round(num_items * val_fraction)))
    train_size = max(1, num_items - val_size)
    if train_size + val_size > num_items:
        train_size = num_items - val_size
    perm = torch.randperm(num_items)
    train_idx = perm[:train_size].tolist()
    val_idx = perm[train_size:train_size + val_size].tolist()
    train_ds = torch.utils.data.Subset(ds, train_idx)
    val_ds = torch.utils.data.Subset(ds, val_idx)
    train_loader = DataLoader(train_ds, batch_size=int(script_args.batch_size), shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=int(script_args.batch_size), shuffle=False, drop_last=False)

    model = _make_model(args, device)
    init_ckpt = script_args.resume or script_args.checkpoint
    model.load_state_dict(torch.load(init_ckpt, map_location=device), strict=True)
    trainable_names = _set_trainable_camera_only(
        model,
        train_shared_visual_encoder=bool(script_args.train_shared_visual_encoder),
    )
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable camera parameters")
    print(
        f"[camera-pretrain] trainable_params={sum(p.numel() for p in trainable)} "
        f"trainable_tensors={len(trainable_names)}"
    )
    print("[camera-pretrain] trainable_prefix_sample=" + ", ".join(trainable_names[:16]))
    optim = torch.optim.AdamW(trainable, lr=float(script_args.lr), weight_decay=float(script_args.weight_decay))
    model.train()

    best_out = Path(script_args.best_out) if script_args.best_out else Path(script_args.out).with_name(Path(script_args.out).stem + "_best.pth")
    best_out.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(int(script_args.epochs)):
        total_loss = 0.0
        total_count = 0
        pbar = tqdm(train_loader, ncols=90, desc=f"epoch {epoch + 1}/{script_args.epochs}")
        for depth, state, cam_state, cam_motion, teacher in pbar:
            depth = depth.to(device)
            state = state.to(device)
            cam_state = cam_state.to(device)
            cam_motion = cam_motion.to(device)
            teacher = teacher.to(device)

            optim.zero_grad(set_to_none=True)
            pred = _camera_forward_sequence(model, depth, state, cam_state, cam_motion)
            loss = F.smooth_l1_loss(pred, teacher)
            if float(script_args.temporal_smooth) > 0.0 and pred.shape[1] > 1:
                loss = loss + float(script_args.temporal_smooth) * pred.diff(dim=1).pow(2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            optim.step()

            total_loss += float(loss.detach().cpu().item()) * int(depth.shape[0])
            total_count += int(depth.shape[0])
            pbar.set_postfix(loss=f"{float(loss.detach().cpu().item()):.4f}")
        train_loss = total_loss / max(total_count, 1)

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        val_mae = torch.zeros(3, device=device)
        with torch.no_grad():
            for depth, state, cam_state, cam_motion, teacher in val_loader:
                depth = depth.to(device)
                state = state.to(device)
                cam_state = cam_state.to(device)
                cam_motion = cam_motion.to(device)
                teacher = teacher.to(device)
                pred = _camera_forward_sequence(model, depth, state, cam_state, cam_motion)
                loss = F.smooth_l1_loss(pred, teacher)
                if float(script_args.temporal_smooth) > 0.0 and pred.shape[1] > 1:
                    loss = loss + float(script_args.temporal_smooth) * pred.diff(dim=1).pow(2).mean()
                val_loss_sum += float(loss.detach().cpu().item()) * int(depth.shape[0])
                val_count += int(depth.shape[0])
                val_mae += (pred - teacher).abs().mean(dim=(0, 1)).detach() * int(depth.shape[0])
        val_loss = val_loss_sum / max(val_count, 1)
        val_mae = val_mae / max(val_count, 1)
        print(
            f"[camera-pretrain] epoch={epoch + 1} "
            f"train={train_loss:.6f} val={val_loss:.6f} "
            f"mae_p/e/g={float(val_mae[0]):.4f}/{float(val_mae[1]):.4f}/{float(val_mae[2]):.4f}"
        )
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_out)
            print(f"[camera-pretrain] best saved: {best_out}")
        model.train()

    out = Path(script_args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    print(f"[camera-pretrain] saved: {out}")


if __name__ == "__main__":
    main()
