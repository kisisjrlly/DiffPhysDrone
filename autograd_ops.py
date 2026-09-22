"""Autograd bridge for the differentiable quadrotor dynamics.

The legacy D455/differentiable-depth sensor autograd functions were removed
from this grayscale/IMX900 branch. Camera differentiability now lives entirely
in sensors/imx900_camera.py, with physical coefficients supplied by the
IMX900 calibration profile.
"""

import os
import sys

import torch

try:
    import quadsim_cuda
except ModuleNotFoundError:
    _src_dir = os.path.join(os.path.dirname(__file__), "src")
    if _src_dir not in sys.path:
        sys.path.insert(0, _src_dir)
    import quadsim_cuda


class RunFunction(torch.autograd.Function):
    """Differentiable quadrotor dynamics backed by the CUDA extension."""

    @staticmethod
    def forward(
        ctx,
        R,
        dg,
        z_drag_coef,
        drag_2,
        pitch_ctl_delay,
        act_pred,
        act,
        p,
        v,
        v_wind,
        a,
        grad_decay,
        ctl_dt,
        airmode,
    ):
        act_next, p_next, v_next, a_next = quadsim_cuda.run_forward(
            R,
            dg,
            z_drag_coef,
            drag_2,
            pitch_ctl_delay,
            act_pred,
            act,
            p,
            v,
            v_wind,
            a,
            ctl_dt,
            airmode,
        )
        ctx.save_for_backward(
            R,
            dg,
            z_drag_coef,
            drag_2,
            pitch_ctl_delay,
            v,
            v_wind,
            act_next,
        )
        ctx.grad_decay = grad_decay
        ctx.ctl_dt = ctl_dt
        return act_next, p_next, v_next, a_next

    @staticmethod
    def backward(ctx, d_act_next, d_p_next, d_v_next, d_a_next):
        (
            R,
            dg,
            z_drag_coef,
            drag_2,
            pitch_ctl_delay,
            v,
            v_wind,
            act_next,
        ) = ctx.saved_tensors
        d_act_pred, d_act, d_p, d_v, d_a = quadsim_cuda.run_backward(
            R,
            dg,
            z_drag_coef,
            drag_2,
            pitch_ctl_delay,
            v,
            v_wind,
            act_next,
            d_act_next,
            d_p_next,
            d_v_next,
            d_a_next,
            ctx.grad_decay,
            ctx.ctl_dt,
        )
        return (
            None,
            None,
            None,
            None,
            None,
            d_act_pred,
            d_act,
            d_p,
            d_v,
            None,
            d_a,
            None,
            None,
            None,
        )


run = RunFunction.apply
