import math
import torch
import quadsim_cuda

# =============================================================================
# 物理引擎 CUDA 扩展测试脚本 (Physics Engine CUDA Extension Test Script)
# 
# 该脚本用于验证自定义的 CUDA 物理前向和反向传播函数 (quadsim_cuda) 
# 是否与纯 PyTorch 实现的结果完全一致。这对于确保可微物理引擎的正确性至关重要。
# =============================================================================

# 梯度衰减函数 (Gradient Decay Function)
# 用于在 BPTT (Backpropagation Through Time) 中缓解梯度爆炸问题
class GDecay(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.alpha, None

g_decay = GDecay.apply

# 初始化随机测试数据 (Initialize random test data)
# 使用双精度 (torch.double) 以确保数值比较的精度
R = torch.randn((64, 3, 3), dtype=torch.double, device='cuda')
dg = torch.randn((64, 3), dtype=torch.double, device='cuda')
z_drag_coef = torch.randn((64, 1), dtype=torch.double, device='cuda')
drag_2 = torch.randn((64, 2), dtype=torch.double, device='cuda')
pitch_ctl_delay = torch.randn((64, 1), dtype=torch.double, device='cuda')
g_std = torch.tensor([[0, 0, -9.80665]], dtype=torch.double, device='cuda')

# 需要计算梯度的变量 (Variables requiring gradients)
act_pred = torch.randn((64, 3), dtype=torch.double, device='cuda', requires_grad=True)
act = torch.randn((64, 3), dtype=torch.double, device='cuda', requires_grad=True)
p = torch.randn((64, 3), dtype=torch.double, device='cuda', requires_grad=True)
v = torch.randn((64, 3), dtype=torch.double, device='cuda', requires_grad=True)
v_wind = torch.randn((64, 3), dtype=torch.double, device='cuda', requires_grad=True)
a = torch.randn((64, 3), dtype=torch.double, device='cuda', requires_grad=True)

grad_decay = 0.4
ctl_dt = 1/15

# 纯 PyTorch 实现的物理前向传播 (Pure PyTorch implementation of physics forward pass)
# 用于作为基准 (Ground truth) 来验证 CUDA 实现
def run_forward_pytorch(R, dg, z_drag_coef, drag_2, pitch_ctl_delay, act_pred, act, p, v, v_wind, a, ctl_dt):
    # 1. 计算控制延迟 (Calculate control delay)
    alpha = torch.exp(-pitch_ctl_delay * ctl_dt)
    act_next = act_pred * (1 - alpha) + act * alpha
    
    # 2. 计算空气阻力 (Calculate air drag)
    # 将速度转换到无人机局部坐标系 (Convert velocity to local frame)
    v_fwd_s, v_left_s, v_up_s = (v.add(-v_wind)[:, None] @ R).unbind(-1)
    # 二次阻力项 (Quadratic drag term)
    drag = drag_2[:, :1] * (v_fwd_s.abs() * v_fwd_s * R[..., 0] + v_left_s.abs() * v_left_s * R[..., 1] + v_up_s.abs() * v_up_s * R[..., 2] * z_drag_coef)
    # 线性阻力项 (Linear drag term)
    drag += drag_2[:, 1:] * (v_fwd_s * R[..., 0] + v_left_s * R[..., 1] + v_up_s * R[..., 2] * z_drag_coef)
    
    # 3. 计算下一步的加速度、位置和速度 (Calculate next acceleration, position, and velocity)
    a_next = act_next + dg - drag
    p_next = g_decay(p, grad_decay ** ctl_dt) + v * ctl_dt + 0.5 * a * ctl_dt**2
    v_next = g_decay(v, grad_decay ** ctl_dt) + (a + a_next) / 2 * ctl_dt
    return act_next, p_next, v_next, a_next

# =============================================================================
# 1. 测试前向传播 (Test Forward Pass)
# =============================================================================
# 调用 CUDA 实现
act_next, p_next, v_next, a_next = quadsim_cuda.run_forward(
    R, dg, z_drag_coef, drag_2, pitch_ctl_delay, act_pred, act, p, v, v_wind, a, ctl_dt, 0)

# 调用 PyTorch 实现
_act_next, _p_next, _v_next, _a_next = run_forward_pytorch(
    R, dg, z_drag_coef, drag_2, pitch_ctl_delay, act_pred, act, p, v, v_wind, a, ctl_dt)

# 验证结果是否一致 (Verify results match)
assert torch.allclose(act_next, _act_next)
assert torch.allclose(a_next, _a_next)
assert torch.allclose(p_next, _p_next)
assert torch.allclose(v_next, _v_next)
print("Forward pass test passed!")

# =============================================================================
# 2. 测试反向传播 (Test Backward Pass)
# =============================================================================
# 随机生成上游传来的梯度 (Randomly generate upstream gradients)
d_act_next = torch.randn_like(act_next)
d_p_next = torch.randn_like(p_next)
d_v_next = torch.randn_like(v_next)
d_a_next = torch.randn_like(a_next)

# 使用 PyTorch 的自动求导计算基准梯度 (Calculate ground truth gradients using PyTorch autograd)
torch.autograd.backward(
    (_act_next, _p_next, _v_next, _a_next),
    (d_act_next, d_p_next, d_v_next, d_a_next),
)

# 调用自定义的 CUDA 反向传播函数 (Call custom CUDA backward function)
d_act_pred, d_act, d_p, d_v, d_a = quadsim_cuda.run_backward(
    R, dg, z_drag_coef, drag_2, pitch_ctl_delay, v, v_wind, act_next, d_act_next, d_p_next, d_v_next, d_a_next, grad_decay, ctl_dt)

# 验证自定义 CUDA 梯度与 PyTorch 自动求导的梯度是否一致 (Verify gradients match)
assert torch.allclose(d_act_pred, act_pred.grad)
assert torch.allclose(d_act, act.grad)
assert torch.allclose(d_p, p.grad)
assert torch.allclose(d_v, v.grad)
assert torch.allclose(d_a, a.grad)
print("Backward pass test passed!")
