import torch

# =============================================================================
# 全身运动/轨迹跟踪的微分线性二次型调节器 (Differentiable LQR)
# 这个文件实现了一个批处理形式 (batched) 的可微 LQR 求解器。
# 在项目中，它常被用作一个 "白盒" 的底层控制器 (DMPC: Differentiable Model Predictive Control)。
# 策略网络 (Policy Network) 可以只输出“高层意图” (Intent) 比如目标速度/加速度，
# 然后 LQR 作为带有物理先验的层，计算出具体的底层控制动作 (如桨叶推力)。
# =============================================================================

def build_velocity_tracking_linear_system(batch_size: int, dt: float, device: torch.device):
    """
    构建一个简化的速度跟踪线性系统 (Velocity Tracking Linear System)：
    基于简单的运动学方程: v_{t+1} = v_t + u_t * dt
    状态维度 nx=3 (飞行器当前的局部速度)，控制维度 nu=3 (目标加速度或推力)。
    """
    eye = torch.eye(3, device=device)
    A = eye.unsqueeze(0).repeat(batch_size, 1, 1)
    B = (dt * eye).unsqueeze(0).repeat(batch_size, 1, 1)
    return A, B


def solve_batched_dlqr(
    A: torch.Tensor,
    B: torch.Tensor,
    Q: torch.Tensor,
    R: torch.Tensor,
    x0: torch.Tensor,
    x_ref: torch.Tensor,
    horizon: int = 5,
    reg: float = 1e-4,
):
    """
    批量有限时域 dLQR（Differentiable Linear Quadratic Regulator）。

    Args:
        A: (B, nx, nx) 状态转移矩阵 (State Transition Matrix)
        B: (B, nx, nu) 控制输入矩阵 (Control Input Matrix)
        Q: (B, nx, nx) 状态误差代价矩阵 (State Error Cost)
        R: (B, nu, nu) 控制输出代价矩阵 (Control Effort Cost)
        x0: (B, nx) 当前初始状态 (Current State)
        x_ref: (B, nx) 目标参考状态 (Target Reference State)
        horizon: 预测时域长度 (MPC 预测的未来步数 N)
        reg: 数值稳定正则化参数

    Returns:
        u0: (B, nu) 第一拍控制动作 (MPC 通常只部署第一步规划输出)
        u_seq: (T, B, nu) 完整预测控制序列
        x_seq: (T+1, B, nx) 完整预测状态序列
    """
    Bn, nx, _ = A.shape
    nu = B.shape[-1]

    # 1. 确保 Q/R 半正定 + 正定（通过显式对称化 + 增加对角正则）
    Q = 0.5 * (Q + Q.transpose(-1, -2))
    R = 0.5 * (R + R.transpose(-1, -2))

    eye_x = torch.eye(nx, device=A.device, dtype=A.dtype).unsqueeze(0).expand(Bn, -1, -1)
    eye_u = torch.eye(nu, device=A.device, dtype=A.dtype).unsqueeze(0).expand(Bn, -1, -1)

    Q = Q + reg * eye_x
    R = R + reg * eye_u

    # 2. 从后向前求解 Discrete-time Algebraic Riccati Equation (DARE)
    # 因为是有限时域 (Finite Horizon LQR)，而不是求稳态的无穷时域
    P = Q # 终端时间步的目标 cost 矩阵

    Ks = []
    # 反向 Riccati 递推 (Backward Pass)
    for _ in range(horizon):
        BtP = B.transpose(1, 2) @ P
        # S = R + B^T P_{t+1} B
        S = R + BtP @ B
        # 再次强制 S 对称，增强 torch.linalg.solve 的抗噪性
        S = 0.5 * (S + S.transpose(-1, -2)) + reg * eye_u

        # K_t = S^{-1} (B^T P_{t+1} A) 计算反馈增益矩阵
        K = torch.linalg.solve(S, BtP @ A)
        Ks.append(K)

        # 黎卡提迭代: P_t = Q + A^T P_{t+1} A - A^T P_{t+1} B K_t
        AtPA = A.transpose(1, 2) @ P @ A
        AtPBK = A.transpose(1, 2) @ P @ B @ K
        P = Q + AtPA - AtPBK
        P = 0.5 * (P + P.transpose(-1, -2)) + reg * eye_x

    # 反转回来，让前向部署时是从 t=0 开始
    Ks = Ks[::-1]

    # 3. 前向 rollout (Forward Pass) 计算最优轨迹与具体控制量
    x = x0
    x_seq = [x]
    u_seq = []
    for t in range(horizon):
        # 目标偏差
        dx = (x - x_ref)
        # u = -K * (x - x_ref)
        u = -torch.squeeze(Ks[t] @ dx.unsqueeze(-1), -1)
        u_seq.append(u)
        x = torch.squeeze(A @ x.unsqueeze(-1), -1) + torch.squeeze(B @ u.unsqueeze(-1), -1)
        x_seq.append(x)

    u_seq = torch.stack(u_seq, 0)
    x_seq = torch.stack(x_seq, 0)
    u0 = u_seq[0]
    return u0, u_seq, x_seq
