#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

// ============================================================================
// 更新状态向量 CUDA 内核 (Update State Vector CUDA Kernel)
// 
// 该内核根据当前的推力加速度 (a_thr) 和预测速度 (v_pred) 更新无人机的旋转矩阵 (R_new)。
// 它模拟了无人机为了产生特定方向的推力而改变姿态的过程，并考虑了偏航惯性。
// ============================================================================
template <typename scalar_t>
__global__ void update_state_vec_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> R_new,  // 输出：更新后的旋转矩阵 (Output: updated rotation matrix)
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> R,      // 输入：当前旋转矩阵 (Input: current rotation matrix)
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> a_thr,  // 输入：推力加速度 (Input: thrust acceleration)
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> v_pred, // 输入：预测速度 (Input: predicted velocity)
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> alpha,  // 输入：姿态控制延迟系数 (Input: attitude control delay coefficient)
    float yaw_inertia) {                                                            // 输入：偏航转动惯量 (Input: yaw inertia)
    
    const int b = blockIdx.x * blockDim.x + threadIdx.x; // 获取当前线程处理的 batch 索引
    const int B = R.size(0);
    if (b >= B) return; // 越界检查

    // 1. 计算无人机的向上向量 (Up vector)
    // a_thr = a_thr - self.g_std; (在 Python 端已处理，这里加上重力补偿)
    scalar_t ax = a_thr[b][0];
    scalar_t ay = a_thr[b][1];
    scalar_t az = a_thr[b][2] + 9.80665; // 加上重力加速度 (Add gravity)
    
    // thrust = torch.norm(a_thr, 2, -1, True);
    scalar_t thrust = max((scalar_t)1e-5, sqrt(ax*ax+ay*ay+az*az)); // 计算总推力大小并防除零 (with epsilon guard)
    
    // self.up_vec = a_thr / thrust;
    // 归一化得到向上向量 (Normalize to get the up vector)
    scalar_t ux = ax / thrust;
    scalar_t uy = ay / thrust;
    scalar_t uz = az / thrust;

    // 2. 计算无人机的前向向量 (Forward vector)
    // forward_vec = self.forward_vec * yaw_inertia + v_pred;
    // 结合当前的偏航惯性和预测速度方向来确定新的前向方向
    scalar_t fx = R[b][0][0] * yaw_inertia + v_pred[b][0];
    scalar_t fy = R[b][1][0] * yaw_inertia + v_pred[b][1];
    scalar_t fz = R[b][2][0] * yaw_inertia + v_pred[b][2];
    
    // forward_vec = F.normalize(forward_vec, 2, -1);
    // forward_vec = (1-alpha) * forward_vec + alpha * self.forward_vec
    // 归一化并应用一阶低通滤波 (模拟姿态控制延迟)
    scalar_t t = max((scalar_t)1e-6, sqrt(fx * fx + fy * fy + fz * fz));
    fx = (1 - alpha[b][0]) * (fx / t) + alpha[b][0] * R[b][0][0];
    fy = (1 - alpha[b][0]) * (fy / t) + alpha[b][0] * R[b][1][0];
    fz = (1 - alpha[b][0]) * (fz / t) + alpha[b][0] * R[b][2][0];

    // 3. 确保前向向量与向上向量正交 (Ensure forward vector is orthogonal to up vector)
    // forward_vec[2] = (forward_vec[0] * self_up_vec[0] + forward_vec[1] * self_up_vec[1]) / -self_up_vec[2]
    // 通过调整 Z 分量使得点积为 0 (fx*ux + fy*uy + fz*uz = 0)
    // 防止 uz≈0 时除零导致 NaN/Inf 传播并最终触发 GPU hang
    if (abs(uz) > (scalar_t)1e-4) {
        fz = (fx * ux + fy * uy) / -uz;
    }

    // self.forward_vec = F.normalize(forward_vec, 2, -1);
    // 再次归一化前向向量 (Re-normalize forward vector)
    t = max((scalar_t)1e-6, sqrt(fx * fx + fy * fy + fz * fz));
    fx /= t;
    fy /= t;
    fz /= t;
    
    // 4. 计算左向向量并构建新的旋转矩阵 (Calculate left vector and construct new rotation matrix)
    // self.left_vec = torch.cross(self.up_vec, self.forward_vec);
    // R_new = [forward_vec, left_vec, up_vec]
    R_new[b][0][0] = fx;
    R_new[b][0][1] = uy * fz - uz * fy; // 叉乘计算左向向量 X (Cross product for left vector X)
    R_new[b][0][2] = ux;
    R_new[b][1][0] = fy;
    R_new[b][1][1] = uz * fx - ux * fz; // 叉乘计算左向向量 Y (Cross product for left vector Y)
    R_new[b][1][2] = uy;
    R_new[b][2][0] = fz;
    R_new[b][2][1] = ux * fy - uy * fx; // 叉乘计算左向向量 Z (Cross product for left vector Z)
    R_new[b][2][2] = uz;
}

// ============================================================================
// 物理前向传播 CUDA 内核 (Physics Forward Pass CUDA Kernel)
// 
// 该内核执行单步的无人机物理仿真，包括：
// 1. 应用控制延迟 (一阶低通滤波)
// 2. 计算空气动力学阻力 (线性和二次阻力)
// 3. 使用欧拉法积分更新加速度、速度和位置
// ============================================================================
template <typename scalar_t>
__global__ void run_forward_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> R,               // 旋转矩阵 (Rotation matrix)
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> dg,              // 重力向量 (Gravity vector)
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> z_drag_coef,     // Z轴阻力系数 (Z-axis drag coefficient)
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> drag_2,          // 二次阻力系数 (Quadratic drag coefficient)
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> pitch_ctl_delay, // 俯仰控制延迟 (Pitch control delay)
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> act_pred,        // 预测动作 (Predicted action)
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> act,             // 当前实际动作 (Current actual action)
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> p,               // 当前位置 (Current position)
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> v,               // 当前速度 (Current velocity)
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> v_wind,          // 风速 (Wind velocity)
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> a,               // 当前加速度 (Current acceleration)
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> act_next,        // 输出：下一步动作 (Output: next action)
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> p_next,          // 输出：下一步位置 (Output: next position)
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> v_next,          // 输出：下一步速度 (Output: next velocity)
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> a_next,          // 输出：下一步加速度 (Output: next acceleration)
    float ctl_dt, float airmode_av2a) {                                                      // 控制时间步长和 Airmode 系数
    
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int B = R.size(0);
    if (i >= B) return;

    // 1. 计算控制延迟 (Calculate control delay)
    // alpha = torch.exp(-self.pitch_ctl_delay * ctl_dt)
    scalar_t alpha = exp(-pitch_ctl_delay[i][0] * ctl_dt);
    
    // self.act = act_pred * (1 - alpha) + self.act * alpha
    // 使用一阶低通滤波器平滑动作，模拟电机和电调的响应延迟
    for (int j=0; j<3; j++)
        act_next[i][j] = act_pred[i][j] * (1 - alpha) + act[i][j] * alpha;

    // 2. 计算相对风速并转换到机体坐标系 (Calculate relative wind speed and convert to body frame)
    scalar_t v_rel_wind_x = v[i][0] - v_wind[i][0];
    scalar_t v_rel_wind_y = v[i][1] - v_wind[i][1];
    scalar_t v_rel_wind_z = v[i][2] - v_wind[i][2];
    
    // 投影到机体坐标系的 Z (Up), X (Forward), Y (Left) 轴
    scalar_t v_up_s = v_rel_wind_x * R[i][0][2] + v_rel_wind_y * R[i][1][2] + v_rel_wind_z * R[i][2][2];
    scalar_t v_fwd_s = v_rel_wind_x * R[i][0][0] + v_rel_wind_y * R[i][1][0] + v_rel_wind_z * R[i][2][0];
    scalar_t v_left_s = v_rel_wind_x * R[i][0][1] + v_rel_wind_y * R[i][1][1] + v_rel_wind_z * R[i][2][1];
    
    // 计算速度的平方项 (保留符号) (Calculate squared velocity terms preserving sign)
    scalar_t v_up_2 = v_up_s * abs(v_up_s);
    scalar_t v_fwd_2 = v_fwd_s * abs(v_fwd_s);
    scalar_t v_left_2 = v_left_s * abs(v_left_s);

    // 3. 计算空气阻力加速度 (Calculate aerodynamic drag acceleration)
    scalar_t a_drag_2[3], a_drag_1[3];
    for (int j=0; j<3; j++){
        // 二次阻力项 (Quadratic drag term)
        a_drag_2[j] = v_up_2 * R[i][j][2] * z_drag_coef[i][0] + v_left_2 * R[i][j][1] + v_fwd_2 * R[i][j][0];
        // 线性阻力项 (Linear drag term)
        a_drag_1[j] = v_up_s * R[i][j][2] * z_drag_coef[i][0] + v_left_s * R[i][j][1] + v_fwd_s * R[i][j][0];
    }

    // 4. 计算 Airmode 补偿 (Calculate Airmode compensation)
    // Airmode 允许在零油门时保持姿态控制，这里通过角速度估算额外的加速度补偿
    scalar_t dot = act[i][0] * act_next[i][0] + act[i][1] * act_next[i][1] + (act[i][2] + 9.80665) * (act_next[i][2] + 9.80665);
    scalar_t n1 = act[i][0] * act[i][0] + act[i][1] * act[i][1] + (act[i][2] + 9.80665) * (act[i][2] + 9.80665);
    scalar_t n2 = act_next[i][0] * act_next[i][0] + act_next[i][1] * act_next[i][1] + (act_next[i][2] + 9.80665) * (act_next[i][2] + 9.80665);
    // 计算角速度 (Calculate angular velocity)
    scalar_t av = acos(max(-1., min(1., dot / max(1e-8, sqrt(n1) * sqrt(n2))))) / ctl_dt;

    scalar_t ax = act[i][0];
    scalar_t ay = act[i][1];
    scalar_t az = act[i][2] + 9.80665;
    scalar_t thrust = max((scalar_t)1e-5, sqrt(ax*ax+ay*ay+az*az));
    scalar_t airmode_a[3] = {
        ax / thrust * av * airmode_av2a,
        ay / thrust * av * airmode_av2a,
        az / thrust * av * airmode_av2a};

    // 5. 积分更新状态 (Integrate to update state)
    // a_next = self.act + self.dg - drag + airmode
    for (int j=0; j<3; j++)
        a_next[i][j] = act_next[i][j] + dg[i][j] - a_drag_2[j] * drag_2[i][0] - a_drag_1[j] * drag_2[i][1] + airmode_a[j];
    
    // self.p = p + v * dt + 0.5 * a * dt^2
    for (int j=0; j<3; j++)
        p_next[i][j] = p[i][j] + v[i][j] * ctl_dt + 0.5 * a[i][j] * ctl_dt * ctl_dt;
    
    // self.v = v + 0.5 * (a + a_next) * dt (梯形积分 / Trapezoidal integration)
    for (int j=0; j<3; j++)
        v_next[i][j] = v[i][j] + 0.5 * (a[i][j] + a_next[i][j]) * ctl_dt;
}

// ============================================================================
// 物理反向传播 CUDA 内核 (Physics Backward Pass CUDA Kernel)
// 
// 该内核手动实现了 run_forward_cuda_kernel 的反向传播 (自动微分)。
// 它接收来自下游的梯度 (d_p_next, d_v_next, d_a_next)，并计算对输入变量的梯度。
// 这是实现可微物理引擎的核心，允许梯度穿过物理仿真步骤流向策略网络。
// ============================================================================
// ============================================================================
// 物理反向传播 CUDA 内核 (Physics Backward Pass CUDA Kernel)
// 
// 该内核手动实现了 run_forward_cuda_kernel 的反向传播 (自动微分)。
// 它接收来自下游的梯度 (d_p_next, d_v_next, d_a_next)，并计算对输入变量的梯度。
// 这是实现可微物理引擎的核心，允许梯度穿过物理仿真步骤流向策略网络。
// ============================================================================
template <typename scalar_t>
__global__ void run_backward_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> R,               // 旋转矩阵 (Rotation matrix)
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> dg,              // 重力向量 (Gravity vector)
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> z_drag_coef,     // Z轴阻力系数 (Z-axis drag coefficient)
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> drag_2,          // 二次阻力系数 (Quadratic drag coefficient)
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> pitch_ctl_delay, // 俯仰控制延迟 (Pitch control delay)
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> v,               // 当前速度 (Current velocity)
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> v_wind,          // 风速 (Wind velocity)
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> act_next,        // 下一步动作 (Next action)
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_act_pred,      // 输出：对预测动作的梯度 (Output: gradient w.r.t predicted action)
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_act,           // 输出：对当前动作的梯度 (Output: gradient w.r.t current action)
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_p,             // 输出：对当前位置的梯度 (Output: gradient w.r.t current position)
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_v,             // 输出：对当前速度的梯度 (Output: gradient w.r.t current velocity)
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_a,             // 输出：对当前加速度的梯度 (Output: gradient w.r.t current acceleration)
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> _d_act_next,     // 输入：对下一步动作的梯度 (Input: gradient w.r.t next action)
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_p_next,        // 输入：对下一步位置的梯度 (Input: gradient w.r.t next position)
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_v_next,        // 输入：对下一步速度的梯度 (Input: gradient w.r.t next velocity)
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> _d_a_next,       // 输入：对下一步加速度的梯度 (Input: gradient w.r.t next acceleration)
    float grad_decay, float ctl_dt) {                                                        // 梯度衰减系数和控制时间步长
    
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int B = R.size(0);
    if (i >= B) return;

    // 1. 重新计算前向传播中的中间变量 (Recompute intermediate variables from forward pass)
    // alpha = torch.exp(-self.pitch_ctl_delay * ctl_dt)
    scalar_t alpha = exp(-pitch_ctl_delay[i][0] * ctl_dt);
    
    // 提取下一步动作的各个分量 (Extract components of next action)
    scalar_t act_x = act_next[i][0];
    scalar_t act_y = act_next[i][1];
    scalar_t act_z = act_next[i][2] + 9.80665;

    // 复制输入的梯度到局部数组 (Copy input gradients to local arrays)
    scalar_t d_act_next[3] = {_d_act_next[i][0], _d_act_next[i][1], _d_act_next[i][2]};
    scalar_t d_a_next[3] = {_d_a_next[i][0], _d_a_next[i][1], _d_a_next[i][2]};
    
    // ========================================================================
    // 反向传播开始 (Backward pass starts here)
    // ========================================================================
    
    // 2. 速度更新的反向传播 (Backward for velocity update)
    // 前向: v_next[i][j] = v[i][j] + 0.5 * (a[i][j] + a_next[i][j]) * ctl_dt;
    for (int j=0; j<3; j++){
        // 考虑 BPTT 梯度衰减 (Apply BPTT gradient decay)
        d_v[i][j] = d_v_next[i][j] * pow(grad_decay, ctl_dt);
        d_a[i][j] = 0.5 * ctl_dt * d_v_next[i][j];
        d_a_next[j] += 0.5 * ctl_dt * d_v_next[i][j];
    }
    
    // 3. 位置更新的反向传播 (Backward for position update)
    // 前向: p_next[i][j] = p[i][j] + v[i][j] * ctl_dt + 0.5 * a[i][j] * ctl_dt * ctl_dt;
    for (int j=0; j<3; j++){
        // 考虑 BPTT 梯度衰减 (Apply BPTT gradient decay)
        d_p[i][j] = d_p_next[i][j] * pow(grad_decay, ctl_dt);
        d_v[i][j] += ctl_dt * d_p_next[i][j];
        d_a[i][j] += 0.5 * ctl_dt * ctl_dt * d_p_next[i][j];
    }

    // 4. 加速度更新的反向传播 (Backward for acceleration update)
    scalar_t d_a_drag_2[3];
    scalar_t d_a_drag_1[3];
    for (int j=0; j<3; j++){
        // 前向: a_next[i][j] = act_next[i][j] + dg[i][j] - a_drag_2 - a_drag_1;
        d_act_next[j] += d_a_next[j];
        // 计算对阻力项的梯度 (Calculate gradients w.r.t drag terms)
        d_a_drag_2[j] = -d_a_next[j] * drag_2[i][0];
        d_a_drag_1[j] = -d_a_next[j] * drag_2[i][1];
    }

    // 5. 空气阻力的反向传播 (Backward for aerodynamic drag)
    // 重新计算相对风速 (Recompute relative wind speed)
    scalar_t v_rel_wind_x = v[i][0] - v_wind[i][0];
    scalar_t v_rel_wind_y = v[i][1] - v_wind[i][1];
    scalar_t v_rel_wind_z = v[i][2] - v_wind[i][2];
    scalar_t v_fwd_s = v_rel_wind_x * R[i][0][0] + v_rel_wind_y * R[i][1][0] + v_rel_wind_z * R[i][2][0];
    scalar_t v_left_s = v_rel_wind_x * R[i][0][1] + v_rel_wind_y * R[i][1][1] + v_rel_wind_z * R[i][2][1];
    scalar_t v_up_s = v_rel_wind_x * R[i][0][2] + v_rel_wind_y * R[i][1][2] + v_rel_wind_z * R[i][2][2];
    
    scalar_t d_v_fwd_s = 0;
    scalar_t d_v_left_s = 0;
    scalar_t d_v_up_s = 0;
    for (int j=0; j<3; j++){
        // 前向: a_drag_2[j] = v_up_s * |v_up_s| * R[i][j][2] * z_drag_coef + ...
        // 导数: d(x*|x|)/dx = 2*|x|
        d_v_fwd_s += d_a_drag_2[j] * 2 * abs(v_fwd_s) * R[i][j][0];
        d_v_left_s += d_a_drag_2[j] * 2 * abs(v_left_s) * R[i][j][1];
        d_v_up_s += d_a_drag_2[j] * 2 * abs(v_up_s) * R[i][j][2] * z_drag_coef[i][0];
        
        // 前向: a_drag_1[j] = v_up_s * R[i][j][2] * z_drag_coef + ...
        d_v_fwd_s += d_a_drag_1[j] * R[i][j][0];
        d_v_left_s += d_a_drag_1[j] * R[i][j][1];
        d_v_up_s += d_a_drag_1[j] * R[i][j][2] * z_drag_coef[i][0];
    }

    // 将机体坐标系下的速度梯度转换回世界坐标系 (Convert velocity gradients back to world frame)
    for (int j=0; j<3; j++){
        d_v[i][j] += R[i][j][0] * d_v_fwd_s;
        d_v[i][j] += R[i][j][1] * d_v_left_s;
        d_v[i][j] += R[i][j][2] * d_v_up_s;
    }
    
    // 6. 控制延迟的反向传播 (Backward for control delay)
    for (int j=0; j<3; j++){
        // 前向: act_next[i][j] = act_pred[i][j] * (1 - alpha) + act[i][j] * alpha;
        d_act_pred[i][j] = (1 - alpha) * d_act_next[j];
        d_act[i][j] = alpha * d_act_next[j];
    }
}

} // namespace

// ============================================================================
// C++ 接口函数：物理前向传播 (C++ Interface: Physics Forward Pass)
// 
// 负责分配输出张量内存，并调用 CUDA 内核。
// ============================================================================
std::vector<torch::Tensor> run_forward_cuda(
    torch::Tensor R,
    torch::Tensor dg,
    torch::Tensor z_drag_coef,
    torch::Tensor drag_2,
    torch::Tensor pitch_ctl_delay,
    torch::Tensor act_pred,
    torch::Tensor act,
    torch::Tensor p,
    torch::Tensor v,
    torch::Tensor v_wind,
    torch::Tensor a,
    float ctl_dt,
    float airmode_av2a){

    // 分配输出张量 (Allocate output tensors)
    torch::Tensor act_next = torch::empty_like(act);
    torch::Tensor p_next = torch::empty_like(p);
    torch::Tensor v_next = torch::empty_like(v);
    torch::Tensor a_next = torch::empty_like(a);

    const int threads = 256;
    const int B = R.size(0);
    const dim3 blocks((B + threads - 1) / threads);
    
    // 启动 CUDA 内核 (Launch CUDA kernel)
    AT_DISPATCH_FLOATING_TYPES(R.type(), "run_forward_cuda", ([&] {
        run_forward_cuda_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            R.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            dg.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            z_drag_coef.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            drag_2.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            pitch_ctl_delay.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            act_pred.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            act.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            p.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            v.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            v_wind.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            a.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            act_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            p_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            v_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            a_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            ctl_dt, airmode_av2a);
    }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    at::cuda::getCurrentCUDAStream().synchronize();
    return {act_next, p_next, v_next, a_next};
}

// ============================================================================
// C++ 接口函数：物理反向传播 (C++ Interface: Physics Backward Pass)
// 
// 负责分配梯度张量内存，并调用 CUDA 反向传播内核。
// ============================================================================
std::vector<torch::Tensor> run_backward_cuda(
    torch::Tensor R,
    torch::Tensor dg,
    torch::Tensor z_drag_coef,
    torch::Tensor drag_2,
    torch::Tensor pitch_ctl_delay,
    torch::Tensor v,
    torch::Tensor v_wind,
    torch::Tensor act_next,
    torch::Tensor _d_act_next,
    torch::Tensor d_p_next,
    torch::Tensor d_v_next,
    torch::Tensor _d_a_next,
    float grad_decay,
    float ctl_dt){

    // 分配梯度输出张量 (Allocate gradient output tensors)
    torch::Tensor d_act_pred = torch::empty_like(dg);
    torch::Tensor d_act = torch::empty_like(dg);
    torch::Tensor d_p = torch::empty_like(dg);
    torch::Tensor d_v = torch::empty_like(dg);
    torch::Tensor d_a = torch::empty_like(dg);

    const int threads = 256;
    const int B = R.size(0);
    const dim3 blocks((B + threads - 1) / threads);
    
    // 启动 CUDA 内核 (Launch CUDA kernel)
    AT_DISPATCH_FLOATING_TYPES(R.type(), "run_backward_cuda", ([&] {
        run_backward_cuda_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            R.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            dg.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            z_drag_coef.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            drag_2.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            pitch_ctl_delay.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            v.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            v_wind.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            act_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            d_act_pred.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            d_act.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            d_p.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            d_v.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            d_a.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            _d_act_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            d_p_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            d_v_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            _d_a_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            grad_decay, ctl_dt);
    }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    at::cuda::getCurrentCUDAStream().synchronize();
    // 返回计算得到的梯度 (Return computed gradients)
    return {d_act_pred, d_act, d_p, d_v, d_a};
}

// ============================================================================
// C++ 接口函数：更新状态向量 (C++ Interface: Update State Vector)
// 
// 负责分配新的旋转矩阵内存，并调用 CUDA 内核更新姿态。
// ============================================================================
torch::Tensor update_state_vec_cuda(
    torch::Tensor R,
    torch::Tensor a_thr,
    torch::Tensor v_pred,
    torch::Tensor alpha,
    float yaw_inertia) {
    
    const int threads = 256;
    const int B = a_thr.size(0);
    const dim3 blocks((B + threads - 1) / threads);
    
    // 分配新的旋转矩阵张量 (Allocate new rotation matrix tensor)
    torch::Tensor R_new = torch::empty_like(R);
    
    // 启动 CUDA 内核 (Launch CUDA kernel)
    AT_DISPATCH_FLOATING_TYPES(a_thr.type(), "update_state_vec", ([&] {
        update_state_vec_cuda_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            R_new.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            R.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            a_thr.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            v_pred.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            alpha.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            yaw_inertia);
    }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    at::cuda::getCurrentCUDAStream().synchronize();
    return R_new;
}
