#include <torch/extension.h>

#include <vector>

// ============================================================================
// CUDA 前向声明 (CUDA forward declarations)
// 这些函数在对应的 .cu 文件中实现，这里声明以便 C++ 接口调用
// ============================================================================

// 渲染深度图和光流 (Render depth map and optical flow)
void render_cuda(
    torch::Tensor canvas,       // 输出的深度图画布 (Output depth map canvas)
    torch::Tensor flow,         // 输出的光流图 (Output optical flow)
    torch::Tensor balls,        // 球形障碍物数据 (Spherical obstacles)
    torch::Tensor cylinders,    // 圆柱形障碍物数据 (Cylindrical obstacles)
    torch::Tensor cylinders_h,  // 圆柱形障碍物高度 (Cylinder heights)
    torch::Tensor voxels,       // 体素障碍物数据 (Voxel obstacles)
    torch::Tensor R,            // 当前无人机旋转矩阵 (Current rotation matrix)
    torch::Tensor R_old,        // 上一帧无人机旋转矩阵 (Previous rotation matrix)
    torch::Tensor pos,          // 当前无人机位置 (Current position)
    torch::Tensor pos_old,      // 上一帧无人机位置 (Previous position)
    float drone_radius,         // 无人机半径 (Drone radius)
    int n_drones_per_group,     // 每组无人机数量 (Number of drones per group)
    float fov_x_half_tan);      // 相机水平视场角一半的正切值 (tan(FOV_x / 2))

// 重新渲染的反向传播 (Backward pass for re-rendering)
void rerender_backward_cuda(
    torch::Tensor depth,        // 深度图 (Depth map)
    torch::Tensor dddp,         // 深度对位置的导数 (Derivative of depth w.r.t position)
    float fov_x_half_tan);      // 相机水平视场角一半的正切值 (tan(FOV_x / 2))

// 寻找最近的障碍物点 (Find the nearest obstacle point)
void find_nearest_pt_cuda(
    torch::Tensor nearest_pt,   // 输出的最近点坐标 (Output nearest point coordinates)
    torch::Tensor balls,        // 球形障碍物数据 (Spherical obstacles)
    torch::Tensor cylinders,    // 圆柱形障碍物数据 (Cylindrical obstacles)
    torch::Tensor cylinders_h,  // 圆柱形障碍物高度 (Cylinder heights)
    torch::Tensor voxels,       // 体素障碍物数据 (Voxel obstacles)
    torch::Tensor pos,          // 无人机位置 (Drone position)
    float drone_radius,         // 无人机半径 (Drone radius)
    int n_drones_per_group);    // 每组无人机数量 (Number of drones per group)

// 使用椭球体模型寻找最近的障碍物点 (Find nearest point using ellipsoid model)
void find_nearest_pt_ellipsoid_cuda(
    torch::Tensor nearest_pt,   // 输出的最近点坐标 (Output nearest point coordinates)
    torch::Tensor balls,        // 球形障碍物数据 (Spherical obstacles)
    torch::Tensor cylinders,    // 圆柱形障碍物数据 (Cylindrical obstacles)
    torch::Tensor cylinders_h,  // 圆柱形障碍物高度 (Cylinder heights)
    torch::Tensor voxels,       // 体素障碍物数据 (Voxel obstacles)
    torch::Tensor pos,          // 无人机位置 (Drone position)
    torch::Tensor R_body,       // 无人机机身旋转矩阵 (Drone body rotation matrix)
    float drone_radius,         // 无人机半径 (Drone radius)
    int n_drones_per_group,     // 每组无人机数量 (Number of drones per group)
    float ellipsoid_a,          // 椭球体 XY 半轴 (Ellipsoid semi-axis a)
    float ellipsoid_c);         // 椭球体 Z 半轴 (Ellipsoid semi-axis c)

// 更新状态向量 (Update state vector)
torch::Tensor update_state_vec_cuda(
    torch::Tensor R,            // 旋转矩阵 (Rotation matrix)
    torch::Tensor a_thr,        // 推力加速度 (Thrust acceleration)
    torch::Tensor v_pred,       // 预测速度 (Predicted velocity)
    torch::Tensor alpha,        // 角加速度 (Angular acceleration)
    float yaw_inertia);         // 偏航转动惯量 (Yaw inertia)

// 物理仿真前向传播 (Physics simulation forward pass)
std::vector<torch::Tensor> run_forward_cuda(
    torch::Tensor R,            // 旋转矩阵 (Rotation matrix)
    torch::Tensor dg,           // 重力向量 (Gravity vector)
    torch::Tensor z_drag_coef,  // Z轴阻力系数 (Z-axis drag coefficient)
    torch::Tensor drag_2,       // 二次阻力系数 (Quadratic drag coefficient)
    torch::Tensor pitch_ctl_delay, // 俯仰控制延迟 (Pitch control delay)
    torch::Tensor act_pred,     // 预测动作 (Predicted action)
    torch::Tensor act,          // 实际动作 (Actual action)
    torch::Tensor p,            // 位置 (Position)
    torch::Tensor v,            // 速度 (Velocity)
    torch::Tensor v_wind,       // 风速 (Wind velocity)
    torch::Tensor a,            // 加速度 (Acceleration)
    float ctl_dt,               // 控制时间步长 (Control time step)
    float airmode_av2a);        // 空中模式角速度到加速度的转换系数 (Airmode angular velocity to acceleration coefficient)

// 物理仿真反向传播 (Physics simulation backward pass)
std::vector<torch::Tensor> run_backward_cuda(
    torch::Tensor R,            // 旋转矩阵 (Rotation matrix)
    torch::Tensor dg,           // 重力向量 (Gravity vector)
    torch::Tensor z_drag_coef,  // Z轴阻力系数 (Z-axis drag coefficient)
    torch::Tensor drag_2,       // 二次阻力系数 (Quadratic drag coefficient)
    torch::Tensor pitch_ctl_delay, // 俯仰控制延迟 (Pitch control delay)
    torch::Tensor v,            // 速度 (Velocity)
    torch::Tensor v_wind,       // 风速 (Wind velocity)
    torch::Tensor act_next,     // 下一步动作 (Next action)
    torch::Tensor _d_act_next,  // 下一步动作的梯度 (Gradient of next action)
    torch::Tensor d_p_next,     // 下一步位置的梯度 (Gradient of next position)
    torch::Tensor d_v_next,     // 下一步速度的梯度 (Gradient of next velocity)
    torch::Tensor _d_a_next,    // 下一步加速度的梯度 (Gradient of next acceleration)
    float grad_decay,           // 梯度衰减系数 (Gradient decay factor)
    float ctl_dt);              // 控制时间步长 (Control time step)

// 带有可微 FOV 的渲染前向传播 (Render forward pass with differentiable FOV)
void render_diff_fov_cuda(
    torch::Tensor canvas,       // 输出的深度图画布 (Output depth map canvas)
    torch::Tensor balls,        // 球形障碍物数据 (Spherical obstacles)
    torch::Tensor cylinders,    // 圆柱形障碍物数据 (Cylindrical obstacles)
    torch::Tensor cylinders_h,  // 圆柱形障碍物高度 (Cylinder heights)
    torch::Tensor voxels,       // 体素障碍物数据 (Voxel obstacles)
    torch::Tensor R,            // 旋转矩阵 (Rotation matrix)
    torch::Tensor pos,          // 位置 (Position)
    int n_drones_per_group,     // 每组无人机数量 (Number of drones per group)
    torch::Tensor fov_x_half_tan); // 相机水平视场角一半的正切值 (tan(FOV_x / 2))

// 带有可微 FOV 的渲染反向传播 (Render backward pass with differentiable FOV)
void render_backward_fov_cuda(
    torch::Tensor grad_fov,     // 输出的 FOV 梯度 (Output gradient of FOV)
    torch::Tensor grad_output,  // 损失函数对深度图的梯度 (Gradient of loss w.r.t depth map)
    torch::Tensor canvas,       // 深度图画布 (Depth map canvas)
    torch::Tensor balls,        // 球形障碍物数据 (Spherical obstacles)
    torch::Tensor cylinders,    // 圆柱形障碍物数据 (Cylindrical obstacles)
    torch::Tensor cylinders_h,  // 圆柱形障碍物高度 (Cylinder heights)
    torch::Tensor voxels,       // 体素障碍物数据 (Voxel obstacles)
    torch::Tensor R,            // 旋转矩阵 (Rotation matrix)
    torch::Tensor pos,          // 位置 (Position)
    int n_drones_per_group,     // 每组无人机数量 (Number of drones per group)
    torch::Tensor fov_x_half_tan); // 相机水平视场角一半的正切值 (tan(FOV_x / 2))

// ============================================================================
// PyBind11 模块绑定 (PyBind11 module binding)
// 将 C++ 函数暴露给 Python，使得可以在 Python 中通过 quadsim_cuda 模块调用
// ============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("render", &render_cuda, "render (CUDA)");
  m.def("find_nearest_pt", &find_nearest_pt_cuda, "find_nearest_pt (CUDA)");
  m.def("find_nearest_pt_ellipsoid", &find_nearest_pt_ellipsoid_cuda, "find_nearest_pt_ellipsoid (CUDA)");
  m.def("update_state_vec", &update_state_vec_cuda, "update_state_vec (CUDA)");
  m.def("run_forward", &run_forward_cuda, "run_forward_cuda (CUDA)");
  m.def("run_backward", &run_backward_cuda, "run_backward_cuda (CUDA)");
  m.def("rerender_backward", &rerender_backward_cuda, "rerender_backward_cuda (CUDA)");
  m.def("render_diff_fov", &render_diff_fov_cuda, "render_diff_fov (CUDA)");
  m.def("render_backward_fov", &render_backward_fov_cuda, "render_backward_fov (CUDA)");
}
