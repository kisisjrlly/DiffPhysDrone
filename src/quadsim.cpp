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

// 带法线输出的可微 FOV 前向渲染 (Forward render with normal map output)
void render_diff_fov_with_normal_cuda(
    torch::Tensor canvas,
    torch::Tensor normals,      // 输出法线图 [B,3,H,W]
    torch::Tensor balls,
    torch::Tensor cylinders,
    torch::Tensor cylinders_h,
    torch::Tensor voxels,
    torch::Tensor R,
    torch::Tensor pos,
    int n_drones_per_group,
    torch::Tensor fov_x_half_tan);

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

// 基于法线图的解析 FOV 反向传播 (Analytical backward from normal map)
void render_backward_fov_from_normal_cuda(
    torch::Tensor grad_fov,
    torch::Tensor grad_output,
    torch::Tensor canvas,
    torch::Tensor normals,
    torch::Tensor R,
    torch::Tensor fov_x_half_tan);

// Y 通道渲染（YUV420 的 Y）(Render luma channel Y from YUV420 pipeline)
void render_yuv_y_cuda(
    torch::Tensor canvas,
    torch::Tensor flow,
    torch::Tensor balls,
    torch::Tensor cylinders,
    torch::Tensor cylinders_h,
    torch::Tensor voxels,
    torch::Tensor R,
    torch::Tensor R_old,
    torch::Tensor pos,
    torch::Tensor pos_old,
    float drone_radius,
    int n_drones_per_group,
    float fov_x_half_tan);

// 可微 Y 通道渲染接口 (Differentiable Y-channel rendering interface)
torch::Tensor render_diff_yuv_y_cuda(
    torch::Tensor fov_x_half_tan,
    torch::Tensor exposure,
    torch::Tensor iso,
    torch::Tensor R,
    torch::Tensor pos,
    torch::Tensor balls,
    torch::Tensor cylinders,
    torch::Tensor cylinders_h,
    torch::Tensor voxels,
    int n_drones_per_group,
    int height,
    int width);

// 可微 Y 通道渲染前向（返回 y 与几何深度）
std::vector<torch::Tensor> render_diff_yuv_y_forward_cuda(
    torch::Tensor fov_x_half_tan,
    torch::Tensor exposure,
    torch::Tensor iso,
    torch::Tensor R,
    torch::Tensor pos,
    torch::Tensor balls,
    torch::Tensor cylinders,
    torch::Tensor cylinders_h,
    torch::Tensor voxels,
    int n_drones_per_group,
    int height,
    int width,
    torch::Tensor cam_light_dir,
    torch::Tensor cam_ambient,
    torch::Tensor cam_dir_intensity,
    torch::Tensor cam_fog_beta,
    torch::Tensor cam_airlight,
    torch::Tensor cam_mat_ground,
    torch::Tensor cam_mat_obstacle,
    torch::Tensor cam_mat_spec,
    torch::Tensor cam_dist_k1,
    torch::Tensor cam_dist_k2,
    torch::Tensor cam_flare_strength,
    torch::Tensor cam_gamma,
    torch::Tensor cam_prnu,
    torch::Tensor cam_dsnu,
    torch::Tensor cam_prev_y,
    torch::Tensor cam_use_rolling,
    torch::Tensor v,
    torch::Tensor cam_ae_log_t,
    int64_t cam_profile_mask,
    double cam_vignette_a,
    double cam_vignette_b,
    double cam_black_level,
    double cam_sharpen_amount,
    double cam_base_gain,
    double cam_motion_blur_gain,
    double cam_exposure_t_min,
    double cam_exposure_t_span,
    double cam_exposure_eff_min,
    double cam_exposure_eff_max,
    double cam_iso_gain_base,
    double cam_iso_gain_scale,
    double cam_iso_gain_gamma);

// 可微 Y 通道渲染反向（返回 grad_fov, grad_exposure, grad_iso, grad_focus）
std::vector<torch::Tensor> render_diff_yuv_y_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor depth_raw,
    torch::Tensor fov_x_half_tan,
    torch::Tensor exposure,
    torch::Tensor iso,
    torch::Tensor normals,
    torch::Tensor R,
    torch::Tensor cam_light_dir,
    torch::Tensor cam_ambient,
    torch::Tensor cam_dir_intensity,
    torch::Tensor cam_fog_beta,
    torch::Tensor cam_airlight,
    torch::Tensor cam_mat_ground,
    torch::Tensor cam_mat_obstacle,
    torch::Tensor cam_mat_spec,
    torch::Tensor cam_dist_k1,
    torch::Tensor cam_dist_k2,
    torch::Tensor cam_flare_strength,
    torch::Tensor cam_gamma,
    torch::Tensor cam_prnu,
    torch::Tensor cam_dsnu,
    torch::Tensor cam_prev_y,
    torch::Tensor cam_use_rolling,
    torch::Tensor v,
    torch::Tensor cam_ae_log_t,
    int64_t cam_profile_mask,
    double cam_vignette_a,
    double cam_vignette_b,
    double cam_black_level,
    double cam_sharpen_amount,
    double cam_base_gain,
    double cam_motion_blur_gain,
    double cam_exposure_t_min,
    double cam_exposure_t_span,
    double cam_exposure_eff_min,
    double cam_exposure_eff_max,
    double cam_iso_gain_base,
    double cam_iso_gain_scale,
    double cam_iso_gain_gamma,
    bool need_grad_fov,
    bool need_grad_exposure,
    bool need_grad_iso);

// Active ToF 可微前向（CUDA实现，返回 noisy_depth/conf）
std::vector<torch::Tensor> render_active_tof_forward_cuda(
    torch::Tensor fov_x_half_tan,
    torch::Tensor power,
    torch::Tensor exposure,
    torch::Tensor gain,
    torch::Tensor v,
    torch::Tensor R,
    torch::Tensor pos,
    torch::Tensor balls,
    torch::Tensor cylinders,
    torch::Tensor cylinders_h,
    torch::Tensor voxels,
    int n_drones_per_group,
    int height,
    int width,
    double max_range);

// Active ToF 可微反向（CUDA实现，返回 grad_fov/grad_power/grad_exposure/grad_gain）
std::vector<torch::Tensor> render_active_tof_backward_cuda(
    torch::Tensor grad_noisy_depth,
    torch::Tensor grad_conf,
    torch::Tensor noisy_depth,
    torch::Tensor conf,
    torch::Tensor fov_x_half_tan,
    torch::Tensor power,
    torch::Tensor exposure,
    torch::Tensor gain,
    torch::Tensor v,
    torch::Tensor R,
    torch::Tensor pos,
    torch::Tensor balls,
    torch::Tensor cylinders,
    torch::Tensor cylinders_h,
    torch::Tensor voxels,
    int n_drones_per_group,
    int height,
    int width,
    double max_range);

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
    m.def("render_diff_fov_with_normal", &render_diff_fov_with_normal_cuda, "render_diff_fov_with_normal (CUDA)");
  m.def("render_backward_fov", &render_backward_fov_cuda, "render_backward_fov (CUDA)");
    m.def("render_backward_fov_from_normal", &render_backward_fov_from_normal_cuda, "render_backward_fov_from_normal (CUDA)");
    m.def("render_yuv_y", &render_yuv_y_cuda, "render_yuv_y (CUDA)");
    m.def("render_diff_yuv_y", &render_diff_yuv_y_cuda, "render_diff_yuv_y (CUDA)");
        m.def("render_diff_yuv_y_forward", &render_diff_yuv_y_forward_cuda, "render_diff_yuv_y_forward (CUDA)",
            pybind11::call_guard<pybind11::gil_scoped_release>());
        m.def("render_diff_yuv_y_backward", &render_diff_yuv_y_backward_cuda, "render_diff_yuv_y_backward (CUDA)",
            pybind11::call_guard<pybind11::gil_scoped_release>());
    m.def("render_active_tof_forward", &render_active_tof_forward_cuda, "render_active_tof_forward (CUDA)");
    m.def("render_active_tof_backward", &render_active_tof_backward_cuda, "render_active_tof_backward (CUDA)");
}
