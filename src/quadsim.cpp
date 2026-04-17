#include <torch/extension.h>

#include <vector>

// ============================================================================
// CUDA 前向声明 (CUDA forward declarations)
// ============================================================================

// 深度图渲染 (Render depth map)
void render_depth_cuda(
    torch::Tensor canvas,
    torch::Tensor balls,
    torch::Tensor cylinders,
    torch::Tensor cylinders_h,
    torch::Tensor voxels,
    torch::Tensor R,
    torch::Tensor pos,
    int n_drones_per_group,
    float fov_x_half_tan);

// 寻找最近的障碍物点 (Find the nearest obstacle point)
void find_nearest_pt_cuda(
    torch::Tensor nearest_pt,
    torch::Tensor balls,
    torch::Tensor cylinders,
    torch::Tensor cylinders_h,
    torch::Tensor voxels,
    torch::Tensor pos,
    float drone_radius,
    int n_drones_per_group);

// 使用椭球体模型寻找最近的障碍物点 (Find nearest point using ellipsoid model)
void find_nearest_pt_ellipsoid_cuda(
    torch::Tensor nearest_pt,
    torch::Tensor balls,
    torch::Tensor cylinders,
    torch::Tensor cylinders_h,
    torch::Tensor voxels,
    torch::Tensor pos,
    torch::Tensor R_body,
    float drone_radius,
    int n_drones_per_group,
    float ellipsoid_a,
    float ellipsoid_c);

// 更新状态向量 (Update state vector)
torch::Tensor update_state_vec_cuda(
    torch::Tensor R,
    torch::Tensor a_thr,
    torch::Tensor v_pred,
    torch::Tensor alpha,
    float yaw_inertia);

// 物理仿真前向传播 (Physics simulation forward pass)
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
    float airmode_av2a);

// 物理仿真反向传播 (Physics simulation backward pass)
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
    float ctl_dt);

// diff_depth 可微前向（返回 noisy_depth/quality）
std::vector<torch::Tensor> render_diff_depth_forward_cuda(
    float fov_x_half_tan,
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

// diff_depth 可微反向（返回 grad_power/grad_exposure/grad_gain）
std::vector<torch::Tensor> render_diff_depth_backward_cuda(
    torch::Tensor grad_noisy_depth,
    torch::Tensor grad_quality,
    torch::Tensor noisy_depth,
    torch::Tensor quality,
    float fov_x_half_tan,
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
// ============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("render_depth", &render_depth_cuda, "render_depth (CUDA)");
  m.def("find_nearest_pt", &find_nearest_pt_cuda, "find_nearest_pt (CUDA)");
  m.def("find_nearest_pt_ellipsoid", &find_nearest_pt_ellipsoid_cuda, "find_nearest_pt_ellipsoid (CUDA)");
  m.def("update_state_vec", &update_state_vec_cuda, "update_state_vec (CUDA)");
  m.def("run_forward", &run_forward_cuda, "run_forward_cuda (CUDA)");
  m.def("run_backward", &run_backward_cuda, "run_backward_cuda (CUDA)");
  m.def("render_diff_depth_forward", &render_diff_depth_forward_cuda, "render_diff_depth_forward (CUDA)");
  m.def("render_diff_depth_backward", &render_diff_depth_backward_cuda, "render_diff_depth_backward (CUDA)");
}
