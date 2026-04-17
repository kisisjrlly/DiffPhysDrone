#include <torch/extension.h>
#include <torch/autograd.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

// ============================================================================
// 最近点计算 CUDA 内核 (Nearest Point CUDA Kernel)
// 
// 该内核用于计算无人机到场景中各个障碍物的最近点，用于碰撞检测和惩罚计算。
// ============================================================================
template <typename scalar_t>
__global__ void nearest_pt_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> nearest_pt, // 输出：最近点坐标 (Output: Nearest point coordinates)
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> balls,      // 球体障碍物 (Spherical obstacles)
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> cylinders,  // 垂直圆柱体障碍物 (Vertical cylindrical obstacles)
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> cylinders_h,// 水平圆柱体障碍物 (Horizontal cylindrical obstacles)
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> voxels,     // 体素障碍物 (Voxel obstacles)
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> pos,        // 无人机位置 (Drone positions)
    float drone_radius,                                                                 // 无人机半径 (Drone radius)
    int n_drones_per_group) {                                                           // 每组无人机数量 (Number of drones per group)

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int B = nearest_pt.size(1);
    const int j = idx / B; // 时间步索引 (Time step index)
    if (j >= nearest_pt.size(0)) return;
    const int b = idx % B; // 批次索引 (Batch index)

    // 当前无人机位置 (Current drone position)
    const scalar_t ox = pos[j][b][0];
    const scalar_t oy = pos[j][b][1];
    const scalar_t oz = pos[j][b][2];

    // 初始化最小距离为到地面的距离 (Initialize minimum distance to ground distance)
    scalar_t min_dist = max(1e-3f, oz + 1);
    scalar_t nearest_ptx = ox;
    scalar_t nearest_pty = oy;
    scalar_t nearest_ptz = min(-1., oz - 1e-3f);

    // 1. 计算到其他无人机的最近点 (Calculate nearest point to other drones)
    const int batch_base = (b / n_drones_per_group) * n_drones_per_group;
    for (int i = batch_base; i < batch_base + n_drones_per_group; i++) {
        if (i == b || i >= B) continue; // 跳过自己 (Skip self)
        scalar_t cx = pos[j][i][0];
        scalar_t cy = pos[j][i][1];
        scalar_t cz = pos[j][i][2];
        scalar_t r = 0.15; // 假设其他无人机半径为 0.15 (Assume other drones radius is 0.15)
        
        // 计算距离 (Calculate distance)
        scalar_t dist = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy) + 4 * (oz - cz) * (oz - cz);
        dist = max(1e-3f, sqrt(dist) - r);
        
        // 更新最近点 (Update nearest point if closer)
        if (dist < min_dist) {
            min_dist = dist;
            scalar_t ddx = cx - ox;
            scalar_t ddy = cy - oy;
            scalar_t ddz = cz - oz;
            scalar_t dn = sqrt(ddx * ddx + ddy * ddy + ddz * ddz);
            if (dn > 1e-6f) { ddx /= dn; ddy /= dn; ddz /= dn; }
            nearest_ptx = ox + dist * ddx;
            nearest_pty = oy + dist * ddy;
            nearest_ptz = oz + dist * ddz;
        }
    }

    // 2. 计算到球体障碍物的最近点 (Calculate nearest point to spherical obstacles)
    for (int i = 0; i < balls.size(1); i++) {
        scalar_t cx = balls[batch_base][i][0];
        scalar_t cy = balls[batch_base][i][1];
        scalar_t cz = balls[batch_base][i][2];
        scalar_t r = balls[batch_base][i][3];
        
        // 计算距离 (Calculate distance)
        scalar_t dist = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy) + (oz - cz) * (oz - cz);
        dist = max(1e-3f, sqrt(dist) - r);
        
        // 更新最近点 (Update nearest point if closer)
        if (dist < min_dist) {
            min_dist = dist;
            scalar_t ddx = cx - ox;
            scalar_t ddy = cy - oy;
            scalar_t ddz = cz - oz;
            scalar_t dn = sqrt(ddx * ddx + ddy * ddy + ddz * ddz);
            if (dn > 1e-6f) { ddx /= dn; ddy /= dn; ddz /= dn; }
            nearest_ptx = ox + dist * ddx;
            nearest_pty = oy + dist * ddy;
            nearest_ptz = oz + dist * ddz;
        }
    }

    // 3. 计算到垂直圆柱体障碍物的最近点 (Calculate nearest point to vertical cylindrical obstacles)
    for (int i = 0; i < cylinders.size(1); i++) {
        scalar_t cx = cylinders[batch_base][i][0];
        scalar_t cy = cylinders[batch_base][i][1];
        scalar_t r = cylinders[batch_base][i][2];
        
        // 计算距离 (仅考虑 xy 平面) (Calculate distance in xy plane only)
        scalar_t dist = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy);
        dist = max(1e-3f, sqrt(dist) - r);
        
        // 更新最近点 (Update nearest point if closer)
        if (dist < min_dist) {
            min_dist = dist;
            scalar_t ddx = cx - ox;
            scalar_t ddy = cy - oy;
            scalar_t dn = sqrt(ddx * ddx + ddy * ddy);
            if (dn > 1e-6f) { ddx /= dn; ddy /= dn; }
            nearest_ptx = ox + dist * ddx;
            nearest_pty = oy + dist * ddy;
            nearest_ptz = oz; // z 坐标保持不变 (z coordinate remains unchanged)
        }
    }
    
    // 4. 计算到水平圆柱体障碍物的最近点 (Calculate nearest point to horizontal cylindrical obstacles)
    for (int i = 0; i < cylinders_h.size(1); i++) {
        scalar_t cx = cylinders_h[batch_base][i][0];
        scalar_t cz = cylinders_h[batch_base][i][1];
        scalar_t r = cylinders_h[batch_base][i][2];
        
        // 计算距离 (仅考虑 xz 平面) (Calculate distance in xz plane only)
        scalar_t dist = (ox - cx) * (ox - cx) + (oz - cz) * (oz - cz);
        dist = max(1e-3f, sqrt(dist) - r);
        
        // 更新最近点 (Update nearest point if closer)
        if (dist < min_dist) {
            min_dist = dist;
            scalar_t ddx = cx - ox;
            scalar_t ddz = cz - oz;
            scalar_t dn = sqrt(ddx * ddx + ddz * ddz);
            if (dn > 1e-6f) { ddx /= dn; ddz /= dn; }
            nearest_ptx = ox + dist * ddx;
            nearest_pty = oy; // y 坐标保持不变 (y coordinate remains unchanged)
            nearest_ptz = oz + dist * ddz;
        }
    }

    // 5. 计算到体素/长方体障碍物的最近点 (Calculate nearest point to voxel/box obstacles)
    for (int i = 0; i < voxels.size(1); i++) {
        scalar_t cx = voxels[batch_base][i][0];
        scalar_t cy = voxels[batch_base][i][1];
        scalar_t cz = voxels[batch_base][i][2];
        
        // 限制最大半径以避免穿透 (Limit max radius to avoid penetration)
        scalar_t max_r = max(abs(ox - cx), max(abs(oy - cy), abs(oz - cz))) - 1e-3;
        scalar_t rx = min(max_r, voxels[batch_base][i][3]);
        scalar_t ry = min(max_r, voxels[batch_base][i][4]);
        scalar_t rz = min(max_r, voxels[batch_base][i][5]);
        
        // 计算长方体表面上距离无人机最近的点 (Calculate nearest point on box surface)
        scalar_t ptx = cx + max(-rx, min(rx, ox - cx));
        scalar_t pty = cy + max(-ry, min(ry, oy - cy));
        scalar_t ptz = cz + max(-rz, min(rz, oz - cz));
        
        // 计算距离 (Calculate distance)
        scalar_t dist = (ptx - ox) * (ptx - ox) + (pty - oy) * (pty - oy) + (ptz - oz) * (ptz - oz);
        dist = sqrt(dist);
        
        // 更新最近点 (Update nearest point if closer)
        if (dist < min_dist) {
            min_dist = dist;
            nearest_ptx = ptx;
            nearest_pty = pty;
            nearest_ptz = ptz;
        }
    }
    
    // 将最近点坐标写入输出张量 (Write nearest point coordinates to output tensor)
    nearest_pt[j][b][0] = nearest_ptx;
    nearest_pt[j][b][1] = nearest_pty;
    nearest_pt[j][b][2] = nearest_ptz;
}


// ============================================================================
// 椭球体无人机碰撞检测 (Ellipsoid Drone Collision)
// 
// 将无人机视为机体坐标系下的椭球体，半轴长为 (a, a, c)。
// R_body[B,3,3] 提供机体到世界的旋转矩阵。
// 对于每个障碍物表面点，我们计算沿接触方向的有效椭球体半径，
// 并将其从点到障碍物的距离中减去，以获得更精确的碰撞距离。
// ============================================================================

// 辅助函数：计算沿给定方向的椭球体有效半径 (Helper: Calculate effective ellipsoid radius along given direction)
template <typename scalar_t>
__device__ __forceinline__ scalar_t ellipsoid_radius_along_dir(
    scalar_t dx, scalar_t dy, scalar_t dz, // 世界坐标系下的方向向量 (Direction in world frame)
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t>& R_body, // 旋转矩阵 (Rotation matrix)
    int b, scalar_t ea, scalar_t ec) {     // 批次索引和椭球体半轴长 (Batch index and ellipsoid semi-axes)
    
    // 将世界坐标系下的方向 (dx, dy, dz) 转换到机体坐标系 (Transform direction to body frame via R^T)
    // R_body[b] 的列是世界坐标系下的 [前, 左, 上] 向量 (Columns are [fwd, left, up] in world coords)
    scalar_t bx = R_body[b][0][0]*dx + R_body[b][1][0]*dy + R_body[b][2][0]*dz;
    scalar_t by = R_body[b][0][1]*dx + R_body[b][1][1]*dy + R_body[b][2][1]*dz;
    scalar_t bz = R_body[b][0][2]*dx + R_body[b][1][2]*dy + R_body[b][2][2]*dz;
    
    // 椭球体支撑距离公式: 1/sqrt((bx/a)^2+(by/a)^2+(bz/c)^2) (Ellipsoid support distance formula)
    scalar_t inv_a2 = 1.0f / (ea * ea);
    scalar_t inv_c2 = 1.0f / (ec * ec);
    scalar_t s = bx*bx*inv_a2 + by*by*inv_a2 + bz*bz*inv_c2;
    return 1.0f / sqrt(max(s, 1e-8f));
}

// ============================================================================
// 考虑椭球体形状的最近点计算 CUDA 内核 (Nearest Point Ellipsoid CUDA Kernel)
// ============================================================================
template <typename scalar_t>
__global__ void nearest_pt_ellipsoid_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> nearest_pt, // 输出：最近点坐标 (Output: Nearest point coordinates)
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> balls,      // 球体障碍物 (Spherical obstacles)
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> cylinders,  // 垂直圆柱体障碍物 (Vertical cylindrical obstacles)
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> cylinders_h,// 水平圆柱体障碍物 (Horizontal cylindrical obstacles)
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> voxels,     // 体素障碍物 (Voxel obstacles)
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> pos,        // 无人机位置 (Drone positions)
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> R_body,     // 无人机旋转矩阵 (Drone rotation matrices)
    float drone_radius,                                                                 // 无人机基础半径 (Base drone radius)
    int n_drones_per_group,                                                             // 每组无人机数量 (Number of drones per group)
    float ellipsoid_a,                                                                  // 椭球体水平半轴长 (Ellipsoid horizontal semi-axis)
    float ellipsoid_c) {                                                                // 椭球体垂直半轴长 (Ellipsoid vertical semi-axis)

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int B = nearest_pt.size(1);
    const int j = idx / B; // 时间步索引 (Time step index)
    if (j >= nearest_pt.size(0)) return;
    const int b = idx % B; // 批次索引 (Batch index)

    const scalar_t ea = (scalar_t)ellipsoid_a;
    const scalar_t ec = (scalar_t)ellipsoid_c;

    // 当前无人机位置 (Current drone position)
    const scalar_t ox = pos[j][b][0];
    const scalar_t oy = pos[j][b][1];
    const scalar_t oz = pos[j][b][2];

    // 1. 计算到地面的距离 (Ground plane z = -1: direction is (0, 0, -1))
    scalar_t ground_reff = ellipsoid_radius_along_dir((scalar_t)0, (scalar_t)0, (scalar_t)-1, R_body, b, ea, ec);
    scalar_t min_dist = max(1e-3f, oz + 1 - ground_reff);
    scalar_t nearest_ptx = ox;
    scalar_t nearest_pty = oy;
    scalar_t nearest_ptz = oz - min_dist;

    // 2. 计算到其他无人机的最近点 (Calculate nearest point to other drones)
    const int batch_base = (b / n_drones_per_group) * n_drones_per_group;
    for (int i = batch_base; i < batch_base + n_drones_per_group; i++) {
        if (i == b || i >= B) continue; // 跳过自己 (Skip self)
        scalar_t cx = pos[j][i][0];
        scalar_t cy = pos[j][i][1];
        scalar_t cz = pos[j][i][2];
        scalar_t r = 0.15; // 假设其他无人机半径为 0.15 (Assume other drones radius is 0.15)
        
        // 计算原始距离 (Calculate raw distance)
        scalar_t raw_dist2 = (ox-cx)*(ox-cx) + (oy-cy)*(oy-cy) + 4*(oz-cz)*(oz-cz);
        scalar_t raw_dist = sqrt(raw_dist2);
        scalar_t point_dist = max(1e-3f, raw_dist - r);
        
        // 计算从无人机指向障碍物的方向向量 (Direction from drone to obstacle (toward center))
        scalar_t ddx = (cx - ox), ddy = (cy - oy), ddz = (cz - oz);
        scalar_t dd_norm = sqrt(ddx*ddx + ddy*ddy + ddz*ddz);
        if (dd_norm > 1e-6f) { ddx /= dd_norm; ddy /= dd_norm; ddz /= dd_norm; }
        
        // 计算沿该方向的有效椭球体半径 (Calculate effective ellipsoid radius along this direction)
        scalar_t reff = ellipsoid_radius_along_dir(ddx, ddy, ddz, R_body, b, ea, ec);
        
        // 减去有效半径得到最终距离 (Subtract effective radius to get final distance)
        scalar_t dist = max(1e-3f, point_dist - reff);
        
        // 更新最近点 (Update nearest point if closer)
        if (dist < min_dist) {
            min_dist = dist;
            nearest_ptx = ox + dist * ddx;
            nearest_pty = oy + dist * ddy;
            nearest_ptz = oz + dist * ddz;
        }
    }

    // 3. 计算到球体障碍物的最近点 (Calculate nearest point to spherical obstacles)
    for (int i = 0; i < balls.size(1); i++) {
        scalar_t cx = balls[batch_base][i][0];
        scalar_t cy = balls[batch_base][i][1];
        scalar_t cz = balls[batch_base][i][2];
        scalar_t r = balls[batch_base][i][3];
        scalar_t ddx = cx - ox, ddy = cy - oy, ddz = cz - oz;
        scalar_t dd_norm = sqrt(ddx*ddx + ddy*ddy + ddz*ddz);
        scalar_t point_dist = max(1e-3f, dd_norm - r);
        if (dd_norm > 1e-6f) { ddx /= dd_norm; ddy /= dd_norm; ddz /= dd_norm; }
        scalar_t reff = ellipsoid_radius_along_dir(ddx, ddy, ddz, R_body, b, ea, ec);
        scalar_t dist = max(1e-3f, point_dist - reff);
        if (dist < min_dist) {
            min_dist = dist;
            nearest_ptx = ox + dist * ddx;
            nearest_pty = oy + dist * ddy;
            nearest_ptz = oz + dist * ddz;
        }
    }

    // 4. 计算到垂直圆柱体障碍物的最近点 (Calculate nearest point to vertical cylindrical obstacles)
    for (int i = 0; i < cylinders.size(1); i++) {
        scalar_t cx = cylinders[batch_base][i][0];
        scalar_t cy = cylinders[batch_base][i][1];
        scalar_t r = cylinders[batch_base][i][2];
        
        // 计算水平方向的距离向量 (Calculate horizontal distance vector)
        scalar_t ddx = cx - ox, ddy = cy - oy;
        scalar_t dd_norm = sqrt(ddx*ddx + ddy*ddy);
        scalar_t point_dist = max(1e-3f, dd_norm - r);
        
        // 归一化方向向量 (Normalize direction vector)
        if (dd_norm > 1e-6f) { ddx /= dd_norm; ddy /= dd_norm; }
        else { ddx = 0; ddy = 0; }
        
        // 世界坐标系下的方向: (ddx, ddy, 0) — 水平指向圆柱体轴线 (Direction in world: horizontal toward cylinder axis)
        scalar_t reff = ellipsoid_radius_along_dir(ddx, ddy, (scalar_t)0, R_body, b, ea, ec);
        scalar_t dist = max(1e-3f, point_dist - reff);
        
        // 更新最近点 (Update nearest point if closer)
        if (dist < min_dist) {
            min_dist = dist;
            nearest_ptx = ox + dist * ddx;
            nearest_pty = oy + dist * ddy;
            nearest_ptz = oz;
        }
    }

    // 5. 计算到水平圆柱体障碍物的最近点 (沿 Y 轴) (Calculate nearest point to horizontal cylinders (along Y))
    for (int i = 0; i < cylinders_h.size(1); i++) {
        scalar_t cx = cylinders_h[batch_base][i][0];
        scalar_t cz = cylinders_h[batch_base][i][1];
        scalar_t r = cylinders_h[batch_base][i][2];
        
        // 计算 xz 平面上的距离向量 (Calculate distance vector in xz plane)
        scalar_t ddx = cx - ox, ddz = cz - oz;
        scalar_t dd_norm = sqrt(ddx*ddx + ddz*ddz);
        scalar_t point_dist = max(1e-3f, dd_norm - r);
        
        // 归一化方向向量 (Normalize direction vector)
        if (dd_norm > 1e-6f) { ddx /= dd_norm; ddz /= dd_norm; }
        else { ddx = 0; ddz = 0; }
        
        // 计算有效半径 (Calculate effective radius)
        scalar_t reff = ellipsoid_radius_along_dir(ddx, (scalar_t)0, ddz, R_body, b, ea, ec);
        scalar_t dist = max(1e-3f, point_dist - reff);
        
        // 更新最近点 (Update nearest point if closer)
        if (dist < min_dist) {
            min_dist = dist;
            nearest_ptx = ox + dist * ddx;
            nearest_pty = oy;
            nearest_ptz = oz + dist * ddz;
        }
    }

    // 6. 计算到体素/长方体障碍物的最近点 (AABB) (Calculate nearest point to voxels (AABB))
    for (int i = 0; i < voxels.size(1); i++) {
        scalar_t cx = voxels[batch_base][i][0];
        scalar_t cy = voxels[batch_base][i][1];
        scalar_t cz = voxels[batch_base][i][2];
        
        // 限制最大半径以避免穿透 (Limit max radius to avoid penetration)
        scalar_t max_r = max(abs(ox - cx), max(abs(oy - cy), abs(oz - cz))) - 1e-3;
        scalar_t rx = min(max_r, voxels[batch_base][i][3]);
        scalar_t ry = min(max_r, voxels[batch_base][i][4]);
        scalar_t rz = min(max_r, voxels[batch_base][i][5]);
        
        // 计算长方体表面上距离无人机最近的点 (Calculate nearest point on box surface)
        scalar_t ptx = cx + max(-rx, min(rx, ox - cx));
        scalar_t pty = cy + max(-ry, min(ry, oy - cy));
        scalar_t ptz = cz + max(-rz, min(rz, oz - cz));
        
        // 计算距离向量 (Calculate distance vector)
        scalar_t ddx = ptx - ox, ddy = pty - oy, ddz = ptz - oz;
        scalar_t point_dist = sqrt(ddx*ddx + ddy*ddy + ddz*ddz);
        
        // 归一化方向向量 (Normalize direction vector)
        if (point_dist > 1e-6f) { ddx /= point_dist; ddy /= point_dist; ddz /= point_dist; }
        
        // 计算有效半径 (Calculate effective radius)
        scalar_t reff = (point_dist > 1e-6f) ?
            ellipsoid_radius_along_dir(ddx, ddy, ddz, R_body, b, ea, ec) : ea;
        scalar_t dist = max(0.0f, point_dist - reff);
        
        // 更新最近点 (Update nearest point if closer)
        if (dist < min_dist) {
            min_dist = dist;
            if (point_dist > 1e-6f) {
                nearest_ptx = ox + dist * ddx;
                nearest_pty = oy + dist * ddy;
                nearest_ptz = oz + dist * ddz;
            } else {
                nearest_ptx = ptx;
                nearest_pty = pty;
                nearest_ptz = ptz;
            }
        }
    }
    
    // 将最近点坐标写入输出张量 (Write nearest point coordinates to output tensor)
    nearest_pt[j][b][0] = nearest_ptx;
    nearest_pt[j][b][1] = nearest_pty;
    nearest_pt[j][b][2] = nearest_ptz;
}


// ============================================================================
// 可微视场渲染 (Differentiable FOV Rendering)
// 
// 这些函数用于实现可微的深度图渲染，允许梯度从渲染的图像反向传播到相机位姿。
// ============================================================================

// 设备函数：追踪单条光线穿过所有场景几何体，返回最小交点深度
// (Device function: trace a single ray through all scene geometry, return min intersection depth)
template <typename scalar_t>
__device__ __forceinline__ scalar_t trace_ray_device(
    scalar_t dx, scalar_t dy, scalar_t dz, // 光线方向 (Ray direction)
    scalar_t ox, scalar_t oy, scalar_t oz, // 光线起点 (Ray origin)
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> balls,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> cylinders,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> cylinders_h,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> voxels,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> pos,
    int n_drones_per_group, int batch_base, int bi, int B)
{
    const scalar_t kEps = (scalar_t)1e-8;
    scalar_t min_dist = 100; // 初始化为最大距离 (Initialize to max distance)
    
    // 1. 与地面的交点 (ground plane z = -1)
    if (abs(dz) > kEps) {
        scalar_t gt = (-1 - oz) / dz;
        if (gt > 0) min_dist = gt;
    }

    // 2. 与其他无人机的交点 (other drones (ellipsoid with z scaled by 2))
    for (int i = batch_base; i < batch_base + n_drones_per_group; i++) {
        if (i == bi || i >= B) continue;
        scalar_t cx = pos[i][0], cy = pos[i][1], cz = pos[i][2];
        scalar_t rad = 0.15;
        scalar_t qa = dx*dx + dy*dy + 4*dz*dz;
        scalar_t qb = 2*(dx*(ox-cx) + dy*(oy-cy) + 4*dz*(oz-cz));
        scalar_t qc = (ox-cx)*(ox-cx) + (oy-cy)*(oy-cy) + 4*(oz-cz)*(oz-cz) - rad*rad;
        scalar_t qd = qb*qb - 4*qa*qc;
        if (qa > kEps && qd >= 0) {
            scalar_t qt = (-qb - sqrt(qd)) / (2*qa);
            if (qt > 1e-5) { min_dist = min(min_dist, qt); }
            else { qt = (-qb + sqrt(qd)) / (2*qa); if (qt > 1e-5) min_dist = min(min_dist, qt); }
        }
    }

    // 3. 与球体障碍物的交点 (balls (spheres))
    for (int i = 0; i < balls.size(1); i++) {
        scalar_t cx = balls[batch_base][i][0], cy = balls[batch_base][i][1];
        scalar_t cz = balls[batch_base][i][2], rad = balls[batch_base][i][3];
        scalar_t qa = dx*dx + dy*dy + dz*dz;
        scalar_t qb = 2*(dx*(ox-cx) + dy*(oy-cy) + dz*(oz-cz));
        scalar_t qc = (ox-cx)*(ox-cx) + (oy-cy)*(oy-cy) + (oz-cz)*(oz-cz) - rad*rad;
        scalar_t qd = qb*qb - 4*qa*qc;
        if (qa > kEps && qd >= 0) {
            scalar_t qt = (-qb - sqrt(qd)) / (2*qa);
            if (qt > 1e-5) { min_dist = min(min_dist, qt); }
            else { qt = (-qb + sqrt(qd)) / (2*qa); if (qt > 1e-5) min_dist = min(min_dist, qt); }
        }
    }

    // 4. 与垂直圆柱体障碍物的交点 (vertical cylinders)
    for (int i = 0; i < cylinders.size(1); i++) {
        scalar_t cx = cylinders[batch_base][i][0], cy = cylinders[batch_base][i][1];
        scalar_t rad = cylinders[batch_base][i][2];
        scalar_t qa = dx*dx + dy*dy;
        scalar_t qb = 2*(dx*(ox-cx) + dy*(oy-cy));
        scalar_t qc = (ox-cx)*(ox-cx) + (oy-cy)*(oy-cy) - rad*rad;
        scalar_t qd = qb*qb - 4*qa*qc;
        if (qa > kEps && qd >= 0) {
            scalar_t qt = (-qb - sqrt(qd)) / (2*qa);
            if (qt > 1e-5) { min_dist = min(min_dist, qt); }
            else { qt = (-qb + sqrt(qd)) / (2*qa); if (qt > 1e-5) min_dist = min(min_dist, qt); }
        }
    }

    // 5. 与水平圆柱体障碍物的交点 (horizontal cylinders)
    for (int i = 0; i < cylinders_h.size(1); i++) {
        scalar_t cx = cylinders_h[batch_base][i][0], cz = cylinders_h[batch_base][i][1];
        scalar_t rad = cylinders_h[batch_base][i][2];
        scalar_t qa = dx*dx + dz*dz;
        scalar_t qb = 2*(dx*(ox-cx) + dz*(oz-cz));
        scalar_t qc = (ox-cx)*(ox-cx) + (oz-cz)*(oz-cz) - rad*rad;
        scalar_t qd = qb*qb - 4*qa*qc;
        if (qa > kEps && qd >= 0) {
            scalar_t qt = (-qb - sqrt(qd)) / (2*qa);
            if (qt > 1e-5) { min_dist = min(min_dist, qt); }
            else { qt = (-qb + sqrt(qd)) / (2*qa); if (qt > 1e-5) min_dist = min(min_dist, qt); }
        }
    }

    // 6. 与体素/长方体障碍物的交点 (voxels (AABB))
    for (int i = 0; i < voxels.size(1); i++) {
        scalar_t cx = voxels[batch_base][i][0], cy = voxels[batch_base][i][1];
        scalar_t cz = voxels[batch_base][i][2];
        scalar_t rx = voxels[batch_base][i][3], ry = voxels[batch_base][i][4];
        scalar_t rz = voxels[batch_base][i][5];
        scalar_t tx_min, tx_max, ty_min, ty_max, tz_min, tz_max;
        if (abs(dx) <= kEps) {
            if (ox < cx - rx || ox > cx + rx) continue;
            tx_min = -1e20; tx_max = 1e20;
        } else {
            scalar_t tx1 = (cx - rx - ox) / dx;
            scalar_t tx2 = (cx + rx - ox) / dx;
            tx_min = min(tx1, tx2);
            tx_max = max(tx1, tx2);
        }
        if (abs(dy) <= kEps) {
            if (oy < cy - ry || oy > cy + ry) continue;
            ty_min = -1e20; ty_max = 1e20;
        } else {
            scalar_t ty1 = (cy - ry - oy) / dy;
            scalar_t ty2 = (cy + ry - oy) / dy;
            ty_min = min(ty1, ty2);
            ty_max = max(ty1, ty2);
        }
        if (abs(dz) <= kEps) {
            if (oz < cz - rz || oz > cz + rz) continue;
            tz_min = -1e20; tz_max = 1e20;
        } else {
            scalar_t tz1 = (cz - rz - oz) / dz;
            scalar_t tz2 = (cz + rz - oz) / dz;
            tz_min = min(tz1, tz2);
            tz_max = max(tz1, tz2);
        }
        scalar_t t_min_v = max(max(tx_min, ty_min), tz_min);
        scalar_t t_max_v = min(min(tx_max, ty_max), tz_max);
        if (t_min_v < min_dist && t_min_v < t_max_v && t_min_v > 0)
            min_dist = t_min_v;
    }

    return min_dist;
}


// 设备函数：追踪单条“原始”光线并返回命中法线
// (Device function: trace original ray and return hit normal)
template <typename scalar_t>
__device__ __forceinline__ scalar_t trace_ray_with_normal_device(
    scalar_t dx, scalar_t dy, scalar_t dz,
    scalar_t ox, scalar_t oy, scalar_t oz,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> balls,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> cylinders,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> cylinders_h,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> voxels,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> pos,
    int n_drones_per_group, int batch_base, int bi, int B,
    scalar_t* out_nx, scalar_t* out_ny, scalar_t* out_nz)
{
    const scalar_t kEps = (scalar_t)1e-8;
    scalar_t min_dist = (scalar_t)100;
    scalar_t nx = (scalar_t)0, ny = (scalar_t)0, nz = (scalar_t)0;

    // ground plane z = -1
    if (abs(dz) > kEps) {
        scalar_t gt = (-1 - oz) / dz;
        if (gt > 0 && gt < min_dist) {
            min_dist = gt;
            nx = (scalar_t)0; ny = (scalar_t)0; nz = (scalar_t)1;
        }
    }

    // other drones (ellipsoid with z scaled by 2)
    for (int i = batch_base; i < batch_base + n_drones_per_group; i++) {
        if (i == bi || i >= B) continue;
        scalar_t cx = pos[i][0], cy = pos[i][1], cz = pos[i][2];
        scalar_t rad = (scalar_t)0.15;
        scalar_t qa = dx*dx + dy*dy + 4*dz*dz;
        scalar_t qb = 2*(dx*(ox-cx) + dy*(oy-cy) + 4*dz*(oz-cz));
        scalar_t qc = (ox-cx)*(ox-cx) + (oy-cy)*(oy-cy) + 4*(oz-cz)*(oz-cz) - rad*rad;
        scalar_t qd = qb*qb - 4*qa*qc;
        if (qa > kEps && qd >= 0) {
            scalar_t sqrt_qd = sqrt(qd);
            scalar_t qt = (-qb - sqrt_qd) / (2*qa);
            if (!(qt > (scalar_t)1e-5)) qt = (-qb + sqrt_qd) / (2*qa);
            if (qt > (scalar_t)1e-5 && qt < min_dist) {
                scalar_t px = ox + qt * dx;
                scalar_t py = oy + qt * dy;
                scalar_t pz = oz + qt * dz;
                scalar_t gx = px - cx;
                scalar_t gy = py - cy;
                scalar_t gz = (scalar_t)4 * (pz - cz);
                scalar_t gn = sqrt(gx*gx + gy*gy + gz*gz);
                if (gn > kEps) {
                    nx = gx / gn; ny = gy / gn; nz = gz / gn;
                    min_dist = qt;
                }
            }
        }
    }

    // spheres
    for (int i = 0; i < balls.size(1); i++) {
        scalar_t cx = balls[batch_base][i][0], cy = balls[batch_base][i][1];
        scalar_t cz = balls[batch_base][i][2], rad = balls[batch_base][i][3];
        scalar_t qa = dx*dx + dy*dy + dz*dz;
        scalar_t qb = 2*(dx*(ox-cx) + dy*(oy-cy) + dz*(oz-cz));
        scalar_t qc = (ox-cx)*(ox-cx) + (oy-cy)*(oy-cy) + (oz-cz)*(oz-cz) - rad*rad;
        scalar_t qd = qb*qb - 4*qa*qc;
        if (qa > kEps && qd >= 0) {
            scalar_t sqrt_qd = sqrt(qd);
            scalar_t qt = (-qb - sqrt_qd) / (2*qa);
            if (!(qt > (scalar_t)1e-5)) qt = (-qb + sqrt_qd) / (2*qa);
            if (qt > (scalar_t)1e-5 && qt < min_dist) {
                scalar_t px = ox + qt * dx;
                scalar_t py = oy + qt * dy;
                scalar_t pz = oz + qt * dz;
                scalar_t gx = px - cx;
                scalar_t gy = py - cy;
                scalar_t gz = pz - cz;
                scalar_t gn = sqrt(gx*gx + gy*gy + gz*gz);
                if (gn > kEps) {
                    nx = gx / gn; ny = gy / gn; nz = gz / gn;
                    min_dist = qt;
                }
            }
        }
    }

    // vertical cylinders
    for (int i = 0; i < cylinders.size(1); i++) {
        scalar_t cx = cylinders[batch_base][i][0], cy = cylinders[batch_base][i][1];
        scalar_t rad = cylinders[batch_base][i][2];
        scalar_t qa = dx*dx + dy*dy;
        scalar_t qb = 2*(dx*(ox-cx) + dy*(oy-cy));
        scalar_t qc = (ox-cx)*(ox-cx) + (oy-cy)*(oy-cy) - rad*rad;
        scalar_t qd = qb*qb - 4*qa*qc;
        if (qa > kEps && qd >= 0) {
            scalar_t sqrt_qd = sqrt(qd);
            scalar_t qt = (-qb - sqrt_qd) / (2*qa);
            if (!(qt > (scalar_t)1e-5)) qt = (-qb + sqrt_qd) / (2*qa);
            if (qt > (scalar_t)1e-5 && qt < min_dist) {
                scalar_t px = ox + qt * dx;
                scalar_t py = oy + qt * dy;
                scalar_t gx = px - cx;
                scalar_t gy = py - cy;
                scalar_t gn = sqrt(gx*gx + gy*gy);
                if (gn > kEps) {
                    nx = gx / gn; ny = gy / gn; nz = (scalar_t)0;
                    min_dist = qt;
                }
            }
        }
    }

    // horizontal cylinders
    for (int i = 0; i < cylinders_h.size(1); i++) {
        scalar_t cx = cylinders_h[batch_base][i][0], cz = cylinders_h[batch_base][i][1];
        scalar_t rad = cylinders_h[batch_base][i][2];
        scalar_t qa = dx*dx + dz*dz;
        scalar_t qb = 2*(dx*(ox-cx) + dz*(oz-cz));
        scalar_t qc = (ox-cx)*(ox-cx) + (oz-cz)*(oz-cz) - rad*rad;
        scalar_t qd = qb*qb - 4*qa*qc;
        if (qa > kEps && qd >= 0) {
            scalar_t sqrt_qd = sqrt(qd);
            scalar_t qt = (-qb - sqrt_qd) / (2*qa);
            if (!(qt > (scalar_t)1e-5)) qt = (-qb + sqrt_qd) / (2*qa);
            if (qt > (scalar_t)1e-5 && qt < min_dist) {
                scalar_t px = ox + qt * dx;
                scalar_t pz = oz + qt * dz;
                scalar_t gx = px - cx;
                scalar_t gz = pz - cz;
                scalar_t gn = sqrt(gx*gx + gz*gz);
                if (gn > kEps) {
                    nx = gx / gn; ny = (scalar_t)0; nz = gz / gn;
                    min_dist = qt;
                }
            }
        }
    }

    // voxels (AABB)
    for (int i = 0; i < voxels.size(1); i++) {
        scalar_t cx = voxels[batch_base][i][0], cy = voxels[batch_base][i][1];
        scalar_t cz = voxels[batch_base][i][2];
        scalar_t rx = voxels[batch_base][i][3], ry = voxels[batch_base][i][4];
        scalar_t rz = voxels[batch_base][i][5];
        scalar_t tx_min, tx_max, ty_min, ty_max, tz_min, tz_max;

        if (abs(dx) <= kEps) {
            if (ox < cx - rx || ox > cx + rx) continue;
            tx_min = (scalar_t)-1e20; tx_max = (scalar_t)1e20;
        } else {
            scalar_t tx1 = (cx - rx - ox) / dx;
            scalar_t tx2 = (cx + rx - ox) / dx;
            tx_min = min(tx1, tx2);
            tx_max = max(tx1, tx2);
        }
        if (abs(dy) <= kEps) {
            if (oy < cy - ry || oy > cy + ry) continue;
            ty_min = (scalar_t)-1e20; ty_max = (scalar_t)1e20;
        } else {
            scalar_t ty1 = (cy - ry - oy) / dy;
            scalar_t ty2 = (cy + ry - oy) / dy;
            ty_min = min(ty1, ty2);
            ty_max = max(ty1, ty2);
        }
        if (abs(dz) <= kEps) {
            if (oz < cz - rz || oz > cz + rz) continue;
            tz_min = (scalar_t)-1e20; tz_max = (scalar_t)1e20;
        } else {
            scalar_t tz1 = (cz - rz - oz) / dz;
            scalar_t tz2 = (cz + rz - oz) / dz;
            tz_min = min(tz1, tz2);
            tz_max = max(tz1, tz2);
        }

        scalar_t t_min_v = max(max(tx_min, ty_min), tz_min);
        scalar_t t_max_v = min(min(tx_max, ty_max), tz_max);
        if (t_min_v > 0 && t_min_v < t_max_v && t_min_v < min_dist) {
            scalar_t px = ox + t_min_v * dx;
            scalar_t py = oy + t_min_v * dy;
            scalar_t pz = oz + t_min_v * dz;
            scalar_t lx = px - cx;
            scalar_t ly = py - cy;
            scalar_t lz = pz - cz;
            scalar_t ex = abs(abs(lx) - rx);
            scalar_t ey = abs(abs(ly) - ry);
            scalar_t ez = abs(abs(lz) - rz);
            if (ex <= ey && ex <= ez) {
                nx = (lx >= 0) ? (scalar_t)1 : (scalar_t)-1;
                ny = (scalar_t)0;
                nz = (scalar_t)0;
            } else if (ey <= ex && ey <= ez) {
                nx = (scalar_t)0;
                ny = (ly >= 0) ? (scalar_t)1 : (scalar_t)-1;
                nz = (scalar_t)0;
            } else {
                nx = (scalar_t)0;
                ny = (scalar_t)0;
                nz = (lz >= 0) ? (scalar_t)1 : (scalar_t)-1;
            }
            min_dist = t_min_v;
        }
    }

    *out_nx = nx;
    *out_ny = ny;
    *out_nz = nz;
    return min_dist;
}

// ============================================================================
// 深度图渲染 CUDA 内核 (Depth Rendering CUDA Kernel)
// ============================================================================
template <typename scalar_t>
__global__ void render_depth_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> canvas,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> balls,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> cylinders,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> cylinders_h,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> voxels,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> R,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> pos,
    int n_drones_per_group,
    float fov_x_half_tan) {

    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int B = canvas.size(0);
    const int H = canvas.size(1);
    const int W = canvas.size(2);
    if (c >= B * H * W) return;
    const int b = c / (H * W);
    const int u = (c % (H * W)) / W;
    const int v = c % W;

    const scalar_t fov = (scalar_t)fov_x_half_tan;
    const scalar_t fov_y_ht = fov / W * H;

    const scalar_t fu = (2 * (u + 0.5) / H - 1) * fov_y_ht - 1e-5;
    const scalar_t fv = (2 * (v + 0.5) / W - 1) * fov - 1e-5;
    scalar_t dx = R[b][0][0] - fu * R[b][0][2] - fv * R[b][0][1];
    scalar_t dy = R[b][1][0] - fu * R[b][1][2] - fv * R[b][1][1];
    scalar_t dz = R[b][2][0] - fu * R[b][2][2] - fv * R[b][2][1];

    const int batch_base = (b / n_drones_per_group) * n_drones_per_group;

    canvas[b][u][v] = trace_ray_device(dx, dy, dz,
        pos[b][0], pos[b][1], pos[b][2],
        balls, cylinders, cylinders_h, voxels, pos,
        n_drones_per_group, batch_base, b, B);
}

} // namespace

// ============================================================================
// C++ 接口函数：寻找最近点 (C++ Interface: Find Nearest Point)
// ============================================================================
void find_nearest_pt_cuda(
    torch::Tensor nearest_pt,
    torch::Tensor balls,
    torch::Tensor cylinders,
    torch::Tensor cylinders_h,
    torch::Tensor voxels,
    torch::Tensor pos,
    float drone_radius,
    int n_drones_per_group) {
    
    const int threads = 1024;
    size_t state_size = pos.size(0) * pos.size(1); // 时间步数 * 批次大小 (Time steps * Batch size)
    const dim3 blocks((state_size + threads - 1) / threads);
    
    AT_DISPATCH_FLOATING_TYPES(pos.type(), "nearest_pt_cuda", ([&] {
        nearest_pt_cuda_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            nearest_pt.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            balls.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            cylinders.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            cylinders_h.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            voxels.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            pos.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            drone_radius,
            n_drones_per_group);
    }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    at::cuda::getCurrentCUDAStream().synchronize();
}

// ============================================================================
// C++ 接口函数：寻找最近点 (考虑椭球体形状) (C++ Interface: Find Nearest Point Ellipsoid)
// ============================================================================
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
    float ellipsoid_c) {
    
    const int threads = 1024;
    size_t state_size = pos.size(0) * pos.size(1);
    const dim3 blocks((state_size + threads - 1) / threads);
    
    AT_DISPATCH_FLOATING_TYPES(pos.type(), "nearest_pt_ellipsoid_cuda", ([&] {
        nearest_pt_ellipsoid_cuda_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            nearest_pt.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            balls.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            cylinders.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            cylinders_h.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            voxels.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            pos.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            R_body.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            drone_radius,
            n_drones_per_group,
            ellipsoid_a,
            ellipsoid_c);
    }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    at::cuda::getCurrentCUDAStream().synchronize();
}

// ============================================================================
// C++ 接口函数：深度图渲染 (C++ Interface: Depth Rendering)
// ============================================================================
void render_depth_cuda(
    torch::Tensor canvas,
    torch::Tensor balls,
    torch::Tensor cylinders,
    torch::Tensor cylinders_h,
    torch::Tensor voxels,
    torch::Tensor R,
    torch::Tensor pos,
    int n_drones_per_group,
    float fov_x_half_tan) {

    const int threads = 1024;
    size_t state_size = canvas.numel();
    const dim3 blocks((state_size + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(canvas.type(), "render_depth_cuda", ([&] {
        render_depth_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            canvas.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            balls.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            cylinders.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            cylinders_h.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            voxels.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            R.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            pos.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            n_drones_per_group,
            fov_x_half_tan);
    }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    at::cuda::getCurrentCUDAStream().synchronize();
}


// ============================================================================
// C++ 接口函数：diff_depth 可微前向（CUDA高性能路径）
// 说明：
// - 几何深度：复用 render_depth_cuda（CUDA）
// - 传感器与噪声：使用 ATen 张量算子（GPU 上执行）
// - 返回：noisy_depth, quality
// ============================================================================
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
    double max_range) {

    const auto B = pos.size(0);
    auto depth = torch::empty({B, height, width}, pos.options());
    render_depth_cuda(
        depth, balls, cylinders, cylinders_h, voxels,
        R, pos, n_drones_per_group, fov_x_half_tan);

    depth = torch::clamp(depth, 0.03, 120.0);

    auto power_scaled = (0.01 + power * 0.99).unsqueeze(1).unsqueeze(2);
    auto exp_scaled = (0.05 + exposure * 0.95).unsqueeze(1).unsqueeze(2);
    auto gain_scaled = (1.0 + gain * 9.0).unsqueeze(1).unsqueeze(2);

    auto energy_recv = (power_scaled * exp_scaled) / (depth * depth + 0.1);
    energy_recv = energy_recv * gain_scaled * 100.0;
    energy_recv = energy_recv.clamp_max(1e6);  // 防止 Inf → NaN

    auto quality_raw = torch::tanh(energy_recv * 0.5);

    auto speed = v.norm(2, -1);
    auto motion_blur_factor = (speed * exp_scaled.squeeze(2).squeeze(1) * 0.1).clamp(0.0, 1.0);
    auto mbf = motion_blur_factor.unsqueeze(1).unsqueeze(2);

    auto blur_kernel = at::avg_pool2d(
        depth.unsqueeze(1),
        {3, 3},
        {1, 1},
        {1, 1},
        false,
        true,
        c10::nullopt).squeeze(1);
    auto depth_blurred = depth * (1.0 - mbf) + blur_kernel * mbf;

    auto quality = quality_raw * (1.0 - mbf * 0.8);

    auto noise_std = (0.05 * gain_scaled) / (energy_recv + 1e-3);
    noise_std = noise_std.clamp(0.01, 1.0);

    auto noisy_depth = depth_blurred + torch::randn_like(depth_blurred) * noise_std;
    noisy_depth = noisy_depth.clamp(0.05, max_range);

    return {noisy_depth, quality};
}

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
    double max_range) {

    const auto B = pos.size(0);
    auto opts = pos.options();

    auto go_depth = grad_noisy_depth.contiguous();
    auto go_quality = grad_quality.contiguous();

    // 重新计算几何深度与中间量（与 forward 路径一致）
    auto depth = torch::empty({B, height, width}, opts);
    render_depth_cuda(
        depth, balls, cylinders, cylinders_h, voxels,
        R, pos, n_drones_per_group, fov_x_half_tan);
    depth = torch::clamp(depth, 0.03, 120.0);

    auto ps = (0.01 + power * 0.99).unsqueeze(1).unsqueeze(2);   // power_scaled
    auto es = (0.05 + exposure * 0.95).unsqueeze(1).unsqueeze(2); // exp_scaled
    auto gs = (1.0 + gain * 9.0).unsqueeze(1).unsqueeze(2);       // gain_scaled

    auto d2 = depth * depth;
    auto energy_recv = (ps * es) / (d2 + 0.1);
    energy_recv = energy_recv * gs * 100.0;

    auto quality_raw = torch::tanh(energy_recv * 0.5);

    auto speed = v.norm(2, -1); // (B,)
    auto mbf0 = speed * es.squeeze(2).squeeze(1) * 0.1; // unclamped motion blur factor
    auto mbf = mbf0.clamp(0.0, 1.0);
    auto m = mbf.unsqueeze(1).unsqueeze(2); // (B,1,1)

    auto blur_kernel = at::avg_pool2d(
        depth.unsqueeze(1),
        {3, 3},
        {1, 1},
        {1, 1},
        false,
        true,
        c10::nullopt).squeeze(1);
    auto depth_blurred = depth * (1.0 - m) + blur_kernel * m;

    auto noise_std0 = (0.05 * gs) / (energy_recv + 1e-3);
    auto noise_std = noise_std0.clamp(0.01, 1.0);

    // 由输出近似恢复噪声样本 epsilon（用于重参数化反向）
    auto eps = (noisy_depth - depth_blurred) / noise_std.clamp_min(1e-6);

    // clamp 的有效梯度掩码
    auto mask_depth = ((noisy_depth > 0.05 + 1e-6) & (noisy_depth < (double)max_range - 1e-6)).to(go_depth.scalar_type());
    auto g_noisy = go_depth * mask_depth;

    // noisy_depth = depth_blurred + eps * noise_std
    auto g_depth_blurred = g_noisy;
    auto g_noise_std = g_noisy * eps;

    // quality = quality_raw * (1 - 0.8 m)
    auto g_quality_raw = go_quality * (1.0 - 0.8 * m);
    auto g_m_from_quality = go_quality * (-0.8 * quality_raw);

    // depth_blurred = depth*(1-m) + blur*m
    auto g_m_from_blur = g_depth_blurred * (blur_kernel - depth);
    auto g_m_total = g_m_from_quality + g_m_from_blur;

    auto g_depth = g_depth_blurred * (1.0 - m);
    auto g_blur = g_depth_blurred * m;

    // 近似传播 blur 对 depth 的梯度（与 forward 的 avg_pool 匹配的近似）
    auto g_blur_to_depth = at::avg_pool2d(
        g_blur.unsqueeze(1),
        {3, 3},
        {1, 1},
        {1, 1},
        false,
        true,
        c10::nullopt).squeeze(1);
    g_depth = g_depth + g_blur_to_depth;

    // m = clamp(m0,0,1), m0 = speed * es_scalar * 0.1
    auto mask_m = ((mbf0 > 0.0) & (mbf0 < 1.0)).to(go_depth.scalar_type()); // (B,)
    auto g_m_sum = g_m_total.sum({1, 2}); // (B,)
    auto g_m0 = g_m_sum * mask_m; // (B,)

    // noise_std = clamp(noise_std0, 0.01, 1.0)
    auto mask_ns = ((noise_std0 > 0.01) & (noise_std0 < 1.0)).to(go_depth.scalar_type());
    auto g_noise_std0 = g_noise_std * mask_ns;

    // noise_std0 = 0.05 * gs / (E + 1e-3)
    auto denomE = (energy_recv + 1e-3);
    auto g_gs_from_ns = (g_noise_std0 * (0.05 / denomE)).sum({1, 2});
    auto g_E_from_ns = g_noise_std0 * (-0.05 * gs / (denomE * denomE));

    // quality_raw = tanh(0.5E)
    auto g_E_from_quality = g_quality_raw * (0.5 * (1.0 - quality_raw * quality_raw));
    auto g_E = g_E_from_quality + g_E_from_ns;

    // E = 100 * ps * es * gs / (d^2 + 0.1)
    auto inv_den = 1.0 / (d2 + 0.1);
    auto g_ps = (g_E * (100.0 * es * gs * inv_den)).sum({1, 2});
    auto g_es_from_E = (g_E * (100.0 * ps * gs * inv_den)).sum({1, 2});
    auto g_gs_from_E = (g_E * (100.0 * ps * es * inv_den)).sum({1, 2});

    auto g_depth_from_E = g_E * (-200.0 * ps * es * gs * depth / ((d2 + 0.1) * (d2 + 0.1)));
    g_depth = g_depth + g_depth_from_E;

    // es = 0.05 + 0.95 * exposure;  m0 对 es 的额外依赖
    auto g_es_from_m = g_m0 * (speed * 0.1);
    auto g_es_total = g_es_from_E + g_es_from_m;

    // gs 合并两路
    auto g_gs_total = g_gs_from_E + g_gs_from_ns;

    // 回到原始输入尺度
    auto grad_power = g_ps * 0.99;
    auto grad_exposure = g_es_total * 0.95;
    auto grad_gain = g_gs_total * 9.0;

    return {grad_power, grad_exposure, grad_gain};
}
