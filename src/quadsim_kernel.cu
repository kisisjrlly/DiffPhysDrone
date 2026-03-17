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
// 深度图渲染 CUDA 内核 (Depth Rendering CUDA Kernel)
// 
// 该内核通过光线追踪 (Ray Tracing) 的方式，为每个无人机渲染深度图。
// 它计算从相机中心发出的光线与场景中各种几何体（地面、其他无人机、球体、圆柱体、体素）的交点，
// 并记录最近的交点距离作为深度值。
// ============================================================================
template <typename scalar_t>
__global__ void render_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> canvas,      // 输出：深度图画布 (Output: Depth map canvas) [B, H, W]
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> flow,        // 输出：光流图 (Output: Optical flow) [B, H, W, 2] (当前未使用)
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> balls,       // 场景中的球体障碍物 (Spherical obstacles)
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> cylinders,   // 场景中的垂直圆柱体障碍物 (Vertical cylindrical obstacles)
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> cylinders_h, // 场景中的水平圆柱体障碍物 (Horizontal cylindrical obstacles)
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> voxels,      // 场景中的体素/长方体障碍物 (Voxel/Box obstacles)
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> R,           // 当前相机的旋转矩阵 (Current camera rotation matrix)
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> R_old,       // 上一帧相机的旋转矩阵 (Previous camera rotation matrix)
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> pos,         // 当前相机的位置 (Current camera position)
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> pos_old,     // 上一帧相机的位置 (Previous camera position)
    float drone_radius,                                                                  // 无人机半径 (Drone radius)
    int n_drones_per_group,                                                              // 每组无人机数量 (Number of drones per group)
    float fov_x_half_tan) {                                                              // 水平视场角一半的正切值 (Tan of half horizontal FOV)

    // 计算当前线程对应的像素坐标和批次索引 (Calculate pixel coordinates and batch index for current thread)
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int B = canvas.size(0);
    const int H = canvas.size(1);
    const int W = canvas.size(2);
    if (c >= B * H * W) return;
    const int b = c / (H * W);           // 批次索引 (Batch index)
    const int u = (c % (H * W)) / W;     // 像素行索引 (Pixel row index)
    const int v = c % W;                 // 像素列索引 (Pixel column index)
    
    // 计算相机坐标系下的光线方向 (Calculate ray direction in camera frame)
    const scalar_t fov_y_half_tan = fov_x_half_tan / W * H;
    const scalar_t fu = (2 * (u + 0.5) / H - 1) * fov_y_half_tan - 1e-5;
    const scalar_t fv = (2 * (v + 0.5) / W - 1) * fov_x_half_tan - 1e-5;
    
    // 将光线方向转换到世界坐标系 (Transform ray direction to world frame)
    scalar_t dx = R[b][0][0] - fu * R[b][0][2] - fv * R[b][0][1];
    scalar_t dy = R[b][1][0] - fu * R[b][1][2] - fv * R[b][1][1];
    scalar_t dz = R[b][2][0] - fu * R[b][2][2] - fv * R[b][2][1];
    
    // 光线起点 (Ray origin)
    const scalar_t ox = pos[b][0];
    const scalar_t oy = pos[b][1];
    const scalar_t oz = pos[b][2];

    // 初始化最小距离为无穷大 (Initialize minimum distance to infinity)
    scalar_t min_dist = 100;
    
    // 1. 与地面的交点 (Intersection with ground plane z = -1)
    const scalar_t kEps = (scalar_t)1e-8;
    if (abs(dz) > kEps) {
        scalar_t t = (-1 - oz) / dz;
        if (t > 0) min_dist = t;
    }

    // 2. 与其他无人机的交点 (Intersection with other drones in the same group)
    const int batch_base = (b / n_drones_per_group) * n_drones_per_group;
    for (int i = batch_base; i < batch_base + n_drones_per_group; i++) {
        if (i == b || i >= B) continue; // 跳过自己 (Skip self)
        scalar_t cx = pos[i][0];
        scalar_t cy = pos[i][1];
        scalar_t cz = pos[i][2];
        scalar_t r = 0.15; // 假设其他无人机为半径 0.15 的椭球体 (Assume other drones are ellipsoids)
        
        // 解一元二次方程求交点 (Solve quadratic equation for intersection)
        // (ox + t dx)^2 + (oy + t dy)^2 + 4 (oz + t dz)^2 = r^2
        scalar_t a = dx * dx + dy * dy + 4 * dz * dz;
        scalar_t b = 2 * (dx * (ox - cx) + dy * (oy - cy) + 4 * dz * (oz - cz));
        scalar_t c = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy) + 4 * (oz - cz) * (oz - cz) - r * r;
        scalar_t d = b * b - 4 * a * c;
        if (a > kEps && d >= 0) {
            r = (-b-sqrt(d)) / (2 * a);
            if (r > 1e-5) {
                min_dist = min(min_dist, r);
            } else {
                r = (-b+sqrt(d)) / (2 * a);
                if (r > 1e-5) min_dist = min(min_dist, r);
            }
        }
    }

    // 3. 与球体障碍物的交点 (Intersection with spherical obstacles)
    for (int i = 0; i < balls.size(1); i++) {
        scalar_t cx = balls[batch_base][i][0];
        scalar_t cy = balls[batch_base][i][1];
        scalar_t cz = balls[batch_base][i][2];
        scalar_t r = balls[batch_base][i][3];
        scalar_t a = dx * dx + dy * dy + dz * dz;
        scalar_t b = 2 * (dx * (ox - cx) + dy * (oy - cy) + dz * (oz - cz));
        scalar_t c = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy) + (oz - cz) * (oz - cz) - r * r;
        scalar_t d = b * b - 4 * a * c;
        if (a > kEps && d >= 0) {
            r = (-b-sqrt(d)) / (2 * a);
            if (r > 1e-5) {
                min_dist = min(min_dist, r);
            } else {
                r = (-b+sqrt(d)) / (2 * a);
                if (r > 1e-5) min_dist = min(min_dist, r);
            }
        }
    }

    // 4. 与垂直圆柱体障碍物的交点 (Intersection with vertical cylindrical obstacles)
    for (int i = 0; i < cylinders.size(1); i++) {
        scalar_t cx = cylinders[batch_base][i][0];
        scalar_t cy = cylinders[batch_base][i][1];
        scalar_t r = cylinders[batch_base][i][2];
        scalar_t a = dx * dx + dy * dy;
        scalar_t b = 2 * (dx * (ox - cx) + dy * (oy - cy));
        scalar_t c = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy) - r * r;
        scalar_t d = b * b - 4 * a * c;
        if (a > kEps && d >= 0) {
            r = (-b-sqrt(d)) / (2 * a);
            if (r > 1e-5) {
                min_dist = min(min_dist, r);
            } else {
                r = (-b+sqrt(d)) / (2 * a);
                if (r > 1e-5) min_dist = min(min_dist, r);
            }
        }
    }
    
    // 5. 与水平圆柱体障碍物的交点 (Intersection with horizontal cylindrical obstacles)
    for (int i = 0; i < cylinders_h.size(1); i++) {
        scalar_t cx = cylinders_h[batch_base][i][0];
        scalar_t cz = cylinders_h[batch_base][i][1];
        scalar_t r = cylinders_h[batch_base][i][2];
        scalar_t a = dx * dx + dz * dz;
        scalar_t b = 2 * (dx * (ox - cx) + dz * (oz - cz));
        scalar_t c = (ox - cx) * (ox - cx) + (oz - cz) * (oz - cz) - r * r;
        scalar_t d = b * b - 4 * a * c;
        if (a > kEps && d >= 0) {
            r = (-b-sqrt(d)) / (2 * a);
            if (r > 1e-5) {
                min_dist = min(min_dist, r);
            } else {
                r = (-b+sqrt(d)) / (2 * a);
                if (r > 1e-5) min_dist = min(min_dist, r);
            }
        }
    }

    // 6. 与体素/长方体障碍物的交点 (Intersection with voxel/box obstacles using AABB ray intersection)
    for (int i = 0; i < voxels.size(1); i++) {
        scalar_t cx = voxels[batch_base][i][0];
        scalar_t cy = voxels[batch_base][i][1];
        scalar_t cz = voxels[batch_base][i][2];
        scalar_t rx = voxels[batch_base][i][3];
        scalar_t ry = voxels[batch_base][i][4];
        scalar_t rz = voxels[batch_base][i][5];
        
        // 计算与各个面的交点参数 t (Calculate intersection parameters t for each face)
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
        
        // 找到进入和离开长方体的 t 值 (Find entry and exit t values for the box)
        scalar_t t_min = max(max(tx_min, ty_min), tz_min);
        scalar_t t_max = min(min(tx_max, ty_max), tz_max);
        
        // 如果光线与长方体相交且在相机前方 (If ray intersects box and is in front of camera)
        if (t_min < min_dist && t_min < t_max && t_min > 0)
            min_dist = t_min;
    }

    // 将最小距离写入深度图画布 (Write minimum distance to depth map canvas)
    canvas[b][u][v] = min_dist;
}

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
// 可微视场前向渲染 CUDA 内核 (Differentiable FOV Forward Rendering CUDA Kernel)
// 
// 使用每个批次独立的 FOV 张量进行渲染，使得渲染过程对 FOV 可微。
// ============================================================================
template <typename scalar_t>
__global__ void render_diff_fov_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> canvas,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> balls,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> cylinders,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> cylinders_h,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> voxels,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> R,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> pos,
    int n_drones_per_group,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> fov_x_half_tan) { // 每个批次的 FOV (Per-batch FOV)

    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int B = canvas.size(0);
    const int H = canvas.size(1);
    const int W = canvas.size(2);
    if (c >= B * H * W) return;
    const int b = c / (H * W);
    const int u = (c % (H * W)) / W;
    const int v = c % W;

    // 获取当前批次的 FOV (Get FOV for current batch)
    const scalar_t fov = fov_x_half_tan[b];
    const scalar_t fov_y_ht = fov / W * H;
    
    // 计算光线方向 (Calculate ray direction)
    const scalar_t fu = (2 * (u + 0.5) / H - 1) * fov_y_ht - 1e-5;
    const scalar_t fv = (2 * (v + 0.5) / W - 1) * fov - 1e-5;
    scalar_t dx = R[b][0][0] - fu * R[b][0][2] - fv * R[b][0][1];
    scalar_t dy = R[b][1][0] - fu * R[b][1][2] - fv * R[b][1][1];
    scalar_t dz = R[b][2][0] - fu * R[b][2][2] - fv * R[b][2][1];

    const int batch_base = (b / n_drones_per_group) * n_drones_per_group;
    
    // 调用设备函数进行光线追踪 (Call device function for ray tracing)
    canvas[b][u][v] = trace_ray_device(dx, dy, dz,
        pos[b][0], pos[b][1], pos[b][2],
        balls, cylinders, cylinders_h, voxels, pos,
        n_drones_per_group, batch_base, b, B);
}


// ============================================================================
// 可微视场前向渲染 CUDA 内核（含法线输出）
// (Differentiable FOV forward kernel with normal map output)
// ============================================================================
template <typename scalar_t>
__global__ void render_diff_fov_with_normal_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> canvas,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> normals,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> balls,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> cylinders,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> cylinders_h,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> voxels,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> R,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> pos,
    int n_drones_per_group,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> fov_x_half_tan) {

    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int B = canvas.size(0);
    const int H = canvas.size(1);
    const int W = canvas.size(2);
    if (c >= B * H * W) return;
    const int b = c / (H * W);
    const int u = (c % (H * W)) / W;
    const int v = c % W;

    const scalar_t fov = fov_x_half_tan[b];
    const scalar_t fov_y_ht = fov / W * H;
    const scalar_t fu = (2 * (u + 0.5) / H - 1) * fov_y_ht - 1e-5;
    const scalar_t fv = (2 * (v + 0.5) / W - 1) * fov - 1e-5;

    scalar_t dx = R[b][0][0] - fu * R[b][0][2] - fv * R[b][0][1];
    scalar_t dy = R[b][1][0] - fu * R[b][1][2] - fv * R[b][1][1];
    scalar_t dz = R[b][2][0] - fu * R[b][2][2] - fv * R[b][2][1];

    const int batch_base = (b / n_drones_per_group) * n_drones_per_group;

    scalar_t nx = (scalar_t)0, ny = (scalar_t)0, nz = (scalar_t)0;
    scalar_t depth = trace_ray_with_normal_device(
        dx, dy, dz,
        pos[b][0], pos[b][1], pos[b][2],
        balls, cylinders, cylinders_h, voxels, pos,
        n_drones_per_group, batch_base, b, B,
        &nx, &ny, &nz);

    canvas[b][u][v] = depth;
    normals[b][0][u][v] = nx;
    normals[b][1][u][v] = ny;
    normals[b][2][u][v] = nz;
}


// ============================================================================
// 可微视场反向传播 CUDA 内核 (Differentiable FOV Backward CUDA Kernel)
// 
// 通过有限差分法计算深度对 FOV 的梯度: d(depth)/d(fov)，
// 并使用 atomicAdd 累加每个批次的梯度。
// ============================================================================
template <typename scalar_t>
__global__ void render_backward_fov_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> grad_output, // 输入：来自下游的梯度 (Input: gradient from downstream)
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> canvas,      // 输入：原始深度图 (Input: original depth map)
    scalar_t* __restrict__ grad_fov,                                                     // 输出：对 FOV 的梯度 (Output: gradient w.r.t FOV)
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> balls,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> cylinders,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> cylinders_h,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> voxels,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> R,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> pos,
    int n_drones_per_group,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> fov_x_half_tan) {

    // 3D Grid: x->W, y->H, z->B
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    const int u = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;

    const int B = canvas.size(0);
    const int H = canvas.size(1);
    const int W = canvas.size(2);

    // blockDim = (16,16,1) => 256 threads per block
    __shared__ scalar_t s_grad[256];
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    s_grad[tid] = (scalar_t)0;

    if (b < B && u < H && v < W) {
        const scalar_t go = grad_output[b][u][v];
        if (abs(go) >= (scalar_t)1e-8) {
            const scalar_t fov = fov_x_half_tan[b];
            const scalar_t ox = pos[b][0], oy = pos[b][1], oz = pos[b][2];

            // 原始光线方向 d
            const scalar_t fov_y = fov / W * H;
            const scalar_t fu = (2 * (u + 0.5) / H - 1) * fov_y - 1e-5;
            const scalar_t fv = (2 * (v + 0.5) / W - 1) * fov - 1e-5;
            const scalar_t dx = R[b][0][0] - fu * R[b][0][2] - fv * R[b][0][1];
            const scalar_t dy = R[b][1][0] - fu * R[b][1][2] - fv * R[b][1][1];
            const scalar_t dz = R[b][2][0] - fu * R[b][2][2] - fv * R[b][2][1];

            // d / d(fov)
            const scalar_t d_fv_d_fov = (2 * (v + 0.5) / W - 1);
            const scalar_t d_fu_d_fov = (2 * (u + 0.5) / H - 1) * ((scalar_t)H / W);
            const scalar_t d_dx_d_fov = -d_fu_d_fov * R[b][0][2] - d_fv_d_fov * R[b][0][1];
            const scalar_t d_dy_d_fov = -d_fu_d_fov * R[b][1][2] - d_fv_d_fov * R[b][1][1];
            const scalar_t d_dz_d_fov = -d_fu_d_fov * R[b][2][2] - d_fv_d_fov * R[b][2][1];

            const int batch_base = (b / n_drones_per_group) * n_drones_per_group;
            scalar_t nx = (scalar_t)0, ny = (scalar_t)0, nz = (scalar_t)0;

            // 单次追踪原始光线并拿到命中法线
            scalar_t depth_hit = trace_ray_with_normal_device(
                dx, dy, dz, ox, oy, oz,
                balls, cylinders, cylinders_h, voxels, pos,
                n_drones_per_group, batch_base, b, B,
                &nx, &ny, &nz);

            // 使用前向缓存深度作为 D（与计算图保持一致）
            scalar_t D = canvas[b][u][v];
            if (D < (scalar_t)99.0 && depth_hit < (scalar_t)99.0) {
                scalar_t n_dot_d = nx * dx + ny * dy + nz * dz;
                if (abs(n_dot_d) > (scalar_t)1e-6) {
                    scalar_t n_dot_dd_dfov = nx * d_dx_d_fov + ny * d_dy_d_fov + nz * d_dz_d_fov;
                    scalar_t local_grad = -D * (n_dot_dd_dfov / n_dot_d);
                    s_grad[tid] = go * local_grad;
                }
            }
        }
    }

    __syncthreads();

    // Tree reduction in shared memory
    for (int s = (blockDim.x * blockDim.y) / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_grad[tid] += s_grad[tid + s];
        }
        __syncthreads();
    }

    // Single atomic add per block
    if (tid == 0 && s_grad[0] != (scalar_t)0) {
        atomicAdd(&grad_fov[b], s_grad[0]);
    }
}


// ============================================================================
// 可微视场反向传播 CUDA 内核（基于法线图解析梯度）
// (Differentiable FOV backward kernel from normal map, analytical)
// ============================================================================
template <typename scalar_t>
__global__ void render_backward_fov_from_normal_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> grad_output,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> canvas,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> normals,
    scalar_t* __restrict__ grad_fov,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> R,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> fov_x_half_tan) {

    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    const int u = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;

    const int B = canvas.size(0);
    const int H = canvas.size(1);
    const int W = canvas.size(2);

    __shared__ scalar_t s_grad[256];
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    s_grad[tid] = (scalar_t)0;

    if (b < B && u < H && v < W) {
        const scalar_t go = grad_output[b][u][v];
        const scalar_t D = canvas[b][u][v];

        if (abs(go) >= (scalar_t)1e-8 && D < (scalar_t)99.0) {
            const scalar_t fov = fov_x_half_tan[b];

            // 当前光线方向
            const scalar_t fov_y = fov / W * H;
            const scalar_t fu = (2 * (u + 0.5) / H - 1) * fov_y - 1e-5;
            const scalar_t fv = (2 * (v + 0.5) / W - 1) * fov - 1e-5;
            const scalar_t dx = R[b][0][0] - fu * R[b][0][2] - fv * R[b][0][1];
            const scalar_t dy = R[b][1][0] - fu * R[b][1][2] - fv * R[b][1][1];
            const scalar_t dz = R[b][2][0] - fu * R[b][2][2] - fv * R[b][2][1];

            // d(d)/d(fov)
            const scalar_t d_fv_d_fov = (2 * (v + 0.5) / W - 1);
            const scalar_t d_fu_d_fov = (2 * (u + 0.5) / H - 1) * ((scalar_t)H / W);
            const scalar_t d_dx_d_fov = -d_fu_d_fov * R[b][0][2] - d_fv_d_fov * R[b][0][1];
            const scalar_t d_dy_d_fov = -d_fu_d_fov * R[b][1][2] - d_fv_d_fov * R[b][1][1];
            const scalar_t d_dz_d_fov = -d_fu_d_fov * R[b][2][2] - d_fv_d_fov * R[b][2][1];

            const scalar_t nx = normals[b][0][u][v];
            const scalar_t ny = normals[b][1][u][v];
            const scalar_t nz = normals[b][2][u][v];

            const scalar_t n_dot_d = nx * dx + ny * dy + nz * dz;
            if (abs(n_dot_d) > (scalar_t)5e-2) {
                const scalar_t n_dot_dd_dfov = nx * d_dx_d_fov + ny * d_dy_d_fov + nz * d_dz_d_fov;
                scalar_t local_grad = -D * (n_dot_dd_dfov / n_dot_d);
                local_grad = max((scalar_t)-500.0, min((scalar_t)500.0, local_grad));
                s_grad[tid] = go * local_grad;
            }
        }
    }

    __syncthreads();
    for (int s = (blockDim.x * blockDim.y) / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_grad[tid] += s_grad[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0 && s_grad[0] != (scalar_t)0) {
        atomicAdd(&grad_fov[b], s_grad[0]);
    }
}

// ============================================================================
// 深度图重渲染反向传播 CUDA 内核 (Rerender Backward CUDA Kernel)
// 
// 计算深度图对相机位姿的导数 (dddp)。
// ============================================================================
template <typename scalar_t>
__global__ void rerender_backward_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> depth, // 输入：深度图 (Input: depth map)
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> dddp,  // 输出：深度对位姿的导数 (Output: derivative of depth w.r.t pose)
    float fov_x_half_tan) {

    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int B = dddp.size(0);
    const int H = dddp.size(2);
    const int W = dddp.size(3);
    if (c >= B * H * W) return;
    const int b = c / (H * W);
    const int u = (c % (H * W)) / W;
    const int v = c % W;

    const scalar_t unit = fov_x_half_tan / W;
    
    // 计算 2x2 像素块的平均深度 (Calculate average depth of 2x2 pixel block)
    const scalar_t d = (depth[b][0][u*2][v*2] + depth[b][0][u*2+1][v*2] + depth[b][0][u*2][v*2+1] + depth[b][0][u*2+1][v*2+1]) / 4 * unit;
    
    // 计算深度在 y 和 z 方向的梯度 (Calculate depth gradients in y and z directions)
    const scalar_t dddy = (depth[b][0][u*2][v*2] + depth[b][0][u*2+1][v*2] - depth[b][0][u*2][v*2+1] - depth[b][0][u*2+1][v*2+1]) / 2 / d;
    const scalar_t dddz = (depth[b][0][u*2][v*2] - depth[b][0][u*2+1][v*2] + depth[b][0][u*2][v*2+1] - depth[b][0][u*2+1][v*2+1]) / 2 / d;
    
    // 归一化梯度向量 (Normalize gradient vector)
    const scalar_t dddp_norm = max(8., sqrt(1 + dddy * dddy + dddz * dddz));
    dddp[b][0][u][v] = -1. / dddp_norm;
    dddp[b][1][u][v] = dddy / dddp_norm;
    dddp[b][2][u][v] = dddz / dddp_norm;
}

} // namespace

// ============================================================================
// C++ 接口函数：深度图渲染 (C++ Interface: Depth Rendering)
// 
// 负责计算线程块数量并启动 render_cuda_kernel。
// ============================================================================
void render_cuda(
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
    float fov_x_half_tan) {
    
    const int threads = 1024; // 每个 block 的线程数 (Threads per block)
    size_t state_size = canvas.numel(); // 总像素数 (Total number of pixels)
    const dim3 blocks((state_size + threads - 1) / threads); // 计算 block 数量 (Calculate number of blocks)

    // 启动 CUDA 内核 (Launch CUDA kernel)
    AT_DISPATCH_FLOATING_TYPES(canvas.type(), "render_cuda", ([&] {
        render_cuda_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            canvas.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            flow.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            balls.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            cylinders.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            cylinders_h.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            voxels.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            R.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            R_old.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            pos.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            pos_old.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            drone_radius,
            n_drones_per_group,
            fov_x_half_tan);
    }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ============================================================================
// C++ 接口函数：深度图重渲染反向传播 (C++ Interface: Rerender Backward)
// ============================================================================
void rerender_backward_cuda(
    torch::Tensor depth,
    torch::Tensor dddp,
    float fov_x_half_tan) {
    
    const int threads = 1024;
    size_t state_size = dddp.numel();
    const dim3 blocks((state_size + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(depth.type(), "rerender_backward_cuda", ([&] {
        rerender_backward_cuda_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            depth.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            dddp.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            fov_x_half_tan);
    }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

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
}

// ============================================================================
// C++ 接口函数：可微视场前向渲染 (C++ Interface: Differentiable FOV Forward Rendering)
// ============================================================================
void render_diff_fov_cuda(
    torch::Tensor canvas,
    torch::Tensor balls,
    torch::Tensor cylinders,
    torch::Tensor cylinders_h,
    torch::Tensor voxels,
    torch::Tensor R,
    torch::Tensor pos,
    int n_drones_per_group,
    torch::Tensor fov_x_half_tan) {
    
    const int threads = 1024;
    size_t state_size = canvas.numel();
    const dim3 blocks((state_size + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(canvas.type(), "render_diff_fov_cuda", ([&] {
        render_diff_fov_cuda_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            canvas.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            balls.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            cylinders.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            cylinders_h.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            voxels.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            R.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            pos.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            n_drones_per_group,
            fov_x_half_tan.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>());
    }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ============================================================================
// C++ 接口函数：可微视场前向渲染（含法线输出）
// ============================================================================
void render_diff_fov_with_normal_cuda(
    torch::Tensor canvas,
    torch::Tensor normals,
    torch::Tensor balls,
    torch::Tensor cylinders,
    torch::Tensor cylinders_h,
    torch::Tensor voxels,
    torch::Tensor R,
    torch::Tensor pos,
    int n_drones_per_group,
    torch::Tensor fov_x_half_tan) {

    const int threads = 1024;
    size_t state_size = canvas.numel();
    const dim3 blocks((state_size + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(canvas.type(), "render_diff_fov_with_normal_cuda", ([&] {
        render_diff_fov_with_normal_cuda_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            canvas.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            normals.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            balls.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            cylinders.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            cylinders_h.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            voxels.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            R.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            pos.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            n_drones_per_group,
            fov_x_half_tan.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>());
    }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ============================================================================
// C++ 接口函数：可微视场反向传播 (C++ Interface: Differentiable FOV Backward)
// ============================================================================
void render_backward_fov_cuda(
    torch::Tensor grad_fov,
    torch::Tensor grad_output,
    torch::Tensor canvas,
    torch::Tensor balls,
    torch::Tensor cylinders,
    torch::Tensor cylinders_h,
    torch::Tensor voxels,
    torch::Tensor R,
    torch::Tensor pos,
    int n_drones_per_group,
    torch::Tensor fov_x_half_tan) {

    const int B = canvas.size(0);
    const int H = canvas.size(1);
    const int W = canvas.size(2);

    // 2D block + 3D grid to reduce atomic contention within each batch
    const dim3 threads(16, 16, 1);
    const dim3 blocks(
        (W + threads.x - 1) / threads.x,
        (H + threads.y - 1) / threads.y,
        B);

    AT_DISPATCH_FLOATING_TYPES(canvas.type(), "render_backward_fov_cuda", ([&] {
        render_backward_fov_cuda_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            grad_output.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            canvas.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            grad_fov.data_ptr<scalar_t>(),
            balls.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            cylinders.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            cylinders_h.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            voxels.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            R.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            pos.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            n_drones_per_group,
            fov_x_half_tan.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>());
    }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ============================================================================
// C++ 接口函数：可微视场反向传播（基于法线图解析梯度）
// ============================================================================
void render_backward_fov_from_normal_cuda(
    torch::Tensor grad_fov,
    torch::Tensor grad_output,
    torch::Tensor canvas,
    torch::Tensor normals,
    torch::Tensor R,
    torch::Tensor fov_x_half_tan) {

    const int B = canvas.size(0);
    const int H = canvas.size(1);
    const int W = canvas.size(2);

    const dim3 threads(16, 16, 1);
    const dim3 blocks(
        (W + threads.x - 1) / threads.x,
        (H + threads.y - 1) / threads.y,
        B);

    AT_DISPATCH_FLOATING_TYPES(canvas.type(), "render_backward_fov_from_normal_cuda", ([&] {
        render_backward_fov_from_normal_cuda_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            grad_output.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            canvas.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            normals.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            grad_fov.data_ptr<scalar_t>(),
            R.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            fov_x_half_tan.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>());
    }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ============================================================================
// C++ 接口函数：Y 通道渲染（非可微相机参数）
// 目前复用几何深度渲染内核，输出为单通道主传感器观测。
// ============================================================================
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
    float fov_x_half_tan) {

    // 当前实现以几何渲染作为 Y 通道基础，接口保持 YUV 主相机语义。
    render_cuda(
        canvas, flow, balls, cylinders, cylinders_h, voxels,
        R, R_old, pos, pos_old,
        drone_radius, n_drones_per_group, fov_x_half_tan);
}

// ============================================================================
// C++ 接口函数：Y 通道可微渲染（FOV + exposure/iso 输入）
// 说明：
// - 几何部分复用 render_diff_fov_with_normal_cuda
// - 相机部分采用 ATen 张量算子复刻 Python 链路的低风险近似
// - 反向中 exposure/iso/depth 使用 autograd 链式求导，fov 使用 normal-map 解析反向
// ============================================================================
torch::Tensor render_camera_luma_fused_forward_cuda(
    torch::Tensor depth_raw,
    torch::Tensor normals,
    torch::Tensor exposure,
    torch::Tensor iso,
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
    torch::Tensor cam_gamma,
    torch::Tensor cam_prnu,
    torch::Tensor cam_dsnu,
    torch::Tensor cam_ae_log_t,
    int64_t cam_profile_mask,
    double cam_vignette_a,
    double cam_vignette_b,
    double cam_black_level,
    double cam_sharpen_amount,
    double cam_base_gain,
    double cam_exposure_t_min,
    double cam_exposure_t_span,
    double cam_exposure_eff_min,
    double cam_exposure_eff_max,
    double cam_iso_gain_base,
    double cam_iso_gain_scale,
    double cam_iso_gain_gamma);

std::vector<torch::Tensor> render_camera_luma_fused_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor depth_raw,
    torch::Tensor normals,
    torch::Tensor exposure,
    torch::Tensor iso,
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
    torch::Tensor cam_gamma,
    torch::Tensor cam_prnu,
    torch::Tensor cam_dsnu,
    torch::Tensor cam_ae_log_t,
    int64_t cam_profile_mask,
    double cam_vignette_a,
    double cam_vignette_b,
    double cam_black_level,
    double cam_sharpen_amount,
    double cam_base_gain,
    double cam_exposure_t_min,
    double cam_exposure_t_span,
    double cam_exposure_eff_min,
    double cam_exposure_eff_max,
    double cam_iso_gain_base,
    double cam_iso_gain_scale,
    double cam_iso_gain_gamma,
    bool need_grad_exposure,
    bool need_grad_iso);

static inline bool use_fused_camera_fast_path(int64_t cam_profile_mask) {
    const int64_t mask = cam_profile_mask & 0x3f;
    // 融合路径优先覆盖 low/high（无 flare/motion/rolling）
    return (mask & ((int64_t(1) << 3) | (int64_t(1) << 4) | (int64_t(1) << 5))) == 0;
}

static torch::Tensor build_y_from_depth_full(
    torch::Tensor depth_raw,
    torch::Tensor normals,
    torch::Tensor exposure,
    torch::Tensor iso,
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
    bool cam_enable_shadow,
    bool cam_enable_specular,
    bool cam_enable_distortion,
    bool cam_enable_flare,
    bool cam_enable_motion_blur,
    bool cam_enable_rolling,
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
    double cam_iso_gain_gamma) {

    auto depth = depth_raw.clamp(0.03, 120.0);
    const auto B = depth.size(0);
    const auto H = depth.size(1);
    const auto W = depth.size(2);

    auto nx = normals.select(1, 0);
    auto ny = normals.select(1, 1);
    auto nz = normals.select(1, 2);
    auto nz_abs = torch::abs(nz);

    auto w_ground = ((nz_abs - 0.55) / 0.45).clamp(0.0, 1.0);
    auto albedo =
        w_ground * cam_mat_ground.unsqueeze(1).unsqueeze(2)
        + (1.0 - w_ground) * cam_mat_obstacle.unsqueeze(1).unsqueeze(2);

    auto Lx = cam_light_dir.select(1, 0).unsqueeze(1).unsqueeze(2);
    auto Ly = cam_light_dir.select(1, 1).unsqueeze(1).unsqueeze(2);
    auto Lz = cam_light_dir.select(1, 2).unsqueeze(1).unsqueeze(2);
    auto ndotl = (nx * Lx + ny * Ly + nz * Lz).clamp_min(0.0);

    torch::Tensor shadow = torch::ones_like(depth);
    if (cam_enable_shadow) {
        shadow = (0.35 + 0.65 * ndotl).clamp(0.2, 1.0);
    }

    torch::Tensor specular = torch::zeros_like(depth);
    if (cam_enable_specular) {
        specular = cam_mat_spec.unsqueeze(1).unsqueeze(2) * torch::pow(ndotl, 24.0);
    }

    auto irradiance =
        albedo * (cam_ambient.unsqueeze(1).unsqueeze(2) + cam_dir_intensity.unsqueeze(1).unsqueeze(2) * ndotl * shadow)
        + specular;

    auto trans = torch::exp(-cam_fog_beta.unsqueeze(1).unsqueeze(2) * depth);
    irradiance = (irradiance * trans + cam_airlight.unsqueeze(1).unsqueeze(2) * (1.0 - trans)).clamp(0.0, 4.0);

    auto yy = torch::linspace(-1.0, 1.0, H, depth.options()).view({H, 1});
    auto xx = torch::linspace(-1.0, 1.0, W, depth.options()).view({1, W});
    auto r2 = yy * yy + xx * xx;
    auto vignette = (1.0 - cam_vignette_a * r2 - cam_vignette_b * (r2 * r2)).clamp(0.25, 1.0);
    auto lens_y = irradiance * vignette.unsqueeze(0);

    if (cam_enable_distortion) {
        auto r2b = r2.unsqueeze(0);
        auto radial = 1.0 + cam_dist_k1.unsqueeze(1).unsqueeze(2) * r2b + cam_dist_k2.unsqueeze(1).unsqueeze(2) * (r2b * r2b);
        lens_y = (lens_y * radial.clamp(0.7, 1.3)).clamp(0.0, 4.0);
    }

    if (cam_enable_flare) {
        auto bright = torch::relu(lens_y - 0.82);
        auto flare = at::avg_pool2d(
            bright.unsqueeze(1),
            {9, 9},
            {1, 1},
            {4, 4},
            false,
            true,
            c10::nullopt).squeeze(1);
        lens_y = lens_y + cam_flare_strength.unsqueeze(1).unsqueeze(2) * flare;
    }

    auto exposure01 = exposure.clamp(0.0, 1.0);
    auto iso01 = iso.clamp(0.0, 1.0);
    auto t_cmd = cam_exposure_t_min + cam_exposure_t_span * exposure01;
    auto t_ae = torch::exp(cam_ae_log_t);
    auto t_eff = (t_cmd * t_ae).clamp(cam_exposure_eff_min, cam_exposure_eff_max);
    auto iso_gain = cam_iso_gain_base + cam_iso_gain_scale * torch::pow(iso01, cam_iso_gain_gamma);

    auto electrons = lens_y * t_eff.unsqueeze(1).unsqueeze(2) * cam_base_gain;
    electrons = electrons * iso_gain.unsqueeze(1).unsqueeze(2);
    auto raw = electrons * (1.0 + cam_prnu) + cam_dsnu;

    auto x = torch::relu(raw - cam_black_level);
    x = x / (1.0 + x);

    auto denoise_strength = 0.08 + 0.28 * iso01;
    auto smooth = at::avg_pool2d(
        x.unsqueeze(1),
        {3, 3},
        {1, 1},
        {1, 1},
        false,
        true,
        c10::nullopt).squeeze(1);
    x = x * (1.0 - denoise_strength.unsqueeze(1).unsqueeze(2)) + smooth * denoise_strength.unsqueeze(1).unsqueeze(2);

    auto blur_small = at::avg_pool2d(
        x.unsqueeze(1),
        {3, 3},
        {1, 1},
        {1, 1},
        false,
        true,
        c10::nullopt).squeeze(1);
    x = x + cam_sharpen_amount * (x - blur_small);

    auto gamma = cam_gamma.unsqueeze(1).unsqueeze(2).clamp_min(1e-3);
    auto x_lin = x.clamp(0.0, 1.0);
    auto x_safe = x_lin.clamp(1e-6, 1.0);
    auto x_gamma = torch::pow(x_safe, 1.0 / gamma);
    auto y = torch::where(x_lin > 0, x_gamma, torch::zeros_like(x_gamma));

    if (cam_enable_motion_blur) {
        auto speed = v.norm(2, -1);
        auto t_norm = (t_cmd / 3.0).clamp(0.0, 1.0);
        auto blur_alpha = (speed * cam_motion_blur_gain * t_norm).clamp(0.0, 0.72);

        auto yg = y * (1.0 - blur_alpha.unsqueeze(1).unsqueeze(2)) + cam_prev_y * blur_alpha.unsqueeze(1).unsqueeze(2);

        if (cam_enable_rolling) {
            auto row = torch::linspace(0.0, 1.0, H, depth.options()).view({1, H, 1}).expand({B, H, W});
            auto a_roll = blur_alpha.unsqueeze(1).unsqueeze(2) * row;
            auto yr = y * (1.0 - a_roll) + cam_prev_y * a_roll;
            auto use_roll = cam_use_rolling.unsqueeze(1).unsqueeze(2);
            y = yg * (1.0 - use_roll) + yr * use_roll;
        } else {
            y = yg;
        }
    }

    return y.clamp(0.0, 1.0);
}

static torch::Tensor build_y_from_depth_profiled(
    torch::Tensor depth_raw,
    torch::Tensor normals,
    torch::Tensor exposure,
    torch::Tensor iso,
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
    double cam_iso_gain_gamma) {
    const int64_t mask = cam_profile_mask & 0x3f;
    if (mask == 63) {
        return build_y_from_depth_full(
            depth_raw, normals, exposure, iso,
            cam_light_dir, cam_ambient, cam_dir_intensity,
            cam_fog_beta, cam_airlight,
            cam_mat_ground, cam_mat_obstacle, cam_mat_spec,
            cam_dist_k1, cam_dist_k2, cam_flare_strength,
            cam_gamma, cam_prnu, cam_dsnu,
            cam_prev_y, cam_use_rolling, v, cam_ae_log_t,
            true, true, true, true, true, true,
            cam_vignette_a, cam_vignette_b,
            cam_black_level, cam_sharpen_amount, cam_base_gain, cam_motion_blur_gain,
            cam_exposure_t_min, cam_exposure_t_span,
            cam_exposure_eff_min, cam_exposure_eff_max,
            cam_iso_gain_base, cam_iso_gain_scale, cam_iso_gain_gamma);
    }
    if (mask == 7) {
        return build_y_from_depth_full(
            depth_raw, normals, exposure, iso,
            cam_light_dir, cam_ambient, cam_dir_intensity,
            cam_fog_beta, cam_airlight,
            cam_mat_ground, cam_mat_obstacle, cam_mat_spec,
            cam_dist_k1, cam_dist_k2, cam_flare_strength,
            cam_gamma, cam_prnu, cam_dsnu,
            cam_prev_y, cam_use_rolling, v, cam_ae_log_t,
            true, true, true, false, false, false,
            cam_vignette_a, cam_vignette_b,
            cam_black_level, cam_sharpen_amount, cam_base_gain, cam_motion_blur_gain,
            cam_exposure_t_min, cam_exposure_t_span,
            cam_exposure_eff_min, cam_exposure_eff_max,
            cam_iso_gain_base, cam_iso_gain_scale, cam_iso_gain_gamma);
    }
    if (mask == 2) {
        return build_y_from_depth_full(
            depth_raw, normals, exposure, iso,
            cam_light_dir, cam_ambient, cam_dir_intensity,
            cam_fog_beta, cam_airlight,
            cam_mat_ground, cam_mat_obstacle, cam_mat_spec,
            cam_dist_k1, cam_dist_k2, cam_flare_strength,
            cam_gamma, cam_prnu, cam_dsnu,
            cam_prev_y, cam_use_rolling, v, cam_ae_log_t,
            false, true, false, false, false, false,
            cam_vignette_a, cam_vignette_b,
            cam_black_level, cam_sharpen_amount, cam_base_gain, cam_motion_blur_gain,
            cam_exposure_t_min, cam_exposure_t_span,
            cam_exposure_eff_min, cam_exposure_eff_max,
            cam_iso_gain_base, cam_iso_gain_scale, cam_iso_gain_gamma);
    }

    const bool cam_enable_shadow = (mask & (1 << 0)) != 0;
    const bool cam_enable_specular = (mask & (1 << 1)) != 0;
    const bool cam_enable_distortion = (mask & (1 << 2)) != 0;
    const bool cam_enable_flare = (mask & (1 << 3)) != 0;
    const bool cam_enable_motion_blur = (mask & (1 << 4)) != 0;
    const bool cam_enable_rolling = (mask & (1 << 5)) != 0;

    return build_y_from_depth_full(
        depth_raw, normals, exposure, iso,
        cam_light_dir, cam_ambient, cam_dir_intensity,
        cam_fog_beta, cam_airlight,
        cam_mat_ground, cam_mat_obstacle, cam_mat_spec,
        cam_dist_k1, cam_dist_k2, cam_flare_strength,
        cam_gamma, cam_prnu, cam_dsnu,
        cam_prev_y, cam_use_rolling, v, cam_ae_log_t,
        cam_enable_shadow, cam_enable_specular, cam_enable_distortion,
        cam_enable_flare, cam_enable_motion_blur, cam_enable_rolling,
        cam_vignette_a, cam_vignette_b,
        cam_black_level, cam_sharpen_amount, cam_base_gain, cam_motion_blur_gain,
        cam_exposure_t_min, cam_exposure_t_span,
        cam_exposure_eff_min, cam_exposure_eff_max,
        cam_iso_gain_base, cam_iso_gain_scale, cam_iso_gain_gamma);
}

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
    int width) {
    const auto B = pos.size(0);
    auto opts = pos.options();
    auto cam_light_dir = torch::zeros({B, 3}, opts);
    cam_light_dir.select(1, 2).fill_(1.0);
    auto cam_ambient = torch::full({B}, 0.2, opts);
    auto cam_dir_intensity = torch::full({B}, 1.0, opts);
    auto cam_fog_beta = torch::full({B}, 0.02, opts);
    auto cam_airlight = torch::full({B}, 0.4, opts);
    auto cam_mat_ground = torch::full({B}, 0.4, opts);
    auto cam_mat_obstacle = torch::full({B}, 0.6, opts);
    auto cam_mat_spec = torch::full({B}, 0.08, opts);
    auto cam_dist_k1 = torch::zeros({B}, opts);
    auto cam_dist_k2 = torch::zeros({B}, opts);
    auto cam_flare_strength = torch::zeros({B}, opts);
    auto cam_gamma = torch::full({B}, 2.2, opts);
    auto cam_prnu = torch::zeros({B, height, width}, opts);
    auto cam_dsnu = torch::zeros({B, height, width}, opts);
    auto cam_prev_y = torch::zeros({B, height, width}, opts);
    auto cam_use_rolling = torch::zeros({B}, opts);
    auto v = torch::zeros({B, 3}, opts);
    auto cam_ae_log_t = torch::zeros({B}, opts);

    auto out = render_diff_yuv_y_forward_cuda(
        fov_x_half_tan, exposure, iso,
        R, pos, balls, cylinders, cylinders_h, voxels,
        n_drones_per_group, height, width,
        cam_light_dir, cam_ambient, cam_dir_intensity,
        cam_fog_beta, cam_airlight,
        cam_mat_ground, cam_mat_obstacle, cam_mat_spec,
        cam_dist_k1, cam_dist_k2, cam_flare_strength,
        cam_gamma, cam_prnu, cam_dsnu,
        cam_prev_y, cam_use_rolling, v, cam_ae_log_t,
        0,
        0.28, 0.22,
        0.01, 0.35, 0.14, 0.09,
        0.25, 2.75,
        0.15, 4.0,
        1.0, 10.0, 1.2);
    return out[0];
}

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
    double cam_iso_gain_gamma) {

    const auto B = pos.size(0);
    auto depth_raw = torch::empty({B, height, width}, pos.options());
    auto normals = torch::empty({B, 3, height, width}, pos.options());
    render_diff_fov_with_normal_cuda(
        depth_raw, normals, balls, cylinders, cylinders_h, voxels,
        R, pos, n_drones_per_group, fov_x_half_tan);

    torch::Tensor y;
    if (use_fused_camera_fast_path(cam_profile_mask)) {
        y = render_camera_luma_fused_forward_cuda(
            depth_raw, normals, exposure, iso,
            cam_light_dir, cam_ambient, cam_dir_intensity,
            cam_fog_beta, cam_airlight,
            cam_mat_ground, cam_mat_obstacle, cam_mat_spec,
            cam_dist_k1, cam_dist_k2,
            cam_gamma, cam_prnu, cam_dsnu, cam_ae_log_t,
            cam_profile_mask,
            cam_vignette_a, cam_vignette_b,
            cam_black_level, cam_sharpen_amount, cam_base_gain,
            cam_exposure_t_min, cam_exposure_t_span,
            cam_exposure_eff_min, cam_exposure_eff_max,
            cam_iso_gain_base, cam_iso_gain_scale, cam_iso_gain_gamma);
    } else {
        y = build_y_from_depth_profiled(
            depth_raw, normals, exposure, iso,
            cam_light_dir, cam_ambient, cam_dir_intensity,
            cam_fog_beta, cam_airlight,
            cam_mat_ground, cam_mat_obstacle, cam_mat_spec,
            cam_dist_k1, cam_dist_k2, cam_flare_strength,
            cam_gamma, cam_prnu, cam_dsnu,
            cam_prev_y, cam_use_rolling, v, cam_ae_log_t,
            cam_profile_mask,
            cam_vignette_a, cam_vignette_b,
            cam_black_level, cam_sharpen_amount, cam_base_gain, cam_motion_blur_gain,
            cam_exposure_t_min, cam_exposure_t_span,
            cam_exposure_eff_min, cam_exposure_eff_max,
            cam_iso_gain_base, cam_iso_gain_scale, cam_iso_gain_gamma);
    }
    return {y, depth_raw, normals};
}

std::vector<torch::Tensor> render_diff_yuv_y_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor depth_raw,
    torch::Tensor fov_x_half_tan,
    torch::Tensor exposure,
    torch::Tensor iso,
    torch::Tensor normals,
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
    double cam_iso_gain_gamma,
    bool need_grad_fov,
    bool need_grad_exposure,
    bool need_grad_iso) {

    auto go = grad_output.contiguous();
    auto d = depth_raw.contiguous();

    torch::Tensor grad_depth = torch::zeros_like(d);
    torch::Tensor grad_exposure = torch::zeros_like(exposure);
    torch::Tensor grad_iso = torch::zeros_like(iso);

    const bool have_geom_cache = depth_raw.numel() > 0 && normals.numel() > 0;
    if (use_fused_camera_fast_path(cam_profile_mask)) {
        torch::Tensor d_used = d;
        torch::Tensor n_used = normals;
        if (!have_geom_cache) {
            d_used = torch::empty({pos.size(0), height, width}, pos.options());
            n_used = torch::empty({pos.size(0), 3, height, width}, pos.options());
            render_diff_fov_with_normal_cuda(
                d_used, n_used,
                balls, cylinders, cylinders_h, voxels,
                R, pos,
                n_drones_per_group,
                fov_x_half_tan);
        }
        auto fused_grads = render_camera_luma_fused_backward_cuda(
            go,
            d_used,
            n_used,
            exposure,
            iso,
            cam_light_dir, cam_ambient, cam_dir_intensity,
            cam_fog_beta, cam_airlight,
            cam_mat_ground, cam_mat_obstacle, cam_mat_spec,
            cam_dist_k1, cam_dist_k2,
            cam_gamma, cam_prnu, cam_dsnu,
            cam_ae_log_t,
            cam_profile_mask,
            cam_vignette_a, cam_vignette_b,
            cam_black_level, cam_sharpen_amount, cam_base_gain,
            cam_exposure_t_min, cam_exposure_t_span,
            cam_exposure_eff_min, cam_exposure_eff_max,
            cam_iso_gain_base, cam_iso_gain_scale, cam_iso_gain_gamma,
            need_grad_exposure, need_grad_iso);
        grad_depth = fused_grads[0];
        if (need_grad_exposure) grad_exposure = fused_grads[1];
        if (need_grad_iso) grad_iso = fused_grads[2];
    } else {

    const bool need_depth = need_grad_fov;
    if (need_depth || need_grad_exposure || need_grad_iso) {
        torch::autograd::AutoGradMode enable_grad(true);

        torch::Tensor d_var = need_depth ? d.detach() : d;
        torch::Tensor e_var = need_grad_exposure ? exposure.detach() : exposure;
        torch::Tensor i_var = need_grad_iso ? iso.detach() : iso;

        if (need_depth) d_var.set_requires_grad(true);
        if (need_grad_exposure) e_var.set_requires_grad(true);
        if (need_grad_iso) i_var.set_requires_grad(true);

        auto y_var = build_y_from_depth_profiled(
            d_var, normals, e_var, i_var,
            cam_light_dir, cam_ambient, cam_dir_intensity,
            cam_fog_beta, cam_airlight,
            cam_mat_ground, cam_mat_obstacle, cam_mat_spec,
            cam_dist_k1, cam_dist_k2, cam_flare_strength,
            cam_gamma, cam_prnu, cam_dsnu,
            cam_prev_y, cam_use_rolling, v, cam_ae_log_t,
            cam_profile_mask,
            cam_vignette_a, cam_vignette_b,
            cam_black_level, cam_sharpen_amount, cam_base_gain, cam_motion_blur_gain,
            cam_exposure_t_min, cam_exposure_t_span,
            cam_exposure_eff_min, cam_exposure_eff_max,
            cam_iso_gain_base, cam_iso_gain_scale, cam_iso_gain_gamma);

        std::vector<torch::Tensor> grad_inputs;
        if (need_depth) grad_inputs.push_back(d_var);
        if (need_grad_exposure) grad_inputs.push_back(e_var);
        if (need_grad_iso) grad_inputs.push_back(i_var);

        std::vector<torch::Tensor> grads = torch::autograd::grad(
            {y_var},
            grad_inputs,
            {go},
            false,
            false,
            true);

        int gi = 0;
        if (need_depth) {
            grad_depth = grads[gi].defined() ? grads[gi] : torch::zeros_like(d);
            gi++;
        }
        if (need_grad_exposure) {
            grad_exposure = grads[gi].defined() ? grads[gi] : torch::zeros_like(exposure);
            gi++;
        }
        if (need_grad_iso) {
            grad_iso = grads[gi].defined() ? grads[gi] : torch::zeros_like(iso);
        }
    }
    }

    auto grad_fov = torch::zeros_like(fov_x_half_tan);
    if (need_grad_fov) {
        render_backward_fov_from_normal_cuda(
            grad_fov,
            grad_depth.contiguous(),
            d,
            normals,
            R,
            fov_x_half_tan);
    }

    return {grad_fov, grad_exposure, grad_iso};
}

// ============================================================================
// C++ 接口函数：Active ToF 可微前向（CUDA高性能路径）
// 说明：
// - 几何深度：复用 render_diff_fov_cuda（CUDA）
// - 传感器与噪声：使用 ATen 张量算子（GPU 上执行）
// - 返回：noisy_depth, confidence
// ============================================================================
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
    double max_range) {

    const auto B = pos.size(0);
    auto depth = torch::empty({B, height, width}, pos.options());
    render_diff_fov_cuda(
        depth, balls, cylinders, cylinders_h, voxels,
        R, pos, n_drones_per_group, fov_x_half_tan);

    depth = torch::clamp(depth, 0.03, 120.0);

    auto power_scaled = (0.01 + power * 0.99).unsqueeze(1).unsqueeze(2);
    auto exp_scaled = (0.05 + exposure * 0.95).unsqueeze(1).unsqueeze(2);
    auto gain_scaled = (1.0 + gain * 9.0).unsqueeze(1).unsqueeze(2);

    auto energy_recv = (power_scaled * exp_scaled) / (depth * depth + 0.1);
    energy_recv = energy_recv * gain_scaled * 100.0;

    auto conf_raw = torch::tanh(energy_recv * 0.5);

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

    auto conf = conf_raw * (1.0 - mbf * 0.8);

    auto noise_std = (0.05 * gain_scaled) / (energy_recv + 1e-3);
    noise_std = noise_std.clamp(0.01, 1.0);

    auto noisy_depth = depth_blurred + torch::randn_like(depth_blurred) * noise_std;
    noisy_depth = noisy_depth.clamp(0.05, max_range);

    return {noisy_depth, conf};
}

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
    double max_range) {

    const auto B = pos.size(0);
    auto opts = pos.options();

    auto go_depth = grad_noisy_depth.contiguous();
    auto go_conf = grad_conf.contiguous();

    // 重新计算几何深度与中间量（与 forward 路径一致）
    auto depth = torch::empty({B, height, width}, opts);
    render_diff_fov_cuda(
        depth, balls, cylinders, cylinders_h, voxels,
        R, pos, n_drones_per_group, fov_x_half_tan);
    depth = torch::clamp(depth, 0.03, 120.0);

    auto ps = (0.01 + power * 0.99).unsqueeze(1).unsqueeze(2);   // power_scaled
    auto es = (0.05 + exposure * 0.95).unsqueeze(1).unsqueeze(2); // exp_scaled
    auto gs = (1.0 + gain * 9.0).unsqueeze(1).unsqueeze(2);       // gain_scaled

    auto d2 = depth * depth;
    auto energy_recv = (ps * es) / (d2 + 0.1);
    energy_recv = energy_recv * gs * 100.0;

    auto conf_raw = torch::tanh(energy_recv * 0.5);

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

    // conf = conf_raw * (1 - 0.8 m)
    auto g_conf_raw = go_conf * (1.0 - 0.8 * m);
    auto g_m_from_conf = go_conf * (-0.8 * conf_raw);

    // depth_blurred = depth*(1-m) + blur*m
    auto g_m_from_blur = g_depth_blurred * (blur_kernel - depth);
    auto g_m_total = g_m_from_conf + g_m_from_blur;

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

    // conf_raw = tanh(0.5E)
    auto g_E_from_conf = g_conf_raw * (0.5 * (1.0 - conf_raw * conf_raw));
    auto g_E = g_E_from_conf + g_E_from_ns;

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

    // 几何链路回传到 fov（供未来扩展；当前 active_tof 调用里 fov 通常不需梯度）
    auto grad_fov = torch::zeros_like(fov_x_half_tan);
    render_backward_fov_cuda(
        grad_fov,
        g_depth.contiguous(),
        depth.contiguous(),
        balls,
        cylinders,
        cylinders_h,
        voxels,
        R,
        pos,
        n_drones_per_group,
        fov_x_half_tan);

    return {grad_fov, grad_power, grad_exposure, grad_gain};
}
