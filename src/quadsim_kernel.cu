#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

template <typename scalar_t>
__global__ void render_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> canvas,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> flow,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> balls,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> cylinders,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> cylinders_h,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> voxels,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> R,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> R_old,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> pos,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> pos_old,
    float drone_radius,
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
    const scalar_t fov_y_half_tan = fov_x_half_tan / W * H;
    const scalar_t fu = (2 * (u + 0.5) / H - 1) * fov_y_half_tan - 1e-5;
    const scalar_t fv = (2 * (v + 0.5) / W - 1) * fov_x_half_tan - 1e-5;
    scalar_t dx = R[b][0][0] - fu * R[b][0][2] - fv * R[b][0][1];
    scalar_t dy = R[b][1][0] - fu * R[b][1][2] - fv * R[b][1][1];
    scalar_t dz = R[b][2][0] - fu * R[b][2][2] - fv * R[b][2][1];
    const scalar_t ox = pos[b][0];
    const scalar_t oy = pos[b][1];
    const scalar_t oz = pos[b][2];

    scalar_t min_dist = 100;
    scalar_t  t = (-1 - oz) / dz;
    if (t > 0) min_dist = t;

    // others
    const int batch_base = (b / n_drones_per_group) * n_drones_per_group;
    for (int i = batch_base; i < batch_base + n_drones_per_group; i++) {
        if (i == b || i >= B) continue;
        scalar_t cx = pos[i][0];
        scalar_t cy = pos[i][1];
        scalar_t cz = pos[i][2];
        scalar_t r = 0.15;
        // (ox + t dx)^2 + (oy + t dy)^2 + 4 (oz + t dz)^2 = r^2
        scalar_t a = dx * dx + dy * dy + 4 * dz * dz;
        scalar_t b = 2 * (dx * (ox - cx) + dy * (oy - cy) + 4 * dz * (oz - cz));
        scalar_t c = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy) + 4 * (oz - cz) * (oz - cz) - r * r;
        scalar_t d = b * b - 4 * a * c;
        if (d >= 0) {
            r = (-b-sqrt(d)) / (2 * a);
            if (r > 1e-5) {
                min_dist = min(min_dist, r);
            } else {
                r = (-b+sqrt(d)) / (2 * a);
                if (r > 1e-5) min_dist = min(min_dist, r);
            }
        }
    }

    // balls
    for (int i = 0; i < balls.size(1); i++) {
        scalar_t cx = balls[batch_base][i][0];
        scalar_t cy = balls[batch_base][i][1];
        scalar_t cz = balls[batch_base][i][2];
        scalar_t r = balls[batch_base][i][3];
        scalar_t a = dx * dx + dy * dy + dz * dz;
        scalar_t b = 2 * (dx * (ox - cx) + dy * (oy - cy) + dz * (oz - cz));
        scalar_t c = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy) + (oz - cz) * (oz - cz) - r * r;
        scalar_t d = b * b - 4 * a * c;
        if (d >= 0) {
            r = (-b-sqrt(d)) / (2 * a);
            if (r > 1e-5) {
                min_dist = min(min_dist, r);
            } else {
                r = (-b+sqrt(d)) / (2 * a);
                if (r > 1e-5) min_dist = min(min_dist, r);
            }
        }
    }

    // cylinders
    for (int i = 0; i < cylinders.size(1); i++) {
        scalar_t cx = cylinders[batch_base][i][0];
        scalar_t cy = cylinders[batch_base][i][1];
        scalar_t r = cylinders[batch_base][i][2];
        scalar_t a = dx * dx + dy * dy;
        scalar_t b = 2 * (dx * (ox - cx) + dy * (oy - cy));
        scalar_t c = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy) - r * r;
        scalar_t d = b * b - 4 * a * c;
        if (d >= 0) {
            r = (-b-sqrt(d)) / (2 * a);
            if (r > 1e-5) {
                min_dist = min(min_dist, r);
            } else {
                r = (-b+sqrt(d)) / (2 * a);
                if (r > 1e-5) min_dist = min(min_dist, r);
            }
        }
    }
    for (int i = 0; i < cylinders_h.size(1); i++) {
        scalar_t cx = cylinders_h[batch_base][i][0];
        scalar_t cz = cylinders_h[batch_base][i][1];
        scalar_t r = cylinders_h[batch_base][i][2];
        scalar_t a = dx * dx + dz * dz;
        scalar_t b = 2 * (dx * (ox - cx) + dz * (oz - cz));
        scalar_t c = (ox - cx) * (ox - cx) + (oz - cz) * (oz - cz) - r * r;
        scalar_t d = b * b - 4 * a * c;
        if (d >= 0) {
            r = (-b-sqrt(d)) / (2 * a);
            if (r > 1e-5) {
                min_dist = min(min_dist, r);
            } else {
                r = (-b+sqrt(d)) / (2 * a);
                if (r > 1e-5) min_dist = min(min_dist, r);
            }
        }
    }

    // balls
    for (int i = 0; i < voxels.size(1); i++) {
        scalar_t cx = voxels[batch_base][i][0];
        scalar_t cy = voxels[batch_base][i][1];
        scalar_t cz = voxels[batch_base][i][2];
        scalar_t rx = voxels[batch_base][i][3];
        scalar_t ry = voxels[batch_base][i][4];
        scalar_t rz = voxels[batch_base][i][5];
        scalar_t tx1 = (cx - rx - ox) / dx;
        scalar_t tx2 = (cx + rx - ox) / dx;
        scalar_t tx_min = min(tx1, tx2);
        scalar_t tx_max = max(tx1, tx2);
        scalar_t ty1 = (cy - ry - oy) / dy;
        scalar_t ty2 = (cy + ry - oy) / dy;
        scalar_t ty_min = min(ty1, ty2);
        scalar_t ty_max = max(ty1, ty2);
        scalar_t tz1 = (cz - rz - oz) / dz;
        scalar_t tz2 = (cz + rz - oz) / dz;
        scalar_t tz_min = min(tz1, tz2);
        scalar_t tz_max = max(tz1, tz2);
        scalar_t t_min = max(max(tx_min, ty_min), tz_min);
        scalar_t t_max = min(min(tx_max, ty_max), tz_max);
        if (t_min < min_dist && t_min < t_max && t_min > 0)
            min_dist = t_min;
    }

    canvas[b][u][v] = min_dist;
}

template <typename scalar_t>
__global__ void nearest_pt_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> nearest_pt,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> balls,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> cylinders,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> cylinders_h,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> voxels,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> pos,
    float drone_radius,
    int n_drones_per_group) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int B = nearest_pt.size(1);
    const int j = idx / B;
    if (j >= nearest_pt.size(0)) return;
    const int b = idx % B;
    // assert(j < pos.size(0));
    // assert(b < pos.size(1));

    const scalar_t ox = pos[j][b][0];
    const scalar_t oy = pos[j][b][1];
    const scalar_t oz = pos[j][b][2];

    scalar_t min_dist = max(1e-3f, oz + 1);
    scalar_t nearest_ptx = ox;
    scalar_t nearest_pty = oy;
    scalar_t nearest_ptz = min(-1., oz - 1e-3f);

    // others
    const int batch_base = (b / n_drones_per_group) * n_drones_per_group;
    for (int i = batch_base; i < batch_base + n_drones_per_group; i++) {
        if (i == b || i >= B) continue;
        scalar_t cx = pos[j][i][0];
        scalar_t cy = pos[j][i][1];
        scalar_t cz = pos[j][i][2];
        scalar_t r = 0.15;
        scalar_t dist = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy) + 4 * (oz - cz) * (oz - cz);
        dist = max(1e-3f, sqrt(dist) - r);
        if (dist < min_dist) {
            min_dist = dist;
            nearest_ptx = ox + dist * (cx - ox);
            nearest_pty = oy + dist * (cy - oy);
            nearest_ptz = oz + dist * (cz - oz);
        }
    }

    // balls
    for (int i = 0; i < balls.size(1); i++) {
        scalar_t cx = balls[batch_base][i][0];
        scalar_t cy = balls[batch_base][i][1];
        scalar_t cz = balls[batch_base][i][2];
        scalar_t r = balls[batch_base][i][3];
        scalar_t dist = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy) + (oz - cz) * (oz - cz);
        dist = max(1e-3f, sqrt(dist) - r);
        if (dist < min_dist) {
            min_dist = dist;
            nearest_ptx = ox + dist * (cx - ox);
            nearest_pty = oy + dist * (cy - oy);
            nearest_ptz = oz + dist * (cz - oz);
        }
    }

    // cylinders
    for (int i = 0; i < cylinders.size(1); i++) {
        scalar_t cx = cylinders[batch_base][i][0];
        scalar_t cy = cylinders[batch_base][i][1];
        scalar_t r = cylinders[batch_base][i][2];
        scalar_t dist = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy);
        dist = max(1e-3f, sqrt(dist) - r);
        if (dist < min_dist) {
            min_dist = dist;
            nearest_ptx = ox + dist * (cx - ox);
            nearest_pty = oy + dist * (cy - oy);
            nearest_ptz = oz;
        }
    }
    for (int i = 0; i < cylinders_h.size(1); i++) {
        scalar_t cx = cylinders_h[batch_base][i][0];
        scalar_t cz = cylinders_h[batch_base][i][1];
        scalar_t r = cylinders_h[batch_base][i][2];
        scalar_t dist = (ox - cx) * (ox - cx) + (oz - cz) * (oz - cz);
        dist = max(1e-3f, sqrt(dist) - r);
        if (dist < min_dist) {
            min_dist = dist;
            nearest_ptx = ox + dist * (cx - ox);
            nearest_pty = oy;
            nearest_ptz = oz + dist * (cz - oz);
        }
    }

    // voxels
    for (int i = 0; i < voxels.size(1); i++) {
        scalar_t cx = voxels[batch_base][i][0];
        scalar_t cy = voxels[batch_base][i][1];
        scalar_t cz = voxels[batch_base][i][2];
        scalar_t max_r = max(abs(ox - cx), max(abs(oy - cy), abs(oz - cz))) - 1e-3;
        scalar_t rx = min(max_r, voxels[batch_base][i][3]);
        scalar_t ry = min(max_r, voxels[batch_base][i][4]);
        scalar_t rz = min(max_r, voxels[batch_base][i][5]);
        scalar_t ptx = cx + max(-rx, min(rx, ox - cx));
        scalar_t pty = cy + max(-ry, min(ry, oy - cy));
        scalar_t ptz = cz + max(-rz, min(rz, oz - cz));
        scalar_t dist = (ptx - ox) * (ptx - ox) + (pty - oy) * (pty - oy) + (ptz - oz) * (ptz - oz);
        dist = sqrt(dist);
        if (dist < min_dist) {
            min_dist = dist;
            nearest_ptx = ptx;
            nearest_pty = pty;
            nearest_ptz = ptz;
        }
    }
    nearest_pt[j][b][0] = nearest_ptx;
    nearest_pt[j][b][1] = nearest_pty;
    nearest_pt[j][b][2] = nearest_ptz;
}


// ==================== Ellipsoid Drone Collision ====================
// Treats the drone as an oriented ellipsoid with semi-axes (a, a, c) in body frame.
// R_body[B,3,3] gives the body-to-world rotation (columns = body axes in world).
// For each obstacle surface point, we compute the effective ellipsoid radius along
// the contact direction and subtract it from the point-to-obstacle distance.

template <typename scalar_t>
__device__ __forceinline__ scalar_t ellipsoid_radius_along_dir(
    scalar_t dx, scalar_t dy, scalar_t dz,
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t>& R_body,
    int b, scalar_t ea, scalar_t ec) {
    // Transform world-frame direction (dx, dy, dz) into body frame via R^T
    // R_body[b] columns are [fwd, left, up] in world coords
    scalar_t bx = R_body[b][0][0]*dx + R_body[b][1][0]*dy + R_body[b][2][0]*dz;
    scalar_t by = R_body[b][0][1]*dx + R_body[b][1][1]*dy + R_body[b][2][1]*dz;
    scalar_t bz = R_body[b][0][2]*dx + R_body[b][1][2]*dy + R_body[b][2][2]*dz;
    // Ellipsoid support distance: 1/sqrt((bx/a)^2+(by/a)^2+(bz/c)^2)
    scalar_t inv_a2 = 1.0f / (ea * ea);
    scalar_t inv_c2 = 1.0f / (ec * ec);
    scalar_t s = bx*bx*inv_a2 + by*by*inv_a2 + bz*bz*inv_c2;
    return 1.0f / sqrt(max(s, 1e-8f));
}

template <typename scalar_t>
__global__ void nearest_pt_ellipsoid_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> nearest_pt,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> balls,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> cylinders,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> cylinders_h,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> voxels,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> pos,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> R_body,
    float drone_radius,
    int n_drones_per_group,
    float ellipsoid_a,
    float ellipsoid_c) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int B = nearest_pt.size(1);
    const int j = idx / B;
    if (j >= nearest_pt.size(0)) return;
    const int b = idx % B;

    const scalar_t ea = (scalar_t)ellipsoid_a;
    const scalar_t ec = (scalar_t)ellipsoid_c;

    const scalar_t ox = pos[j][b][0];
    const scalar_t oy = pos[j][b][1];
    const scalar_t oz = pos[j][b][2];

    // Ground plane z = -1: direction is (0, 0, -1)
    scalar_t ground_reff = ellipsoid_radius_along_dir((scalar_t)0, (scalar_t)0, (scalar_t)-1, R_body, b, ea, ec);
    scalar_t min_dist = max(1e-3f, oz + 1 - ground_reff);
    scalar_t nearest_ptx = ox;
    scalar_t nearest_pty = oy;
    scalar_t nearest_ptz = oz - min_dist;

    // other drones
    const int batch_base = (b / n_drones_per_group) * n_drones_per_group;
    for (int i = batch_base; i < batch_base + n_drones_per_group; i++) {
        if (i == b || i >= B) continue;
        scalar_t cx = pos[j][i][0];
        scalar_t cy = pos[j][i][1];
        scalar_t cz = pos[j][i][2];
        scalar_t r = 0.15;
        scalar_t raw_dist2 = (ox-cx)*(ox-cx) + (oy-cy)*(oy-cy) + 4*(oz-cz)*(oz-cz);
        scalar_t raw_dist = sqrt(raw_dist2);
        scalar_t point_dist = max(1e-3f, raw_dist - r);
        // Direction from drone to obstacle (toward center)
        scalar_t ddx = (cx - ox), ddy = (cy - oy), ddz = (cz - oz);
        scalar_t dd_norm = sqrt(ddx*ddx + ddy*ddy + ddz*ddz);
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

    // balls
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

    // vertical cylinders
    for (int i = 0; i < cylinders.size(1); i++) {
        scalar_t cx = cylinders[batch_base][i][0];
        scalar_t cy = cylinders[batch_base][i][1];
        scalar_t r = cylinders[batch_base][i][2];
        scalar_t ddx = cx - ox, ddy = cy - oy;
        scalar_t dd_norm = sqrt(ddx*ddx + ddy*ddy);
        scalar_t point_dist = max(1e-3f, dd_norm - r);
        if (dd_norm > 1e-6f) { ddx /= dd_norm; ddy /= dd_norm; }
        else { ddx = 0; ddy = 0; }
        // Direction in world: (ddx, ddy, 0) — horizontal toward cylinder axis
        scalar_t reff = ellipsoid_radius_along_dir(ddx, ddy, (scalar_t)0, R_body, b, ea, ec);
        scalar_t dist = max(1e-3f, point_dist - reff);
        if (dist < min_dist) {
            min_dist = dist;
            nearest_ptx = ox + dist * ddx;
            nearest_pty = oy + dist * ddy;
            nearest_ptz = oz;
        }
    }

    // horizontal cylinders (along Y)
    for (int i = 0; i < cylinders_h.size(1); i++) {
        scalar_t cx = cylinders_h[batch_base][i][0];
        scalar_t cz = cylinders_h[batch_base][i][1];
        scalar_t r = cylinders_h[batch_base][i][2];
        scalar_t ddx = cx - ox, ddz = cz - oz;
        scalar_t dd_norm = sqrt(ddx*ddx + ddz*ddz);
        scalar_t point_dist = max(1e-3f, dd_norm - r);
        if (dd_norm > 1e-6f) { ddx /= dd_norm; ddz /= dd_norm; }
        else { ddx = 0; ddz = 0; }
        scalar_t reff = ellipsoid_radius_along_dir(ddx, (scalar_t)0, ddz, R_body, b, ea, ec);
        scalar_t dist = max(1e-3f, point_dist - reff);
        if (dist < min_dist) {
            min_dist = dist;
            nearest_ptx = ox + dist * ddx;
            nearest_pty = oy;
            nearest_ptz = oz + dist * ddz;
        }
    }

    // voxels (AABB)
    for (int i = 0; i < voxels.size(1); i++) {
        scalar_t cx = voxels[batch_base][i][0];
        scalar_t cy = voxels[batch_base][i][1];
        scalar_t cz = voxels[batch_base][i][2];
        scalar_t max_r = max(abs(ox - cx), max(abs(oy - cy), abs(oz - cz))) - 1e-3;
        scalar_t rx = min(max_r, voxels[batch_base][i][3]);
        scalar_t ry = min(max_r, voxels[batch_base][i][4]);
        scalar_t rz = min(max_r, voxels[batch_base][i][5]);
        scalar_t ptx = cx + max(-rx, min(rx, ox - cx));
        scalar_t pty = cy + max(-ry, min(ry, oy - cy));
        scalar_t ptz = cz + max(-rz, min(rz, oz - cz));
        scalar_t ddx = ptx - ox, ddy = pty - oy, ddz = ptz - oz;
        scalar_t point_dist = sqrt(ddx*ddx + ddy*ddy + ddz*ddz);
        if (point_dist > 1e-6f) { ddx /= point_dist; ddy /= point_dist; ddz /= point_dist; }
        scalar_t reff = (point_dist > 1e-6f) ?
            ellipsoid_radius_along_dir(ddx, ddy, ddz, R_body, b, ea, ec) : ea;
        scalar_t dist = max(0.0f, point_dist - reff);
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
    nearest_pt[j][b][0] = nearest_ptx;
    nearest_pt[j][b][1] = nearest_pty;
    nearest_pt[j][b][2] = nearest_ptz;
}


// ==================== Differentiable FOV Rendering ====================

// Device function: trace a single ray through all scene geometry, return min intersection depth
template <typename scalar_t>
__device__ __forceinline__ scalar_t trace_ray_device(
    scalar_t dx, scalar_t dy, scalar_t dz,
    scalar_t ox, scalar_t oy, scalar_t oz,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> balls,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> cylinders,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> cylinders_h,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> voxels,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> pos,
    int n_drones_per_group, int batch_base, int bi, int B)
{
    scalar_t min_dist = 100;
    // ground plane z = -1
    scalar_t gt = (-1 - oz) / dz;
    if (gt > 0) min_dist = gt;

    // other drones (ellipsoid with z scaled by 2)
    for (int i = batch_base; i < batch_base + n_drones_per_group; i++) {
        if (i == bi || i >= B) continue;
        scalar_t cx = pos[i][0], cy = pos[i][1], cz = pos[i][2];
        scalar_t rad = 0.15;
        scalar_t qa = dx*dx + dy*dy + 4*dz*dz;
        scalar_t qb = 2*(dx*(ox-cx) + dy*(oy-cy) + 4*dz*(oz-cz));
        scalar_t qc = (ox-cx)*(ox-cx) + (oy-cy)*(oy-cy) + 4*(oz-cz)*(oz-cz) - rad*rad;
        scalar_t qd = qb*qb - 4*qa*qc;
        if (qd >= 0) {
            scalar_t qt = (-qb - sqrt(qd)) / (2*qa);
            if (qt > 1e-5) { min_dist = min(min_dist, qt); }
            else { qt = (-qb + sqrt(qd)) / (2*qa); if (qt > 1e-5) min_dist = min(min_dist, qt); }
        }
    }

    // balls (spheres)
    for (int i = 0; i < balls.size(1); i++) {
        scalar_t cx = balls[batch_base][i][0], cy = balls[batch_base][i][1];
        scalar_t cz = balls[batch_base][i][2], rad = balls[batch_base][i][3];
        scalar_t qa = dx*dx + dy*dy + dz*dz;
        scalar_t qb = 2*(dx*(ox-cx) + dy*(oy-cy) + dz*(oz-cz));
        scalar_t qc = (ox-cx)*(ox-cx) + (oy-cy)*(oy-cy) + (oz-cz)*(oz-cz) - rad*rad;
        scalar_t qd = qb*qb - 4*qa*qc;
        if (qd >= 0) {
            scalar_t qt = (-qb - sqrt(qd)) / (2*qa);
            if (qt > 1e-5) { min_dist = min(min_dist, qt); }
            else { qt = (-qb + sqrt(qd)) / (2*qa); if (qt > 1e-5) min_dist = min(min_dist, qt); }
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
        if (qd >= 0) {
            scalar_t qt = (-qb - sqrt(qd)) / (2*qa);
            if (qt > 1e-5) { min_dist = min(min_dist, qt); }
            else { qt = (-qb + sqrt(qd)) / (2*qa); if (qt > 1e-5) min_dist = min(min_dist, qt); }
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
        if (qd >= 0) {
            scalar_t qt = (-qb - sqrt(qd)) / (2*qa);
            if (qt > 1e-5) { min_dist = min(min_dist, qt); }
            else { qt = (-qb + sqrt(qd)) / (2*qa); if (qt > 1e-5) min_dist = min(min_dist, qt); }
        }
    }

    // voxels (AABB)
    for (int i = 0; i < voxels.size(1); i++) {
        scalar_t cx = voxels[batch_base][i][0], cy = voxels[batch_base][i][1];
        scalar_t cz = voxels[batch_base][i][2];
        scalar_t rx = voxels[batch_base][i][3], ry = voxels[batch_base][i][4];
        scalar_t rz = voxels[batch_base][i][5];
        scalar_t tx1 = (cx - rx - ox) / dx, tx2 = (cx + rx - ox) / dx;
        scalar_t tx_min = min(tx1, tx2), tx_max = max(tx1, tx2);
        scalar_t ty1 = (cy - ry - oy) / dy, ty2 = (cy + ry - oy) / dy;
        scalar_t ty_min = min(ty1, ty2), ty_max = max(ty1, ty2);
        scalar_t tz1 = (cz - rz - oz) / dz, tz2 = (cz + rz - oz) / dz;
        scalar_t tz_min = min(tz1, tz2), tz_max = max(tz1, tz2);
        scalar_t t_min_v = max(max(tx_min, ty_min), tz_min);
        scalar_t t_max_v = min(min(tx_max, ty_max), tz_max);
        if (t_min_v < min_dist && t_min_v < t_max_v && t_min_v > 0)
            min_dist = t_min_v;
    }

    return min_dist;
}


// Forward kernel: render with per-batch FOV tensor (differentiable w.r.t. FOV)
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
    canvas[b][u][v] = trace_ray_device(dx, dy, dz,
        pos[b][0], pos[b][1], pos[b][2],
        balls, cylinders, cylinders_h, voxels, pos,
        n_drones_per_group, batch_base, b, B);
}


// Backward kernel: compute d(depth)/d(fov) via finite differences, accumulate per-batch with atomicAdd
template <typename scalar_t>
__global__ void render_backward_fov_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> grad_output,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> canvas,
    scalar_t* __restrict__ grad_fov,
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

    const scalar_t go = grad_output[b][u][v];
    if (abs(go) < 1e-8) return;  // skip zero-gradient pixels

    const scalar_t fov = fov_x_half_tan[b];
    const scalar_t eps = (scalar_t)1e-3;
    const scalar_t fov_p = fov + eps;
    const scalar_t ox = pos[b][0], oy = pos[b][1], oz = pos[b][2];

    // Perturbed ray direction at fov + eps
    const scalar_t fov_y_p = fov_p / W * H;
    const scalar_t fu_p = (2 * (u + 0.5) / H - 1) * fov_y_p - 1e-5;
    const scalar_t fv_p = (2 * (v + 0.5) / W - 1) * fov_p - 1e-5;
    scalar_t dx_p = R[b][0][0] - fu_p * R[b][0][2] - fv_p * R[b][0][1];
    scalar_t dy_p = R[b][1][0] - fu_p * R[b][1][2] - fv_p * R[b][1][1];
    scalar_t dz_p = R[b][2][0] - fu_p * R[b][2][2] - fv_p * R[b][2][1];

    const int batch_base = (b / n_drones_per_group) * n_drones_per_group;
    scalar_t depth_p = trace_ray_device(dx_p, dy_p, dz_p, ox, oy, oz,
        balls, cylinders, cylinders_h, voxels, pos,
        n_drones_per_group, batch_base, b, B);

    scalar_t depth_orig = canvas[b][u][v];
    scalar_t local_grad = (depth_p - depth_orig) / eps;
    atomicAdd(&grad_fov[b], go * local_grad);
}


template <typename scalar_t>
__global__ void rerender_backward_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> depth,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> dddp,
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
    const scalar_t d = (depth[b][0][u*2][v*2] + depth[b][0][u*2+1][v*2] + depth[b][0][u*2][v*2+1] + depth[b][0][u*2+1][v*2+1]) / 4 * unit;
    const scalar_t dddy = (depth[b][0][u*2][v*2] + depth[b][0][u*2+1][v*2] - depth[b][0][u*2][v*2+1] - depth[b][0][u*2+1][v*2+1]) / 2 / d;
    const scalar_t dddz = (depth[b][0][u*2][v*2] - depth[b][0][u*2+1][v*2] + depth[b][0][u*2][v*2+1] - depth[b][0][u*2+1][v*2+1]) / 2 / d;
    // if ReRender.diff_kernel is None:
    //     unit = 0.637 / depth.size(3)
    //     ReRender.diff_kernel = torch.tensor([
    //         [[1, -1], [1, -1]],
    //         [[1, 1], [-1, -1]],
    //         [[unit, unit], [unit, unit]],
    //     ], device=device).mul(0.5)[:, None]
    // ddepthdyz = F.conv2d(depth, ReRender.diff_kernel, None, 2)
    // depth = ddepthdyz[:, 2:]
    // ddepthdyz = torch.cat([
    //     torch.full_like(depth, -1.),
    //     ddepthdyz[:, :2] / depth,
    // ], 1)
    const scalar_t dddp_norm = max(8., sqrt(1 + dddy * dddy + dddz * dddz));
    dddp[b][0][u][v] = -1. / dddp_norm;
    dddp[b][1][u][v] = dddy / dddp_norm;
    dddp[b][2][u][v] = dddz / dddp_norm;
    // ddepthdyz /= ddepthdyz.norm(2, 1, True).clamp_min(8);
}

} // namespace

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
    const int threads = 1024;
    size_t state_size = canvas.numel();
    const dim3 blocks((state_size + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(canvas.type(), "render_cuda", ([&] {
        render_cuda_kernel<scalar_t><<<blocks, threads>>>(
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
}

void rerender_backward_cuda(
    torch::Tensor depth,
    torch::Tensor dddp,
    float fov_x_half_tan) {
    const int threads = 1024;
    size_t state_size = dddp.numel();
    const dim3 blocks((state_size + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(depth.type(), "rerender_backward_cuda", ([&] {
        rerender_backward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            depth.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            dddp.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            fov_x_half_tan);
    }));
}

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
    size_t state_size = pos.size(0) * pos.size(1);
    const dim3 blocks((state_size + threads - 1) / threads);
    AT_DISPATCH_FLOATING_TYPES(pos.type(), "nearest_pt_cuda", ([&] {
        nearest_pt_cuda_kernel<scalar_t><<<blocks, threads>>>(
            nearest_pt.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            balls.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            cylinders.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            cylinders_h.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            voxels.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            pos.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            drone_radius,
            n_drones_per_group);
    }));
}

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
        nearest_pt_ellipsoid_cuda_kernel<scalar_t><<<blocks, threads>>>(
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
}

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
        render_diff_fov_cuda_kernel<scalar_t><<<blocks, threads>>>(
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
}

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
    const int threads = 1024;
    size_t state_size = canvas.numel();
    const dim3 blocks((state_size + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(canvas.type(), "render_backward_fov_cuda", ([&] {
        render_backward_fov_cuda_kernel<scalar_t><<<blocks, threads>>>(
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
}
