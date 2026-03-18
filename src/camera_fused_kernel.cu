#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

constexpr int kBX = 32;
constexpr int kBY = 8;

__device__ __forceinline__ float clampf(float x, float a, float b) {
    return x < a ? a : (x > b ? b : x);
}

__device__ __forceinline__ float safe_powf(float x, float p) {
    return powf(fmaxf(x, 1e-12f), p);
}

template <typename scalar_t>
__global__ void fused_render_to_x_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> x_out,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> depth_raw,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> normals,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> exposure,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> iso,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> cam_light_dir,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> cam_ambient,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> cam_dir_intensity,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> cam_fog_beta,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> cam_airlight,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> cam_mat_ground,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> cam_mat_obstacle,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> cam_mat_spec,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> cam_dist_k1,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> cam_dist_k2,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> cam_prnu,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> cam_dsnu,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> cam_ae_log_t,
    int cam_enable_shadow,
    int cam_enable_specular,
    int cam_enable_distortion,
    float cam_vignette_a,
    float cam_vignette_b,
    float cam_black_level,
    float cam_base_gain,
    float cam_exposure_t_min,
    float cam_exposure_t_span,
    float cam_exposure_eff_min,
    float cam_exposure_eff_max,
    float cam_iso_gain_base,
    float cam_iso_gain_scale,
    float cam_iso_gain_gamma) {

    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    const int u = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;

    const int B = depth_raw.size(0);
    const int H = depth_raw.size(1);
    const int W = depth_raw.size(2);
    if (b >= B || u >= H || v >= W) return;

    const float depth = clampf((float)depth_raw[b][u][v], 0.03f, 120.0f);
    const float nx = (float)normals[b][0][u][v];
    const float ny = (float)normals[b][1][u][v];
    const float nz = (float)normals[b][2][u][v];

    const float nz_abs = fabsf(nz);
    const float w_ground = clampf((nz_abs - 0.55f) / 0.45f, 0.0f, 1.0f);

    const float albedo = w_ground * (float)cam_mat_ground[b] + (1.0f - w_ground) * (float)cam_mat_obstacle[b];

    const float Lx = (float)cam_light_dir[b][0];
    const float Ly = (float)cam_light_dir[b][1];
    const float Lz = (float)cam_light_dir[b][2];
    const float ndotl = fmaxf(nx * Lx + ny * Ly + nz * Lz, 0.0f);

    float shadow = 1.0f;
    if (cam_enable_shadow) {
        shadow = clampf(0.35f + 0.65f * ndotl, 0.2f, 1.0f);
    }

    float specular = 0.0f;
    if (cam_enable_specular) {
        specular = (float)cam_mat_spec[b] * safe_powf(ndotl, 24.0f);
    }

    float irr = albedo * ((float)cam_ambient[b] + (float)cam_dir_intensity[b] * ndotl * shadow) + specular;
    const float trans = expf(-(float)cam_fog_beta[b] * depth);
    irr = clampf(irr * trans + (float)cam_airlight[b] * (1.0f - trans), 0.0f, 4.0f);

    const float yu = (2.0f * ((float)u + 0.5f) / (float)H - 1.0f);
    const float xv = (2.0f * ((float)v + 0.5f) / (float)W - 1.0f);
    const float r2 = yu * yu + xv * xv;
    const float vignette = clampf(1.0f - cam_vignette_a * r2 - cam_vignette_b * r2 * r2, 0.25f, 1.0f);

    float lens_y = irr * vignette;
    if (cam_enable_distortion) {
        const float radial = 1.0f + (float)cam_dist_k1[b] * r2 + (float)cam_dist_k2[b] * r2 * r2;
        lens_y = clampf(lens_y * clampf(radial, 0.7f, 1.3f), 0.0f, 4.0f);
    }

    const float exposure01 = clampf((float)exposure[b], 0.0f, 1.0f);
    const float iso01 = clampf((float)iso[b], 0.0f, 1.0f);

    const float t_cmd = cam_exposure_t_min + cam_exposure_t_span * exposure01;
    const float t_ae = expf((float)cam_ae_log_t[b]);
    const float t_eff = clampf(t_cmd * t_ae, cam_exposure_eff_min, cam_exposure_eff_max);
    const float iso_gain = cam_iso_gain_base + cam_iso_gain_scale * safe_powf(iso01, cam_iso_gain_gamma);

    float electrons = lens_y * t_eff * cam_base_gain;
    electrons *= iso_gain;

    const float raw = electrons * (1.0f + (float)cam_prnu[b][u][v]) + (float)cam_dsnu[b][u][v];
    const float t = fmaxf(raw - cam_black_level, 0.0f);
    const float x = t / (1.0f + t);

    x_out[b][u][v] = (scalar_t)x;
}

template <typename scalar_t>
__global__ void gaussian_h3_shared_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> out,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> in) {

    __shared__ scalar_t tile[kBY][kBX + 2];

    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    const int u = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;

    const int B = in.size(0);
    const int H = in.size(1);
    const int W = in.size(2);
    const bool valid = (b < B && u < H);

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int vc = min(max(v, 0), W - 1);
    const int uc = valid ? u : 0;
    const int bc = valid ? b : 0;
    if (valid && v < W) {
        tile[ty][tx + 1] = in[bc][uc][vc];
    }

    if (tx == 0 && valid) {
        const int vl = min(max(v - 1, 0), W - 1);
        tile[ty][0] = in[bc][uc][vl];
    }
    if (tx == blockDim.x - 1 && valid) {
        const int vr = min(max(v + 1, 0), W - 1);
        tile[ty][blockDim.x + 1] = in[bc][uc][vr];
    }

    __syncthreads();

    if (valid && v < W) {
        const scalar_t l = tile[ty][tx];
        const scalar_t c = tile[ty][tx + 1];
        const scalar_t r = tile[ty][tx + 2];
        out[b][u][v] = (scalar_t)0.25 * l + (scalar_t)0.5 * c + (scalar_t)0.25 * r;
    }
}

template <typename scalar_t>
__global__ void gaussian_v3_shared_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> out,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> in) {

    __shared__ scalar_t tile[kBY + 2][kBX];

    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    const int u = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;

    const int B = in.size(0);
    const int H = in.size(1);
    const int W = in.size(2);
    const bool valid = (b < B && v < W);

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int vc = valid ? v : 0;
    const int bc = valid ? b : 0;
    const int uc = min(max(u, 0), H - 1);
    if (valid && u < H) {
        tile[ty + 1][tx] = in[bc][uc][vc];
    }

    if (ty == 0 && valid) {
        const int ut = min(max(u - 1, 0), H - 1);
        tile[0][tx] = in[bc][ut][vc];
    }
    if (ty == blockDim.y - 1 && valid) {
        const int ub = min(max(u + 1, 0), H - 1);
        tile[blockDim.y + 1][tx] = in[bc][ub][vc];
    }

    __syncthreads();

    if (valid && u < H) {
        const scalar_t t = tile[ty][tx];
        const scalar_t c = tile[ty + 1][tx];
        const scalar_t d = tile[ty + 2][tx];
        out[b][u][v] = (scalar_t)0.25 * t + (scalar_t)0.5 * c + (scalar_t)0.25 * d;
    }
}

template <typename scalar_t>
__global__ void fused_isp_to_y_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> y_out,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> x_in,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> smooth,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> iso,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> cam_gamma,
    float cam_sharpen_amount) {

    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    const int u = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;

    const int B = x_in.size(0);
    const int H = x_in.size(1);
    const int W = x_in.size(2);
    if (b >= B || u >= H || v >= W) return;

    const float x = (float)x_in[b][u][v];
    const float s = (float)smooth[b][u][v];
    const float iso01 = clampf((float)iso[b], 0.0f, 1.0f);
    const float denoise = 0.08f + 0.28f * iso01;

    float x1 = x * (1.0f - denoise) + s * denoise;
    x1 = x1 + cam_sharpen_amount * (x1 - s);

    const float gamma = fmaxf((float)cam_gamma[b], 1e-3f);
    const float x_lin = clampf(x1, 0.0f, 1.0f);
    float y = 0.0f;
    if (x_lin > 0.0f) {
        const float x_safe = fmaxf(x_lin, 1e-6f);
        y = powf(x_safe, 1.0f / gamma);
    }
    y_out[b][u][v] = (scalar_t)clampf(y, 0.0f, 1.0f);
}

template <typename scalar_t>
__global__ void fused_isp_backward_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> grad_x_direct,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> grad_smooth_direct,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> grad_y,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> x_in,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> smooth,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> iso,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> cam_gamma,
    float cam_sharpen_amount) {

    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    const int u = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;

    const int B = x_in.size(0);
    const int H = x_in.size(1);
    const int W = x_in.size(2);
    if (b >= B || u >= H || v >= W) return;

    const float go = (float)grad_y[b][u][v];
    const float x = (float)x_in[b][u][v];
    const float s = (float)smooth[b][u][v];
    const float iso01 = clampf((float)iso[b], 0.0f, 1.0f);
    const float denoise = 0.08f + 0.28f * iso01;

    const float dx2_dx = (1.0f - denoise) * (1.0f + cam_sharpen_amount);
    const float dx2_ds = denoise * (1.0f + cam_sharpen_amount) - cam_sharpen_amount;

    float x1 = x * (1.0f - denoise) + s * denoise;
    x1 = x1 + cam_sharpen_amount * (x1 - s);

    float dy_dx2 = 0.0f;
    const float gamma = fmaxf((float)cam_gamma[b], 1e-3f);
    const float x_lin = clampf(x1, 0.0f, 1.0f);
    if (x_lin > 1e-6f && x_lin < 1.0f) {
        dy_dx2 = (1.0f / gamma) * powf(x_lin, 1.0f / gamma - 1.0f);
    }

    const float g = go * dy_dx2;
    grad_x_direct[b][u][v] = (scalar_t)(g * dx2_dx);
    grad_smooth_direct[b][u][v] = (scalar_t)(g * dx2_ds);
}

template <typename scalar_t>
__global__ void fused_raw_backward_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> grad_depth,
    scalar_t* __restrict__ grad_exposure,
    scalar_t* __restrict__ grad_iso,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> grad_x,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> x_in,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> depth_raw,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> normals,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> exposure,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> iso,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> cam_light_dir,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> cam_ambient,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> cam_dir_intensity,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> cam_fog_beta,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> cam_airlight,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> cam_mat_ground,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> cam_mat_obstacle,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> cam_mat_spec,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> cam_dist_k1,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> cam_dist_k2,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> cam_prnu,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> cam_ae_log_t,
    int cam_enable_shadow,
    int cam_enable_specular,
    int cam_enable_distortion,
    float cam_vignette_a,
    float cam_vignette_b,
    float cam_black_level,
    float cam_base_gain,
    float cam_exposure_t_min,
    float cam_exposure_t_span,
    float cam_exposure_eff_min,
    float cam_exposure_eff_max,
    float cam_iso_gain_base,
    float cam_iso_gain_scale,
    float cam_iso_gain_gamma,
    int need_grad_exposure,
    int need_grad_iso) {

    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    const int u = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;

    const int B = depth_raw.size(0);
    const int H = depth_raw.size(1);
    const int W = depth_raw.size(2);
    if (b >= B || u >= H || v >= W) return;

    const float depth_in = (float)depth_raw[b][u][v];
    const float depth = clampf(depth_in, 0.03f, 120.0f);
    const float depth_mask = (depth_in > 0.03f && depth_in < 120.0f) ? 1.0f : 0.0f;

    const float nx = (float)normals[b][0][u][v];
    const float ny = (float)normals[b][1][u][v];
    const float nz = (float)normals[b][2][u][v];

    const float nz_abs = fabsf(nz);
    const float w_ground = clampf((nz_abs - 0.55f) / 0.45f, 0.0f, 1.0f);
    const float albedo = w_ground * (float)cam_mat_ground[b] + (1.0f - w_ground) * (float)cam_mat_obstacle[b];

    const float Lx = (float)cam_light_dir[b][0];
    const float Ly = (float)cam_light_dir[b][1];
    const float Lz = (float)cam_light_dir[b][2];
    const float ndotl = fmaxf(nx * Lx + ny * Ly + nz * Lz, 0.0f);

    float shadow = 1.0f;
    if (cam_enable_shadow) {
        shadow = clampf(0.35f + 0.65f * ndotl, 0.2f, 1.0f);
    }

    float specular = 0.0f;
    if (cam_enable_specular) {
        specular = (float)cam_mat_spec[b] * safe_powf(ndotl, 24.0f);
    }

    const float irr = albedo * ((float)cam_ambient[b] + (float)cam_dir_intensity[b] * ndotl * shadow) + specular;
    const float trans = expf(-(float)cam_fog_beta[b] * depth);

    const float yu = (2.0f * ((float)u + 0.5f) / (float)H - 1.0f);
    const float xv = (2.0f * ((float)v + 0.5f) / (float)W - 1.0f);
    const float r2 = yu * yu + xv * xv;
    const float vignette = clampf(1.0f - cam_vignette_a * r2 - cam_vignette_b * r2 * r2, 0.25f, 1.0f);

    float radial = 1.0f;
    if (cam_enable_distortion) {
        radial = clampf(1.0f + (float)cam_dist_k1[b] * r2 + (float)cam_dist_k2[b] * r2 * r2, 0.7f, 1.3f);
    }
    const float vfactor = vignette * radial;

    const float irr2 = clampf(irr * trans + (float)cam_airlight[b] * (1.0f - trans), 0.0f, 4.0f);
    const float lens = irr2 * vfactor;

    const float exp01 = clampf((float)exposure[b], 0.0f, 1.0f);
    const float iso01 = clampf((float)iso[b], 0.0f, 1.0f);

    const float t_cmd = cam_exposure_t_min + cam_exposure_t_span * exp01;
    const float t_ae = expf((float)cam_ae_log_t[b]);
    const float t_mul = t_cmd * t_ae;
    const float t_eff = clampf(t_mul, cam_exposure_eff_min, cam_exposure_eff_max);
    const float t_eff_mask = (t_mul > cam_exposure_eff_min && t_mul < cam_exposure_eff_max) ? 1.0f : 0.0f;

    const float iso_gain = cam_iso_gain_base + cam_iso_gain_scale * safe_powf(iso01, cam_iso_gain_gamma);
    const float iso_mask = (iso[b] > (scalar_t)0.0 && iso[b] < (scalar_t)1.0) ? 1.0f : 0.0f;
    const float exp_mask = (exposure[b] > (scalar_t)0.0 && exposure[b] < (scalar_t)1.0) ? 1.0f : 0.0f;

    const float x = clampf((float)x_in[b][u][v], 0.0f, 1.0f - 1e-6f);
    const float gx = (float)grad_x[b][u][v];

    float g_raw = 0.0f;
    if (x > 0.0f) {
        const float one_minus_x = 1.0f - x;
        g_raw = gx * one_minus_x * one_minus_x;
    }

    const float g_electrons = g_raw * (1.0f + (float)cam_prnu[b][u][v]);

    const float g_lens = g_electrons * t_eff * cam_base_gain * iso_gain;

    const float g_irr2 = g_lens * vfactor;
    const float g_depth = g_irr2 * (irr - (float)cam_airlight[b]) * (-(float)cam_fog_beta[b]) * trans * depth_mask;

    grad_depth[b][u][v] = (scalar_t)g_depth;

    if (need_grad_exposure) {
        const float d_t_eff_d_exp = cam_exposure_t_span * t_ae * t_eff_mask * exp_mask;
        const float g_t_eff = g_electrons * lens * cam_base_gain * iso_gain;
        atomicAdd(&grad_exposure[b], (scalar_t)(g_t_eff * d_t_eff_d_exp));
    }

    if (need_grad_iso) {
        const float d_iso_gain_d_iso = cam_iso_gain_scale * cam_iso_gain_gamma * safe_powf(iso01, cam_iso_gain_gamma - 1.0f) * iso_mask;
        const float g_iso_gain = g_electrons * lens * t_eff * cam_base_gain;
        atomicAdd(&grad_iso[b], (scalar_t)(g_iso_gain * d_iso_gain_d_iso));
    }
}

void gaussian_blur3_separable_cuda(torch::Tensor out, torch::Tensor in) {
    const int B = in.size(0);
    const int H = in.size(1);
    const int W = in.size(2);

    auto tmp = torch::empty_like(in);
    const dim3 threads(kBX, kBY, 1);
    const dim3 blocks((W + kBX - 1) / kBX, (H + kBY - 1) / kBY, B);

    AT_DISPATCH_FLOATING_TYPES(in.scalar_type(), "gaussian_h3_shared_kernel", ([&] {
        gaussian_h3_shared_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            tmp.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            in.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>());
    }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    AT_DISPATCH_FLOATING_TYPES(in.scalar_type(), "gaussian_v3_shared_kernel", ([&] {
        gaussian_v3_shared_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            out.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            tmp.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>());
    }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace

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
    double cam_iso_gain_gamma) {

    const int B = depth_raw.size(0);
    const int H = depth_raw.size(1);
    const int W = depth_raw.size(2);

    auto x = torch::empty_like(depth_raw);
    auto smooth = torch::empty_like(depth_raw);
    auto y = torch::empty_like(depth_raw);

    const int enable_shadow = (cam_profile_mask & (1 << 0)) ? 1 : 0;
    const int enable_specular = (cam_profile_mask & (1 << 1)) ? 1 : 0;
    const int enable_distortion = (cam_profile_mask & (1 << 2)) ? 1 : 0;

    const dim3 threads(kBX, kBY, 1);
    const dim3 blocks((W + kBX - 1) / kBX, (H + kBY - 1) / kBY, B);

    AT_DISPATCH_FLOATING_TYPES(depth_raw.scalar_type(), "fused_render_to_x_kernel", ([&] {
        fused_render_to_x_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            x.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            depth_raw.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            normals.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            exposure.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            iso.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            cam_light_dir.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            cam_ambient.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            cam_dir_intensity.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            cam_fog_beta.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            cam_airlight.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            cam_mat_ground.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            cam_mat_obstacle.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            cam_mat_spec.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            cam_dist_k1.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            cam_dist_k2.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            cam_prnu.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            cam_dsnu.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            cam_ae_log_t.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            enable_shadow,
            enable_specular,
            enable_distortion,
            (float)cam_vignette_a,
            (float)cam_vignette_b,
            (float)cam_black_level,
            (float)cam_base_gain,
            (float)cam_exposure_t_min,
            (float)cam_exposure_t_span,
            (float)cam_exposure_eff_min,
            (float)cam_exposure_eff_max,
            (float)cam_iso_gain_base,
            (float)cam_iso_gain_scale,
            (float)cam_iso_gain_gamma);
    }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    gaussian_blur3_separable_cuda(smooth, x);

    AT_DISPATCH_FLOATING_TYPES(depth_raw.scalar_type(), "fused_isp_to_y_kernel", ([&] {
        fused_isp_to_y_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            y.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            x.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            smooth.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            iso.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            cam_gamma.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            (float)cam_sharpen_amount);
    }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return y;
}

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
    bool need_grad_iso) {

    const int B = depth_raw.size(0);
    const int H = depth_raw.size(1);
    const int W = depth_raw.size(2);

    auto x = torch::empty_like(depth_raw);
    auto smooth = torch::empty_like(depth_raw);
    auto grad_x_direct = torch::empty_like(depth_raw);
    auto grad_smooth_direct = torch::empty_like(depth_raw);
    auto grad_smooth_via = torch::empty_like(depth_raw);
    auto grad_x_total = torch::empty_like(depth_raw);

    auto grad_depth = torch::zeros_like(depth_raw);
    auto grad_exposure = torch::zeros_like(exposure);
    auto grad_iso = torch::zeros_like(iso);

    const int enable_shadow = (cam_profile_mask & (1 << 0)) ? 1 : 0;
    const int enable_specular = (cam_profile_mask & (1 << 1)) ? 1 : 0;
    const int enable_distortion = (cam_profile_mask & (1 << 2)) ? 1 : 0;

    const dim3 threads(kBX, kBY, 1);
    const dim3 blocks((W + kBX - 1) / kBX, (H + kBY - 1) / kBY, B);

    AT_DISPATCH_FLOATING_TYPES(depth_raw.scalar_type(), "fused_render_to_x_kernel_backward_recompute", ([&] {
        fused_render_to_x_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            x.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            depth_raw.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            normals.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            exposure.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            iso.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            cam_light_dir.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            cam_ambient.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            cam_dir_intensity.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            cam_fog_beta.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            cam_airlight.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            cam_mat_ground.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            cam_mat_obstacle.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            cam_mat_spec.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            cam_dist_k1.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            cam_dist_k2.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            cam_prnu.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            cam_dsnu.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            cam_ae_log_t.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            enable_shadow,
            enable_specular,
            enable_distortion,
            (float)cam_vignette_a,
            (float)cam_vignette_b,
            (float)cam_black_level,
            (float)cam_base_gain,
            (float)cam_exposure_t_min,
            (float)cam_exposure_t_span,
            (float)cam_exposure_eff_min,
            (float)cam_exposure_eff_max,
            (float)cam_iso_gain_base,
            (float)cam_iso_gain_scale,
            (float)cam_iso_gain_gamma);
    }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    gaussian_blur3_separable_cuda(smooth, x);

    auto go = grad_output.contiguous();
    AT_DISPATCH_FLOATING_TYPES(depth_raw.scalar_type(), "fused_isp_backward_kernel", ([&] {
        fused_isp_backward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            grad_x_direct.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            grad_smooth_direct.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            go.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            x.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            smooth.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            iso.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            cam_gamma.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            (float)cam_sharpen_amount);
    }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    gaussian_blur3_separable_cuda(grad_smooth_via, grad_smooth_direct);
    grad_x_total = grad_x_direct + grad_smooth_via;

    AT_DISPATCH_FLOATING_TYPES(depth_raw.scalar_type(), "fused_raw_backward_kernel", ([&] {
        fused_raw_backward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            grad_depth.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            grad_exposure.data_ptr<scalar_t>(),
            grad_iso.data_ptr<scalar_t>(),
            grad_x_total.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            x.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            depth_raw.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            normals.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            exposure.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            iso.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            cam_light_dir.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            cam_ambient.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            cam_dir_intensity.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            cam_fog_beta.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            cam_airlight.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            cam_mat_ground.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            cam_mat_obstacle.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            cam_mat_spec.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            cam_dist_k1.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            cam_dist_k2.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            cam_prnu.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            cam_ae_log_t.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            enable_shadow,
            enable_specular,
            enable_distortion,
            (float)cam_vignette_a,
            (float)cam_vignette_b,
            (float)cam_black_level,
            (float)cam_base_gain,
            (float)cam_exposure_t_min,
            (float)cam_exposure_t_span,
            (float)cam_exposure_eff_min,
            (float)cam_exposure_eff_max,
            (float)cam_iso_gain_base,
            (float)cam_iso_gain_scale,
            (float)cam_iso_gain_gamma,
            need_grad_exposure ? 1 : 0,
            need_grad_iso ? 1 : 0);
    }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {grad_depth, grad_exposure, grad_iso};
}
