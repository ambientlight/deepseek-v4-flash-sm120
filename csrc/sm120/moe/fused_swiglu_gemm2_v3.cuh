// =====================================================================
// SM120 Fused SwiGLU + GEMM2 v3 — Optimized for small K, large N
//
// GEMM2 shape: [6 slots × 4096 N × 512 K]
// K=512 is small → can't split K across 32 lanes efficiently
// N=4096 is large → many N outputs per CTA
//
// Design: Each warp owns 1 N output. All 32 lanes cooperate on K=512.
// K/32 = 16 elements per lane. Process all 16 in one go with uint64 load.
// SwiGLU fused into input load: x_k = silu(gate_k) * up_k
//
// Grid: (num_slots, ceil(N / BLOCK_N))  where BLOCK_N = 4 (4 warps)
// Block: 128 threads = 4 warps
// CTAs: 6 × ceil(4096/4) = 6144 — massive occupancy
// =====================================================================
#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>

namespace dsv4_kernel {
namespace sm120 {
namespace v3 {

// Reuse c_FP4_LUT from fused_fp4_moe_gemv.cuh (same translation unit)

static constexpr int SG2_BLOCK_N = 4;
static constexpr int SG2_WARPS = 4;
static constexpr int SG2_THREADS = SG2_WARPS * 32;

__device__ __forceinline__ float device_silu_v3(float x) {
    return x / (1.0f + expf(-x));
}

__global__ __launch_bounds__(128, 8)
void fused_swiglu_gemm2_v3_kernel(
    const __nv_bfloat16* __restrict__ intermediate, // [num_slots, 2*I]
    const uint8_t* __restrict__ B_packed,    // [E, K_out, I/2]
    const float* __restrict__ B_scale,       // [E, K_out, I/32]
    __nv_bfloat16* __restrict__ C,           // [num_slots, K_out]
    const int* __restrict__ expert_ids,      // [num_slots]
    int K_out, int I,
    float clamp_limit,
    int stride_bn,           // I/2
    int stride_bsn,          // I/32
    int64_t expert_b_stride,
    int64_t expert_bs_stride)
{
    const int slot = blockIdx.x;
    const int n_block = blockIdx.y * SG2_BLOCK_N;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int gn = n_block + warp_id;

    // Register LUT
    float lut[16];
    lut[0] = 0.0f; lut[1] = 0.5f; lut[2] = 1.0f; lut[3] = 1.5f;
    lut[4] = 2.0f; lut[5] = 3.0f; lut[6] = 4.0f; lut[7] = 6.0f;
    lut[8] = -0.0f; lut[9] = -0.5f; lut[10] = -1.0f; lut[11] = -1.5f;
    lut[12] = -2.0f; lut[13] = -3.0f; lut[14] = -4.0f; lut[15] = -6.0f;

    if (gn >= K_out) return;

    const int exp = expert_ids[slot];
    const __nv_bfloat16* gate_row = intermediate + slot * (2 * I);
    const __nv_bfloat16* up_row = gate_row + I;
    const uint8_t* B_row = B_packed + exp * expert_b_stride + gn * stride_bn;
    const float* Bs_row = B_scale + exp * expert_bs_stride + gn * stride_bsn;
    const bool do_clamp = (clamp_limit > 0.0f);

    // 32 lanes, each handles I/32 contiguous K elements
    float acc = 0.0f;
    const int elems_per_lane = I / 32;
    const int my_k = lane * elems_per_lane;
    const int my_k_end = my_k + elems_per_lane;

    // For I=512, elems_per_lane=16. Process with uint64 (16 FP4 = 8 bytes)
    for (int k = my_k; k < my_k_end; k += 16) {
        float scale = Bs_row[k / 32];
        uint64_t packed8 = *reinterpret_cast<const uint64_t*>(B_row + k / 2);

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            uint8_t byte = (packed8 >> (i * 8)) & 0xFF;
            int ki = k + i * 2;

            // Fused SwiGLU: x = silu(gate[ki]) * up[ki]
            float g0 = __bfloat162float(gate_row[ki]);
            float u0 = __bfloat162float(up_row[ki]);
            float g1 = __bfloat162float(gate_row[ki + 1]);
            float u1 = __bfloat162float(up_row[ki + 1]);

            if (do_clamp) {
                g0 = fminf(g0, clamp_limit);
                u0 = fmaxf(fminf(u0, clamp_limit), -clamp_limit);
                g1 = fminf(g1, clamp_limit);
                u1 = fmaxf(fminf(u1, clamp_limit), -clamp_limit);
            }

            float x0 = device_silu_v3(g0) * u0;
            float x1 = device_silu_v3(g1) * u1;

            float b0 = lut[byte & 0xF] * scale;
            float b1 = lut[(byte >> 4) & 0xF] * scale;

            acc += x0 * b0 + x1 * b1;
        }
    }

    // Warp-shuffle reduction
    #pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1) {
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, offset);
    }

    if (lane == 0) {
        C[slot * K_out + gn] = __float2bfloat16(acc);
    }
}

inline void launch_fused_swiglu_gemm2_v3(
    const __nv_bfloat16* intermediate,
    const uint8_t* B_packed,
    const float* B_scale,
    __nv_bfloat16* C,
    const int* expert_ids,
    int num_slots, int K_out, int I,
    float clamp_limit,
    int stride_bn, int stride_bsn,
    int64_t expert_b_stride, int64_t expert_bs_stride,
    cudaStream_t stream)
{
    dim3 grid(num_slots, (K_out + SG2_BLOCK_N - 1) / SG2_BLOCK_N);
    dim3 block(SG2_THREADS);

    fused_swiglu_gemm2_v3_kernel<<<grid, block, 0, stream>>>(
        intermediate, B_packed, B_scale, C, expert_ids,
        K_out, I, clamp_limit,
        stride_bn, stride_bsn, expert_b_stride, expert_bs_stride);
}

}  // namespace v3
}  // namespace sm120
}  // namespace dsv4_kernel
