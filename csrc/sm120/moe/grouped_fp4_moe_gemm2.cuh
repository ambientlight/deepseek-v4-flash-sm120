// =====================================================================
// SM120 Grouped FP4 SwiGLU + GEMM2 — Weight Reuse Across Tokens
//
// Same grouped approach as GEMM1: slots sorted by expert, weight loaded
// once per expert, all M_e token rows processed against cached weight.
// SwiGLU fused into K-loop: x_k = silu(gate_k) * up_k
//
// GEMM2 shape per expert: [M_e, K_out=4096] = [M_e, I=512] × W2[K_out, I/2]
// K_out=4096 is the N (output) dimension, I=512 is the K (reduction) dimension.
// =====================================================================
#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>

namespace dsv4_kernel {
namespace sm120 {
namespace grouped {

static constexpr int GS2_BLOCK_M = 16;
static constexpr int GS2_BLOCK_N = 4;
static constexpr int GS2_WARPS = 4;
static constexpr int GS2_THREADS = GS2_WARPS * 32;

__device__ __forceinline__ float g_device_silu(float x) {
    return x / (1.0f + expf(-x));
}

__global__ __launch_bounds__(128, 6)
void grouped_fp4_swiglu_gemm2_kernel(
    const __nv_bfloat16* __restrict__ intermediate,  // [num_slots, 2*I]
    const uint8_t* __restrict__ B_packed,             // [E, K_out, I/2]
    const float* __restrict__ B_scale,                // [E, K_out, I/32]
    __nv_bfloat16* __restrict__ C,                    // [num_slots, K_out]
    const int32_t* __restrict__ sorted_slot_ids,      // [num_slots] sorted by expert
    const int32_t* __restrict__ expert_offsets,        // [E+1]
    int K_out, int I,
    int num_experts,
    int num_slots,
    float clamp_limit,
    int stride_bn,
    int stride_bsn,
    int64_t expert_b_stride,
    int64_t expert_bs_stride)
{
    const int m_tile = blockIdx.x;
    const int n_block = blockIdx.y * GS2_BLOCK_N;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int gn = n_block + warp_id;

    float lut[16];
    lut[0] = 0.0f; lut[1] = 0.5f; lut[2] = 1.0f; lut[3] = 1.5f;
    lut[4] = 2.0f; lut[5] = 3.0f; lut[6] = 4.0f; lut[7] = 6.0f;
    lut[8] = -0.0f; lut[9] = -0.5f; lut[10] = -1.0f; lut[11] = -1.5f;
    lut[12] = -2.0f; lut[13] = -3.0f; lut[14] = -4.0f; lut[15] = -6.0f;

    if (gn >= K_out) return;

    const int m_start = m_tile * GS2_BLOCK_M;
    const bool do_clamp = (clamp_limit > 0.0f);
    const int elems_per_lane = I / 32;

    // Track current expert — switch weight pointers on boundary
    int cur_expert = -1;
    const uint8_t* B_row = nullptr;
    const float* Bs_row = nullptr;

    for (int mi = 0; mi < GS2_BLOCK_M; mi++) {
        int slot_global = m_start + mi;
        if (slot_global >= num_slots) break;

        int slot_expert = find_expert(expert_offsets, num_experts, slot_global);
        if (slot_expert != cur_expert) {
            cur_expert = slot_expert;
            B_row = B_packed + cur_expert * expert_b_stride + gn * stride_bn;
            Bs_row = B_scale + cur_expert * expert_bs_stride + gn * stride_bsn;
        }

        int sorted_slot = sorted_slot_ids[slot_global];
        const __nv_bfloat16* gate_row = intermediate + sorted_slot * (2 * I);
        const __nv_bfloat16* up_row = gate_row + I;

        float acc = 0.0f;
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

                float x0 = g_device_silu(g0) * u0;
                float x1 = g_device_silu(g1) * u1;

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
            C[sorted_slot * K_out + gn] = __float2bfloat16(acc);
        }
    }
}

inline void launch_grouped_fp4_swiglu_gemm2(
    const __nv_bfloat16* intermediate,
    const uint8_t* B_packed,
    const float* B_scale,
    __nv_bfloat16* C,
    const int32_t* sorted_slot_ids,
    const int32_t* expert_offsets,
    int num_slots, int K_out, int I,
    int num_experts,
    float clamp_limit,
    int stride_bn, int stride_bsn,
    int64_t expert_b_stride, int64_t expert_bs_stride,
    cudaStream_t stream)
{
    int total_m_tiles = (num_slots + GS2_BLOCK_M - 1) / GS2_BLOCK_M;
    dim3 grid(total_m_tiles, (K_out + GS2_BLOCK_N - 1) / GS2_BLOCK_N);
    dim3 block(GS2_THREADS);

    grouped_fp4_swiglu_gemm2_kernel<<<grid, block, 0, stream>>>(
        intermediate, B_packed, B_scale, C,
        sorted_slot_ids, expert_offsets,
        K_out, I, num_experts, num_slots, clamp_limit,
        stride_bn, stride_bsn, expert_b_stride, expert_bs_stride);
}

}  // namespace grouped
}  // namespace sm120
}  // namespace dsv4_kernel
