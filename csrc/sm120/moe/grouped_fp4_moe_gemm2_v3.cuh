// =====================================================================
// SM120 Grouped FP4 SwiGLU+GEMM2 v3 — Register Blocking
//
// Same approach as GEMM1 v3: load weight once in registers,
// multiply against multiple M rows. SwiGLU fused into activation load.
// =====================================================================
#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>

namespace dsv4_kernel {
namespace sm120 {
namespace grouped_v3 {

// g3_find_expert already defined in grouped_fp4_moe_gemm_v3.cuh (same TU)

static constexpr int GS3_BLOCK_M = 8;
static constexpr int GS3_BLOCK_N = 4;
static constexpr int GS3_WARPS = 4;
static constexpr int GS3_THREADS = GS3_WARPS * 32;

__device__ __forceinline__ float g3_silu(float x) {
    return x / (1.0f + expf(-x));
}

__global__ __launch_bounds__(128, 6)
void grouped_fp4_swiglu_gemm2_v3_kernel(
    const __nv_bfloat16* __restrict__ intermediate,
    const uint8_t* __restrict__ B_packed,
    const float* __restrict__ B_scale,
    __nv_bfloat16* __restrict__ C,
    const int32_t* __restrict__ sorted_slot_ids,
    const int32_t* __restrict__ expert_offsets,
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
    const int n_block = blockIdx.y * GS3_BLOCK_N;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int gn = n_block + warp_id;

    if (gn >= K_out) return;

    const int m_start = m_tile * GS3_BLOCK_M;
    const bool do_clamp = (clamp_limit > 0.0f);

    float lut[16];
    lut[0] = 0.0f; lut[1] = 0.5f; lut[2] = 1.0f; lut[3] = 1.5f;
    lut[4] = 2.0f; lut[5] = 3.0f; lut[6] = 4.0f; lut[7] = 6.0f;
    lut[8] = -0.0f; lut[9] = -0.5f; lut[10] = -1.0f; lut[11] = -1.5f;
    lut[12] = -2.0f; lut[13] = -3.0f; lut[14] = -4.0f; lut[15] = -6.0f;

    int row_sorted_slot[GS3_BLOCK_M];
    int row_expert[GS3_BLOCK_M];
    int valid_m = 0;

    for (int mi = 0; mi < GS3_BLOCK_M; mi++) {
        int sg = m_start + mi;
        if (sg >= num_slots) break;
        row_sorted_slot[mi] = sorted_slot_ids[sg];
        row_expert[mi] = g3_find_expert(expert_offsets, num_experts, sg);
        valid_m = mi + 1;
    }

    if (valid_m == 0) return;

    float acc[GS3_BLOCK_M];
    #pragma unroll
    for (int mi = 0; mi < GS3_BLOCK_M; mi++) acc[mi] = 0.0f;

    const int elems_per_lane = I / 32;
    const int my_k = lane * elems_per_lane;
    const int my_k_end = my_k + elems_per_lane;

    int mi = 0;
    while (mi < valid_m) {
        int cur_exp = row_expert[mi];
        int group_start = mi;
        while (mi < valid_m && row_expert[mi] == cur_exp) mi++;
        int group_end = mi;

        const uint8_t* B_row = B_packed + cur_exp * expert_b_stride + (int64_t)gn * stride_bn;
        const float* Bs_row = B_scale + cur_exp * expert_bs_stride + (int64_t)gn * stride_bsn;

        // For I=512, elems_per_lane=16. Process with uint64 (16 FP4)
        for (int k = my_k; k < my_k_end; k += 16) {
            float scale = Bs_row[k / 32];
            uint64_t packed8 = *reinterpret_cast<const uint64_t*>(B_row + k / 2);

            // Dequant to registers
            float w_vals[16];
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                uint8_t byte = (packed8 >> (i * 8)) & 0xFF;
                w_vals[i * 2]     = lut[byte & 0xF] * scale;
                w_vals[i * 2 + 1] = lut[(byte >> 4) & 0xF] * scale;
            }

            // Apply to each M row with fused SwiGLU
            for (int g = group_start; g < group_end; g++) {
                int ss = row_sorted_slot[g];
                const __nv_bfloat16* gate_row = intermediate + (int64_t)ss * (2 * I);
                const __nv_bfloat16* up_row = gate_row + I;

                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    int ki = k + i;
                    float g0 = __bfloat162float(gate_row[ki]);
                    float u0 = __bfloat162float(up_row[ki]);
                    if (do_clamp) {
                        g0 = fminf(g0, clamp_limit);
                        u0 = fmaxf(fminf(u0, clamp_limit), -clamp_limit);
                    }
                    acc[g] += g3_silu(g0) * u0 * w_vals[i];
                }
            }
        }
    }

    // Warp-shuffle reduction
    #pragma unroll
    for (int mi = 0; mi < GS3_BLOCK_M; mi++) {
        #pragma unroll
        for (int offset = 16; offset >= 1; offset >>= 1) {
            acc[mi] += __shfl_xor_sync(0xFFFFFFFF, acc[mi], offset);
        }
    }

    if (lane == 0) {
        for (int mi = 0; mi < valid_m; mi++) {
            C[row_sorted_slot[mi] * K_out + gn] = __float2bfloat16(acc[mi]);
        }
    }
}

inline void launch_grouped_fp4_swiglu_gemm2_v3(
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
    int total_m_tiles = (num_slots + GS3_BLOCK_M - 1) / GS3_BLOCK_M;
    dim3 grid(total_m_tiles, (K_out + GS3_BLOCK_N - 1) / GS3_BLOCK_N);
    dim3 block(GS3_THREADS);

    grouped_fp4_swiglu_gemm2_v3_kernel<<<grid, block, 0, stream>>>(
        intermediate, B_packed, B_scale, C,
        sorted_slot_ids, expert_offsets,
        K_out, I, num_experts, num_slots, clamp_limit,
        stride_bn, stride_bsn, expert_b_stride, expert_bs_stride);
}

}  // namespace grouped_v3
}  // namespace sm120
}  // namespace dsv4_kernel
