// =====================================================================
// SM120 Grouped FP4 SwiGLU+GEMM2 v2 — SMEM Weight Caching
//
// Same SMEM approach as GEMM1 v2: weight tile cached, M rows share it.
// SwiGLU fused: reads gate[k] and up[k] from intermediate, computes
// silu(gate)*up per element before multiplying with cached weight.
// =====================================================================
#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>

namespace dsv4_kernel {
namespace sm120 {
namespace grouped_v2 {

// Reuse g2_find_expert from grouped_fp4_moe_gemm_v2.cuh (same TU)

static constexpr int GS2v2_BLOCK_M = 16;
static constexpr int GS2v2_BLOCK_N = 4;
static constexpr int GS2v2_WARPS = 4;
static constexpr int GS2v2_THREADS = GS2v2_WARPS * 32;
// K_TILE for GEMM2: I=512, use 128 (4 tiles) or 64 (8 tiles)
static constexpr int GS2v2_K_TILE = 64;

__device__ __forceinline__ float g2_silu(float x) {
    return x / (1.0f + expf(-x));
}

__global__ __launch_bounds__(128, 4)
void grouped_fp4_swiglu_gemm2_v2_kernel(
    const __nv_bfloat16* __restrict__ intermediate,  // [num_slots, 2*I]
    const uint8_t* __restrict__ B_packed,             // [E, K_out, I/2]
    const float* __restrict__ B_scale,                // [E, K_out, I/32]
    __nv_bfloat16* __restrict__ C,                    // [num_slots, K_out]
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
    const int n_block = blockIdx.y * GS2v2_BLOCK_N;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int gn = n_block + warp_id;

    if (gn >= K_out) return;

    const int m_start = m_tile * GS2v2_BLOCK_M;
    const bool do_clamp = (clamp_limit > 0.0f);

    float lut[16];
    lut[0] = 0.0f; lut[1] = 0.5f; lut[2] = 1.0f; lut[3] = 1.5f;
    lut[4] = 2.0f; lut[5] = 3.0f; lut[6] = 4.0f; lut[7] = 6.0f;
    lut[8] = -0.0f; lut[9] = -0.5f; lut[10] = -1.0f; lut[11] = -1.5f;
    lut[12] = -2.0f; lut[13] = -3.0f; lut[14] = -4.0f; lut[15] = -6.0f;

    __shared__ float s_weight[GS2v2_WARPS][GS2v2_K_TILE];

    // Collect M rows
    int row_expert[GS2v2_BLOCK_M];
    int row_sorted_slot[GS2v2_BLOCK_M];
    int valid_m = 0;

    for (int mi = 0; mi < GS2v2_BLOCK_M; mi++) {
        int sg = m_start + mi;
        if (sg >= num_slots) break;
        row_expert[mi] = g2_find_expert(expert_offsets, num_experts, sg);
        row_sorted_slot[mi] = sorted_slot_ids[sg];
        valid_m = mi + 1;
    }

    if (valid_m == 0) return;

    float acc[GS2v2_BLOCK_M];
    #pragma unroll
    for (int mi = 0; mi < GS2v2_BLOCK_M; mi++) acc[mi] = 0.0f;

    // K-loop over I dimension
    for (int k_start = 0; k_start < I; k_start += GS2v2_K_TILE) {
        const int k_len = min(GS2v2_K_TILE, I - k_start);

        int mi = 0;
        while (mi < valid_m) {
            int cur_exp = row_expert[mi];
            int group_start = mi;
            while (mi < valid_m && row_expert[mi] == cur_exp) mi++;
            int group_end = mi;

            // Load weight tile into SMEM
            const uint8_t* b_base = B_packed + cur_exp * expert_b_stride
                                  + (int64_t)gn * stride_bn + k_start / 2;
            const float* bs_base = B_scale + cur_exp * expert_bs_stride
                                 + (int64_t)gn * stride_bsn;

            for (int ki = lane; ki < k_len; ki += 32) {
                int global_k = k_start + ki;
                int byte_off = ki / 2;
                uint8_t packed_byte = b_base[byte_off];
                uint8_t nibble = (ki & 1) ? ((packed_byte >> 4) & 0xF) : (packed_byte & 0xF);
                float scale = bs_base[global_k / 32];
                s_weight[warp_id][ki] = lut[nibble] * scale;
            }
            __syncthreads();

            // Compute: for each M row, apply SwiGLU then multiply with cached weight
            for (int g = group_start; g < group_end; g++) {
                int ss = row_sorted_slot[g];
                const __nv_bfloat16* gate_row = intermediate + (int64_t)ss * (2 * I) + k_start;
                const __nv_bfloat16* up_row = gate_row + I;
                float local_acc = 0.0f;

                for (int ki = lane; ki < k_len; ki += 32) {
                    float g0 = __bfloat162float(gate_row[ki]);
                    float u0 = __bfloat162float(up_row[ki]);

                    if (do_clamp) {
                        g0 = fminf(g0, clamp_limit);
                        u0 = fmaxf(fminf(u0, clamp_limit), -clamp_limit);
                    }

                    float x = g2_silu(g0) * u0;
                    local_acc += x * s_weight[warp_id][ki];
                }

                #pragma unroll
                for (int offset = 16; offset >= 1; offset >>= 1) {
                    local_acc += __shfl_xor_sync(0xFFFFFFFF, local_acc, offset);
                }

                if (lane == 0) {
                    acc[g] += local_acc;
                }
            }
            __syncthreads();
        }
    }

    if (lane == 0) {
        for (int mi = 0; mi < valid_m; mi++) {
            C[row_sorted_slot[mi] * K_out + gn] = __float2bfloat16(acc[mi]);
        }
    }
}

inline void launch_grouped_fp4_swiglu_gemm2_v2(
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
    int total_m_tiles = (num_slots + GS2v2_BLOCK_M - 1) / GS2v2_BLOCK_M;
    dim3 grid(total_m_tiles, (K_out + GS2v2_BLOCK_N - 1) / GS2v2_BLOCK_N);
    dim3 block(GS2v2_THREADS);

    grouped_fp4_swiglu_gemm2_v2_kernel<<<grid, block, 0, stream>>>(
        intermediate, B_packed, B_scale, C,
        sorted_slot_ids, expert_offsets,
        K_out, I, num_experts, num_slots, clamp_limit,
        stride_bn, stride_bsn, expert_b_stride, expert_bs_stride);
}

}  // namespace grouped_v2
}  // namespace sm120
}  // namespace dsv4_kernel
