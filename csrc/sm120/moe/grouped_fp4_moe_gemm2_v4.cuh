// =====================================================================
// SM120 Grouped FP4 SwiGLU+GEMM2 v4 — SMEM Weight Tiling
//
// Same SMEM approach as GEMM1 v4.
// K_TILE = 512 = I (entire K in one tile for GEMM2)
//   32 lanes × 16 FP4/lane = 512. Each lane loads 8 bytes (uint64).
//   SMEM: 512 × 4 × 4 warps = 8 KB
// SwiGLU fused: gate/up loaded from intermediate per M row.
// =====================================================================
#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>

namespace dsv4_kernel {
namespace sm120 {
namespace grouped_v4 {

// g4_find_expert already defined in grouped_fp4_moe_gemm_v4.cuh

static constexpr int GS4_BLOCK_M = 16;
static constexpr int GS4_BLOCK_N = 4;
static constexpr int GS4_WARPS = 4;
static constexpr int GS4_THREADS = GS4_WARPS * 32;
static constexpr int GS4_K_TILE = 512;  // Entire I=512 in one tile

__device__ __forceinline__ float g4_silu(float x) {
    return x / (1.0f + expf(-x));
}

__global__ __launch_bounds__(128, 4)
void grouped_fp4_swiglu_gemm2_v4_kernel(
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
    const int n_block = blockIdx.y * GS4_BLOCK_N;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int gn = n_block + warp_id;

    if (gn >= K_out) return;

    const int m_start = m_tile * GS4_BLOCK_M;
    const bool do_clamp = (clamp_limit > 0.0f);

    float lut[16];
    lut[0] = 0.0f; lut[1] = 0.5f; lut[2] = 1.0f; lut[3] = 1.5f;
    lut[4] = 2.0f; lut[5] = 3.0f; lut[6] = 4.0f; lut[7] = 6.0f;
    lut[8] = -0.0f; lut[9] = -0.5f; lut[10] = -1.0f; lut[11] = -1.5f;
    lut[12] = -2.0f; lut[13] = -3.0f; lut[14] = -4.0f; lut[15] = -6.0f;

    // SMEM: each warp caches I=512 weight floats
    // 4 warps × 512 × 4 = 8 KB
    __shared__ float s_weight[GS4_WARPS][GS4_K_TILE];

    int row_sorted_slot[GS4_BLOCK_M];
    int row_expert_id[GS4_BLOCK_M];
    int valid_m = 0;

    for (int mi = 0; mi < GS4_BLOCK_M; mi++) {
        int sg = m_start + mi;
        if (sg >= num_slots) break;
        row_sorted_slot[mi] = sorted_slot_ids[sg];
        row_expert_id[mi] = g4_find_expert(expert_offsets, num_experts, sg);
        valid_m = mi + 1;
    }

    if (valid_m == 0) return;

    float acc[GS4_BLOCK_M];
    #pragma unroll
    for (int mi = 0; mi < GS4_BLOCK_M; mi++) acc[mi] = 0.0f;

    // For GEMM2: I=512, one K-tile covers everything
    // But support general case with K-loop
    for (int k_start = 0; k_start < I; k_start += GS4_K_TILE) {
        const int k_tile_len = min(GS4_K_TILE, I - k_start);

        int mi = 0;
        while (mi < valid_m) {
            int cur_exp = row_expert_id[mi];
            int group_start = mi;
            while (mi < valid_m && row_expert_id[mi] == cur_exp) mi++;
            int group_end = mi;

            // === PHASE 1: Load weight tile ===
            const uint8_t* b_row = B_packed + cur_exp * expert_b_stride
                                 + (int64_t)gn * stride_bn + k_start / 2;
            const float* bs_row = B_scale + cur_exp * expert_bs_stride
                                + (int64_t)gn * stride_bsn;

            // For I=512: 32 lanes, each loads 16 FP4 (= 8 bytes = uint64)
            // lane i handles elements [i*16, i*16+16)
            {
                const int my_k_in_tile = lane * 16;
                if (my_k_in_tile < k_tile_len) {
                    const int global_k = k_start + my_k_in_tile;
                    const float scale = bs_row[global_k / 32];

                    uint64_t packed = *reinterpret_cast<const uint64_t*>(b_row + my_k_in_tile / 2);

                    #pragma unroll
                    for (int i = 0; i < 8; i++) {
                        uint8_t byte_val = (packed >> (i * 8)) & 0xFF;
                        int idx = my_k_in_tile + i * 2;
                        s_weight[warp_id][idx]     = lut[byte_val & 0xF] * scale;
                        s_weight[warp_id][idx + 1] = lut[(byte_val >> 4) & 0xF] * scale;
                    }
                }
            }
            __syncwarp();

            // === PHASE 2: Compute with fused SwiGLU ===
            for (int g = group_start; g < group_end; g++) {
                int ss = row_sorted_slot[g];
                const __nv_bfloat16* gate_row = intermediate + (int64_t)ss * (2 * I) + k_start;
                const __nv_bfloat16* up_row = gate_row + I;
                float local_acc = 0.0f;

                for (int ki = lane; ki < k_tile_len; ki += 32) {
                    float g0 = __bfloat162float(gate_row[ki]);
                    float u0 = __bfloat162float(up_row[ki]);
                    if (do_clamp) {
                        g0 = fminf(g0, clamp_limit);
                        u0 = fmaxf(fminf(u0, clamp_limit), -clamp_limit);
                    }
                    float x = g4_silu(g0) * u0;
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
            __syncwarp();
        }
    }

    if (lane == 0) {
        for (int mi = 0; mi < valid_m; mi++) {
            C[row_sorted_slot[mi] * K_out + gn] = __float2bfloat16(acc[mi]);
        }
    }
}

inline void launch_grouped_fp4_swiglu_gemm2_v4(
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
    int total_m_tiles = (num_slots + GS4_BLOCK_M - 1) / GS4_BLOCK_M;
    dim3 grid(total_m_tiles, (K_out + GS4_BLOCK_N - 1) / GS4_BLOCK_N);
    dim3 block(GS4_THREADS);

    grouped_fp4_swiglu_gemm2_v4_kernel<<<grid, block, 0, stream>>>(
        intermediate, B_packed, B_scale, C,
        sorted_slot_ids, expert_offsets,
        K_out, I, num_experts, num_slots, clamp_limit,
        stride_bn, stride_bsn, expert_b_stride, expert_bs_stride);
}

}  // namespace grouped_v4
}  // namespace sm120
}  // namespace dsv4_kernel
