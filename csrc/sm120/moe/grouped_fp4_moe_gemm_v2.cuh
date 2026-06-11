// =====================================================================
// SM120 Grouped FP4 MoE GEMM1 v2 — SMEM Weight Caching
//
// Key optimization: load weight K-tile into SMEM as dequanted float32,
// then ALL M rows read from SMEM instead of GMEM.
//
// Weight BW savings: BLOCK_M× less weight loads from GMEM.
// At M=294 with 64 experts (~4.6 tokens/expert avg): ~4.6× BW savings.
//
// Structure:
//   Grid: (total_m_tiles, ceil(N / BLOCK_N))
//   Block: 128 threads = 4 warps, each warp = 1 N output
//   SMEM: [4 warps][K_TILE] float = 4 × 128 × 4 = 2 KB
//
//   K-loop per CTA:
//     1. Cooperative dequant: B_packed[expert, gn, k_tile] → s_weight[warp][K_TILE]
//     2. __syncthreads()
//     3. For each of BLOCK_M rows: A[tok, k_tile] × s_weight → acc
//     4. __syncthreads()
//   Write acc to C
// =====================================================================
#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace dsv4_kernel {
namespace sm120 {
namespace grouped_v2 {

static constexpr int G2_BLOCK_M = 16;
static constexpr int G2_BLOCK_N = 4;
static constexpr int G2_WARPS = 4;
static constexpr int G2_THREADS = G2_WARPS * 32;
static constexpr int G2_K_TILE = 128;

__device__ __forceinline__ int g2_find_expert(
    const int32_t* __restrict__ expert_offsets,
    int num_experts,
    int global_slot_idx)
{
    int lo = 0, hi = num_experts - 1;
    while (lo < hi) {
        int mid = (lo + hi + 1) / 2;
        if (expert_offsets[mid] <= global_slot_idx)
            lo = mid;
        else
            hi = mid - 1;
    }
    return lo;
}

__global__ __launch_bounds__(128, 4)
void grouped_fp4_moe_gemm1_v2_kernel(
    const __nv_bfloat16* __restrict__ A,
    const uint8_t* __restrict__ B_packed,
    const float* __restrict__ B_scale,
    __nv_bfloat16* __restrict__ C,
    const int32_t* __restrict__ sorted_slot_ids,
    const int32_t* __restrict__ token_ids,
    const int32_t* __restrict__ expert_offsets,
    int N, int K,
    int num_experts,
    int num_slots,
    int stride_bn,
    int stride_bsn,
    int64_t expert_b_stride,
    int64_t expert_bs_stride)
{
    const int m_tile = blockIdx.x;
    const int n_block = blockIdx.y * G2_BLOCK_N;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int gn = n_block + warp_id;

    if (gn >= N) return;

    const int m_start = m_tile * G2_BLOCK_M;

    // FP4 LUT
    float lut[16];
    lut[0] = 0.0f; lut[1] = 0.5f; lut[2] = 1.0f; lut[3] = 1.5f;
    lut[4] = 2.0f; lut[5] = 3.0f; lut[6] = 4.0f; lut[7] = 6.0f;
    lut[8] = -0.0f; lut[9] = -0.5f; lut[10] = -1.0f; lut[11] = -1.5f;
    lut[12] = -2.0f; lut[13] = -3.0f; lut[14] = -4.0f; lut[15] = -6.0f;

    // SMEM for dequanted weights: each warp caches its own N row's K-tile
    __shared__ float s_weight[G2_WARPS][G2_K_TILE];

    // Collect M rows info
    int row_expert[G2_BLOCK_M];
    int row_sorted_slot[G2_BLOCK_M];
    int row_tok[G2_BLOCK_M];
    int valid_m = 0;

    for (int mi = 0; mi < G2_BLOCK_M; mi++) {
        int sg = m_start + mi;
        if (sg >= num_slots) break;
        row_expert[mi] = g2_find_expert(expert_offsets, num_experts, sg);
        row_sorted_slot[mi] = sorted_slot_ids[sg];
        row_tok[mi] = token_ids[row_sorted_slot[mi]];
        valid_m = mi + 1;
    }

    if (valid_m == 0) return;

    // Accumulators
    float acc[G2_BLOCK_M];
    #pragma unroll
    for (int mi = 0; mi < G2_BLOCK_M; mi++) acc[mi] = 0.0f;

    // K-loop
    for (int k_start = 0; k_start < K; k_start += G2_K_TILE) {
        const int k_len = min(G2_K_TILE, K - k_start);

        // Process M rows grouped by expert
        int mi = 0;
        while (mi < valid_m) {
            int cur_exp = row_expert[mi];
            int group_start = mi;
            while (mi < valid_m && row_expert[mi] == cur_exp) mi++;
            int group_end = mi;

            // === PHASE 1: Load weight tile into SMEM ===
            // B_packed layout: [E, N, K/2] — row gn of expert cur_exp
            const uint8_t* b_base = B_packed + cur_exp * expert_b_stride
                                  + (int64_t)gn * stride_bn + k_start / 2;
            const float* bs_base = B_scale + cur_exp * expert_bs_stride
                                 + (int64_t)gn * stride_bsn;

            // Each of 32 lanes loads ceil(k_len/32) elements
            // Decode FP4 → float with scale
            for (int ki = lane; ki < k_len; ki += 32) {
                int global_k = k_start + ki;
                int byte_off = ki / 2;
                uint8_t packed_byte = b_base[byte_off];
                uint8_t nibble = (ki & 1) ? ((packed_byte >> 4) & 0xF) : (packed_byte & 0xF);
                float scale = bs_base[global_k / 32];
                s_weight[warp_id][ki] = lut[nibble] * scale;
            }
            __syncthreads();

            // === PHASE 2: Compute A × s_weight for each M row ===
            for (int g = group_start; g < group_end; g++) {
                const __nv_bfloat16* A_row = A + (int64_t)row_tok[g] * K + k_start;
                float local_acc = 0.0f;

                // Strided access: each lane handles every 32nd element
                for (int ki = lane; ki < k_len; ki += 32) {
                    local_acc += __bfloat162float(A_row[ki]) * s_weight[warp_id][ki];
                }

                // Warp-shuffle reduction
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

    // Write outputs
    if (lane == 0) {
        for (int mi = 0; mi < valid_m; mi++) {
            C[row_sorted_slot[mi] * N + gn] = __float2bfloat16(acc[mi]);
        }
    }
}

inline void launch_grouped_fp4_moe_gemm1_v2(
    const __nv_bfloat16* A,
    const uint8_t* B_packed,
    const float* B_scale,
    __nv_bfloat16* C,
    const int32_t* sorted_slot_ids,
    const int32_t* token_ids,
    const int32_t* expert_offsets,
    int num_slots, int N, int K,
    int num_experts,
    int stride_bn, int stride_bsn,
    int64_t expert_b_stride, int64_t expert_bs_stride,
    cudaStream_t stream)
{
    int total_m_tiles = (num_slots + G2_BLOCK_M - 1) / G2_BLOCK_M;
    dim3 grid(total_m_tiles, (N + G2_BLOCK_N - 1) / G2_BLOCK_N);
    dim3 block(G2_THREADS);

    grouped_fp4_moe_gemm1_v2_kernel<<<grid, block, 0, stream>>>(
        A, B_packed, B_scale, C,
        sorted_slot_ids, token_ids, expert_offsets,
        N, K, num_experts, num_slots,
        stride_bn, stride_bsn, expert_b_stride, expert_bs_stride);
}

}  // namespace grouped_v2
}  // namespace sm120
}  // namespace dsv4_kernel
