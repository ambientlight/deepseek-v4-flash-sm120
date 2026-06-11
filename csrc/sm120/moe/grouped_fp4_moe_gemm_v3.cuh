// =====================================================================
// SM120 Grouped FP4 MoE GEMM1 v3 — Register Blocking + Weight Reuse
//
// Key insight: keep v3's fast uint4 vectorized weight loading, but
// process multiple M rows per weight load. Each lane loads the weight
// K-chunk ONCE, then multiplies against multiple activation rows.
//
// Design:
//   Grid X: total M-tiles (num_slots / BLOCK_M)
//   Grid Y: N-tiles (ceil(N / BLOCK_N))
//   Each warp: 1 N output, all 32 lanes on K
//   Per K-chunk: load weight → loop over M rows → accumulate
//
// vs v3 GEMV: M× fewer weight loads (weight stays in registers)
// vs grouped v2: no SMEM, no syncthreads, vectorized loads preserved
// =====================================================================
#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace dsv4_kernel {
namespace sm120 {
namespace grouped_v3 {

static constexpr int G3_BLOCK_M = 8;    // M rows per CTA (in registers)
static constexpr int G3_BLOCK_N = 4;
static constexpr int G3_WARPS = 4;
static constexpr int G3_THREADS = G3_WARPS * 32;

__device__ __forceinline__ int g3_find_expert(
    const int32_t* __restrict__ expert_offsets,
    int num_experts,
    int idx)
{
    int lo = 0, hi = num_experts - 1;
    while (lo < hi) {
        int mid = (lo + hi + 1) / 2;
        if (expert_offsets[mid] <= idx)
            lo = mid;
        else
            hi = mid - 1;
    }
    return lo;
}

__global__ __launch_bounds__(128, 6)
void grouped_fp4_moe_gemm1_v3_kernel(
    const __nv_bfloat16* __restrict__ A,
    const uint8_t* __restrict__ B_packed,     // [E, N, K/2]
    const float* __restrict__ B_scale,        // [E, N, K/32]
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
    const int n_block = blockIdx.y * G3_BLOCK_N;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int gn = n_block + warp_id;

    if (gn >= N) return;

    const int m_start = m_tile * G3_BLOCK_M;

    // FP4 LUT
    float lut[16];
    lut[0] = 0.0f; lut[1] = 0.5f; lut[2] = 1.0f; lut[3] = 1.5f;
    lut[4] = 2.0f; lut[5] = 3.0f; lut[6] = 4.0f; lut[7] = 6.0f;
    lut[8] = -0.0f; lut[9] = -0.5f; lut[10] = -1.0f; lut[11] = -1.5f;
    lut[12] = -2.0f; lut[13] = -3.0f; lut[14] = -4.0f; lut[15] = -6.0f;

    // Collect valid M rows
    int row_sorted_slot[G3_BLOCK_M];
    int row_tok[G3_BLOCK_M];
    int row_expert[G3_BLOCK_M];
    int valid_m = 0;

    for (int mi = 0; mi < G3_BLOCK_M; mi++) {
        int sg = m_start + mi;
        if (sg >= num_slots) break;
        row_sorted_slot[mi] = sorted_slot_ids[sg];
        row_tok[mi] = token_ids[row_sorted_slot[mi]];
        row_expert[mi] = g3_find_expert(expert_offsets, num_experts, sg);
        valid_m = mi + 1;
    }

    if (valid_m == 0) return;

    // Accumulators in registers — one per M row
    float acc[G3_BLOCK_M];
    #pragma unroll
    for (int mi = 0; mi < G3_BLOCK_M; mi++) acc[mi] = 0.0f;

    // K-loop: each lane handles K/32 contiguous elements
    const int elems_per_lane = K / 32;
    const int my_k = lane * elems_per_lane;
    const int my_k_end = my_k + elems_per_lane;

    // Process M rows grouped by expert within this tile
    int mi = 0;
    while (mi < valid_m) {
        int cur_exp = row_expert[mi];
        int group_start = mi;
        while (mi < valid_m && row_expert[mi] == cur_exp) mi++;
        int group_end = mi;

        // Weight pointers for this expert (ONE load for all rows in group)
        const uint8_t* B_row = B_packed + cur_exp * expert_b_stride + (int64_t)gn * stride_bn;
        const float* Bs_row = B_scale + cur_exp * expert_bs_stride + (int64_t)gn * stride_bsn;

        // Main K-loop: load weight ONCE in registers, multiply against all M rows
        int k = my_k;

        // uint4 path: 32 FP4 per iteration
        for (; k + 32 <= my_k_end; k += 32) {
            float scale = Bs_row[k / 32];
            uint4 packed16 = *reinterpret_cast<const uint4*>(B_row + k / 2);
            uint32_t words[4] = {packed16.x, packed16.y, packed16.z, packed16.w};

            // Dequant weight bytes into registers (8 floats from 4 bytes)
            float w_vals[32];
            #pragma unroll
            for (int wi = 0; wi < 4; wi++) {
                uint32_t word = words[wi];
                #pragma unroll
                for (int bi = 0; bi < 4; bi++) {
                    uint8_t byte = (word >> (bi * 8)) & 0xFF;
                    w_vals[wi * 8 + bi * 2]     = lut[byte & 0xF] * scale;
                    w_vals[wi * 8 + bi * 2 + 1] = lut[(byte >> 4) & 0xF] * scale;
                }
            }

            // Now multiply with each M row's activation
            for (int g = group_start; g < group_end; g++) {
                const __nv_bfloat16* A_row = A + (int64_t)row_tok[g] * K;

                #pragma unroll
                for (int wi = 0; wi < 4; wi++) {
                    uint4 a_packed = *reinterpret_cast<const uint4*>(&A_row[k + wi * 8]);
                    const __nv_bfloat16* a_ptr = reinterpret_cast<const __nv_bfloat16*>(&a_packed);

                    #pragma unroll
                    for (int bi = 0; bi < 8; bi++) {
                        acc[g] += __bfloat162float(a_ptr[bi]) * w_vals[wi * 8 + bi];
                    }
                }
            }
        }

        // uint64 remainder (16 FP4)
        for (; k + 16 <= my_k_end; k += 16) {
            float scale = Bs_row[k / 32];
            uint64_t packed8 = *reinterpret_cast<const uint64_t*>(B_row + k / 2);

            float w_vals[16];
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                uint8_t byte = (packed8 >> (i * 8)) & 0xFF;
                w_vals[i * 2]     = lut[byte & 0xF] * scale;
                w_vals[i * 2 + 1] = lut[(byte >> 4) & 0xF] * scale;
            }

            for (int g = group_start; g < group_end; g++) {
                const __nv_bfloat16* A_row = A + (int64_t)row_tok[g] * K;
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    acc[g] += __bfloat162float(A_row[k + i]) * w_vals[i];
                }
            }
        }

        // uint32 tail (8 FP4)
        for (; k + 8 <= my_k_end; k += 8) {
            float scale = Bs_row[k / 32];
            uint32_t packed4 = *reinterpret_cast<const uint32_t*>(B_row + k / 2);

            float w_vals[8];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                uint8_t byte = (packed4 >> (i * 8)) & 0xFF;
                w_vals[i * 2]     = lut[byte & 0xF] * scale;
                w_vals[i * 2 + 1] = lut[(byte >> 4) & 0xF] * scale;
            }

            for (int g = group_start; g < group_end; g++) {
                const __nv_bfloat16* A_row = A + (int64_t)row_tok[g] * K;
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    acc[g] += __bfloat162float(A_row[k + i]) * w_vals[i];
                }
            }
        }
    }

    // Warp-shuffle reduction for all M rows
    #pragma unroll
    for (int mi = 0; mi < G3_BLOCK_M; mi++) {
        #pragma unroll
        for (int offset = 16; offset >= 1; offset >>= 1) {
            acc[mi] += __shfl_xor_sync(0xFFFFFFFF, acc[mi], offset);
        }
    }

    // Write outputs
    if (lane == 0) {
        for (int mi = 0; mi < valid_m; mi++) {
            C[row_sorted_slot[mi] * N + gn] = __float2bfloat16(acc[mi]);
        }
    }
}

inline void launch_grouped_fp4_moe_gemm1_v3(
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
    int total_m_tiles = (num_slots + G3_BLOCK_M - 1) / G3_BLOCK_M;
    dim3 grid(total_m_tiles, (N + G3_BLOCK_N - 1) / G3_BLOCK_N);
    dim3 block(G3_THREADS);

    grouped_fp4_moe_gemm1_v3_kernel<<<grid, block, 0, stream>>>(
        A, B_packed, B_scale, C,
        sorted_slot_ids, token_ids, expert_offsets,
        N, K, num_experts, num_slots,
        stride_bn, stride_bsn, expert_b_stride, expert_bs_stride);
}

}  // namespace grouped_v3
}  // namespace sm120
}  // namespace dsv4_kernel
