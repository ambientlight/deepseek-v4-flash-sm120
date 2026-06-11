// =====================================================================
// SM120 Grouped FP4 MoE GEMM1 v4 — SMEM Weight Tiling + Vectorized Load
//
// The correct SMEM approach: cooperative vectorized weight loading.
//
// Per CTA (4 warps = 4 N outputs):
//   K-loop (K/K_TILE iterations):
//     Phase 1 — LOAD: 32 lanes cooperatively load B[gn, k_tile] via uint4
//               Each lane: 1 uint4 = 32 FP4, dequant → 32 floats to SMEM
//               1 scale per 32 elements → 1 scale per lane. Clean.
//               __syncwarp() after load.
//     Phase 2 — COMPUTE: for each of BLOCK_M rows:
//               Load A[tok, k_tile] from GMEM (strided per-lane)
//               FMA with SMEM weight data
//               Warp-shuffle partial reduction within tile
//
// K_TILE = 1024: 32 lanes × 32 FP4/lane = 1024 elements
//   SMEM: 1024 × 4 bytes × 4 warps = 16 KB (fits easily)
//   K iterations: 4096/1024 = 4 (GEMM1), 512/512 = 1 (GEMM2)
//   Load: 1 uint4 per lane per tile (512 bytes total per warp)
//
// Weight BW: loaded once per expert group (BLOCK_M rows share)
//   vs v3 GEMV: BLOCK_M× fewer weight loads
//   At M_per_expert=28: 28× less weight bandwidth
// =====================================================================
#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace dsv4_kernel {
namespace sm120 {
namespace grouped_v4 {

static constexpr int G4_BLOCK_M = 16;
static constexpr int G4_BLOCK_N = 4;
static constexpr int G4_WARPS = 4;
static constexpr int G4_THREADS = G4_WARPS * 32;
static constexpr int G4_K_TILE = 1024;  // 32 lanes × 32 FP4/lane

__device__ __forceinline__ int g4_find_expert(
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

__global__ __launch_bounds__(128, 4)  // 16KB SMEM → 4 CTAs/SM
void grouped_fp4_moe_gemm1_v4_kernel(
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
    const int n_block = blockIdx.y * G4_BLOCK_N;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int gn = n_block + warp_id;

    if (gn >= N) return;

    const int m_start = m_tile * G4_BLOCK_M;

    // FP4 LUT in registers
    float lut[16];
    lut[0] = 0.0f; lut[1] = 0.5f; lut[2] = 1.0f; lut[3] = 1.5f;
    lut[4] = 2.0f; lut[5] = 3.0f; lut[6] = 4.0f; lut[7] = 6.0f;
    lut[8] = -0.0f; lut[9] = -0.5f; lut[10] = -1.0f; lut[11] = -1.5f;
    lut[12] = -2.0f; lut[13] = -3.0f; lut[14] = -4.0f; lut[15] = -6.0f;

    // SMEM: each warp caches its own weight K-tile
    // 4 warps × 1024 floats = 16 KB
    __shared__ float s_weight[G4_WARPS][G4_K_TILE];

    // Collect M rows for this tile
    int row_sorted_slot[G4_BLOCK_M];
    int row_tok[G4_BLOCK_M];
    int row_expert_id[G4_BLOCK_M];
    int valid_m = 0;

    for (int mi = 0; mi < G4_BLOCK_M; mi++) {
        int sg = m_start + mi;
        if (sg >= num_slots) break;
        row_sorted_slot[mi] = sorted_slot_ids[sg];
        row_tok[mi] = token_ids[row_sorted_slot[mi]];
        row_expert_id[mi] = g4_find_expert(expert_offsets, num_experts, sg);
        valid_m = mi + 1;
    }

    if (valid_m == 0) return;

    // Accumulators
    float acc[G4_BLOCK_M];
    #pragma unroll
    for (int mi = 0; mi < G4_BLOCK_M; mi++) acc[mi] = 0.0f;

    // K-loop
    for (int k_start = 0; k_start < K; k_start += G4_K_TILE) {
        const int k_tile_len = min(G4_K_TILE, K - k_start);

        // Process M rows grouped by expert
        int mi = 0;
        while (mi < valid_m) {
            int cur_exp = row_expert_id[mi];
            int group_start = mi;
            while (mi < valid_m && row_expert_id[mi] == cur_exp) mi++;
            int group_end = mi;

            // === PHASE 1: Cooperative vectorized weight load into SMEM ===
            // B_packed[expert, gn, K/2]: row gn, starting at k_start
            const uint8_t* b_row = B_packed + cur_exp * expert_b_stride
                                 + (int64_t)gn * stride_bn + k_start / 2;
            const float* bs_row = B_scale + cur_exp * expert_bs_stride
                                + (int64_t)gn * stride_bsn;

            // Each lane loads 32 contiguous FP4 (= 1 scale group) via uint4
            // Lane i handles elements [i*32, i*32+32) within k_tile
            {
                const int my_k_in_tile = lane * 32;  // 0, 32, 64, ..., 992
                if (my_k_in_tile < k_tile_len) {
                    const int global_k = k_start + my_k_in_tile;
                    const float scale = bs_row[global_k / 32];

                    // Load 16 bytes = 32 FP4 packed
                    uint4 packed = *reinterpret_cast<const uint4*>(b_row + my_k_in_tile / 2);
                    uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};

                    // Dequant 32 FP4 → 32 floats with scale, store to SMEM
                    #pragma unroll
                    for (int w = 0; w < 4; w++) {
                        uint32_t word = words[w];
                        #pragma unroll
                        for (int b = 0; b < 4; b++) {
                            uint8_t byte_val = (word >> (b * 8)) & 0xFF;
                            int idx = my_k_in_tile + w * 8 + b * 2;
                            s_weight[warp_id][idx]     = lut[byte_val & 0xF] * scale;
                            s_weight[warp_id][idx + 1] = lut[(byte_val >> 4) & 0xF] * scale;
                        }
                    }
                }
            }
            // Ensure all lanes in warp have finished writing SMEM
            __syncwarp();

            // === PHASE 2: Compute A × s_weight for each M row in group ===
            for (int g = group_start; g < group_end; g++) {
                const __nv_bfloat16* A_row = A + (int64_t)row_tok[g] * K + k_start;
                float local_acc = 0.0f;

                // Each lane processes every 32nd element (strided access)
                // This gives coalesced SMEM reads (consecutive lanes → consecutive banks)
                for (int ki = lane; ki < k_tile_len; ki += 32) {
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
            __syncwarp();  // Ensure compute done before next expert's SMEM overwrite
        }
    }

    // Write outputs
    if (lane == 0) {
        for (int mi = 0; mi < valid_m; mi++) {
            C[row_sorted_slot[mi] * N + gn] = __float2bfloat16(acc[mi]);
        }
    }
}

inline void launch_grouped_fp4_moe_gemm1_v4(
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
    int total_m_tiles = (num_slots + G4_BLOCK_M - 1) / G4_BLOCK_M;
    dim3 grid(total_m_tiles, (N + G4_BLOCK_N - 1) / G4_BLOCK_N);
    dim3 block(G4_THREADS);

    grouped_fp4_moe_gemm1_v4_kernel<<<grid, block, 0, stream>>>(
        A, B_packed, B_scale, C,
        sorted_slot_ids, token_ids, expert_offsets,
        N, K, num_experts, num_slots,
        stride_bn, stride_bsn, expert_b_stride, expert_bs_stride);
}

}  // namespace grouped_v4
}  // namespace sm120
}  // namespace dsv4_kernel
