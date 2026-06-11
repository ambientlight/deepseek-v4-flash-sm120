// =====================================================================
// SM120 BF16-Activation × FP4-Weight MoE GEMM (bf16 HMMA)
//
// For each (token, expert) slot:
//   C[slot, :] = A_bf16[token] @ dequant(B_fp4[expert]).T
//
// Dequant is done in-register during SMEM load:
//   bf16_val = __float2bfloat16(FP4_LUT[nibble] * scale_f32)
//
// Uses mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
// No activation quantization. No weight repacking. Zero extra memory.
//
// Grid: (num_slots, ceil(N / BLOCK_N))
// Block: 128 threads (4 warps)
// =====================================================================
#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace dsv4_kernel {
namespace sm120 {

// FP4 E2M1 dequant lookup table (16 entries, in constant memory)
__device__ __constant__ float FP4_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// Tile sizes
static constexpr int MOE2_MMA_M = 16;
static constexpr int MOE2_MMA_N = 8;
static constexpr int MOE2_MMA_K = 16;    // bf16 HMMA k=16
static constexpr int MOE2_BLOCK_N = 64;  // 8 N-tiles per CTA
static constexpr int MOE2_K_TILE = 64;   // Load 64 K elements per stage
static constexpr int MOE2_NUM_WARPS = 4;
static constexpr int MOE2_BLOCK_SIZE = MOE2_NUM_WARPS * 32;  // 128 threads
static constexpr int MOE2_N_TILES = MOE2_BLOCK_N / MOE2_MMA_N;  // 8
static constexpr int MOE2_MMA_PER_K = MOE2_K_TILE / MOE2_MMA_K; // 4

// SMEM: dequanted bf16 tiles
// A: [16, K_TILE] bf16 = 16 × 64 × 2 = 2 KB (only row 0 populated for M=1)
// B: [BLOCK_N, K_TILE] bf16 = 64 × 64 × 2 = 8 KB (dequanted from FP4)
// Total: ~10 KB — fits trivially in 99 KB
struct Moe2Smem {
    __nv_bfloat16 sA[MOE2_MMA_M][MOE2_K_TILE];     // 2 KB
    __nv_bfloat16 sB[MOE2_BLOCK_N][MOE2_K_TILE];    // 8 KB
};  // ~10 KB

__device__ __forceinline__ __nv_bfloat16 fp4_to_bf16(uint8_t nibble, float scale) {
    return __float2bfloat16(FP4_LUT[nibble & 0xF] * scale);
}

__global__ void fp4_moe_bf16_gemm_kernel(
    const __nv_bfloat16* __restrict__ A,    // [M_total, K] bf16
    const uint8_t* __restrict__ B_packed,   // [E, N, K/2] packed FP4
    const float* __restrict__ B_scale,      // [E, N, K/32] float32
    __nv_bfloat16* __restrict__ C,          // [num_slots, N] bf16
    const int* __restrict__ token_ids,      // [num_slots] int32
    const int* __restrict__ expert_ids,     // [num_slots] int32
    int N, int K,
    int stride_bn,           // B packed stride for N: K/2
    int stride_bsn,          // B scale stride for N: K/32
    int64_t expert_b_stride, // B stride between experts: N * K/2
    int64_t expert_bs_stride)// B_scale stride between experts: N * K/32
{
    extern __shared__ char _smem[];
    Moe2Smem &smem = *reinterpret_cast<Moe2Smem*>(_smem);

    const int slot = blockIdx.x;
    const int n_block = blockIdx.y * MOE2_BLOCK_N;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid % 32;
    const int g = lane >> 2;   // row in MMA (0-7)
    const int t = lane & 3;   // col group in MMA

    // Look up this slot's token and expert
    const int tok = token_ids[slot];
    const int exp = expert_ids[slot];

    // Base pointers
    const __nv_bfloat16* A_row = A + tok * K;
    const uint8_t* B_exp = B_packed + exp * expert_b_stride;
    const float* Bs_exp = B_scale + exp * expert_bs_stride;

    // Accumulators: each warp handles N_TILES / NUM_WARPS = 2 N-tiles
    static constexpr int N_PER_WARP = MOE2_N_TILES / MOE2_NUM_WARPS;  // 2
    float acc[N_PER_WARP][4];
    #pragma unroll
    for (int i = 0; i < N_PER_WARP; i++)
        acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;

    // K-tile loop
    for (int k_start = 0; k_start < K; k_start += MOE2_K_TILE) {

        // ---- Load A tile: [16, K_TILE] bf16 from activation ----
        // For M=1: only row 0 has data, rest zero
        for (int idx = tid; idx < MOE2_MMA_M * MOE2_K_TILE; idx += MOE2_BLOCK_SIZE) {
            int m = idx / MOE2_K_TILE;
            int k = idx % MOE2_K_TILE;
            int gk = k_start + k;
            smem.sA[m][k] = (m == 0 && gk < K) ? A_row[gk] : __float2bfloat16(0.0f);
        }

        // ---- Load B tile: [BLOCK_N, K_TILE] bf16 (dequant from FP4) ----
        // Each thread loads and dequants multiple (n, k) elements
        for (int idx = tid; idx < MOE2_BLOCK_N * MOE2_K_TILE; idx += MOE2_BLOCK_SIZE) {
            int n = idx / MOE2_K_TILE;
            int k = idx % MOE2_K_TILE;
            int gn = n_block + n;
            int gk = k_start + k;

            if (gn < N && gk < K) {
                // Packed byte containing two FP4 values
                int byte_idx = gk / 2;
                uint8_t packed = B_exp[gn * stride_bn + byte_idx];
                uint8_t nibble = (gk & 1) ? (packed >> 4) : (packed & 0xF);

                // Scale for this 32-element group
                int scale_idx = gk / 32;
                float scale = Bs_exp[gn * stride_bsn + scale_idx];

                smem.sB[n][k] = fp4_to_bf16(nibble, scale);
            } else {
                smem.sB[n][k] = __float2bfloat16(0.0f);
            }
        }

        __syncthreads();

        // ---- Compute: bf16 HMMA m16n8k16 ----
        #pragma unroll
        for (int kk = 0; kk < MOE2_MMA_PER_K; kk++) {
            int k_base = kk * MOE2_MMA_K;

            // Load A fragment from SMEM (m16n8k16 layout)
            // A: [16, 16] bf16 → 4 × uint32 (a0,a1,a2,a3)
            unsigned int a0 = *reinterpret_cast<const unsigned int*>(
                &smem.sA[g][k_base + 2*t]);
            unsigned int a1 = *reinterpret_cast<const unsigned int*>(
                &smem.sA[g + 8][k_base + 2*t]);
            unsigned int a2 = *reinterpret_cast<const unsigned int*>(
                &smem.sA[g][k_base + 2*t + 8]);
            unsigned int a3 = *reinterpret_cast<const unsigned int*>(
                &smem.sA[g + 8][k_base + 2*t + 8]);

            #pragma unroll
            for (int ni = 0; ni < N_PER_WARP; ni++) {
                int local_n = (warp_id * N_PER_WARP + ni) * MOE2_MMA_N;

                // Load B fragment (m16n8k16 col-major)
                unsigned int b0 = *reinterpret_cast<const unsigned int*>(
                    &smem.sB[local_n + g][k_base + 2*t]);
                unsigned int b1 = *reinterpret_cast<const unsigned int*>(
                    &smem.sB[local_n + g][k_base + 2*t + 8]);

                // BF16 HMMA m16n8k16
                float d0 = acc[ni][0], d1 = acc[ni][1];
                float d2 = acc[ni][2], d3 = acc[ni][3];
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                      "r"(b0), "r"(b1),
                      "f"(d0), "f"(d1), "f"(d2), "f"(d3)
                );
                acc[ni][0] = d0; acc[ni][1] = d1;
                acc[ni][2] = d2; acc[ni][3] = d3;
            }
        }

        __syncthreads();
    }

    // ---- Write output (only row 0 valid for M=1) ----
    #pragma unroll
    for (int ni = 0; ni < N_PER_WARP; ni++) {
        int base_n = n_block + (warp_id * N_PER_WARP + ni) * MOE2_MMA_N;
        int n0 = base_n + 2 * t;
        int n1 = base_n + 2 * t + 1;

        // g=0 → row 0, g+8 → row 8 (invalid for M=1)
        if (g == 0 && n0 < N)
            C[slot * N + n0] = __float2bfloat16(acc[ni][0]);
        if (g == 0 && n1 < N)
            C[slot * N + n1] = __float2bfloat16(acc[ni][2]);
    }
}

// Host launcher
inline void launch_fp4_moe_bf16_gemm(
    const __nv_bfloat16* A,
    const uint8_t* B_packed,
    const float* B_scale,
    __nv_bfloat16* C,
    const int* token_ids,
    const int* expert_ids,
    int num_slots, int N, int K,
    int stride_bn,
    int stride_bsn,
    int64_t expert_b_stride,
    int64_t expert_bs_stride,
    cudaStream_t stream)
{
    dim3 grid(num_slots, (N + MOE2_BLOCK_N - 1) / MOE2_BLOCK_N);
    dim3 block(MOE2_BLOCK_SIZE);
    int smem_size = sizeof(Moe2Smem);

    fp4_moe_bf16_gemm_kernel<<<grid, block, smem_size, stream>>>(
        A, B_packed, B_scale, C, token_ids, expert_ids,
        N, K, stride_bn, stride_bsn, expert_b_stride, expert_bs_stride);
}

}  // namespace sm120
}  // namespace dsv4_kernel
