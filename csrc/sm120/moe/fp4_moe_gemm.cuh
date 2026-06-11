// =====================================================================
// SM120 FP4 MoE GEMM — Expert-Indexed with Native Tensor Core MMA
//
// For MoE decode: each (token, expert) slot runs an independent GEMM:
//   output[slot] = A[token_id] @ B[expert_id].T
//
// Uses native FP4 block-scaled MMA: mma.kind::mxf8f6f4.block_scale.m16n8k32
//
// Grid: (num_slots, ceil(N / BLOCK_N))
// Block: 128 threads (4 warps)
//
// For decode M=1: only first row of BLOCK_M=16 is valid; rest zero-padded.
// =====================================================================
#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace dsv4_kernel {
namespace sm120 {

// Tile sizes for MoE decode (M_per_slot = 1)
static constexpr int MOE_MMA_M = 16;
static constexpr int MOE_MMA_N = 8;
static constexpr int MOE_MMA_K = 32;
static constexpr int MOE_BLOCK_M = 16;     // Single MMA M-tile (only row 0 used for M=1)
static constexpr int MOE_BLOCK_N = 64;     // 8 N-tiles per CTA
static constexpr int MOE_K_TILE = 128;     // 4 MMA ops per K-tile
static constexpr int MOE_MMA_PER_K = MOE_K_TILE / MOE_MMA_K;  // 4
static constexpr int MOE_NUM_WARPS = 4;
static constexpr int MOE_BLOCK_SIZE = MOE_NUM_WARPS * 32;  // 128 threads
static constexpr int MOE_SCALE_GROUP = 32;
static constexpr int MOE_N_TILES = MOE_BLOCK_N / MOE_MMA_N;  // 8

// SMEM layout (single buffer — K_TILE is large enough for pipelining gain to be minimal at M=1)
struct MoeSmem {
    uint8_t sA[MOE_BLOCK_M][MOE_K_TILE / 2];     // 16 × 64 = 1 KB
    uint8_t sB[MOE_BLOCK_N][MOE_K_TILE / 2];     // 64 × 64 = 4 KB
    uint8_t sScaleA[MOE_BLOCK_M][MOE_MMA_PER_K]; // 16 × 4 = 64 B
    uint8_t sScaleB[MOE_BLOCK_N][MOE_MMA_PER_K]; // 64 × 4 = 256 B
};  // ~5.3 KB — fits easily in 99 KB

// ---- Fragment loading (same logic as fp4_gemm_opt) ----
__device__ __forceinline__ void moe_load_A_frag(
    const uint8_t sA[][MOE_K_TILE/2],
    int local_m, int k_mma, int t0, int t1,
    uint32_t &a0, uint32_t &a1, uint32_t &a2, uint32_t &a3)
{
    int k_base = k_mma * MOE_MMA_K;
    uint8_t a_bytes[16];
    #pragma unroll
    for (int v = 0; v < 16; v++) {
        int v0 = v & 3;
        int v1 = (v >> 2) & 1;
        int v2 = v >> 3;
        int phys = t0*64 + t1 + v0*16 + v1*8 + v2*256;
        int m_idx = phys >> 5;
        int k_idx = phys & 31;
        int gk = k_base + k_idx;
        int byte_idx = gk >> 1;
        uint8_t packed = sA[local_m + m_idx][byte_idx];
        a_bytes[v] = (gk & 1) ? (packed >> 4) : (packed & 0xF);
    }
    a0 = *reinterpret_cast<uint32_t*>(&a_bytes[0]);
    a1 = *reinterpret_cast<uint32_t*>(&a_bytes[4]);
    a2 = *reinterpret_cast<uint32_t*>(&a_bytes[8]);
    a3 = *reinterpret_cast<uint32_t*>(&a_bytes[12]);
    a0 <<= 2; a1 <<= 2; a2 <<= 2; a3 <<= 2;
}

__device__ __forceinline__ void moe_load_B_frag(
    const uint8_t sB[][MOE_K_TILE/2],
    int local_n, int k_mma, int t0, int t1,
    uint32_t &b0, uint32_t &b1)
{
    int k_base = k_mma * MOE_MMA_K;
    uint8_t b_bytes[8];
    #pragma unroll
    for (int v = 0; v < 8; v++) {
        int v0 = v & 3;
        int v1 = v >> 2;
        int phys = t0*32 + t1 + v0*8 + v1*128;
        int n_idx = phys >> 5;
        int k_idx = phys & 31;
        int gk = k_base + k_idx;
        int byte_idx = gk >> 1;
        uint8_t packed = sB[local_n + n_idx][byte_idx];
        b_bytes[v] = (gk & 1) ? (packed >> 4) : (packed & 0xF);
    }
    b0 = *reinterpret_cast<uint32_t*>(&b_bytes[0]);
    b1 = *reinterpret_cast<uint32_t*>(&b_bytes[4]);
    b0 <<= 2; b1 <<= 2;
}

__device__ __forceinline__ void moe_fp4_mma(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float &d0, float &d1, float &d2, float &d3,
    uint8_t sfa, uint8_t sfb)
{
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X"
        ".m16n8k32.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
        "{%14},{%15,%16},{%17},{%18,%19};\n"
        : "=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),
          "r"(b0),"r"(b1),
          "f"(d0),"f"(d1),"f"(d2),"f"(d3),
          "r"(uint32_t(sfa)),"h"(uint16_t(0)),"h"(uint16_t(0)),
          "r"(uint32_t(sfb)),"h"(uint16_t(0)),"h"(uint16_t(0))
    );
}

// =====================================================================
// Main MoE GEMM kernel
//
// A_packed: [M_total, K/2] packed FP4 (quantized activations)
// A_scale:  [M_total, K/32] UE8M0 (activation scales)
// B_packed: [E, N, K/2] packed FP4 (expert weights)
// B_scale:  [E, N, K/32] UE8M0 (weight scales)
// C:        [num_slots, N] bf16 (output)
// token_ids: [num_slots] int32 — index into A
// expert_ids: [num_slots] int32 — index into B (expert dimension)
// =====================================================================
__global__ void fp4_moe_gemm_kernel(
    const uint8_t* __restrict__ A_packed,   // [M_total, K/2]
    const uint8_t* __restrict__ A_scale,    // [M_total, K/32]
    const uint8_t* __restrict__ B_packed,   // [E, N, K/2]
    const uint8_t* __restrict__ B_scale,    // [E, N, K/32]
    __nv_bfloat16* __restrict__ C,          // [num_slots, N]
    const int* __restrict__ token_ids,      // [num_slots]
    const int* __restrict__ expert_ids,     // [num_slots]
    int N, int K,
    int stride_bn,     // B stride for N dim: K/2
    int stride_bsn,    // B_scale stride for N dim: K/32
    int64_t expert_b_stride,   // B stride between experts: N * K/2
    int64_t expert_bs_stride)  // B_scale stride between experts: N * K/32
{
    extern __shared__ char _smem[];
    MoeSmem &smem = *reinterpret_cast<MoeSmem*>(_smem);

    const int slot = blockIdx.x;
    const int n_block = blockIdx.y * MOE_BLOCK_N;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid % 32;
    const int t0 = lane % 4;
    const int t1 = lane / 4;
    const int g = lane >> 2;

    // Look up which token and expert this slot uses
    const int tok = token_ids[slot];
    const int exp = expert_ids[slot];

    // Base pointers for this slot's A and B
    const uint8_t* A_row = A_packed + tok * (K / 2);
    const uint8_t* A_sc_row = A_scale + tok * (K / MOE_SCALE_GROUP);
    const uint8_t* B_exp = B_packed + exp * expert_b_stride;
    const uint8_t* B_sc_exp = B_scale + exp * expert_bs_stride;

    // Accumulators: each warp handles MOE_N_TILES / MOE_NUM_WARPS N-tiles
    // With 4 warps and 8 N-tiles: 2 N-tiles per warp
    static constexpr int N_PER_WARP = MOE_N_TILES / MOE_NUM_WARPS;  // 2
    float acc[N_PER_WARP][4];
    #pragma unroll
    for (int i = 0; i < N_PER_WARP; i++)
        acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;

    int num_k_tiles = K / MOE_K_TILE;

    for (int kt = 0; kt < num_k_tiles; kt++) {
        int k_start = kt * MOE_K_TILE;

        // ---- Load A tile: [BLOCK_M, K_TILE/2] ----
        // For M=1 decode: only row 0 has real data; rows 1-15 are zero
        for (int idx = tid; idx < MOE_BLOCK_M * (MOE_K_TILE/2); idx += MOE_BLOCK_SIZE) {
            int m = idx / (MOE_K_TILE/2);
            int kb = idx % (MOE_K_TILE/2);
            smem.sA[m][kb] = (m == 0) ? A_row[k_start/2 + kb] : 0;
        }

        // ---- Load B tile: [BLOCK_N, K_TILE/2] ----
        for (int idx = tid; idx < MOE_BLOCK_N * (MOE_K_TILE/2); idx += MOE_BLOCK_SIZE) {
            int n = idx / (MOE_K_TILE/2);
            int kb = idx % (MOE_K_TILE/2);
            int gn = n_block + n;
            if (gn < N) {
                smem.sB[n][kb] = B_exp[gn * stride_bn + k_start/2 + kb];
            } else {
                smem.sB[n][kb] = 0;
            }
        }

        // ---- Load scales ----
        for (int idx = tid; idx < MOE_BLOCK_M * MOE_MMA_PER_K; idx += MOE_BLOCK_SIZE) {
            int m = idx / MOE_MMA_PER_K;
            int ki = idx % MOE_MMA_PER_K;
            int k_sc = (k_start + ki * MOE_MMA_K) / MOE_SCALE_GROUP;
            smem.sScaleA[m][ki] = (m == 0) ? A_sc_row[k_sc] : 127;
        }
        for (int idx = tid; idx < MOE_BLOCK_N * MOE_MMA_PER_K; idx += MOE_BLOCK_SIZE) {
            int n = idx / MOE_MMA_PER_K;
            int ki = idx % MOE_MMA_PER_K;
            int gn = n_block + n;
            int k_sc = (k_start + ki * MOE_MMA_K) / MOE_SCALE_GROUP;
            smem.sScaleB[n][ki] = (gn < N) ? B_sc_exp[gn * stride_bsn + k_sc] : 127;
        }

        __syncthreads();

        // ---- Compute: 4 MMA iterations per K-tile ----
        #pragma unroll
        for (int kk = 0; kk < MOE_MMA_PER_K; kk++) {
            // Load A fragment (shared across all warps — same row 0)
            uint32_t a0, a1, a2, a3;
            moe_load_A_frag(smem.sA, 0, kk, t0, t1, a0, a1, a2, a3);
            uint8_t sfa = smem.sScaleA[g][kk];  // g = lane>>2, maps to M row

            // Each warp processes its N-tile slice
            #pragma unroll
            for (int ni = 0; ni < N_PER_WARP; ni++) {
                int local_n = (warp_id * N_PER_WARP + ni) * MOE_MMA_N;
                uint32_t b0, b1;
                moe_load_B_frag(smem.sB, local_n, kk, t0, t1, b0, b1);
                uint8_t sfb = smem.sScaleB[local_n + (g & 7)][kk]; // within N-tile

                moe_fp4_mma(a0, a1, a2, a3, b0, b1,
                           acc[ni][0], acc[ni][1], acc[ni][2], acc[ni][3],
                           sfa, sfb);
            }
        }

        __syncthreads();
    }

    // ---- Write output (only row 0 is valid for M=1 decode) ----
    // MMA output layout: d0=[g, 2*t0], d1=[g+8, 2*t0], d2=[g, 2*t0+1], d3=[g+8, 2*t0+1]
    // For M=1: only g==0 elements are valid (row 0 of the 16-row M-tile)
    #pragma unroll
    for (int ni = 0; ni < N_PER_WARP; ni++) {
        int base_n = n_block + (warp_id * N_PER_WARP + ni) * MOE_MMA_N;
        int n0 = base_n + 2 * t0;
        int n1 = base_n + 2 * t0 + 1;

        // g = lane >> 2: only g==0 corresponds to row 0
        if (g == 0 && n0 < N) {
            C[slot * N + n0] = __float2bfloat16(acc[ni][0]);
        }
        if (g == 0 && n1 < N) {
            C[slot * N + n1] = __float2bfloat16(acc[ni][2]);
        }
        // d1, d3 are for g+8 = row 8 — invalid for M=1
    }
}

// Host launcher
inline void launch_fp4_moe_gemm(
    const uint8_t* A_packed,
    const uint8_t* A_scale,
    const uint8_t* B_packed,
    const uint8_t* B_scale,
    __nv_bfloat16* C,
    const int* token_ids,
    const int* expert_ids,
    int num_slots, int N, int K,
    int stride_bn,        // K/2
    int stride_bsn,       // K/32
    int64_t expert_b_stride,   // N * K/2
    int64_t expert_bs_stride,  // N * K/32
    cudaStream_t stream)
{
    dim3 grid(num_slots, (N + MOE_BLOCK_N - 1) / MOE_BLOCK_N);
    dim3 block(MOE_BLOCK_SIZE);
    int smem_size = sizeof(MoeSmem);

    fp4_moe_gemm_kernel<<<grid, block, smem_size, stream>>>(
        A_packed, A_scale, B_packed, B_scale, C,
        token_ids, expert_ids,
        N, K, stride_bn, stride_bsn, expert_b_stride, expert_bs_stride);
}

}  // namespace sm120
}  // namespace dsv4_kernel
