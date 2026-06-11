// =====================================================================
// SM120 Fused FP4 MoE GEMV v3 — Maximum Bandwidth
//
// Design: 1 warp = 1 N output. All 32 lanes cooperate on K reduction.
// Each lane processes K/32 contiguous elements → coalesced B access
// across lanes (adjacent lanes read adjacent N rows in different CTAs).
//
// Key optimizations:
// 1. uint4 (16-byte) vectorized weight loads — max memory transactions
// 2. Vectorized bf16x8 activation loads via uint4
// 3. Branchless FP4 decode via register LUT (shared across warp)
// 4. Scale loaded once per 32-element group, NOT per iteration
// 5. Warp-shuffle reduction — zero shared memory
// 6. BLOCK_N=4 with 4 warps per CTA — 8 CTAs per SM (high occupancy)
//    GEMM1: 6 × ceil(1024/4) = 1536 CTAs
//    GEMM2: 6 × ceil(4096/4) = 6144 CTAs
// =====================================================================
#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace dsv4_kernel {
namespace sm120 {
namespace v3 {

static constexpr int V3_BLOCK_N = 4;
static constexpr int V3_WARPS = 4;
static constexpr int V3_THREADS = V3_WARPS * 32;

// Branchless FP4 decode using register-resident LUT
// The LUT is loaded into a float register array at kernel start
__device__ __forceinline__ float fp4_lut_decode(const float* __restrict__ lut, uint8_t nibble) {
    return lut[nibble & 0xF];
}

__global__ __launch_bounds__(128, 8)  // 128 threads, target 8 CTAs/SM
void fused_fp4_moe_gemv_v3_kernel(
    const __nv_bfloat16* __restrict__ A,
    const uint8_t* __restrict__ B_packed,
    const float* __restrict__ B_scale,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ token_ids,
    const int* __restrict__ expert_ids,
    int N, int K,
    int stride_bn,           // K/2
    int stride_bsn,          // K/32
    int64_t expert_b_stride, // N * K/2
    int64_t expert_bs_stride)// N * K/32
{
    const int slot = blockIdx.x;
    const int n_block = blockIdx.y * V3_BLOCK_N;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int gn = n_block + warp_id;

    // Load FP4 LUT into registers (16 floats — fits in register file)
    float lut[16];
    lut[0] = 0.0f; lut[1] = 0.5f; lut[2] = 1.0f; lut[3] = 1.5f;
    lut[4] = 2.0f; lut[5] = 3.0f; lut[6] = 4.0f; lut[7] = 6.0f;
    lut[8] = -0.0f; lut[9] = -0.5f; lut[10] = -1.0f; lut[11] = -1.5f;
    lut[12] = -2.0f; lut[13] = -3.0f; lut[14] = -4.0f; lut[15] = -6.0f;

    if (gn >= N) return;

    const int tok = token_ids[slot];
    const int exp = expert_ids[slot];

    const __nv_bfloat16* A_row = A + tok * K;
    const uint8_t* B_row = B_packed + exp * expert_b_stride + gn * stride_bn;
    const float* Bs_row = B_scale + exp * expert_bs_stride + gn * stride_bsn;

    // Each of 32 lanes handles K/32 contiguous elements
    float acc = 0.0f;
    const int elems_per_lane = K / 32;
    const int my_k = lane * elems_per_lane;
    const int my_k_end = my_k + elems_per_lane;

    // Process in chunks — use uint4 (32 FP4) when possible, uint64 (16 FP4) otherwise
    int k = my_k;

    // Main loop: 32 elements per iter (uint4 = 16 bytes = 32 FP4)
    for (; k + 32 <= my_k_end; k += 32) {
        float scale = Bs_row[k / 32];
        uint4 packed16 = *reinterpret_cast<const uint4*>(B_row + k / 2);
        uint32_t words[4] = {packed16.x, packed16.y, packed16.z, packed16.w};

        #pragma unroll
        for (int wi = 0; wi < 4; wi++) {
            uint4 a_packed = *reinterpret_cast<const uint4*>(&A_row[k + wi * 8]);
            const __nv_bfloat16* a_ptr = reinterpret_cast<const __nv_bfloat16*>(&a_packed);
            uint32_t word = words[wi];

            #pragma unroll
            for (int bi = 0; bi < 4; bi++) {
                uint8_t byte = (word >> (bi * 8)) & 0xFF;
                acc += __bfloat162float(a_ptr[bi*2])   * lut[byte & 0xF] * scale
                     + __bfloat162float(a_ptr[bi*2+1]) * lut[(byte>>4) & 0xF] * scale;
            }
        }
    }

    // Remainder: 16 elements (uint64 = 8 bytes = 16 FP4)
    for (; k + 16 <= my_k_end; k += 16) {
        float scale = Bs_row[k / 32];
        uint64_t packed8 = *reinterpret_cast<const uint64_t*>(B_row + k / 2);

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            uint8_t byte = (packed8 >> (i * 8)) & 0xFF;
            float a0 = __bfloat162float(A_row[k + i*2]);
            float a1 = __bfloat162float(A_row[k + i*2 + 1]);
            acc += a0 * lut[byte & 0xF] * scale + a1 * lut[(byte>>4) & 0xF] * scale;
        }
    }

    // Tail: 8 elements (uint32 = 4 bytes = 8 FP4)
    for (; k + 8 <= my_k_end; k += 8) {
        float scale = Bs_row[k / 32];
        uint32_t packed4 = *reinterpret_cast<const uint32_t*>(B_row + k / 2);

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            uint8_t byte = (packed4 >> (i * 8)) & 0xFF;
            float a0 = __bfloat162float(A_row[k + i*2]);
            float a1 = __bfloat162float(A_row[k + i*2 + 1]);
            acc += a0 * lut[byte & 0xF] * scale + a1 * lut[(byte>>4) & 0xF] * scale;
        }
    }

    // Warp-shuffle reduction across 32 lanes
    #pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1) {
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, offset);
    }

    if (lane == 0) {
        C[slot * N + gn] = __float2bfloat16(acc);
    }
}

inline void launch_fused_fp4_moe_gemv_v3(
    const __nv_bfloat16* A,
    const uint8_t* B_packed,
    const float* B_scale,
    __nv_bfloat16* C,
    const int* token_ids,
    const int* expert_ids,
    int num_slots, int N, int K,
    int stride_bn, int stride_bsn,
    int64_t expert_b_stride, int64_t expert_bs_stride,
    cudaStream_t stream)
{
    dim3 grid(num_slots, (N + V3_BLOCK_N - 1) / V3_BLOCK_N);
    dim3 block(V3_THREADS);

    fused_fp4_moe_gemv_v3_kernel<<<grid, block, 0, stream>>>(
        A, B_packed, B_scale, C, token_ids, expert_ids,
        N, K, stride_bn, stride_bsn, expert_b_stride, expert_bs_stride);
}

}  // namespace v3
}  // namespace sm120
}  // namespace dsv4_kernel
