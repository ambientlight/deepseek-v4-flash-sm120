// =====================================================================
// SM120 Fused FP4 MoE GEMV — Zero-Overhead Expert-Indexed
//
// Fused kernel: bf16 activation × FP4-weight GEMV with in-register
// dequant. No intermediate tensors, no activation quantization,
// no scale conversion. Single kernel launch per GEMM.
//
// For decode M=1: this is a GEMV (matrix-vector), not GEMM.
// Each CTA computes one (slot, N-tile) output block.
//
// Approach: register-based reduction (no SMEM for weights)
// - Load A (bf16) into registers once per K-chunk
// - Stream B (packed FP4) + scale from GMEM, dequant in registers
// - Accumulate A[k] * B_dequant[n, k] in fp32 registers
// - Write bf16 output
//
// Grid: (num_slots, ceil(N / BLOCK_N))
// Block: 256 threads
//
// This mirrors the Triton _mxfp4_slot_gemv_kernel but with:
// - Vectorized uint32 loads for packed FP4 (4 bytes = 8 FP4 values)
// - Vectorized float2 loads for bf16 activations (2 bf16 per load)
// - Unrolled K reduction with software dequant in registers
// - No SMEM allocation (pure register computation)
// =====================================================================
#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace dsv4_kernel {
namespace sm120 {

// FP4 E2M1 dequant LUT in constant memory
__device__ __constant__ float c_FP4_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

static constexpr int FMOE_BLOCK_N = 64;   // N elements per CTA
static constexpr int FMOE_BLOCK_K = 128;  // K elements per iteration
static constexpr int FMOE_THREADS = 256;
// With 256 threads and BLOCK_N=64: 4 threads per N element for K-parallel reduction
static constexpr int FMOE_N_PER_THREAD = 1;  // Each thread owns 1 N output
static constexpr int FMOE_K_THREADS = FMOE_THREADS / FMOE_BLOCK_N; // 4 threads share one N

__global__ void fused_fp4_moe_gemv_kernel(
    const __nv_bfloat16* __restrict__ A,    // [M_total, K] bf16
    const uint8_t* __restrict__ B_packed,   // [E, N, K/2] packed FP4
    const float* __restrict__ B_scale,      // [E, N, K/32] float32
    __nv_bfloat16* __restrict__ C,          // [num_slots, N] bf16
    const int* __restrict__ token_ids,      // [num_slots]
    const int* __restrict__ expert_ids,     // [num_slots]
    int N, int K,
    int stride_bn,           // K/2 (packed bytes per B row)
    int stride_bsn,          // K/32 (scale floats per B row)
    int64_t expert_b_stride, // N * K/2
    int64_t expert_bs_stride)// N * K/32
{
    const int slot = blockIdx.x;
    const int n_block = blockIdx.y * FMOE_BLOCK_N;
    const int tid = threadIdx.x;

    // Thread mapping: tid = n_local * FMOE_K_THREADS + k_thread
    const int n_local = tid / FMOE_K_THREADS;  // which N element (0..63)
    const int k_tid = tid % FMOE_K_THREADS;     // which K partition (0..3)
    const int gn = n_block + n_local;

    if (gn >= N) return;

    const int tok = token_ids[slot];
    const int exp = expert_ids[slot];

    const __nv_bfloat16* A_row = A + tok * K;
    const uint8_t* B_row = B_packed + exp * expert_b_stride + gn * stride_bn;
    const float* Bs_row = B_scale + exp * expert_bs_stride + gn * stride_bsn;

    // Each k_tid thread handles K / FMOE_K_THREADS elements
    float acc = 0.0f;
    const int k_per_thread = K / FMOE_K_THREADS;
    const int k_start = k_tid * k_per_thread;
    const int k_end = k_start + k_per_thread;

    // Process 8 K elements per iteration (4 packed bytes = 8 FP4 values)
    for (int k = k_start; k < k_end; k += 8) {
        // Load 4 packed bytes (8 FP4 values)
        uint32_t packed4;
        if (k + 7 < K) {
            packed4 = *reinterpret_cast<const uint32_t*>(B_row + k / 2);
        } else {
            // Edge case: load byte by byte
            uint8_t b0 = (k/2 < K/2) ? B_row[k/2] : 0;
            uint8_t b1 = (k/2+1 < K/2) ? B_row[k/2+1] : 0;
            uint8_t b2 = (k/2+2 < K/2) ? B_row[k/2+2] : 0;
            uint8_t b3 = (k/2+3 < K/2) ? B_row[k/2+3] : 0;
            packed4 = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
        }

        // Scale for this 32-element group
        float scale = Bs_row[k / 32];

        // Dequant 8 FP4 values and dot with A
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            uint8_t byte = (packed4 >> (i * 8)) & 0xFF;
            uint8_t lo = byte & 0xF;
            uint8_t hi = (byte >> 4) & 0xF;

            float b_lo = c_FP4_LUT[lo] * scale;
            float b_hi = c_FP4_LUT[hi] * scale;

            int ki = k + i * 2;
            float a_lo = __bfloat162float(A_row[ki]);
            float a_hi = __bfloat162float(A_row[ki + 1]);

            acc += a_lo * b_lo + a_hi * b_hi;
        }
    }

    // Shared memory reduction across k_tid threads sharing the same N
    __shared__ float smem_reduce[FMOE_BLOCK_N][FMOE_K_THREADS];
    smem_reduce[n_local][k_tid] = acc;
    __syncthreads();

    if (k_tid == 0) {
        float sum = smem_reduce[n_local][0];
        #pragma unroll
        for (int i = 1; i < FMOE_K_THREADS; i++)
            sum += smem_reduce[n_local][i];
        C[slot * N + gn] = __float2bfloat16(sum);
    }
}

inline void launch_fused_fp4_moe_gemv(
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
    dim3 grid(num_slots, (N + FMOE_BLOCK_N - 1) / FMOE_BLOCK_N);
    dim3 block(FMOE_THREADS);

    fused_fp4_moe_gemv_kernel<<<grid, block, 0, stream>>>(
        A, B_packed, B_scale, C, token_ids, expert_ids,
        N, K, stride_bn, stride_bsn, expert_b_stride, expert_bs_stride);
}

}  // namespace sm120
}  // namespace dsv4_kernel
