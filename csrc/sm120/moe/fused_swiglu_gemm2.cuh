// =====================================================================
// SM120 Fused FP4 MoE: GEMM1 + SwiGLU + GEMM2 in one C++ call
//
// Two CUDA kernels internally:
//   1. GEMM1: A_bf16 × W13_fp4 → intermediate [num_slots, 2*I] bf16
//   2. Fused SwiGLU+GEMM2: reads gate/up from intermediate,
//      computes silu(gate)*up on-the-fly, dots with W2_fp4 → output
//
// The SwiGLU is fused INTO the GEMM2 kernel's input load path:
//   for each k: x_k = silu(intermediate[k]) * intermediate[I+k]; acc += x_k * W2[n,k]
//
// This eliminates:
//   - Separate SwiGLU PyTorch ops (gate, clamp, silu, mul, cast)
//   - Intermediate tensor allocation for activated result
//   - 5 kernel launches per MoE layer
// =====================================================================
#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>

namespace dsv4_kernel {
namespace sm120 {

// Reuse the FP4 LUT from fused_fp4_moe_gemv.cuh (same translation unit)
// c_FP4_LUT is defined there, we just reference it here

// SwiGLU-fused GEMM2 kernel
// Input: intermediate [num_slots, 2*I] bf16 (gate in [:I], up in [I:])
// Weight: W2 [E, K_out, I/2] packed FP4 with [E, K_out, I/32] float32 scales
// Output: [num_slots, K_out] bf16
//
// Each thread computes: out[n] = sum_k( silu(gate[k]) * up[k] * W2_dequant[n, k] )
// with optional clamp on gate/up

static constexpr int FMOE2_BLOCK_N = 64;
static constexpr int FMOE2_THREADS = 256;
static constexpr int FMOE2_K_THREADS = FMOE2_THREADS / FMOE2_BLOCK_N; // 4

__device__ __forceinline__ float device_silu(float x) {
    return x / (1.0f + expf(-x));
}

__global__ void fused_swiglu_gemm2_kernel(
    const __nv_bfloat16* __restrict__ intermediate, // [num_slots, 2*I] bf16 (gate|up)
    const uint8_t* __restrict__ B_packed,    // [E, K_out, I/2] packed FP4
    const float* __restrict__ B_scale,       // [E, K_out, I/32] float32
    __nv_bfloat16* __restrict__ C,           // [num_slots, K_out] bf16
    const int* __restrict__ expert_ids,      // [num_slots]
    int K_out, int I,                        // K_out=hidden_size, I=intermediate_size
    float clamp_limit,                       // DeepSeek V4: 10.0, <=0 means no clamp
    int stride_bn,
    int stride_bsn,
    int64_t expert_b_stride,
    int64_t expert_bs_stride)
{
    const int slot = blockIdx.x;
    const int n_block = blockIdx.y * FMOE2_BLOCK_N;
    const int tid = threadIdx.x;
    const int n_local = tid / FMOE2_K_THREADS;
    const int k_tid = tid % FMOE2_K_THREADS;
    const int gn = n_block + n_local;

    if (gn >= K_out) return;

    // For GEMM2, slot == row index directly (no indirection needed)
    const int exp = expert_ids[slot];

    // Intermediate: gate at [slot, 0:I], up at [slot, I:2*I]
    const __nv_bfloat16* gate_row = intermediate + slot * (2 * I);
    const __nv_bfloat16* up_row = gate_row + I;

    const uint8_t* B_row = B_packed + exp * expert_b_stride + gn * stride_bn;
    const float* Bs_row = B_scale + exp * expert_bs_stride + gn * stride_bsn;

    float acc = 0.0f;
    const int k_per_thread = I / FMOE2_K_THREADS;
    const int k_start = k_tid * k_per_thread;
    const int k_end = k_start + k_per_thread;
    const bool do_clamp = (clamp_limit > 0.0f);

    for (int k = k_start; k < k_end; k += 8) {
        // Load W2 packed FP4 (8 values from 4 bytes)
        uint32_t packed4 = *reinterpret_cast<const uint32_t*>(B_row + k / 2);
        float scale = Bs_row[k / 32];

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            uint8_t byte = (packed4 >> (i * 8)) & 0xFF;
            uint8_t lo = byte & 0xF;
            uint8_t hi = (byte >> 4) & 0xF;

            float w_lo = c_FP4_LUT[lo] * scale;
            float w_hi = c_FP4_LUT[hi] * scale;

            int ki = k + i * 2;

            // Fused SwiGLU: x = silu(gate[ki]) * up[ki]
            float g0 = __bfloat162float(gate_row[ki]);
            float u0 = __bfloat162float(up_row[ki]);
            float g1 = __bfloat162float(gate_row[ki + 1]);
            float u1 = __bfloat162float(up_row[ki + 1]);

            if (do_clamp) {
                g0 = fminf(g0, clamp_limit);
                u0 = fmaxf(fminf(u0, clamp_limit), -clamp_limit);
                g1 = fminf(g1, clamp_limit);
                u1 = fmaxf(fminf(u1, clamp_limit), -clamp_limit);
            }

            float x0 = device_silu(g0) * u0;
            float x1 = device_silu(g1) * u1;

            acc += x0 * w_lo + x1 * w_hi;
        }
    }

    // Shared memory reduction
    __shared__ float smem_reduce[FMOE2_BLOCK_N][FMOE2_K_THREADS];
    smem_reduce[n_local][k_tid] = acc;
    __syncthreads();

    if (k_tid == 0) {
        float sum = smem_reduce[n_local][0];
        #pragma unroll
        for (int i = 1; i < FMOE2_K_THREADS; i++)
            sum += smem_reduce[n_local][i];
        C[slot * K_out + gn] = __float2bfloat16(sum);
    }
}

// Launch the fused SwiGLU+GEMM2
inline void launch_fused_swiglu_gemm2(
    const __nv_bfloat16* intermediate,
    const uint8_t* B_packed,
    const float* B_scale,
    __nv_bfloat16* C,
    const int* expert_ids,
    int num_slots, int K_out, int I,
    float clamp_limit,
    int stride_bn, int stride_bsn,
    int64_t expert_b_stride, int64_t expert_bs_stride,
    cudaStream_t stream)
{
    dim3 grid(num_slots, (K_out + FMOE2_BLOCK_N - 1) / FMOE2_BLOCK_N);
    dim3 block(FMOE2_THREADS);

    fused_swiglu_gemm2_kernel<<<grid, block, 0, stream>>>(
        intermediate, B_packed, B_scale, C, expert_ids,
        K_out, I, clamp_limit,
        stride_bn, stride_bsn, expert_b_stride, expert_bs_stride);
}

}  // namespace sm120
}  // namespace dsv4_kernel
