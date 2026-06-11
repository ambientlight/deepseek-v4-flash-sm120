// =====================================================================
// SM120 Sort-by-Expert Kernel — CUDA Graph Safe
//
// Groups (token, expert) slots by expert ID for batched GEMM.
// Output: sorted_slot_ids ordered by expert, expert_offsets[E+1] boundaries.
//
// Algorithm:
//   1. Zero expert_counts
//   2. Histogram: count slots per expert (atomicAdd)
//   3. Prefix sum: compute expert_offsets from counts
//   4. Scatter: write slot indices in sorted order (atomicAdd for position)
//
// All buffers pre-allocated with known sizes → CUDA graph compatible.
// =====================================================================
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace dsv4_kernel {
namespace sm120 {

// Kernel 1: Zero counts + compute histogram
__global__ void sort_by_expert_histogram_kernel(
    const int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ expert_counts,
    int num_slots,
    int num_experts)
{
    // Phase 1: zero expert_counts (first num_experts threads)
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_experts) {
        expert_counts[tid] = 0;
    }
    __threadfence();
    __syncthreads();  // Ensure zeros visible

    // Need grid-level sync — use separate kernel launch instead
    // This kernel ONLY zeros. Histogram is separate.
}

// Actually, let's do this properly with 3 small kernels:

__global__ void zero_counts_kernel(
    int32_t* __restrict__ counts,
    int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) counts[tid] = 0;
}

__global__ void histogram_kernel(
    const int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ expert_counts,
    int num_slots)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_slots) {
        int eid = expert_ids[tid];
        if (eid >= 0) {
            atomicAdd(&expert_counts[eid], 1);
        }
    }
}

// Single-thread prefix sum (num_experts=64, trivially fast)
__global__ void prefix_sum_kernel(
    const int32_t* __restrict__ expert_counts,
    int32_t* __restrict__ expert_offsets,
    int32_t* __restrict__ scatter_counts,  // reset to 0 for scatter phase
    int num_experts)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    int32_t running = 0;
    for (int e = 0; e < num_experts; e++) {
        expert_offsets[e] = running;
        scatter_counts[e] = 0;  // reset for scatter
        running += expert_counts[e];
    }
    expert_offsets[num_experts] = running;
}

// Scatter: write slot indices into sorted order
__global__ void scatter_kernel(
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ expert_offsets,
    int32_t* __restrict__ scatter_counts,  // atomicAdd position within expert
    int32_t* __restrict__ sorted_slot_ids,
    int num_slots)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_slots) {
        int eid = expert_ids[tid];
        if (eid >= 0) {
            int pos = atomicAdd(&scatter_counts[eid], 1);
            sorted_slot_ids[expert_offsets[eid] + pos] = tid;
        }
    }
}

inline void launch_sort_by_expert(
    const int32_t* expert_ids,
    int32_t* sorted_slot_ids,
    int32_t* expert_offsets,    // [num_experts + 1]
    int32_t* expert_counts,     // temp [num_experts]
    int32_t* scatter_counts,    // temp [num_experts]
    int num_slots,
    int num_experts,
    cudaStream_t stream)
{
    const int block = 256;

    // 1. Zero counts
    zero_counts_kernel<<<(num_experts + block - 1) / block, block, 0, stream>>>(
        expert_counts, num_experts);

    // 2. Histogram
    histogram_kernel<<<(num_slots + block - 1) / block, block, 0, stream>>>(
        expert_ids, expert_counts, num_slots);

    // 3. Prefix sum (single block, single thread — 64 experts is trivial)
    prefix_sum_kernel<<<1, 1, 0, stream>>>(
        expert_counts, expert_offsets, scatter_counts, num_experts);

    // 4. Scatter
    scatter_kernel<<<(num_slots + block - 1) / block, block, 0, stream>>>(
        expert_ids, expert_offsets, scatter_counts, sorted_slot_ids, num_slots);
}

}  // namespace sm120
}  // namespace dsv4_kernel
