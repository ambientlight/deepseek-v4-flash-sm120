// =====================================================================
// SM_120 split-KV combine kernel.
//
// Merges partial online-softmax states from multiple split CTAs into
// the final output. Each split produced:
//   lse_accum[split][b*s_q][h_q]     — partial log-sum-exp
//   o_accum[split][b*s_q][h_q][d_v]  — partial unnormalized output (FP32)
//
// The combine kernel runs one thread-block per (batch*s_q, head_group).
// =====================================================================
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "common/cutlass_shim.h"
#include "common/params.h"

namespace dsv4_kernel {
namespace sm120 {

static constexpr int COMBINE_THREADS = 256;

__global__ void dsv4_split_kv_combine(
    const float* __restrict__ lse_accum,    // [num_splits, b*s_q, h_q]
    const float* __restrict__ o_accum,      // [num_splits, b*s_q, h_q, d_v]
    const float* __restrict__ attn_sink,    // [h_q] or nullptr
    float* __restrict__ lse_out,            // [b, s_q, h_q]
    cutlass::bfloat16_t* __restrict__ out,  // [b, s_q, h_q, d_v]
    int num_splits,
    int h_q,
    int d_v,
    int stride_lse_accum_split,  // h_q (or b*s_q*h_q for flat)
    int stride_lse_accum_bs,     // h_q
    int stride_o_accum_split,    // b*s_q*h_q*d_v
    int stride_o_accum_bs,       // h_q*d_v
    int stride_o_accum_h,        // d_v
    int stride_lse_b,
    int stride_lse_sq,
    int stride_o_b,
    int stride_o_sq,
    int stride_o_h,
    int b_times_sq) {

    // Grid: (b*s_q, ceil(h_q/HEADS_PER_BLOCK))
    const int bs_idx = blockIdx.x;
    const int h_base = blockIdx.y * 16;  // process up to 16 heads per block
    const int tid = threadIdx.x;

    if (bs_idx >= b_times_sq) return;

    for (int h = h_base; h < min(h_base + 16, h_q); ++h) {
        // Find global max across splits
        float m = -INFINITY;
        for (int s = 0; s < num_splits; ++s) {
            float partial_lse = lse_accum[s * stride_lse_accum_split +
                                          bs_idx * stride_lse_accum_bs + h];
            m = fmaxf(m, partial_lse);
        }

        // Compute total sum: l = sum_s exp(lse_s - m)
        float l = 0.0f;
        for (int s = 0; s < num_splits; ++s) {
            float partial_lse = lse_accum[s * stride_lse_accum_split +
                                          bs_idx * stride_lse_accum_bs + h];
            if (partial_lse > -INFINITY)
                l += __expf(partial_lse - m);
        }

        // Handle attn_sink
        bool has_sink = attn_sink != nullptr;
        float sink = has_sink ? attn_sink[h] : 0.0f;
        float log_sum = (l > 0.0f) ? (logf(l) + m) : -INFINITY;

        float lse_val;
        float norm_factor;
        if (l == 0.0f) {
            lse_val = has_sink ? sink : -INFINITY;
            norm_factor = 0.0f;
        } else if (has_sink) {
            float mm = fmaxf(log_sum, sink);
            lse_val = mm + logf(__expf(log_sum - mm) + __expf(sink - mm));
            float sink_scale = 1.0f / (1.0f + __expf(sink - log_sum));
            norm_factor = sink_scale / l;
        } else {
            lse_val = log_sum;
            norm_factor = 1.0f / l;
        }

        // Write LSE (only one thread per head)
        if (tid == 0)
            lse_out[bs_idx * stride_lse_sq + h] = lse_val;

        // Combine output: out[v] = norm_factor * sum_s exp(lse_s - m) * o_accum_s[v]
        for (int v = tid; v < d_v; v += COMBINE_THREADS) {
            float acc = 0.0f;
            for (int s = 0; s < num_splits; ++s) {
                float partial_lse = lse_accum[s * stride_lse_accum_split +
                                              bs_idx * stride_lse_accum_bs + h];
                float alpha = (partial_lse > -INFINITY) ? __expf(partial_lse - m) : 0.0f;
                float partial_o = o_accum[s * stride_o_accum_split +
                                          bs_idx * stride_o_accum_bs +
                                          h * stride_o_accum_h + v];
                acc += alpha * partial_o;
            }
            reinterpret_cast<__nv_bfloat16 *>(out)[
                bs_idx * stride_o_sq + h * stride_o_h + v] =
                __float2bfloat16_rn(acc * norm_factor);
        }
    }
}

}  // namespace sm120
}  // namespace dsv4_kernel
