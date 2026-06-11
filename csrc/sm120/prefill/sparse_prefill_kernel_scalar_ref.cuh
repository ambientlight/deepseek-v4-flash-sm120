// =====================================================================
// SM_120 sparse PREFILL kernel for DeepSeek-V4-Flash — SCALAR REFERENCE.
//
// Correctness-first CUDA-core implementation. NOT for production builds;
// use sparse_prefill_kernel.cuh (HMMA) once it lands. Enable by defining
// DSV4_ENABLE_SCALAR_PREFILL_KERNEL.
//
// Architecture mirrors DeepSeek FlashMLA's SM90/SM100 sparse-prefill
// (flashmla-src/csrc/sm90/prefill/sparse/phase1.cuh): grid over
// (s_q_idx, head_block), one query token x a BLOCK_M_HEADS-head group per
// CTA, looping B_TOPK-sized KV blocks; online softmax; gather-mask via
// indices; LSE / max_logits epilogue (natural-log domain).
//
// KEY DIFFERENCE vs sparse_decode: the KV is a FLAT bf16 workspace
// [s_kv, 512] already dequantised by sglang's _forward_prefill_sparse — so
// there is NO FP8 / E8M0 / page-byte unpack. V == K (the full 512 dims).
// =====================================================================
#ifndef DSV4_ENABLE_SCALAR_PREFILL_KERNEL
#error "Scalar sparse prefill kernel must not be included in production builds. " \
       "Define DSV4_ENABLE_SCALAR_PREFILL_KERNEL to use this reference kernel."
#endif

#pragma once

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "common/cutlass_shim.h"
#include "common/defines.h"
#include "common/params.h"

namespace dsv4_kernel {
namespace sm120 {
namespace prefill_scalar {

static constexpr int BLOCK_M_HEADS = 16;   // heads per CTA (M-dim)
static constexpr int KV_CHUNK      = 32;    // KV rows streamed per inner step
static constexpr int HEAD_DIM_QK   = 512;
static constexpr int HEAD_DIM_V    = 512;
static constexpr int NUM_WARPS     = 4;
static constexpr int NUM_THREADS   = NUM_WARPS * 32;

struct SmemLayout {
    __nv_bfloat16 sQ[BLOCK_M_HEADS][HEAD_DIM_QK];   // 16 384 B
    __nv_bfloat16 sK[KV_CHUNK][HEAD_DIM_QK];         // 32 768 B
    float         sP[BLOCK_M_HEADS][KV_CHUNK];       //  2 048 B
    float         sO[BLOCK_M_HEADS][HEAD_DIM_V];     // 32 768 B
    float         sRowMax[BLOCK_M_HEADS];
    float         sRowSum[BLOCK_M_HEADS];
};

// Load one KV chunk of up to KV_CHUNK tokens into smem.sK as BF16
// [KV_CHUNK][512] from the FLAT bf16 workspace `kv[s_kv, 512]`, gathered by
// `indices`. Invalid / out-of-range -> zero row (softmax masks via -inf).
__device__ __forceinline__ void load_kv_chunk(
    SmemLayout &smem,
    const __nv_bfloat16 *kv,    // [s_kv, 512] flat bf16
    int stride_kv_s_kv,         // elements between KV rows (== 512 typically)
    int s_kv,
    const int *indices_base,    // [topk] for this query token
    int token_offset,
    int valid_tokens,
    int warp_id, int lane_id) {

    for (int t = warp_id; t < KV_CHUNK; t += NUM_WARPS) {
        int flat_idx = (t < valid_tokens) ? indices_base[token_offset + t] : -1;
        bool ok = (flat_idx >= 0) && (flat_idx < s_kv);
        if (!ok) {
            for (int v = lane_id; v < HEAD_DIM_QK; v += 32)
                smem.sK[t][v] = __float2bfloat16_rn(0.0f);
            continue;
        }
        const __nv_bfloat16 *row =
            kv + static_cast<size_t>(flat_idx) * static_cast<size_t>(stride_kv_s_kv);
        for (int v = lane_id; v < HEAD_DIM_QK; v += 32)
            smem.sK[t][v] = row[v];
    }
}

// Grid = (s_q * ceil(h_q / HEADS_PER_CTA),). Block = NUM_THREADS.
template <int HEADS_PER_CTA = BLOCK_M_HEADS>
__global__ __launch_bounds__(NUM_THREADS, 1)
void dsv4_sparse_prefill_kernel(SparseAttnPrefillParams params) {
    extern __shared__ __align__(16) unsigned char _smem[];
    SmemLayout &smem = *reinterpret_cast<SmemLayout *>(_smem);

    const int num_head_blocks = (params.h_q + HEADS_PER_CTA - 1) / HEADS_PER_CTA;
    const int s_q_idx  = blockIdx.x / num_head_blocks;
    const int head_bk  = blockIdx.x % num_head_blocks;
    const int head_base = head_bk * HEADS_PER_CTA;
    if (s_q_idx >= params.s_q || head_base >= params.h_q) return;
    const int heads_this_cta = min(HEADS_PER_CTA, params.h_q - head_base);

    const int tid     = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    int my_topk = params.topk;
    if (params.topk_length) my_topk = min(my_topk, params.topk_length[s_q_idx]);
    if (my_topk < 0) my_topk = 0;

    // ---- Load Q for this query token's head group ----
    const cutlass::bfloat16_t *q_base =
        params.q + static_cast<size_t>(s_q_idx) * params.stride_q_s_q +
        head_base * params.stride_q_h_q;
    for (int h = 0; h < heads_this_cta; ++h) {
        const cutlass::bfloat16_t *row = q_base + h * params.stride_q_h_q;
        for (int t = tid; t < HEAD_DIM_QK; t += NUM_THREADS)
            smem.sQ[h][t] = reinterpret_cast<const __nv_bfloat16 *>(row)[t];
    }
    if (tid < BLOCK_M_HEADS) {
        smem.sRowMax[tid] = -INFINITY;
        smem.sRowSum[tid] = 0.0f;
        for (int v = 0; v < HEAD_DIM_V; ++v) smem.sO[tid][v] = 0.0f;
    }
    __syncthreads();

    const int *indices_base =
        params.indices + static_cast<size_t>(s_q_idx) * params.stride_indices_s_q;
    const __nv_bfloat16 *kv = reinterpret_cast<const __nv_bfloat16 *>(params.kv);

    // ---- KV loop over this query's top-k ----
    for (int token_offset = 0; token_offset < my_topk; token_offset += KV_CHUNK) {
        int valid = min(KV_CHUNK, my_topk - token_offset);
        load_kv_chunk(smem, kv, params.stride_kv_s_kv, params.s_kv,
                      indices_base, token_offset, valid, warp_id, lane_id);
        __syncthreads();

        // QK^T (CUDA-core): sP[h][k] = sm_scale * (Q_h . K_k).
        for (int cell = tid; cell < BLOCK_M_HEADS * KV_CHUNK; cell += NUM_THREADS) {
            int h = cell / KV_CHUNK;
            int k = cell % KV_CHUNK;
            if (h >= heads_this_cta || k >= valid) { smem.sP[h][k] = -INFINITY; continue; }
            float acc = 0.0f;
            const __nv_bfloat16 *q_row = smem.sQ[h];
            const __nv_bfloat16 *k_row = smem.sK[k];
            #pragma unroll 4
            for (int d = 0; d < HEAD_DIM_QK; ++d)
                acc += __bfloat162float(q_row[d]) * __bfloat162float(k_row[d]);
            smem.sP[h][k] = acc * params.sm_scale;
        }
        __syncthreads();

        // Online softmax + P @ V accumulate (one warp per head).
        for (int h_local = warp_id; h_local < heads_this_cta; h_local += NUM_WARPS) {
            float chunk_max = -INFINITY;
            #pragma unroll
            for (int k = 0; k < KV_CHUNK; ++k) chunk_max = fmaxf(chunk_max, smem.sP[h_local][k]);
            float old_max = smem.sRowMax[h_local];
            float new_max = fmaxf(old_max, chunk_max);
            float rescale = (old_max == -INFINITY) ? 0.0f : __expf(old_max - new_max);
            if (lane_id == 0) {
                smem.sRowMax[h_local] = new_max;
                smem.sRowSum[h_local] *= rescale;
            }
            for (int v = lane_id; v < HEAD_DIM_V; v += 32) smem.sO[h_local][v] *= rescale;

            float local_sum = 0.0f;
            float exps[KV_CHUNK];
            #pragma unroll
            for (int k = 0; k < KV_CHUNK; ++k) {
                float p = smem.sP[h_local][k];
                float e = (p == -INFINITY) ? 0.0f : __expf(p - new_max);
                exps[k] = e;
                local_sum += e;
            }
            if (lane_id == 0) smem.sRowSum[h_local] += local_sum;

            for (int v = lane_id; v < HEAD_DIM_V; v += 32) {
                float acc = 0.0f;
                #pragma unroll
                for (int k = 0; k < KV_CHUNK; ++k)
                    if (k < valid) acc += exps[k] * __bfloat162float(smem.sK[k][v]);
                smem.sO[h_local][v] += acc;
            }
        }
        __syncthreads();
    }

    // ---- Epilogue: normalise, attn_sink mix, write out / max_logits / lse ----
    for (int h_local = warp_id; h_local < heads_this_cta; h_local += NUM_WARPS) {
        float row_max = smem.sRowMax[h_local];
        float row_sum = smem.sRowSum[h_local];
        bool  has_sink = params.attn_sink != nullptr;
        float sink    = has_sink ? params.attn_sink[head_base + h_local] : 0.0f;

        float max_logits_val, lse_val;
        if (row_sum == 0.0f) {
            // Lonely query: no valid token. DeepSeek convention: O=0,
            // max_logits=-inf, lse=+inf.
            max_logits_val = -INFINITY;
            lse_val = INFINITY;
            for (int v = lane_id; v < HEAD_DIM_V; v += 32) smem.sO[h_local][v] = 0.0f;
        } else {
            max_logits_val = row_max;                      // natural-log row max
            float log_sum = logf(row_sum) + row_max;       // orig lse
            if (has_sink) {
                float lse_for_o =
                    fmaxf(log_sum, sink) +
                    logf(__expf(log_sum - fmaxf(log_sum, sink)) +
                         __expf(sink - fmaxf(log_sum, sink)));
                for (int v = lane_id; v < HEAD_DIM_V; v += 32)
                    smem.sO[h_local][v] =
                        smem.sO[h_local][v] * __expf(log_sum - lse_for_o) / row_sum;
            } else {
                for (int v = lane_id; v < HEAD_DIM_V; v += 32)
                    smem.sO[h_local][v] = smem.sO[h_local][v] / row_sum;
            }
            lse_val = log_sum;   // orig_lse (matches DeepSeek ref: NOT the sink-merged lse)
        }
        if (lane_id == 0) {
            size_t off = static_cast<size_t>(s_q_idx) * params.stride_ml_s_q +
                         head_base + h_local;
            params.max_logits[off] = max_logits_val;
            params.lse[off] = lse_val;
        }
    }
    __syncthreads();

    // Write BF16 output [s_q, h_q, 512].
    cutlass::bfloat16_t *out_base =
        params.out + static_cast<size_t>(s_q_idx) * params.stride_o_s_q +
        head_base * params.stride_o_h_q;
    for (int h_local = 0; h_local < heads_this_cta; ++h_local) {
        cutlass::bfloat16_t *row_out = out_base + h_local * params.stride_o_h_q;
        for (int v = tid; v < HEAD_DIM_V; v += NUM_THREADS)
            reinterpret_cast<__nv_bfloat16 *>(row_out)[v] =
                __float2bfloat16_rn(smem.sO[h_local][v]);
    }
}

}  // namespace prefill_scalar
}  // namespace sm120
}  // namespace dsv4_kernel
