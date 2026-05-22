// =====================================================================
// SM_120 sparse decode kernel for DeepSeek-V4-Flash — HMMA production.
//
// All matmul paths (QK^T and P@V) use HMMA tensor core instructions
// (mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32) on SM_120.
//
// No scalar dot-product loops in the production attention math path.
// Scalar reductions (softmax max/sum/exp, normalization) remain SIMT.
//
// SMEM layout includes skew padding to reduce bank conflicts.
//
// Drop-in replacement for the original scalar kernel — same API,
// same launch config, same on-disk KV format.
// =====================================================================
#pragma once

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "common/cutlass_shim.h"
#include "common/defines.h"
#include "common/params.h"

namespace dsv4_kernel {
namespace sm120 {

static constexpr int BLOCK_M_HEADS = 16;
static constexpr int KV_CHUNK      = 32;
static constexpr int HEAD_DIM_NOPE = 448;
static constexpr int HEAD_DIM_ROPE = 64;
static constexpr int HEAD_DIM_QK   = HEAD_DIM_NOPE + HEAD_DIM_ROPE;  // 512
static constexpr int HEAD_DIM_V    = HEAD_DIM_QK;                    // 512

static constexpr int QUANT_TILE    = 64;
static constexpr int NUM_ACTIVE_SCALES = HEAD_DIM_NOPE / QUANT_TILE;  // 7
static constexpr int NUM_SCALE_SLOTS   = 8;
static constexpr int NOPE_BYTES    = HEAD_DIM_NOPE;                  // 448
static constexpr int ROPE_BYTES    = HEAD_DIM_ROPE * 2;              // 128
static constexpr int NOPE_ROPE_BYTES = NOPE_BYTES + ROPE_BYTES;      // 576
static constexpr int K_BYTES_PER_TOKEN =
    NOPE_ROPE_BYTES + NUM_SCALE_SLOTS;                               // 584

static constexpr int NUM_WARPS     = 4;
static constexpr int NUM_THREADS   = NUM_WARPS * 32;

// HMMA m16n8k16 tile dimensions
static constexpr int MMA_M = 16;
static constexpr int MMA_N = 8;
static constexpr int MMA_K = 16;

// SMEM skew to reduce bank conflicts on BF16 loads (8 bf16 = 16 bytes)
static constexpr int SMEM_SKEW_BF16 = 8;

__device__ __forceinline__ float ue8m0_to_scale(unsigned char b) {
    int e = static_cast<int>(b) - 127;
    return __powf(2.0f, static_cast<float>(e));
}

__device__ __forceinline__ void fp8x4_to_bf16x4(uint32_t bits, float scale,
                                                 __nv_bfloat16 out[4]) {
    __nv_fp8_e4m3 f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        f.__x = static_cast<unsigned char>((bits >> (i * 8)) & 0xFFu);
        out[i] = __float2bfloat16_rn(static_cast<float>(f) * scale);
    }
}

__device__ __forceinline__ unsigned int pack_bf16x2(
    __nv_bfloat16 x0, __nv_bfloat16 x1) {
    uint16_t lo = *reinterpret_cast<uint16_t*>(&x0);
    uint16_t hi = *reinterpret_cast<uint16_t*>(&x1);
    return static_cast<unsigned int>(lo) |
           (static_cast<unsigned int>(hi) << 16);
}

// -------------------------------------------------------------------
// SMEM layout with skew padding and BF16 probability tile for P@V HMMA.
//
// Budget (per CTA):
//   sQ       : 16 × (512+8) × 2 =  16 640 B
//   sK       : 32 × (512+8) × 2 =  33 280 B
//   sP       : 16 × 32 × 4      =   2 048 B
//   sP_bf16  : 16 × (32+8) × 2  =   1 280 B
//   sO       : 16 × 512 × 4     =  32 768 B
//   stats    :                        128 B
//   ────────────────────────────────────────
//   total                        ≈  86 KB (SM120: 101 KB limit)
// -------------------------------------------------------------------
struct SmemLayout {
    __nv_bfloat16 sQ[BLOCK_M_HEADS][HEAD_DIM_QK + SMEM_SKEW_BF16];
    __nv_bfloat16 sK[KV_CHUNK][HEAD_DIM_QK + SMEM_SKEW_BF16];
    float         sP[BLOCK_M_HEADS][KV_CHUNK];
    __nv_bfloat16 sP_bf16[BLOCK_M_HEADS][KV_CHUNK + SMEM_SKEW_BF16];
    float         sO[BLOCK_M_HEADS][HEAD_DIM_V];
    float         sRowMax[BLOCK_M_HEADS];
    float         sRowSum[BLOCK_M_HEADS];
};

// -------------------------------------------------------------------
// Dequantise & load one KV chunk into sK (with skew stride).
// -------------------------------------------------------------------
__device__ __forceinline__ void load_kv_chunk(
    SmemLayout &smem,
    const uint8_t *kv_bytes_base,
    int stride_kv_block,
    int stride_kv_row,
    int page_block_size,
    const int *indices_base,
    int token_offset,
    int valid_tokens,
    int warp_id, int lane_id) {

    constexpr int SK_STRIDE = HEAD_DIM_QK + SMEM_SKEW_BF16;

    for (int t = warp_id; t < KV_CHUNK; t += NUM_WARPS) {
        if (t >= valid_tokens) {
            for (int v = lane_id; v < HEAD_DIM_QK; v += 32)
                smem.sK[t][v] = __float2bfloat16_rn(0.0f);
            continue;
        }
        int flat_idx = indices_base[token_offset + t];
        if (flat_idx < 0) {
            for (int v = lane_id; v < HEAD_DIM_QK; v += 32)
                smem.sK[t][v] = __float2bfloat16_rn(0.0f);
            continue;
        }
        int block_idx    = flat_idx / page_block_size;
        int row_in_block = flat_idx % page_block_size;

        const uint8_t *page_base =
            kv_bytes_base +
            static_cast<size_t>(block_idx) * static_cast<size_t>(stride_kv_block);
        const uint8_t *tok_nope_rope =
            page_base + static_cast<size_t>(row_in_block) * NOPE_ROPE_BYTES;
        const unsigned char *scales_u8 =
            page_base + static_cast<size_t>(page_block_size) * NOPE_ROPE_BYTES +
            static_cast<size_t>(row_in_block) * NUM_SCALE_SLOTS;

        float s[NUM_ACTIVE_SCALES];
        #pragma unroll
        for (int i = 0; i < NUM_ACTIVE_SCALES; ++i)
            s[i] = ue8m0_to_scale(scales_u8[i]);

        const uint32_t *fp8_words =
            reinterpret_cast<const uint32_t *>(tok_nope_rope);
        #pragma unroll
        for (int iter = 0; iter < 4; ++iter) {
            int elem_start = iter * 128 + lane_id * 4;
            if (elem_start >= HEAD_DIM_NOPE) break;
            uint32_t bits = fp8_words[iter * 32 + lane_id];
            int q_tile = elem_start / QUANT_TILE;
            float scale = s[q_tile];
            __nv_bfloat16 out[4];
            fp8x4_to_bf16x4(bits, scale, out);
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                int dst = elem_start + k;
                if (dst < HEAD_DIM_NOPE)
                    smem.sK[t][dst] = out[k];
            }
        }

        const __nv_bfloat16 *rope =
            reinterpret_cast<const __nv_bfloat16 *>(tok_nope_rope + NOPE_BYTES);
        if (lane_id * 2 + 1 < HEAD_DIM_ROPE) {
            smem.sK[t][HEAD_DIM_NOPE + lane_id * 2 + 0] = rope[lane_id * 2 + 0];
            smem.sK[t][HEAD_DIM_NOPE + lane_id * 2 + 1] = rope[lane_id * 2 + 1];
        }
    }
}

// -------------------------------------------------------------------
// HMMA QK^T: sP[16][32] = sQ[16][512] × sK[32][512]^T
//
// Fragment mapping (g = lane>>2, t = lane&3):
//   A: a0={sQ[g][K0+2t:+2]}, a1={sQ[g+8][K0+2t:+2]},
//      a2={sQ[g][K0+2t+8:+2]}, a3={sQ[g+8][K0+2t+8:+2]}
//   B: b0={sK[n_base+g][K0+2t:+2]}, b1={sK[n_base+g][K0+2t+8:+2]}
//   D: d0=P[g][n_base+2t], d1=P[g][n_base+2t+1],
//      d2=P[g+8][n_base+2t], d3=P[g+8][n_base+2t+1]
// -------------------------------------------------------------------
__device__ __forceinline__ void hmma_qk_dot(
    SmemLayout &smem,
    int heads_this_cta,
    int valid,
    float sm_scale,
    int warp_id, int lane_id) {

    int n_base = warp_id * MMA_N;
    int g = lane_id >> 2;
    int t = lane_id & 3;

    unsigned int zero = 0;
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;

    #pragma unroll 4
    for (int K0 = 0; K0 < HEAD_DIM_QK; K0 += MMA_K) {
        unsigned int a0 = *reinterpret_cast<const unsigned int*>(&smem.sQ[g][K0 + 2*t]);
        unsigned int a1 = (g + 8 < heads_this_cta)
            ? *reinterpret_cast<const unsigned int*>(&smem.sQ[g + 8][K0 + 2*t])
            : zero;
        unsigned int a2 = *reinterpret_cast<const unsigned int*>(&smem.sQ[g][K0 + 2*t + 8]);
        unsigned int a3 = (g + 8 < heads_this_cta)
            ? *reinterpret_cast<const unsigned int*>(&smem.sQ[g + 8][K0 + 2*t + 8])
            : zero;

        int b_row = n_base + g;
        unsigned int b0, b1;
        if (b_row < KV_CHUNK && b_row < valid) {
            b0 = *reinterpret_cast<const unsigned int*>(&smem.sK[b_row][K0 + 2*t]);
            b1 = *reinterpret_cast<const unsigned int*>(&smem.sK[b_row][K0 + 2*t + 8]);
        } else {
            b0 = 0; b1 = 0;
        }

        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
        );
    }

    int p_col0 = n_base + 2*t;
    int p_col1 = p_col0 + 1;
    if (g < heads_this_cta) {
        smem.sP[g][p_col0] = (p_col0 < valid) ? d0 * sm_scale : -INFINITY;
        smem.sP[g][p_col1] = (p_col1 < valid) ? d1 * sm_scale : -INFINITY;
    }
    if (g + 8 < heads_this_cta) {
        smem.sP[g + 8][p_col0] = (p_col0 < valid) ? d2 * sm_scale : -INFINITY;
        smem.sP[g + 8][p_col1] = (p_col1 < valid) ? d3 * sm_scale : -INFINITY;
    }
}

// -------------------------------------------------------------------
// Online softmax: compute exps, update running max/sum, rescale sO,
// and store BF16 probabilities in sP_bf16 for HMMA P@V.
// -------------------------------------------------------------------
__device__ __forceinline__ void softmax_prepare_p_bf16_and_rescale_o(
    SmemLayout &smem,
    int heads_this_cta,
    int warp_id, int lane_id) {

    for (int h = warp_id; h < heads_this_cta; h += NUM_WARPS) {
        float chunk_max = -INFINITY;
        #pragma unroll
        for (int k = 0; k < KV_CHUNK; ++k)
            chunk_max = fmaxf(chunk_max, smem.sP[h][k]);

        float old_max = smem.sRowMax[h];
        float new_max = fmaxf(old_max, chunk_max);
        float rescale = (old_max == -INFINITY) ? 0.0f : __expf(old_max - new_max);

        if (lane_id == 0) {
            smem.sRowMax[h] = new_max;
            smem.sRowSum[h] *= rescale;
        }
        for (int v = lane_id; v < HEAD_DIM_V; v += 32)
            smem.sO[h][v] *= rescale;

        float local_sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < KV_CHUNK; ++k) {
            float p = smem.sP[h][k];
            float e = (p == -INFINITY) ? 0.0f : __expf(p - new_max);
            local_sum += e;
            smem.sP_bf16[h][k] = __float2bfloat16_rn(e);
        }
        // Zero skew padding
        #pragma unroll
        for (int k = KV_CHUNK; k < KV_CHUNK + SMEM_SKEW_BF16; ++k)
            smem.sP_bf16[h][k] = __float2bfloat16_rn(0.0f);

        if (lane_id == 0)
            smem.sRowSum[h] += local_sum;
    }
    // Zero inactive head rows in sP_bf16
    for (int h = heads_this_cta + warp_id; h < BLOCK_M_HEADS; h += NUM_WARPS) {
        for (int k = lane_id; k < KV_CHUNK + SMEM_SKEW_BF16; k += 32)
            smem.sP_bf16[h][k] = __float2bfloat16_rn(0.0f);
    }
}

// -------------------------------------------------------------------
// HMMA P@V: sO[16][512] += sP_bf16[16][32] × sK[32][512]
//
// P is A (row-major, M=16, K=32), V is B (logical col-major K×N).
// V storage is sK[token][v] row-major, so B_col[k][n] = sK[k][n].
//
// For B: b0 = pack(sK[k_base+2t][v_base+g], sK[k_base+2t+1][v_base+g])
//        b1 = pack(sK[k_base+2t+8][v_base+g], sK[k_base+2t+9][v_base+g])
// Note: strided loads (non-contiguous) — two elements from different rows.
//
// Each warp iterates over v_base in steps of NUM_WARPS*8.
// KV_CHUNK=32 → 2 HMMA K-steps of 16 per v-tile.
// -------------------------------------------------------------------
__device__ __forceinline__ void hmma_pv_accum(
    SmemLayout &smem,
    int heads_this_cta,
    int warp_id, int lane_id) {

    int g = lane_id >> 2;
    int t = lane_id & 3;
    unsigned int zero = 0;

    for (int v_base = warp_id * MMA_N;
         v_base < HEAD_DIM_V;
         v_base += NUM_WARPS * MMA_N) {

        // Load existing sO accumulators into HMMA registers
        float d0 = (g < heads_this_cta)
            ? smem.sO[g][v_base + 2*t] : 0.0f;
        float d1 = (g < heads_this_cta)
            ? smem.sO[g][v_base + 2*t + 1] : 0.0f;
        float d2 = (g + 8 < heads_this_cta)
            ? smem.sO[g + 8][v_base + 2*t] : 0.0f;
        float d3 = (g + 8 < heads_this_cta)
            ? smem.sO[g + 8][v_base + 2*t + 1] : 0.0f;

        // Iterate over K=32 in two steps of 16
        #pragma unroll
        for (int k_base = 0; k_base < KV_CHUNK; k_base += MMA_K) {
            // A fragment from sP_bf16[16][32+skew]
            unsigned int a0 = (g < heads_this_cta)
                ? *reinterpret_cast<const unsigned int*>(&smem.sP_bf16[g][k_base + 2*t])
                : zero;
            unsigned int a1 = (g + 8 < heads_this_cta)
                ? *reinterpret_cast<const unsigned int*>(&smem.sP_bf16[g + 8][k_base + 2*t])
                : zero;
            unsigned int a2 = (g < heads_this_cta)
                ? *reinterpret_cast<const unsigned int*>(&smem.sP_bf16[g][k_base + 2*t + 8])
                : zero;
            unsigned int a3 = (g + 8 < heads_this_cta)
                ? *reinterpret_cast<const unsigned int*>(&smem.sP_bf16[g + 8][k_base + 2*t + 8])
                : zero;

            // B fragment from V = sK[token][v] — strided loads
            // B_col[k][n] = sK[k][v_base + n], n = g for this lane
            unsigned int b0 = pack_bf16x2(
                smem.sK[k_base + 2*t    ][v_base + g],
                smem.sK[k_base + 2*t + 1][v_base + g]);
            unsigned int b1 = pack_bf16x2(
                smem.sK[k_base + 2*t + 8][v_base + g],
                smem.sK[k_base + 2*t + 9][v_base + g]);

            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
            );
        }

        // Write back accumulated sO
        if (g < heads_this_cta) {
            smem.sO[g][v_base + 2*t]     = d0;
            smem.sO[g][v_base + 2*t + 1] = d1;
        }
        if (g + 8 < heads_this_cta) {
            smem.sO[g + 8][v_base + 2*t]     = d2;
            smem.sO[g + 8][v_base + 2*t + 1] = d3;
        }
    }
}

// -------------------------------------------------------------------
// Main kernel — all matmul paths use HMMA tensor cores.
// -------------------------------------------------------------------
template <int HEADS_PER_CTA = BLOCK_M_HEADS>
__global__ __launch_bounds__(NUM_THREADS, 2)
void dsv4_sparse_decode_kernel(SparseAttnDecodeParams params) {
    extern __shared__ __align__(16) unsigned char _smem[];
    SmemLayout &smem = *reinterpret_cast<SmemLayout *>(_smem);

    const int bs_s_q    = blockIdx.x;
    const int head_bk   = blockIdx.y;
    const int batch_idx = bs_s_q / params.s_q;
    const int s_q_idx   = bs_s_q % params.s_q;
    const int head_base = head_bk * HEADS_PER_CTA;
    if (head_base >= params.h_q) return;
    const int heads_this_cta = min(HEADS_PER_CTA, params.h_q - head_base);

    const int tid     = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    int my_topk = params.topk;
    if (params.topk_length)
        my_topk = min(my_topk, params.topk_length[batch_idx]);
    if (my_topk < 0) my_topk = 0;

    int my_extra_topk = 0;
    if (params.extra_kv) {
        my_extra_topk = params.extra_topk;
        if (params.extra_topk_length)
            my_extra_topk = min(my_extra_topk, params.extra_topk_length[batch_idx]);
        if (my_extra_topk < 0) my_extra_topk = 0;
    }

    // ---- Load Q (with skew stride) ----
    const cutlass::bfloat16_t *q_base =
        params.q + static_cast<size_t>(batch_idx) * params.stride_q_b +
        s_q_idx * params.stride_q_s_q + head_base * params.stride_q_h_q;
    for (int h = 0; h < heads_this_cta; ++h) {
        const cutlass::bfloat16_t *row = q_base + h * params.stride_q_h_q;
        for (int t = tid; t < HEAD_DIM_QK; t += NUM_THREADS)
            smem.sQ[h][t] = reinterpret_cast<const __nv_bfloat16 *>(row)[t];
    }
    // Zero inactive head Q rows
    for (int h = heads_this_cta; h < BLOCK_M_HEADS; ++h) {
        for (int t = tid; t < HEAD_DIM_QK; t += NUM_THREADS)
            smem.sQ[h][t] = __float2bfloat16_rn(0.0f);
    }
    if (tid < BLOCK_M_HEADS) {
        smem.sRowMax[tid] = -INFINITY;
        smem.sRowSum[tid] = 0.0f;
        for (int v = 0; v < HEAD_DIM_V; ++v) smem.sO[tid][v] = 0.0f;
    }
    __syncthreads();

    const int *indices_base =
        params.indices + static_cast<size_t>(batch_idx) * params.stride_indices_b +
        s_q_idx * params.stride_indices_s_q;
    const int *extra_indices_base =
        params.extra_indices
            ? params.extra_indices +
                  static_cast<size_t>(batch_idx) * params.stride_extra_indices_b +
                  s_q_idx * params.stride_extra_indices_s_q
            : nullptr;

    auto process = [&](const uint8_t *kv_bytes_base, int stride_block,
                        int stride_row, int page_block,
                        const int *idx_base, int total_tokens) {
        if (total_tokens <= 0 || idx_base == nullptr) return;
        for (int token_offset = 0; token_offset < total_tokens;
             token_offset += KV_CHUNK) {
            int valid = min(KV_CHUNK, total_tokens - token_offset);

            load_kv_chunk(smem, kv_bytes_base, stride_block, stride_row,
                          page_block, idx_base, token_offset, valid,
                          warp_id, lane_id);
            __syncthreads();

            // QK^T — HMMA
            hmma_qk_dot(smem, heads_this_cta, valid, params.sm_scale,
                        warp_id, lane_id);
            __syncthreads();

            // Softmax + prepare BF16 probabilities + rescale sO
            softmax_prepare_p_bf16_and_rescale_o(
                smem, heads_this_cta, warp_id, lane_id);
            __syncthreads();

            // P@V — HMMA
            hmma_pv_accum(smem, heads_this_cta, warp_id, lane_id);
            __syncthreads();
        }
    };

    // Main KV loop.
    process(reinterpret_cast<const uint8_t *>(params.kv),
            params.stride_kv_block, params.stride_kv_row,
            params.page_block_size, indices_base, my_topk);
    // Optional SWA sidecar.
    if (params.extra_kv) {
        process(reinterpret_cast<const uint8_t *>(params.extra_kv),
                params.stride_extra_kv_block, params.stride_extra_kv_row,
                params.extra_page_block_size, extra_indices_base,
                my_extra_topk);
    }

    // --- Epilogue: normalise, apply attn_sink, writeback ---
    for (int h_local = warp_id; h_local < heads_this_cta; h_local += NUM_WARPS) {
        float row_max = smem.sRowMax[h_local];
        float row_sum = smem.sRowSum[h_local];
        bool  has_sink = params.attn_sink != nullptr;
        float sink    = has_sink ? params.attn_sink[head_base + h_local] : 0.0f;

        float lse_val;
        if (row_sum == 0.0f) {
            lse_val = has_sink ? sink : -INFINITY;
            for (int v = lane_id; v < HEAD_DIM_V; v += 32)
                smem.sO[h_local][v] = 0.0f;
        } else {
            float log_sum = logf(row_sum) + row_max;
            if (has_sink) {
                float sink_scale = 1.0f / (1.0f + __expf(sink - log_sum));
                for (int v = lane_id; v < HEAD_DIM_V; v += 32)
                    smem.sO[h_local][v] =
                        (smem.sO[h_local][v] / row_sum) * sink_scale;
                float m = fmaxf(log_sum, sink);
                lse_val = m + logf(__expf(log_sum - m) + __expf(sink - m));
            } else {
                for (int v = lane_id; v < HEAD_DIM_V; v += 32)
                    smem.sO[h_local][v] = smem.sO[h_local][v] / row_sum;
                lse_val = log_sum;
            }
        }
        if (lane_id == 0) {
            params.lse[static_cast<size_t>(batch_idx) * params.stride_lse_b +
                       s_q_idx * params.stride_lse_s_q + head_base + h_local] =
                lse_val;
        }
    }
    __syncthreads();

    // Write out BF16 output [b, s_q, h_q, 512].
    cutlass::bfloat16_t *out_base =
        params.out + static_cast<size_t>(batch_idx) * params.stride_o_b +
        s_q_idx * params.stride_o_s_q + head_base * params.stride_o_h_q;
    for (int h_local = 0; h_local < heads_this_cta; ++h_local) {
        cutlass::bfloat16_t *row_out = out_base + h_local * params.stride_o_h_q;
        for (int v = tid; v < HEAD_DIM_V; v += NUM_THREADS)
            reinterpret_cast<__nv_bfloat16 *>(row_out)[v] =
                __float2bfloat16_rn(smem.sO[h_local][v]);
    }
}

}  // namespace sm120
}  // namespace dsv4_kernel
