// =====================================================================
// SM_120 sparse PREFILL kernel for DeepSeek-V4-Flash — HMMA tensor core.
//
// Production kernel. Mirrors DeepSeek FlashMLA's SM90/SM100 sparse-prefill
// *architecture* (flashmla-src/csrc/sm90/prefill/sparse/phase1.cuh): grid over
// (query-token, head-block), B_TOPK KV blocks, online softmax, gather-mask,
// LSE/max_logits epilogue — realised with the SM120 primitives our decode
// kernel proved (mma.sync m16n8k16 + register-resident O), since SM120 has no
// WGMMA/TMA (SM90) or tcgen05 (SM100).
//
// softmax_and_rescale_reg_o and hmma_pv_accum_reg are KV-source-agnostic — they
// read smem.sQ/sK/sSP — so we reuse the decode versions verbatim. hmma_qk_dot is
// the decode version plus a per-KV-row validity mask (smem.sKvalid, mirroring
// DeepSeek's is_kv_valid): an invalid gather index scores -inf rather than 0, so
// max_logits / lse stay exact. The other datapath change is load_kv_chunk:
// prefill KV is a FLAT bf16 workspace [s_kv, 512] (sglang pre-dequantised), so
// no FP8 / E8M0 / page unpack. V == K (full 512). Grid maps query tokens.
//
// Selected by building the prefill TU with -DDSV4_PREFILL_USE_HMMA.
// =====================================================================
#pragma once

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "common/cutlass_shim.h"
#include "common/defines.h"
#include "common/params.h"

namespace dsv4_kernel {
namespace sm120 {
namespace prefill_hmma {

static constexpr int BLOCK_M_HEADS = 16;    // heads per CTA (MMA M-dim)
static constexpr int KV_CHUNK      = 64;    // = NUM_WARPS * MMA_N (the QK^T N-tiling)
static constexpr int HEAD_DIM_QK   = 512;
static constexpr int HEAD_DIM_V    = 512;

static constexpr int NUM_WARPS     = 8;
static constexpr int NUM_THREADS   = NUM_WARPS * 32;

static constexpr int MMA_M = 16;
static constexpr int MMA_N = 8;
static constexpr int MMA_K = 16;

static constexpr int V_TILES_PER_WARP = HEAD_DIM_V / (NUM_WARPS * MMA_N);  // 16

__device__ __forceinline__ unsigned int pack_bf16x2(
    __nv_bfloat16 x0, __nv_bfloat16 x1) {
    uint16_t lo = *reinterpret_cast<uint16_t*>(&x0);
    uint16_t hi = *reinterpret_cast<uint16_t*>(&x1);
    return static_cast<unsigned int>(lo) | (static_cast<unsigned int>(hi) << 16);
}

// score (f32) and prob (bf16) are kept in SEPARATE storage, NOT a union: all
// warps redundantly read score while writing prob in softmax_and_rescale_reg_o,
// and an overlay would let a prob write clobber a score byte another warp is
// still reading (a real WAR hazard racecheck flags). +2 KB smem buys a clean
// dependency: score is read-only there, prob write-only.
struct ScoreProbStorage {
    float         score[BLOCK_M_HEADS][KV_CHUNK];
    __nv_bfloat16 prob[BLOCK_M_HEADS][KV_CHUNK];
};

struct SmemLayout {
    __nv_bfloat16 sQ[BLOCK_M_HEADS][HEAD_DIM_QK];   // 16 KB
    __nv_bfloat16 sK[KV_CHUNK][HEAD_DIM_QK];         // 32 KB
    ScoreProbStorage sSP;                            //  6 KB
    bool          sKvalid[KV_CHUNK];                 // per-KV-row gather validity
};

// ---- load one KV chunk from the FLAT bf16 workspace, gathered by indices ----
__device__ __forceinline__ void load_kv_chunk(
    SmemLayout &smem,
    const __nv_bfloat16 *kv,    // [s_kv, 512]
    int stride_kv_s_kv,
    int s_kv,
    const int *indices_base,
    int token_offset,
    int valid_tokens,
    int warp_id, int lane_id) {

    for (int tok = warp_id; tok < KV_CHUNK; tok += NUM_WARPS) {
        int flat_idx = (tok < valid_tokens) ? indices_base[token_offset + tok] : -1;
        bool ok = (flat_idx >= 0) && (flat_idx < s_kv);
        if (lane_id == 0) smem.sKvalid[tok] = ok;
        if (!ok) {
            for (int v = lane_id; v < HEAD_DIM_QK; v += 32)
                smem.sK[tok][v] = __float2bfloat16_rn(0.0f);
            continue;
        }
        const __nv_bfloat16 *row =
            kv + static_cast<size_t>(flat_idx) * static_cast<size_t>(stride_kv_s_kv);
        for (int v = lane_id; v < HEAD_DIM_QK; v += 32)
            smem.sK[tok][v] = row[v];
    }
}

// ---- HMMA QK^T -> sSP.score[16][32]  (identical to decode) ----
__device__ __forceinline__ void hmma_qk_dot(
    SmemLayout &smem, int heads_this_cta, int valid, float sm_scale,
    int warp_id, int lane_id) {
    int n_base = warp_id * MMA_N;
    int g = lane_id >> 2, t = lane_id & 3;
    unsigned int zero = 0;
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
    #pragma unroll 4
    for (int K0 = 0; K0 < HEAD_DIM_QK; K0 += MMA_K) {
        unsigned int a0 = *reinterpret_cast<const unsigned int*>(&smem.sQ[g][K0 + 2*t]);
        unsigned int a1 = (g + 8 < heads_this_cta)
            ? *reinterpret_cast<const unsigned int*>(&smem.sQ[g + 8][K0 + 2*t]) : zero;
        unsigned int a2 = *reinterpret_cast<const unsigned int*>(&smem.sQ[g][K0 + 2*t + 8]);
        unsigned int a3 = (g + 8 < heads_this_cta)
            ? *reinterpret_cast<const unsigned int*>(&smem.sQ[g + 8][K0 + 2*t + 8]) : zero;
        int b_row = n_base + g;
        unsigned int b0, b1;
        if (b_row < KV_CHUNK && b_row < valid) {
            b0 = *reinterpret_cast<const unsigned int*>(&smem.sK[b_row][K0 + 2*t]);
            b1 = *reinterpret_cast<const unsigned int*>(&smem.sK[b_row][K0 + 2*t + 8]);
        } else { b0 = 0; b1 = 0; }
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
    // sKvalid[p_col] is false for out-of-chunk (p_col>=valid) AND invalid-gather
    // rows (idx<0 || idx>=s_kv), so it subsumes the chunk-range mask. Invalid ->
    // -inf (DeepSeek is_kv_valid semantics) so max_logits / lse stay exact.
    int p_col0 = n_base + 2*t, p_col1 = p_col0 + 1;
    bool v0 = smem.sKvalid[p_col0], v1 = smem.sKvalid[p_col1];
    if (g < heads_this_cta) {
        smem.sSP.score[g][p_col0] = v0 ? d0 * sm_scale : -INFINITY;
        smem.sSP.score[g][p_col1] = v1 ? d1 * sm_scale : -INFINITY;
    }
    if (g + 8 < heads_this_cta) {
        smem.sSP.score[g + 8][p_col0] = v0 ? d2 * sm_scale : -INFINITY;
        smem.sSP.score[g + 8][p_col1] = v1 ? d3 * sm_scale : -INFINITY;
    }
}

// ---- online softmax + score->prob + rescale reg O  (identical to decode) ----
__device__ __forceinline__ void softmax_and_rescale_reg_o(
    SmemLayout &smem, int heads_this_cta, int warp_id, int lane_id,
    float *o0, float *o1, float *o2, float *o3,
    float &row_max_g, float &row_sum_g, float &row_max_g8, float &row_sum_g8) {
    int g = lane_id >> 2;
    auto process_head = [&](int h, float &rmax, float &rsum, float *oa, float *ob) {
        if (h >= heads_this_cta) return;
        float chunk_max = -INFINITY;
        #pragma unroll
        for (int k = 0; k < KV_CHUNK; ++k) chunk_max = fmaxf(chunk_max, smem.sSP.score[h][k]);
        float old_max = rmax, new_max = fmaxf(old_max, chunk_max);
        float rescale = (old_max == -INFINITY) ? 0.0f : __expf(old_max - new_max);
        rmax = new_max; rsum *= rescale;
        #pragma unroll
        for (int i = 0; i < V_TILES_PER_WARP; ++i) { oa[i] *= rescale; ob[i] *= rescale; }
        float local_sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < KV_CHUNK; ++k) {
            float p = smem.sSP.score[h][k];
            float e = (p == -INFINITY) ? 0.0f : __expf(p - new_max);
            local_sum += e;
            // All warps recompute identical softmax; only warp 0 writes the shared
            // prob smem (covers all 16 heads via g and g+8). Gating the write to
            // one warp removes the redundant cross-warp WAW racecheck flags, while
            // the O-register rescale + rsum above stay per-warp (distinct O tiles).
            if (warp_id == 0) smem.sSP.prob[h][k] = __float2bfloat16_rn(e);
        }
        rsum += local_sum;
    };
    process_head(g,     row_max_g,  row_sum_g,  o0, o1);
    process_head(g + 8, row_max_g8, row_sum_g8, o2, o3);
}

// ---- HMMA P@V into register-resident O  (identical to decode) ----
__device__ __forceinline__ void hmma_pv_accum_reg(
    SmemLayout &smem, int heads_this_cta, int warp_id, int lane_id,
    float *o0, float *o1, float *o2, float *o3) {
    int g = lane_id >> 2, t = lane_id & 3;
    unsigned int zero = 0;
    #pragma unroll
    for (int i = 0; i < V_TILES_PER_WARP; ++i) {
        int v_base = (warp_id + i * NUM_WARPS) * MMA_N;
        float d0 = o0[i], d1 = o1[i], d2 = o2[i], d3 = o3[i];
        #pragma unroll
        for (int k_base = 0; k_base < KV_CHUNK; k_base += MMA_K) {
            unsigned int a0 = (g < heads_this_cta)
                ? *reinterpret_cast<const unsigned int*>(&smem.sSP.prob[g][k_base + 2*t]) : zero;
            unsigned int a1 = (g + 8 < heads_this_cta)
                ? *reinterpret_cast<const unsigned int*>(&smem.sSP.prob[g + 8][k_base + 2*t]) : zero;
            unsigned int a2 = (g < heads_this_cta)
                ? *reinterpret_cast<const unsigned int*>(&smem.sSP.prob[g][k_base + 2*t + 8]) : zero;
            unsigned int a3 = (g + 8 < heads_this_cta)
                ? *reinterpret_cast<const unsigned int*>(&smem.sSP.prob[g + 8][k_base + 2*t + 8]) : zero;
            unsigned int b0 = pack_bf16x2(smem.sK[k_base + 2*t    ][v_base + g],
                                          smem.sK[k_base + 2*t + 1][v_base + g]);
            unsigned int b1 = pack_bf16x2(smem.sK[k_base + 2*t + 8][v_base + g],
                                          smem.sK[k_base + 2*t + 9][v_base + g]);
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
        o0[i] = d0; o1[i] = d1; o2[i] = d2; o3[i] = d3;
    }
}

// ---- Main kernel: grid (s_q * head_blocks,). One query token x head group. ----
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
    const int g = lane_id >> 2;
    const int t = lane_id & 3;

    int my_topk = params.topk;
    if (params.topk_length) my_topk = min(my_topk, params.topk_length[s_q_idx]);
    if (my_topk < 0) my_topk = 0;

    float o0[V_TILES_PER_WARP], o1[V_TILES_PER_WARP];
    float o2[V_TILES_PER_WARP], o3[V_TILES_PER_WARP];
    #pragma unroll
    for (int i = 0; i < V_TILES_PER_WARP; ++i) { o0[i]=0.f; o1[i]=0.f; o2[i]=0.f; o3[i]=0.f; }
    float row_max_g  = -INFINITY, row_sum_g  = 0.0f;
    float row_max_g8 = -INFINITY, row_sum_g8 = 0.0f;

    // ---- Load Q for this query token's head group ----
    const cutlass::bfloat16_t *q_base =
        params.q + static_cast<size_t>(s_q_idx) * params.stride_q_s_q +
        head_base * params.stride_q_h_q;
    for (int h = 0; h < heads_this_cta; ++h) {
        const cutlass::bfloat16_t *row = q_base + h * params.stride_q_h_q;
        for (int d = tid; d < HEAD_DIM_QK; d += NUM_THREADS)
            smem.sQ[h][d] = reinterpret_cast<const __nv_bfloat16 *>(row)[d];
    }
    for (int h = heads_this_cta; h < BLOCK_M_HEADS; ++h)
        for (int d = tid; d < HEAD_DIM_QK; d += NUM_THREADS)
            smem.sQ[h][d] = __float2bfloat16_rn(0.0f);
    __syncthreads();

    const int *indices_base =
        params.indices + static_cast<size_t>(s_q_idx) * params.stride_indices_s_q;
    const __nv_bfloat16 *kv = reinterpret_cast<const __nv_bfloat16 *>(params.kv);

    for (int token_offset = 0; token_offset < my_topk; token_offset += KV_CHUNK) {
        int valid = min(KV_CHUNK, my_topk - token_offset);
        load_kv_chunk(smem, kv, params.stride_kv_s_kv, params.s_kv,
                      indices_base, token_offset, valid, warp_id, lane_id);
        __syncthreads();
        hmma_qk_dot(smem, heads_this_cta, valid, params.sm_scale, warp_id, lane_id);
        __syncthreads();
        softmax_and_rescale_reg_o(smem, heads_this_cta, warp_id, lane_id,
                                  o0, o1, o2, o3, row_max_g, row_sum_g,
                                  row_max_g8, row_sum_g8);
        __syncthreads();
        hmma_pv_accum_reg(smem, heads_this_cta, warp_id, lane_id, o0, o1, o2, o3);
        __syncthreads();
    }

    // ---- Epilogue: normalise (V==K), attn_sink mix, write out/max_logits/lse ----
    bool has_sink = params.attn_sink != nullptr;
    auto write_final = [&](int h, float rmax, float rsum, float *oa, float *ob) {
        if (h >= heads_this_cta) return;
        float sink_val = has_sink ? params.attn_sink[head_base + h] : 0.0f;
        float max_logits_val, lse_val, norm_factor;
        if (rsum == 0.0f) {
            // Lonely query: O=0, max_logits=-inf, lse=+inf (DeepSeek convention).
            max_logits_val = -INFINITY; lse_val = INFINITY; norm_factor = 0.0f;
        } else {
            max_logits_val = rmax;                  // natural-log row max
            float log_sum = logf(rsum) + rmax;      // orig_lse
            if (has_sink) {
                norm_factor = (1.0f / (1.0f + __expf(sink_val - log_sum))) / rsum;
            } else {
                norm_factor = 1.0f / rsum;
            }
            lse_val = log_sum;   // orig_lse (DeepSeek ref reports orig_lse, not sink-merged)
        }
        // Per-head softmax state (rmax/rsum) is replicated across the 4 lanes
        // sharing g = lane_id>>2, so head h's stats live on lanes 4h..4h+3 — write
        // from the t==0 lane (warp 0) of that head, NOT lane_id==0 (which only owns
        // heads g==0 and g+8==8).
        if (warp_id == 0 && t == 0) {
            size_t off = static_cast<size_t>(s_q_idx) * params.stride_ml_s_q + head_base + h;
            params.max_logits[off] = max_logits_val;
            params.lse[off] = lse_val;
        }
        cutlass::bfloat16_t *out_row =
            params.out + static_cast<size_t>(s_q_idx) * params.stride_o_s_q +
            (head_base + h) * params.stride_o_h_q;
        #pragma unroll
        for (int i = 0; i < V_TILES_PER_WARP; ++i) {
            int v_base = (warp_id + i * NUM_WARPS) * MMA_N;
            reinterpret_cast<__nv_bfloat16 *>(out_row)[v_base + 2*t]     =
                __float2bfloat16_rn(oa[i] * norm_factor);
            reinterpret_cast<__nv_bfloat16 *>(out_row)[v_base + 2*t + 1] =
                __float2bfloat16_rn(ob[i] * norm_factor);
        }
    };
    write_final(g,     row_max_g,  row_sum_g,  o0, o1);
    write_final(g + 8, row_max_g8, row_sum_g8, o2, o3);
}

}  // namespace prefill_hmma
}  // namespace sm120
}  // namespace dsv4_kernel
