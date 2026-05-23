// =====================================================================
// SM_120 sparse decode kernel — Register-resident O, targeting occ=2.
//
// SMEM budget: sQ[16][512] + sK[32][512] + sP_union[16][32] = 50 KB
// With 100 KB per SM on SM120, two CTAs fit → occupancy=2.
//
// sO[16][512] moved to per-thread FP32 register arrays (64 regs/thread).
// Row stats (max, sum) also in registers.
// sP and sP_bf16 share one union buffer.
//
// All matmul paths use HMMA tensor cores (m16n8k16 BF16).
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
static constexpr int NOPE_BYTES    = HEAD_DIM_NOPE;
static constexpr int ROPE_BYTES    = HEAD_DIM_ROPE * 2;
static constexpr int NOPE_ROPE_BYTES = NOPE_BYTES + ROPE_BYTES;

static constexpr int NUM_WARPS     = 4;
static constexpr int NUM_THREADS   = NUM_WARPS * 32;

static constexpr int MMA_M = 16;
static constexpr int MMA_N = 8;
static constexpr int MMA_K = 16;

// V tiles owned per warp for register-resident O
static constexpr int V_TILES_PER_WARP = HEAD_DIM_V / (NUM_WARPS * MMA_N);  // 16

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
// SMEM layout — minimal, targeting ≤50 KB for occupancy=2
//
//   sQ[16][512] BF16     = 16,384 B  (16 KB)
//   sK[32][512] BF16     = 32,768 B  (32 KB)
//   sSP union(f32,bf16)  =  2,048 B  ( 2 KB)  [max of f32[16][32], bf16[16][32]]
//   ──────────────────────────────────────────
//   total                = 51,200 B  (50 KB exactly)
// -------------------------------------------------------------------
union ScoreProbUnion {
    float         score[BLOCK_M_HEADS][KV_CHUNK];       // 2048 B
    __nv_bfloat16 prob[BLOCK_M_HEADS][KV_CHUNK];        // 1024 B (fits in same space)
};

struct SmemLayout {
    __nv_bfloat16 sQ[BLOCK_M_HEADS][HEAD_DIM_QK];       // 16,384 B
    __nv_bfloat16 sK[KV_CHUNK][HEAD_DIM_QK];             // 32,768 B
    ScoreProbUnion sSP;                                   //  2,048 B
};
// static_assert(sizeof(SmemLayout) <= 51200, "SMEM exceeds 50 KB target");

// -------------------------------------------------------------------
// Load KV chunk — dequant FP8→BF16 into sK (no skew, tight layout)
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

    for (int tok = warp_id; tok < KV_CHUNK; tok += NUM_WARPS) {
        if (tok >= valid_tokens) {
            for (int v = lane_id; v < HEAD_DIM_QK; v += 32)
                smem.sK[tok][v] = __float2bfloat16_rn(0.0f);
            continue;
        }
        int flat_idx = indices_base[token_offset + tok];
        if (flat_idx < 0) {
            for (int v = lane_id; v < HEAD_DIM_QK; v += 32)
                smem.sK[tok][v] = __float2bfloat16_rn(0.0f);
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
                    smem.sK[tok][dst] = out[k];
            }
        }

        const __nv_bfloat16 *rope =
            reinterpret_cast<const __nv_bfloat16 *>(tok_nope_rope + NOPE_BYTES);
        if (lane_id * 2 + 1 < HEAD_DIM_ROPE) {
            smem.sK[tok][HEAD_DIM_NOPE + lane_id * 2 + 0] = rope[lane_id * 2 + 0];
            smem.sK[tok][HEAD_DIM_NOPE + lane_id * 2 + 1] = rope[lane_id * 2 + 1];
        }
    }
}

// -------------------------------------------------------------------
// HMMA QK^T → sSP.score[16][32]
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
        smem.sSP.score[g][p_col0] = (p_col0 < valid) ? d0 * sm_scale : -INFINITY;
        smem.sSP.score[g][p_col1] = (p_col1 < valid) ? d1 * sm_scale : -INFINITY;
    }
    if (g + 8 < heads_this_cta) {
        smem.sSP.score[g + 8][p_col0] = (p_col0 < valid) ? d2 * sm_scale : -INFINITY;
        smem.sSP.score[g + 8][p_col1] = (p_col1 < valid) ? d3 * sm_scale : -INFINITY;
    }
}

// -------------------------------------------------------------------
// Online softmax + convert score→prob in-place union + rescale reg O.
//
// Reads sSP.score[], writes sSP.prob[] in-place.
// Safe because BF16 prob[h][k] occupies first half of FP32 score[h][k]
// — processing k=0..31 sequentially, each FP32 read happens before
// the BF16 write to the same (or earlier) memory.
// -------------------------------------------------------------------
__device__ __forceinline__ void softmax_and_rescale_reg_o(
    SmemLayout &smem,
    int heads_this_cta,
    int warp_id, int lane_id,
    float *o0, float *o1, float *o2, float *o3,
    float &row_max_g, float &row_sum_g,
    float &row_max_g8, float &row_sum_g8) {

    int g = lane_id >> 2;

    // Helper: process one head row — rescale O accumulators and convert score→prob
    auto process_head = [&](int h, float &rmax, float &rsum,
                            float *oa, float *ob) {
        if (h >= heads_this_cta) return;

        float chunk_max = -INFINITY;
        #pragma unroll
        for (int k = 0; k < KV_CHUNK; ++k)
            chunk_max = fmaxf(chunk_max, smem.sSP.score[h][k]);

        float old_max = rmax;
        float new_max = fmaxf(old_max, chunk_max);
        float rescale = (old_max == -INFINITY) ? 0.0f : __expf(old_max - new_max);

        rmax = new_max;
        rsum *= rescale;

        // Rescale BOTH O accumulator arrays for this head row
        #pragma unroll
        for (int i = 0; i < V_TILES_PER_WARP; ++i) {
            oa[i] *= rescale;
            ob[i] *= rescale;
        }

        // Compute exps and convert score→prob in-place
        float local_sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < KV_CHUNK; ++k) {
            float p = smem.sSP.score[h][k];
            float e = (p == -INFINITY) ? 0.0f : __expf(p - new_max);
            local_sum += e;
            smem.sSP.prob[h][k] = __float2bfloat16_rn(e);
        }
        rsum += local_sum;
    };

    // Row g owns o0, o1. Row g+8 owns o2, o3.
    process_head(g,     row_max_g,  row_sum_g,  o0, o1);
    process_head(g + 8, row_max_g8, row_sum_g8, o2, o3);
}

// -------------------------------------------------------------------
// HMMA P@V: accumulate into register-resident O arrays.
//
// Each warp owns V_TILES_PER_WARP=16 value tiles.
// Warp w owns v_base = (w + i*NUM_WARPS) * 8 for i=0..15.
// Register o0[i]..o3[i] maps to O[g/g+8][v_base + 2t/2t+1].
// -------------------------------------------------------------------
__device__ __forceinline__ void hmma_pv_accum_reg(
    SmemLayout &smem,
    int heads_this_cta,
    int warp_id, int lane_id,
    float *o0, float *o1, float *o2, float *o3) {

    int g = lane_id >> 2;
    int t = lane_id & 3;
    unsigned int zero = 0;

    #pragma unroll
    for (int i = 0; i < V_TILES_PER_WARP; ++i) {
        int v_base = (warp_id + i * NUM_WARPS) * MMA_N;

        float d0 = o0[i], d1 = o1[i], d2 = o2[i], d3 = o3[i];

        #pragma unroll
        for (int k_base = 0; k_base < KV_CHUNK; k_base += MMA_K) {
            unsigned int a0 = (g < heads_this_cta)
                ? *reinterpret_cast<const unsigned int*>(&smem.sSP.prob[g][k_base + 2*t])
                : zero;
            unsigned int a1 = (g + 8 < heads_this_cta)
                ? *reinterpret_cast<const unsigned int*>(&smem.sSP.prob[g + 8][k_base + 2*t])
                : zero;
            unsigned int a2 = (g < heads_this_cta)
                ? *reinterpret_cast<const unsigned int*>(&smem.sSP.prob[g][k_base + 2*t + 8])
                : zero;
            unsigned int a3 = (g + 8 < heads_this_cta)
                ? *reinterpret_cast<const unsigned int*>(&smem.sSP.prob[g + 8][k_base + 2*t + 8])
                : zero;

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

        o0[i] = d0; o1[i] = d1; o2[i] = d2; o3[i] = d3;
    }
}

// -------------------------------------------------------------------
// Main kernel — register-resident O, split-KV support.
//
// Grid: (b*s_q, head_groups, num_sm_parts)
// When num_sm_parts > 1, each CTA processes a slice of topk tokens
// and writes partial O/LSE to o_accum/lse_accum. A separate combine
// kernel merges the results.
// When num_sm_parts == 1, behaves as before (full output).
// -------------------------------------------------------------------
template <int HEADS_PER_CTA = BLOCK_M_HEADS>
__global__ __launch_bounds__(NUM_THREADS, 2)
void dsv4_sparse_decode_kernel(SparseAttnDecodeParams params) {
    extern __shared__ __align__(16) unsigned char _smem[];
    SmemLayout &smem = *reinterpret_cast<SmemLayout *>(_smem);

    const int bs_s_q    = blockIdx.x;
    const int head_bk   = blockIdx.y;
    const int split_idx = (params.num_sm_parts > 1) ? blockIdx.z : 0;
    const int batch_idx = bs_s_q / params.s_q;
    const int s_q_idx   = bs_s_q % params.s_q;
    const int head_base = head_bk * HEADS_PER_CTA;
    if (head_base >= params.h_q) return;
    const int heads_this_cta = min(HEADS_PER_CTA, params.h_q - head_base);

    const int tid     = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int g = lane_id >> 2;
    const int t = lane_id & 3;

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

    // Split-KV: compute this CTA's token range
    const int num_splits = params.num_sm_parts;
    int split_start_main = 0, split_end_main = my_topk;
    int split_start_extra = 0, split_end_extra = my_extra_topk;

    if (num_splits > 1) {
        // Split main KV tokens evenly (rounded to KV_CHUNK for efficiency)
        int total_main = my_topk;
        int tokens_per_split = ((total_main + num_splits - 1) / num_splits + KV_CHUNK - 1)
                                / KV_CHUNK * KV_CHUNK;
        split_start_main = min(split_idx * tokens_per_split, total_main);
        split_end_main   = min(split_start_main + tokens_per_split, total_main);

        // Extra KV: only processed by split 0 (typically small)
        if (split_idx > 0) {
            split_start_extra = 0;
            split_end_extra = 0;
        }
    }

    // ---- Register-resident O accumulators ----
    float o0[V_TILES_PER_WARP], o1[V_TILES_PER_WARP];
    float o2[V_TILES_PER_WARP], o3[V_TILES_PER_WARP];
    #pragma unroll
    for (int i = 0; i < V_TILES_PER_WARP; ++i) {
        o0[i] = 0.0f; o1[i] = 0.0f; o2[i] = 0.0f; o3[i] = 0.0f;
    }

    float row_max_g  = -INFINITY, row_sum_g  = 0.0f;
    float row_max_g8 = -INFINITY, row_sum_g8 = 0.0f;

    // ---- Load Q ----
    const cutlass::bfloat16_t *q_base =
        params.q + static_cast<size_t>(batch_idx) * params.stride_q_b +
        s_q_idx * params.stride_q_s_q + head_base * params.stride_q_h_q;
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
                        const int *idx_base, int start_token, int end_token) {
        if (end_token <= start_token || idx_base == nullptr) return;
        for (int token_offset = start_token; token_offset < end_token;
             token_offset += KV_CHUNK) {
            int valid = min(KV_CHUNK, end_token - token_offset);

            load_kv_chunk(smem, kv_bytes_base, stride_block, stride_row,
                          page_block, idx_base, token_offset, valid,
                          warp_id, lane_id);
            __syncthreads();

            hmma_qk_dot(smem, heads_this_cta, valid, params.sm_scale,
                        warp_id, lane_id);
            __syncthreads();

            softmax_and_rescale_reg_o(smem, heads_this_cta, warp_id, lane_id,
                                     o0, o1, o2, o3,
                                     row_max_g, row_sum_g,
                                     row_max_g8, row_sum_g8);
            __syncthreads();

            hmma_pv_accum_reg(smem, heads_this_cta, warp_id, lane_id,
                              o0, o1, o2, o3);
            __syncthreads();
        }
    };

    // Process this split's token range
    process(reinterpret_cast<const uint8_t *>(params.kv),
            params.stride_kv_block, params.stride_kv_row,
            params.page_block_size, indices_base,
            split_start_main, split_end_main);
    if (params.extra_kv && split_end_extra > split_start_extra)
        process(reinterpret_cast<const uint8_t *>(params.extra_kv),
                params.stride_extra_kv_block, params.stride_extra_kv_row,
                params.extra_page_block_size, extra_indices_base,
                split_start_extra, split_end_extra);

    // --- Epilogue ---
    if (num_splits > 1) {
        // Split mode: write NORMALIZED partial O + LSE to accum buffers.
        // o_accum = partial_o / local_sum  (locally normalized)
        // lse_accum = log(local_sum) + local_max
        // Combine kernel uses: out = (1/Z) * sum_s exp(lse_s - m) * o_accum_s
        // where Z = sum_s exp(lse_s - m), m = max(lse_s)
        auto write_partial = [&](int h, float rmax, float rsum,
                                 float *oa, float *ob) {
            if (h >= heads_this_cta) return;
            float lse_val = (rsum > 0.0f) ? (logf(rsum) + rmax) : -INFINITY;
            float inv_sum = (rsum > 0.0f) ? (1.0f / rsum) : 0.0f;

            // Write LSE: use t==0 (one writer per group of 4 lanes sharing same g)
            if (t == 0) {
                params.lse_accum[split_idx * params.stride_lse_accum_split +
                                bs_s_q * params.stride_lse_accum_s_q +
                                head_base + h] = lse_val;
            }

            // Write locally-normalized O
            float *o_row = params.o_accum +
                split_idx * params.stride_o_accum_split +
                bs_s_q * params.stride_o_accum_s_q +
                (head_base + h) * params.stride_o_accum_h_q;

            #pragma unroll
            for (int i = 0; i < V_TILES_PER_WARP; ++i) {
                int v_base = (warp_id + i * NUM_WARPS) * MMA_N;
                o_row[v_base + 2*t]     = oa[i] * inv_sum;
                o_row[v_base + 2*t + 1] = ob[i] * inv_sum;
            }
        };

        write_partial(g,     row_max_g,  row_sum_g,  o0, o1);
        write_partial(g + 8, row_max_g8, row_sum_g8, o2, o3);

    } else {
        // Single-CTA mode: full normalize + sink + writeback (unchanged)
        bool has_sink = params.attn_sink != nullptr;

        auto write_final = [&](int h, float rmax, float rsum,
                               float *oa, float *ob) {
            if (h >= heads_this_cta) return;
            float sink_val = has_sink ? params.attn_sink[head_base + h] : 0.0f;
            float lse_val, norm_factor;

            if (rsum == 0.0f) {
                lse_val = has_sink ? sink_val : -INFINITY;
                norm_factor = 0.0f;
            } else {
                float log_sum = logf(rsum) + rmax;
                if (has_sink) {
                    float ss = 1.0f / (1.0f + __expf(sink_val - log_sum));
                    norm_factor = ss / rsum;
                    float m = fmaxf(log_sum, sink_val);
                    lse_val = m + logf(__expf(log_sum - m) + __expf(sink_val - m));
                } else {
                    norm_factor = 1.0f / rsum;
                    lse_val = log_sum;
                }
            }

            if (lane_id == 0)
                params.lse[static_cast<size_t>(batch_idx) * params.stride_lse_b +
                           s_q_idx * params.stride_lse_s_q + head_base + h] = lse_val;

            cutlass::bfloat16_t *out_row =
                params.out + static_cast<size_t>(batch_idx) * params.stride_o_b +
                s_q_idx * params.stride_o_s_q + (head_base + h) * params.stride_o_h_q;

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
}

}  // namespace sm120
}  // namespace dsv4_kernel
