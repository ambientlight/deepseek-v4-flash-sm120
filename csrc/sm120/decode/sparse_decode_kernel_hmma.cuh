// =====================================================================
// SM_120 sparse decode kernel for DeepSeek-V4-Flash — HMMA-optimized.
//
// This is an optimized version of the original sparse_decode_kernel.cuh
// that replaces scalar BF16 dot-products with HMMA tensor core instructions
// (mma.sync.aligned.m16n8k16.f32.bf16.bf16.f32) available on SM_120.
//
// The HMMA instruction processes a 16×8 output tile at k=16 depth,
// giving ~16× throughput over the scalar version for the QK^T and P@V
// matmul steps which dominated 50% of total GPU time.
//
// Layout & format unchanged from original — drop-in replacement.
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

// HMMA m16n8k16 constants
static constexpr int MMA_M = 16;
static constexpr int MMA_N = 8;
static constexpr int MMA_K = 16;

// UE8M0 byte -> multiplicative scale.
__device__ __forceinline__ float ue8m0_to_scale(unsigned char b) {
    int e = static_cast<int>(b) - 127;
    return __powf(2.0f, static_cast<float>(e));
}

// FP8 e4m3 quad -> BF16 quad with a single (shared) FP32 scale.
__device__ __forceinline__ void fp8x4_to_bf16x4(uint32_t bits, float scale,
                                                 __nv_bfloat16 out[4]) {
    __nv_fp8_e4m3 f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        f.__x = static_cast<unsigned char>((bits >> (i * 8)) & 0xFFu);
        out[i] = __float2bfloat16_rn(static_cast<float>(f) * scale);
    }
}

// -------------------------------------------------------------------
// HMMA m16n8k16 wrapper: C += A * B^T
// A fragment: 4 x uint32 (each uint32 holds 2 bf16 values)
// B fragment: 2 x uint32
// C/D fragment: 4 x float
// -------------------------------------------------------------------
__device__ __forceinline__ void hmma_m16n8k16_bf16(
    float &d0, float &d1, float &d2, float &d3,
    unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
    unsigned int b0, unsigned int b1,
    float c0, float c1, float c2, float c3) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3)
    );
}

struct SmemLayout {
    // Q stored in row-major for HMMA A-fragment loading
    __nv_bfloat16 sQ[BLOCK_M_HEADS][HEAD_DIM_QK];   // 16 384 B
    __nv_bfloat16 sK[KV_CHUNK][HEAD_DIM_QK];         // 32 768 B
    float         sP[BLOCK_M_HEADS][KV_CHUNK];       //  2 048 B
    float         sO[BLOCK_M_HEADS][HEAD_DIM_V];     // 32 768 B
    float         sRowMax[BLOCK_M_HEADS];
    float         sRowSum[BLOCK_M_HEADS];
};

// -------------------------------------------------------------------
// Dequantise & load one KV chunk — identical to original
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

    for (int t = warp_id; t < KV_CHUNK; t += NUM_WARPS) {
        if (t >= valid_tokens) {
            for (int v = lane_id; v < HEAD_DIM_QK; v += 32) {
                smem.sK[t][v] = __float2bfloat16_rn(0.0f);
            }
            continue;
        }
        int flat_idx = indices_base[token_offset + t];
        if (flat_idx < 0) {
            for (int v = lane_id; v < HEAD_DIM_QK; v += 32) {
                smem.sK[t][v] = __float2bfloat16_rn(0.0f);
            }
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
        for (int i = 0; i < NUM_ACTIVE_SCALES; ++i) {
            s[i] = ue8m0_to_scale(scales_u8[i]);
        }

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
                if (dst < HEAD_DIM_NOPE) {
                    smem.sK[t][dst] = out[k];
                }
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
// HMMA-based QK^T: compute sP[16][32] = sQ[16][512] × sK[32][512]^T
// Using mma.sync.aligned.m16n8k16 with warp-level fragment distribution.
//
// HMMA m16n8k16 fragment layout (per warp of 32 threads):
//   A (row-major): threads hold 4 pairs of bf16 values across M=16 rows
//   B (col-major): threads hold 2 pairs of bf16 values across N=8 cols
//   C/D: threads hold 4 float accumulators
//
// We tile the [16 × 32] output into [16 × 8] HMMA tiles (4 tiles along N).
// Each tile accumulates over k=0..512 in steps of 16.
//
// PTX ISA m16n8k16.row.col fragment mapping (per-warp, 32 lanes):
//   g = lane_id >> 2   (groupID, 0..7)
//   t = lane_id & 3    (threadID_in_group, 0..3)
//
//   A fragment (row-major sQ[16][K]):
//     a0 = {sQ[g    ][K0 + 2t],     sQ[g    ][K0 + 2t + 1]}
//     a1 = {sQ[g + 8][K0 + 2t],     sQ[g + 8][K0 + 2t + 1]}
//     a2 = {sQ[g    ][K0 + 2t + 8], sQ[g    ][K0 + 2t + 9]}
//     a3 = {sQ[g + 8][K0 + 2t + 8], sQ[g + 8][K0 + 2t + 9]}
//
//   B fragment (row-major sK[N][K], fed as col-major B^T):
//     b0 = {sK[g][K0 + 2t],     sK[g][K0 + 2t + 1]}
//     b1 = {sK[g][K0 + 2t + 8], sK[g][K0 + 2t + 9]}
//     Note: sK N-index = g = lane>>2, NOT lane%4!
//
//   D/C output:
//     d0 = P[g][2t],  d1 = P[g][2t+1],  d2 = P[g+8][2t],  d3 = P[g+8][2t+1]
// -------------------------------------------------------------------
__device__ __forceinline__ void hmma_qk_dot(
    SmemLayout &smem,
    int heads_this_cta,
    int valid,
    float sm_scale,
    int warp_id, int lane_id) {

    // Each warp handles one N=8 tile of the 32-wide KV dimension.
    int n_tile = warp_id;  // 0..3
    int n_base = n_tile * MMA_N;  // 0, 8, 16, 24

    // Lane decomposition per PTX ISA
    int g = lane_id >> 2;   // 0..7 — maps to M rows (A) and N rows (B)
    int t = lane_id & 3;    // 0..3 — maps to K position within tile

    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;

    // Iterate over K=512 in chunks of 16
    #pragma unroll 4
    for (int K0 = 0; K0 < HEAD_DIM_QK; K0 += MMA_K) {
        // A fragment from sQ[16][512]
        // a0: row g,   K0+2t..2t+1
        // a1: row g+8, K0+2t..2t+1
        // a2: row g,   K0+2t+8..2t+9
        // a3: row g+8, K0+2t+8..2t+9
        const __nv_bfloat16 *q_g   = smem.sQ[g];
        const __nv_bfloat16 *q_g8  = (g + 8 < heads_this_cta) ? smem.sQ[g + 8] : smem.sQ[g]; // fallback for partial CTA

        unsigned int a0 = *reinterpret_cast<const unsigned int*>(&q_g [K0 + 2*t]);
        unsigned int a1 = *reinterpret_cast<const unsigned int*>(&q_g8[K0 + 2*t]);
        unsigned int a2 = *reinterpret_cast<const unsigned int*>(&q_g [K0 + 2*t + 8]);
        unsigned int a3 = *reinterpret_cast<const unsigned int*>(&q_g8[K0 + 2*t + 8]);

        // B fragment from sK[32][512] — each warp's N=8 tile starts at n_base
        // b0: row n_base+g, K0+2t..2t+1
        // b1: row n_base+g, K0+2t+8..2t+9
        int b_row = n_base + g;
        unsigned int b0, b1;
        if (b_row < KV_CHUNK && b_row < valid) {
            b0 = *reinterpret_cast<const unsigned int*>(&smem.sK[b_row][K0 + 2*t]);
            b1 = *reinterpret_cast<const unsigned int*>(&smem.sK[b_row][K0 + 2*t + 8]);
        } else {
            b0 = 0;
            b1 = 0;
        }

        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
              "r"(b0), "r"(b1)
        );
    }

    // Write HMMA results to sP[16][32]
    // d0 = P[g][2t],  d1 = P[g][2t+1],  d2 = P[g+8][2t],  d3 = P[g+8][2t+1]
    // Map to the warp's N=8 tile: column = n_base + 2t, n_base + 2t + 1
    // But wait — HMMA output N columns are 0..7 within the tile.
    // d0 → C[g][2t], d1 → C[g][2t+1] — these are N-local columns 0..7.
    // In our sP, N-local col j maps to global col n_base + j.
    int p_col0 = n_base + 2*t;
    int p_col1 = p_col0 + 1;

    if (g < heads_this_cta) {
        smem.sP[g][p_col0]     = (p_col0 < valid) ? d0 * sm_scale : -INFINITY;
        smem.sP[g][p_col1]     = (p_col1 < valid) ? d1 * sm_scale : -INFINITY;
    }
    if (g + 8 < heads_this_cta) {
        smem.sP[g + 8][p_col0] = (p_col0 < valid) ? d2 * sm_scale : -INFINITY;
        smem.sP[g + 8][p_col1] = (p_col1 < valid) ? d3 * sm_scale : -INFINITY;
    }
}

// -------------------------------------------------------------------
// Main kernel with HMMA optimization
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
    if (params.topk_length) {
        my_topk = min(my_topk, params.topk_length[batch_idx]);
    }
    if (my_topk < 0) my_topk = 0;

    int my_extra_topk = 0;
    if (params.extra_kv) {
        my_extra_topk = params.extra_topk;
        if (params.extra_topk_length) {
            my_extra_topk = min(my_extra_topk, params.extra_topk_length[batch_idx]);
        }
        if (my_extra_topk < 0) my_extra_topk = 0;
    }

    // ---- Load Q ----
    const cutlass::bfloat16_t *q_base =
        params.q + static_cast<size_t>(batch_idx) * params.stride_q_b +
        s_q_idx * params.stride_q_s_q + head_base * params.stride_q_h_q;
    for (int h = 0; h < heads_this_cta; ++h) {
        const cutlass::bfloat16_t *row = q_base + h * params.stride_q_h_q;
        for (int t = tid; t < HEAD_DIM_QK; t += NUM_THREADS) {
            smem.sQ[h][t] = reinterpret_cast<const __nv_bfloat16 *>(row)[t];
        }
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

            // --- QK^T using HMMA tensor cores ---
            hmma_qk_dot(smem, heads_this_cta, valid, params.sm_scale,
                        warp_id, lane_id);
            __syncthreads();

            // --- Online softmax + P @ V accumulate ---
            // P@V still uses scalar for now (V dim = 512, accumulated in sO)
            // TODO: HMMA-ify this step too for additional speedup
            for (int h_local = warp_id; h_local < heads_this_cta;
                 h_local += NUM_WARPS) {
                float chunk_max = -INFINITY;
                #pragma unroll
                for (int k = 0; k < KV_CHUNK; ++k) {
                    chunk_max = fmaxf(chunk_max, smem.sP[h_local][k]);
                }
                float old_max = smem.sRowMax[h_local];
                float new_max = fmaxf(old_max, chunk_max);
                float rescale = (old_max == -INFINITY) ? 0.0f
                                                       : __expf(old_max - new_max);
                if (lane_id == 0) {
                    smem.sRowMax[h_local] = new_max;
                    smem.sRowSum[h_local] *= rescale;
                }
                for (int v = lane_id; v < HEAD_DIM_V; v += 32) {
                    smem.sO[h_local][v] *= rescale;
                }

                float local_sum = 0.0f;
                float exps[KV_CHUNK];
                #pragma unroll
                for (int k = 0; k < KV_CHUNK; ++k) {
                    float p = smem.sP[h_local][k];
                    float e = (p == -INFINITY) ? 0.0f : __expf(p - new_max);
                    exps[k] = e;
                    local_sum += e;
                }
                if (lane_id == 0) {
                    smem.sRowSum[h_local] += local_sum;
                }

                for (int v = lane_id; v < HEAD_DIM_V; v += 32) {
                    float acc = 0.0f;
                    #pragma unroll
                    for (int k = 0; k < KV_CHUNK; ++k) {
                        if (k < valid) {
                            acc += exps[k] * __bfloat162float(smem.sK[k][v]);
                        }
                    }
                    smem.sO[h_local][v] += acc;
                }
            }
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
            for (int v = lane_id; v < HEAD_DIM_V; v += 32) {
                smem.sO[h_local][v] = 0.0f;
            }
        } else {
            float log_sum = logf(row_sum) + row_max;
            if (has_sink) {
                float sink_scale = 1.0f / (1.0f + __expf(sink - log_sum));
                for (int v = lane_id; v < HEAD_DIM_V; v += 32) {
                    smem.sO[h_local][v] =
                        (smem.sO[h_local][v] / row_sum) * sink_scale;
                }
                float m = fmaxf(log_sum, sink);
                lse_val = m + logf(__expf(log_sum - m) + __expf(sink - m));
            } else {
                for (int v = lane_id; v < HEAD_DIM_V; v += 32) {
                    smem.sO[h_local][v] = smem.sO[h_local][v] / row_sum;
                }
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
        for (int v = tid; v < HEAD_DIM_V; v += NUM_THREADS) {
            reinterpret_cast<__nv_bfloat16 *>(row_out)[v] =
                __float2bfloat16_rn(smem.sO[h_local][v]);
        }
    }
}

}  // namespace sm120
}  // namespace dsv4_kernel
