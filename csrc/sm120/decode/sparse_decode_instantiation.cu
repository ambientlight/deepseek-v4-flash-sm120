// CUDA translation unit: pulls in the kernel template and provides the
// launch entry point seen by the C++-only api/sparse_decode.cpp.

#include "sm120/decode/sparse_decode.h"

#ifdef DSV4_ENABLE_SCALAR_REFERENCE_KERNEL
#include "sm120/decode/sparse_decode_kernel_scalar_ref.cuh"
#else
#include "sm120/decode/sparse_decode_kernel.cuh"
#endif

#include "sm120/decode/split_kv_combine.cuh"

namespace dsv4_kernel {
namespace sm120 {

void launch_dsv4_sparse_decode_v32(const SparseAttnDecodeParams &params) {
    int num_head_blocks = ceil_div(params.h_q, BLOCK_M_HEADS);
    int num_splits = max(params.num_sm_parts, 1);

    dim3 grid(static_cast<unsigned>(params.b * params.s_q),
              static_cast<unsigned>(num_head_blocks),
              static_cast<unsigned>(num_splits));
    dim3 block(NUM_THREADS, 1, 1);
    size_t smem_size = sizeof(SmemLayout);
    DSV4_CHECK_CUDA(cudaFuncSetAttribute(
        dsv4_sparse_decode_kernel<BLOCK_M_HEADS>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_size)));
    dsv4_sparse_decode_kernel<BLOCK_M_HEADS>
        <<<grid, block, smem_size, params.stream>>>(params);
    DSV4_CHECK_CUDA_LAUNCH();

    // If split-KV was used, launch the combine kernel
    if (num_splits > 1) {
        int b_times_sq = params.b * params.s_q;
        dim3 combine_grid(static_cast<unsigned>(b_times_sq),
                          static_cast<unsigned>(num_head_blocks));
        dim3 combine_block(COMBINE_THREADS);
        dsv4_split_kv_combine<<<combine_grid, combine_block, 0, params.stream>>>(
            params.lse_accum,
            params.o_accum,
            params.attn_sink,
            params.lse,
            params.out,
            num_splits,
            params.h_q,
            params.d_v,
            params.stride_lse_accum_split,
            params.h_q,  // stride_lse_accum_bs = h_q
            params.stride_o_accum_split,
            params.stride_o_accum_s_q,
            params.stride_o_accum_h_q,
            params.stride_lse_b,
            params.stride_lse_s_q,
            params.stride_o_b,
            params.stride_o_s_q,
            params.stride_o_h_q,
            b_times_sq);
        DSV4_CHECK_CUDA_LAUNCH();
    }
}

}  // namespace sm120
}  // namespace dsv4_kernel
