// CUDA translation unit: pulls in the prefill kernel template and provides
// the launch entry point seen by the C++-only api/sparse_prefill.cpp.
//
// Stage 1 builds the scalar reference kernel (correctness oracle). Stage 2's
// HMMA kernel (sparse_prefill_kernel.cuh) replaces it once it lands; until
// then, building with DSV4_ENABLE_SCALAR_PREFILL_KERNEL selects the scalar
// path. Define it in setup.py for the prefill TU only.

#include "sm120/prefill/sparse_prefill.h"

#if defined(DSV4_PREFILL_USE_HMMA)
#include "sm120/prefill/sparse_prefill_kernel.cuh"
namespace dsv4_kernel { namespace sm120 {
namespace prefill_active = prefill_hmma;
}}  // namespace
#else
#ifndef DSV4_ENABLE_SCALAR_PREFILL_KERNEL
#define DSV4_ENABLE_SCALAR_PREFILL_KERNEL
#endif
#include "sm120/prefill/sparse_prefill_kernel_scalar_ref.cuh"
namespace dsv4_kernel { namespace sm120 {
namespace prefill_active = prefill_scalar;
}}  // namespace
#endif

namespace dsv4_kernel {
namespace sm120 {

void launch_dsv4_sparse_prefill(const SparseAttnPrefillParams &params) {
    using namespace prefill_active;
    int num_head_blocks = ceil_div(params.h_q, BLOCK_M_HEADS);
    dim3 grid(static_cast<unsigned>(params.s_q * num_head_blocks), 1, 1);
    dim3 block(NUM_THREADS, 1, 1);
    size_t smem_size = sizeof(SmemLayout);
    DSV4_CHECK_CUDA(cudaFuncSetAttribute(
        dsv4_sparse_prefill_kernel<BLOCK_M_HEADS>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_size)));
    dsv4_sparse_prefill_kernel<BLOCK_M_HEADS>
        <<<grid, block, smem_size, params.stream>>>(params);
    DSV4_CHECK_CUDA_LAUNCH();
}

}  // namespace sm120
}  // namespace dsv4_kernel
