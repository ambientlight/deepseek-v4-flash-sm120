#include "api/moe_gemm.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "sm120/moe/fused_fp4_moe_gemv.cuh"
#include "sm120/moe/fused_fp4_moe_gemv_v3.cuh"
#include "sm120/moe/fused_swiglu_gemm2.cuh"
#include "sm120/moe/fused_swiglu_gemm2_v3.cuh"
#include "sm120/moe/sort_by_expert.cuh"
#include "sm120/moe/grouped_fp4_moe_gemm.cuh"
#include "sm120/moe/grouped_fp4_moe_gemm2.cuh"
#include "sm120/moe/grouped_fp4_moe_gemm_v2.cuh"
#include "sm120/moe/grouped_fp4_moe_gemm2_v2.cuh"
#include "sm120/moe/grouped_fp4_moe_gemm_v3.cuh"
#include "sm120/moe/grouped_fp4_moe_gemm2_v3.cuh"
#include "sm120/moe/grouped_fp4_moe_gemm_v4.cuh"
#include "sm120/moe/grouped_fp4_moe_gemm2_v4.cuh"

namespace dsv4_kernel {

at::Tensor fp4_moe_gemm(
    const at::Tensor &A_bf16,
    const at::Tensor &B_packed,
    const at::Tensor &B_scale,
    const at::Tensor &token_ids,
    const at::Tensor &expert_ids,
    int N, int K)
{
    const int num_slots = token_ids.size(0);
    at::cuda::CUDAGuard guard(A_bf16.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    auto C = at::empty({num_slots, N}, A_bf16.options());

    int stride_bn = K / 2;
    int stride_bsn = K / 32;
    int64_t expert_b_stride = static_cast<int64_t>(N) * stride_bn;
    int64_t expert_bs_stride = static_cast<int64_t>(N) * stride_bsn;

    sm120::launch_fused_fp4_moe_gemv(
        reinterpret_cast<const __nv_bfloat16*>(A_bf16.data_ptr()),
        reinterpret_cast<const uint8_t*>(B_packed.data_ptr()),
        B_scale.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
        token_ids.data_ptr<int>(),
        expert_ids.data_ptr<int>(),
        num_slots, N, K,
        stride_bn, stride_bsn, expert_b_stride, expert_bs_stride,
        stream);

    return C;
}

at::Tensor fp4_moe_fused_forward(
    const at::Tensor &hidden_states,    // [M, K] bf16
    const at::Tensor &w13_packed,       // [E, 2*I, K/2] uint8
    const at::Tensor &w13_scale,        // [E, 2*I, K/32] float32
    const at::Tensor &w2_packed,        // [E, K_out, I/2] uint8
    const at::Tensor &w2_scale,         // [E, K_out, I/32] float32
    const at::Tensor &token_ids,        // [num_slots] int32
    const at::Tensor &expert_ids,       // [num_slots] int32
    int hidden_size,                    // K
    int intermediate_size,              // I
    float clamp_limit)
{
    TORCH_CHECK(hidden_states.scalar_type() == at::kBFloat16, "hidden_states must be bf16");
    TORCH_CHECK(w13_scale.scalar_type() == at::kFloat, "w13_scale must be float32");
    TORCH_CHECK(w2_scale.scalar_type() == at::kFloat, "w2_scale must be float32");

    const int num_slots = token_ids.size(0);
    const int K = hidden_size;
    const int I = intermediate_size;

    at::cuda::CUDAGuard guard(hidden_states.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    // Step 1: GEMM1 — hidden_states × W13 → intermediate [num_slots, 2*I]
    // Use v3 kernel (uint4 loads, warp-shuffle, 718 GB/s) for GEMM1 (K large)
    auto intermediate = at::empty({num_slots, 2 * I}, hidden_states.options());

    {
        int N1 = 2 * I;
        int stride_bn = K / 2;
        int stride_bsn = K / 32;
        int64_t eb = static_cast<int64_t>(N1) * stride_bn;
        int64_t ebs = static_cast<int64_t>(N1) * stride_bsn;

        sm120::v3::launch_fused_fp4_moe_gemv_v3(
            reinterpret_cast<const __nv_bfloat16*>(hidden_states.data_ptr()),
            reinterpret_cast<const uint8_t*>(w13_packed.data_ptr()),
            w13_scale.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(intermediate.data_ptr()),
            token_ids.data_ptr<int>(),
            expert_ids.data_ptr<int>(),
            num_slots, N1, K,
            stride_bn, stride_bsn, eb, ebs,
            stream);
    }

    // Step 2: Fused SwiGLU + GEMM2 — v3 kernel (warp-per-N, all lanes on K)
    auto output = at::empty({num_slots, K}, hidden_states.options());

    {
        int stride_bn = I / 2;
        int stride_bsn = I / 32;
        int64_t eb = static_cast<int64_t>(K) * stride_bn;
        int64_t ebs = static_cast<int64_t>(K) * stride_bsn;

        sm120::v3::launch_fused_swiglu_gemm2_v3(
            reinterpret_cast<const __nv_bfloat16*>(intermediate.data_ptr()),
            reinterpret_cast<const uint8_t*>(w2_packed.data_ptr()),
            w2_scale.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
            expert_ids.data_ptr<int>(),
            num_slots, K, I,
            clamp_limit,
            stride_bn, stride_bsn, eb, ebs,
            stream);
    }

    return output;
}

at::Tensor fp4_moe_grouped_forward(
    const at::Tensor &hidden_states,    // [M, K] bf16
    const at::Tensor &w13_packed,       // [E, 2*I, K/2] uint8
    const at::Tensor &w13_scale,        // [E, 2*I, K/32] float32
    const at::Tensor &w2_packed,        // [E, K_out, I/2] uint8
    const at::Tensor &w2_scale,         // [E, K_out, I/32] float32
    const at::Tensor &token_ids,        // [num_slots] int32
    const at::Tensor &expert_ids,       // [num_slots] int32
    int hidden_size,                    // K
    int intermediate_size,              // I
    int num_experts,
    float clamp_limit)
{
    TORCH_CHECK(hidden_states.scalar_type() == at::kBFloat16, "hidden_states must be bf16");
    TORCH_CHECK(w13_scale.scalar_type() == at::kFloat, "w13_scale must be float32");
    TORCH_CHECK(w2_scale.scalar_type() == at::kFloat, "w2_scale must be float32");

    const int num_slots = token_ids.size(0);
    const int K = hidden_size;
    const int I = intermediate_size;

    at::cuda::CUDAGuard guard(hidden_states.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    // Step 0: Sort slots by expert for grouped GEMM
    auto sorted_slot_ids = at::empty({num_slots}, token_ids.options());
    auto expert_offsets = at::empty({num_experts + 1}, token_ids.options());
    auto expert_counts = at::empty({num_experts}, token_ids.options());
    auto scatter_counts = at::empty({num_experts}, token_ids.options());

    sm120::launch_sort_by_expert(
        expert_ids.data_ptr<int32_t>(),
        sorted_slot_ids.data_ptr<int32_t>(),
        expert_offsets.data_ptr<int32_t>(),
        expert_counts.data_ptr<int32_t>(),
        scatter_counts.data_ptr<int32_t>(),
        num_slots, num_experts, stream);

    // Step 1: Grouped GEMM1 — hidden_states × W13 → intermediate [num_slots, 2*I]
    auto intermediate = at::empty({num_slots, 2 * I}, hidden_states.options());

    {
        int N1 = 2 * I;
        int stride_bn = K / 2;
        int stride_bsn = K / 32;
        int64_t eb = static_cast<int64_t>(N1) * stride_bn;
        int64_t ebs = static_cast<int64_t>(N1) * stride_bsn;

        sm120::grouped_v4::launch_grouped_fp4_moe_gemm1_v4(
            reinterpret_cast<const __nv_bfloat16*>(hidden_states.data_ptr()),
            reinterpret_cast<const uint8_t*>(w13_packed.data_ptr()),
            w13_scale.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(intermediate.data_ptr()),
            sorted_slot_ids.data_ptr<int32_t>(),
            token_ids.data_ptr<int32_t>(),
            expert_offsets.data_ptr<int32_t>(),
            num_slots, N1, K, num_experts,
            stride_bn, stride_bsn, eb, ebs,
            stream);
    }

    // Step 2: Grouped SwiGLU + GEMM2
    auto output = at::empty({num_slots, K}, hidden_states.options());

    {
        int stride_bn = I / 2;
        int stride_bsn = I / 32;
        int64_t eb = static_cast<int64_t>(K) * stride_bn;
        int64_t ebs = static_cast<int64_t>(K) * stride_bsn;

        sm120::grouped_v4::launch_grouped_fp4_swiglu_gemm2_v4(
            reinterpret_cast<const __nv_bfloat16*>(intermediate.data_ptr()),
            reinterpret_cast<const uint8_t*>(w2_packed.data_ptr()),
            w2_scale.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
            sorted_slot_ids.data_ptr<int32_t>(),
            expert_offsets.data_ptr<int32_t>(),
            num_slots, K, I, num_experts,
            clamp_limit,
            stride_bn, stride_bsn, eb, ebs,
            stream);
    }

    return output;
}

}  // namespace dsv4_kernel
