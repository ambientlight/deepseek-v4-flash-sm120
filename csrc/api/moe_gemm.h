#pragma once
#include <torch/extension.h>

namespace dsv4_kernel {

// Single GEMM: A_bf16 × dequant(B_fp4) → C_bf16
at::Tensor fp4_moe_gemm(
    const at::Tensor &A_bf16,
    const at::Tensor &B_packed,
    const at::Tensor &B_scale,
    const at::Tensor &token_ids,
    const at::Tensor &expert_ids,
    int N, int K);

// Fused MoE forward: GEMM1 → SwiGLU → GEMM2, one C++ call (per-slot GEMV)
at::Tensor fp4_moe_fused_forward(
    const at::Tensor &hidden_states,    // [M, K] bf16
    const at::Tensor &w13_packed,       // [E, 2*I, K/2] uint8 (gate+up)
    const at::Tensor &w13_scale,        // [E, 2*I, K/32] float32
    const at::Tensor &w2_packed,        // [E, K_out, I/2] uint8 (down)
    const at::Tensor &w2_scale,         // [E, K_out, I/32] float32
    const at::Tensor &token_ids,        // [num_slots] int32
    const at::Tensor &expert_ids,       // [num_slots] int32
    int hidden_size,                    // K = K_out
    int intermediate_size,              // I
    float clamp_limit);                 // DeepSeek V4: 10.0

// Grouped MoE forward: sort by expert → grouped GEMM1 → SwiGLU → grouped GEMM2
// For prefill (M>8): loads expert weights once per expert instead of once per slot.
at::Tensor fp4_moe_grouped_forward(
    const at::Tensor &hidden_states,    // [M, K] bf16
    const at::Tensor &w13_packed,       // [E, 2*I, K/2] uint8 (gate+up)
    const at::Tensor &w13_scale,        // [E, 2*I, K/32] float32
    const at::Tensor &w2_packed,        // [E, K_out, I/2] uint8 (down)
    const at::Tensor &w2_scale,         // [E, K_out, I/32] float32
    const at::Tensor &token_ids,        // [num_slots] int32
    const at::Tensor &expert_ids,       // [num_slots] int32
    int hidden_size,                    // K = K_out
    int intermediate_size,              // I
    int num_experts,                    // E (local, after TP split)
    float clamp_limit);                 // DeepSeek V4: 10.0

}  // namespace dsv4_kernel
