#pragma once

#include <ATen/core/Tensor.h>
#include <optional>
#include <tuple>

namespace dsv4_kernel {

// Python-visible signature identical to sgl_kernel.flash_mla.flash_mla_sparse_fwd:
//   (out[s_q,h_q,d_v] bf16, max_logits[s_q,h_q] f32, lse[s_q,h_q] f32)
std::tuple<at::Tensor, at::Tensor, at::Tensor>
sparse_prefill_fwd(
    const at::Tensor &q,                       // [s_q, h_q, d_qk] bf16
    const at::Tensor &kv,                      // [s_kv, h_kv=1, d_qk] bf16 (flat)
    const at::Tensor &indices,                 // [s_q, h_kv=1, topk] int32
    double sm_scale,
    int d_v,
    const std::optional<at::Tensor> &attn_sink,    // [h_q] f32
    const std::optional<at::Tensor> &topk_length); // [s_q] int32

}  // namespace dsv4_kernel
