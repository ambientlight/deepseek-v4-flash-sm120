// pybind11 entry point for the SM_120 DeepSeek-V4 sparse decode kernel.
// Exposes the same module name (`deepseek_v4_kernel.cuda`) and symbol
// (`sparse_decode_fwd`) that `flash_mla.cuda` publishes so that we can
// monkey-patch it at runtime.

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "api/sparse_decode.h"
#include "api/sparse_prefill.h"
#include "api/moe_gemm.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "DeepSeek-V4-Flash sparse decode + MoE kernels (SM_120).";

    m.def("sparse_decode_fwd", &dsv4_kernel::sparse_decode_fwd,
          pybind11::arg("q"),
          pybind11::arg("kv"),
          pybind11::arg("indices"),
          pybind11::arg("topk_length"),
          pybind11::arg("attn_sink"),
          pybind11::arg("tile_scheduler_metadata"),
          pybind11::arg("num_splits"),
          pybind11::arg("extra_kv"),
          pybind11::arg("extra_indices"),
          pybind11::arg("extra_topk_length"),
          pybind11::arg("d_v"),
          pybind11::arg("sm_scale"),
          "Sparse MLA decode forward (SM_120 / Blackwell workstation).");

    m.def("sparse_prefill_fwd", &dsv4_kernel::sparse_prefill_fwd,
          pybind11::arg("q"),
          pybind11::arg("kv"),
          pybind11::arg("indices"),
          pybind11::arg("sm_scale"),
          pybind11::arg("d_v"),
          pybind11::arg("attn_sink"),
          pybind11::arg("topk_length"),
          "Sparse MLA prefill forward (SM_120 / Blackwell workstation). "
          "Drop-in for sgl_kernel.flash_mla.flash_mla_sparse_fwd.");

    m.def("fp4_moe_gemm", &dsv4_kernel::fp4_moe_gemm,
          pybind11::arg("A_bf16"),
          pybind11::arg("B_packed"),
          pybind11::arg("B_scale"),
          pybind11::arg("token_ids"),
          pybind11::arg("expert_ids"),
          pybind11::arg("N"),
          pybind11::arg("K"),
          "FP4 MoE GEMM with native SM120 tensor cores.");

    m.def("fp4_moe_fused_forward", &dsv4_kernel::fp4_moe_fused_forward,
          pybind11::arg("hidden_states"),
          pybind11::arg("w13_packed"),
          pybind11::arg("w13_scale"),
          pybind11::arg("w2_packed"),
          pybind11::arg("w2_scale"),
          pybind11::arg("token_ids"),
          pybind11::arg("expert_ids"),
          pybind11::arg("hidden_size"),
          pybind11::arg("intermediate_size"),
          pybind11::arg("clamp_limit"),
          "Fused FP4 MoE: GEMM1 + SwiGLU + GEMM2 in one call (per-slot GEMV).");

    m.def("fp4_moe_grouped_forward", &dsv4_kernel::fp4_moe_grouped_forward,
          pybind11::arg("hidden_states"),
          pybind11::arg("w13_packed"),
          pybind11::arg("w13_scale"),
          pybind11::arg("w2_packed"),
          pybind11::arg("w2_scale"),
          pybind11::arg("token_ids"),
          pybind11::arg("expert_ids"),
          pybind11::arg("hidden_size"),
          pybind11::arg("intermediate_size"),
          pybind11::arg("num_experts"),
          pybind11::arg("clamp_limit"),
          "Grouped FP4 MoE: sort by expert + batched GEMM (for prefill M>8).");
}
