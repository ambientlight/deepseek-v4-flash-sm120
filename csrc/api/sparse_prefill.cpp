#include "api/sparse_prefill.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <limits>

#include "common/defines.h"
#include "common/params.h"
#include "sm120/prefill/sparse_prefill.h"

namespace dsv4_kernel {

namespace {
inline int strideToInt(int64_t s) {
    TORCH_CHECK(s <= std::numeric_limits<int>::max(), "stride overflow");
    return static_cast<int>(s);
}
}  // namespace

// Drop-in for sgl_kernel.flash_mla.flash_mla_sparse_fwd on SM_120.
//   q:       [s_q, h_q, d_qk] bf16
//   kv:      [s_kv, h_kv=1, d_qk] or [s_kv, d_qk] bf16 (flat, dequantised)
//   indices: [s_q, h_kv=1, topk] or [s_q, topk] int32
//   -> (out[s_q,h_q,d_v] bf16, max_logits[s_q,h_q] f32, lse[s_q,h_q] f32)
std::tuple<at::Tensor, at::Tensor, at::Tensor>
sparse_prefill_fwd(
    const at::Tensor &q,
    const at::Tensor &kv,
    const at::Tensor &indices,
    double sm_scale,
    int d_v,
    const std::optional<at::Tensor> &attn_sink,
    const std::optional<at::Tensor> &topk_length) {

    TORCH_CHECK(q.is_cuda() && kv.is_cuda(), "q and kv must be CUDA tensors");
    TORCH_CHECK(q.dim() == 3, "q must be 3-D [s_q, h_q, d_qk]");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16, "q must be bf16");
    TORCH_CHECK(kv.scalar_type() == at::kBFloat16, "kv must be bf16 (dequantised workspace)");
    TORCH_CHECK(indices.scalar_type() == at::kInt, "indices must be int32");

    // Normalise kv to flat [s_kv, d_qk] and indices to [s_q, topk].
    at::Tensor kv_flat = (kv.dim() == 3) ? kv.squeeze(1) : kv;
    at::Tensor idx2 = (indices.dim() == 3) ? indices.squeeze(1) : indices;
    TORCH_CHECK(kv_flat.dim() == 2, "kv must be [s_kv, d_qk] (or [s_kv,1,d_qk])");
    TORCH_CHECK(idx2.dim() == 2, "indices must be [s_q, topk] (or [s_q,1,topk])");

    const int s_q  = q.size(0);
    const int h_q  = q.size(1);
    const int d_qk = q.size(2);
    const int s_kv = kv_flat.size(0);
    const int topk = idx2.size(1);

    TORCH_CHECK(d_qk == 512, "DeepSeek-V4-Flash expects d_qk=512, got ", d_qk);
    TORCH_CHECK(kv_flat.size(1) == 512, "kv last dim must be 512, got ", kv_flat.size(1));
    TORCH_CHECK(d_v == 512, "Only d_v=512 supported");
    TORCH_CHECK(idx2.size(0) == s_q, "indices s_q (", idx2.size(0),
                ") must match q s_q (", s_q, ")");

    // The kernel reads q[h][d], kv_flat[row][v], indices[off+tok] and writes
    // out[h][v] assuming the *inner* (d_qk / topk / d_v) dims are stride-1.
    // Only the leading row strides are passed through as params, so guard the
    // contiguity the kernel relies on rather than silently reading garbage.
    TORCH_CHECK(q.stride(2) == 1,
                "q must be contiguous in d_qk (stride(2)==1); got ", q.stride(2));
    TORCH_CHECK(kv_flat.stride(1) == 1,
                "kv must be contiguous in d_qk (stride==1); got ", kv_flat.stride(1));
    TORCH_CHECK(idx2.stride(1) == 1,
                "indices must be contiguous in topk (stride==1); got ", idx2.stride(1));
    if (attn_sink.has_value()) {
        TORCH_CHECK(attn_sink->scalar_type() == at::kFloat,
                    "attn_sink must be float32");
        TORCH_CHECK(attn_sink->numel() == h_q,
                    "attn_sink must have h_q (", h_q, ") elements; got ",
                    attn_sink->numel());
    }
    if (topk_length.has_value()) {
        TORCH_CHECK(topk_length->scalar_type() == at::kInt,
                    "topk_length must be int32");
        TORCH_CHECK(topk_length->numel() == s_q,
                    "topk_length must have s_q (", s_q, ") elements; got ",
                    topk_length->numel());
    }

    at::cuda::CUDAGuard g(q.device());
    auto opts = q.options();
    auto out = at::empty({s_q, h_q, d_v}, opts);
    auto max_logits = at::empty({s_q, h_q}, opts.dtype(at::kFloat));
    auto lse = at::empty({s_q, h_q}, opts.dtype(at::kFloat));

    SparseAttnPrefillParams p{};
    p.s_q = s_q;
    p.s_kv = s_kv;
    p.h_q = h_q;
    p.h_kv = 1;
    p.d_qk = d_qk;
    p.d_v = d_v;
    p.topk = topk;
    p.sm_scale = static_cast<float>(sm_scale);
    p.sm_scale_div_log2 = p.sm_scale * LOG_2_E;

    p.q = reinterpret_cast<cutlass::bfloat16_t *>(q.data_ptr());
    p.kv = reinterpret_cast<cutlass::bfloat16_t *>(kv_flat.data_ptr());
    p.indices = idx2.data_ptr<int>();
    p.attn_sink = attn_sink.has_value() ? attn_sink->data_ptr<float>() : nullptr;
    p.topk_length = topk_length.has_value() ? topk_length->data_ptr<int>() : nullptr;

    p.out = reinterpret_cast<cutlass::bfloat16_t *>(out.data_ptr());
    p.max_logits = max_logits.data_ptr<float>();
    p.lse = lse.data_ptr<float>();

    p.stride_q_s_q = strideToInt(q.stride(0));
    p.stride_q_h_q = strideToInt(q.stride(1));
    p.stride_kv_s_kv = strideToInt(kv_flat.stride(0));
    p.stride_indices_s_q = strideToInt(idx2.stride(0));
    p.stride_o_s_q = strideToInt(out.stride(0));
    p.stride_o_h_q = strideToInt(out.stride(1));
    p.stride_ml_s_q = strideToInt(max_logits.stride(0));
    p.stride_lse_s_q = strideToInt(lse.stride(0));

    Arch arch;
    p.num_sm = arch.num_sms;
    p.stream = at::cuda::getCurrentCUDAStream();

    if (arch.is_sm120() || std::getenv("DSV4_KERNEL_FORCE") != nullptr) {
        sm120::launch_dsv4_sparse_prefill(p);
    } else {
        TORCH_CHECK(false,
                    "dsv4_kernel.sparse_prefill_fwd only implements SM_120 "
                    "(Blackwell workstation). Current device major=",
                    arch.major, " minor=", arch.minor);
    }

    return std::make_tuple(out, max_logits, lse);
}

}  // namespace dsv4_kernel
