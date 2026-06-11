"""Correctness regression for the SM_120 sparse PREFILL kernel.

Gates Stages 1-2 of the prefill hillclimb. Compares
``deepseek_v4_kernel.ops.sparse_prefill_fwd`` against the pure-torch
``sparse_prefill_reference`` (which mirrors DeepSeek FlashMLA's own
``ref_sparse_attn_fwd``). Skips cleanly off SM_120 / before the op exists.
"""
from __future__ import annotations

import math

import pytest
import torch

from reference import make_fake_prefill_batch, sparse_prefill_reference

try:
    from deepseek_v4_kernel.ops import sparse_prefill_fwd
    _HAVE_PREFILL = True
except Exception:  # op not built yet (Stage 0/1 in progress)
    _HAVE_PREFILL = False


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA device required",
)


def _skip_unless_ready():
    major, _ = torch.cuda.get_device_capability(0)
    if major != 12:
        pytest.skip(f"SM_120 required; got {torch.cuda.get_device_capability(0)}")
    if not _HAVE_PREFILL:
        pytest.skip("deepseek_v4_kernel.ops.sparse_prefill_fwd not built yet")


# DeepSeek's own tolerances (flashmla-src/tests/test_flash_mla_sparse_prefill.py):
#   out: cos_diff_tol 7e-6; here we assert cosine >= 0.999 + a loose abs/rel.
def _assert_close(name, got, ref, atol, rtol):
    torch.testing.assert_close(got.float(), ref.float(), atol=atol, rtol=rtol)


@pytest.mark.parametrize("s_q", [1, 64, 1024, 16384])
@pytest.mark.parametrize("h_q", [64, 128])
@pytest.mark.parametrize("topk", [64, 256, 2048])
def test_matches_reference(s_q: int, h_q: int, topk: int):
    _skip_unless_ready()
    torch.manual_seed(0)
    s_kv = max(4096, topk * 4)
    q, kv, idx, _, _ = make_fake_prefill_batch(
        s_q=s_q, h_q=h_q, s_kv=s_kv, topk=topk,
        with_topk_length=False, with_attn_sink=False,
    )
    sm_scale = 1.0 / math.sqrt(512)
    # The kernel always runs the full batch (it streams KV); s_q=16384 is the
    # production trigger (> _LARGE_INDEXER_QUERY_THRESHOLD = 11673). The torch
    # reference, by contrast, materialises a dense [s_q, topk, 512] f32 gather
    # (~64 GiB at the largest shape), so compare against a bounded row-slice of
    # the reference while still exercising the kernel at full s_q.
    out, mx, lse = sparse_prefill_fwd(q, kv, idx.unsqueeze(1), sm_scale, 512, None, None)
    ref_rows = min(s_q, 1024)
    out_ref, mx_ref, lse_ref = sparse_prefill_reference(
        q[:ref_rows], kv, idx[:ref_rows], sm_scale=sm_scale
    )
    out_s, mx_s, lse_s = out[:ref_rows], mx[:ref_rows], lse[:ref_rows]

    cos = torch.nn.functional.cosine_similarity(
        out_s.float().flatten(), out_ref.float().flatten(), dim=0
    ).item()
    assert cos >= 0.999, f"out cos {cos:.6f} < 0.999"
    _assert_close("out", out_s, out_ref, atol=5e-3, rtol=5e-3)
    # max_logits is the pre-softmax row max (natural-log domain); the HMMA
    # kernel reproduces it ~exactly once invalid gathers score -inf.
    fin_mx = torch.isfinite(mx_ref)
    if fin_mx.any():
        torch.testing.assert_close(mx_s[fin_mx], mx_ref[fin_mx], atol=1e-2, rtol=1e-2)
    # lse: bf16-input / fp32-accum logsumexp differs slightly from torch over
    # few terms; sglang discards lse (o, _, _) so a loose check suffices.
    fin = torch.isfinite(lse_ref)
    if fin.any():
        torch.testing.assert_close(lse_s[fin], lse_ref[fin], atol=1e-2, rtol=1e-2)


def test_topk_length_mask():
    _skip_unless_ready()
    torch.manual_seed(1)
    q, kv, idx, _, tl = make_fake_prefill_batch(
        s_q=128, h_q=64, s_kv=4096, topk=512, with_topk_length=True,
    )
    sm_scale = 1.0 / math.sqrt(512)
    out_ref, _, _ = sparse_prefill_reference(q, kv, idx, sm_scale=sm_scale, topk_length=tl)
    out, _, _ = sparse_prefill_fwd(q, kv, idx.unsqueeze(1), sm_scale, 512, None, tl)
    _assert_close("out", out, out_ref, atol=5e-3, rtol=5e-3)


def test_attn_sink():
    _skip_unless_ready()
    torch.manual_seed(2)
    q, kv, idx, sink, _ = make_fake_prefill_batch(
        s_q=128, h_q=64, s_kv=4096, topk=256, with_attn_sink=True,
    )
    sm_scale = 1.0 / math.sqrt(512)
    out_ref, _, _ = sparse_prefill_reference(q, kv, idx, sm_scale=sm_scale, attn_sink=sink)
    out, _, _ = sparse_prefill_fwd(q, kv, idx.unsqueeze(1), sm_scale, 512, sink, None)
    _assert_close("out", out, out_ref, atol=5e-3, rtol=5e-3)
