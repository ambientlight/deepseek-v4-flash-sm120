"""Pure-PyTorch reference for the DeepSeek-V4-Flash sparse decode kernel.

Used both as a unit-test oracle and for debugging the SM_120 kernel.
Semantics match sglang's `flash_mla_with_kvcache_entrypoint` call for
DeepSeek-V4-Flash (sglang.deepseek_v4_memory_pool + flash_mla MODEL1).

Per-page layout (page_block_size = P):
  [0 .. P*576)              nope_rope section
      per token (576 B):
           0..447  FP8 e4m3 NoPE        (448 B)
         448..575  BF16 RoPE            (64 values = 128 B)
  [P*576 .. P*(576+8))      scale section
      per token  :  7 UE8M0 bytes + 1 pad (8 bytes)
                    scale_i = 2**(byte_i - 127), covers NoPE[i*64 .. (i+1)*64)

Q / output : BF16 [b, s_q, h_q, 512]  (448 NoPE || 64 RoPE)

Attention: softmax(QK^T * sm_scale) over the gathered `indices`, with
optional `topk_length` mask and optional `attn_sink` (log-domain mix).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

HEAD_DIM_NOPE = 448
HEAD_DIM_ROPE = 64
HEAD_DIM_QK   = HEAD_DIM_NOPE + HEAD_DIM_ROPE  # 512
HEAD_DIM_V    = HEAD_DIM_QK                     # 512
QUANT_TILE         = 64
NUM_ACTIVE_SCALES  = HEAD_DIM_NOPE // QUANT_TILE   # 7
NUM_SCALE_SLOTS    = 8                              # 7 active + 1 pad
NOPE_BYTES         = HEAD_DIM_NOPE                  # 448
ROPE_BYTES         = HEAD_DIM_ROPE * 2              # 128
NOPE_ROPE_BYTES    = NOPE_BYTES + ROPE_BYTES        # 576
BYTES_PER_TOKEN    = NOPE_ROPE_BYTES + NUM_SCALE_SLOTS  # 584 (cosmetic)


def _ue8m0_to_scale(b: torch.Tensor) -> torch.Tensor:
    """uint8 tensor -> fp32 scale tensor via 2**(b - 127)."""
    return torch.pow(2.0, b.to(torch.float32) - 127.0)


def _unpack_nope_rope_scale(nope_bytes: torch.Tensor,
                             rope_bytes: torch.Tensor,
                             scale_u8: torch.Tensor) -> torch.Tensor:
    """Assemble the dequantised BF16 [N, 512] rows from the separated
    nope / rope / scale sections produced by sglang's quant_k_cache_v4."""
    n = nope_bytes.size(0)
    nope_fp8 = nope_bytes.contiguous().view(dtype=torch.float8_e4m3fn)
    rope = rope_bytes.contiguous().view(dtype=torch.bfloat16).view(n, HEAD_DIM_ROPE)

    scales = _ue8m0_to_scale(scale_u8[:, :NUM_ACTIVE_SCALES]).view(n, NUM_ACTIVE_SCALES)
    nope = nope_fp8.float().view(n, NUM_ACTIVE_SCALES, QUANT_TILE)
    nope = nope * scales.unsqueeze(-1)
    nope = nope.view(n, HEAD_DIM_NOPE).to(torch.bfloat16)
    return torch.cat([nope, rope], dim=-1)


def _pages_to_bf16(kv_pages: torch.Tensor, page_block: int) -> torch.Tensor:
    """Decode every token of every page into a dense [num_blocks*page, 512] BF16
    tensor.  `kv_pages` is a uint8 tensor of shape
      [num_blocks, page_block*NOPE_ROPE_BYTES + page_block*NUM_SCALE_SLOTS]
    i.e. the raw per-page byte arena, with nope_rope section first and
    scale section second (flash_mla MODEL1 layout)."""
    num_blocks = kv_pages.size(0)
    total_tokens = num_blocks * page_block

    nope_rope_section = kv_pages[:, : page_block * NOPE_ROPE_BYTES]
    scale_section = kv_pages[:, page_block * NOPE_ROPE_BYTES :
                             page_block * NOPE_ROPE_BYTES +
                             page_block * NUM_SCALE_SLOTS]

    nope_rope = nope_rope_section.contiguous().view(total_tokens, NOPE_ROPE_BYTES)
    nope_bytes = nope_rope[:, :NOPE_BYTES]
    rope_bytes = nope_rope[:, NOPE_BYTES:]
    scale_u8 = scale_section.contiguous().view(total_tokens, NUM_SCALE_SLOTS)
    return _unpack_nope_rope_scale(nope_bytes, rope_bytes, scale_u8)


def sparse_decode_reference(
    q: torch.Tensor,                 # [b, s_q, h_q, 512] bf16
    kv_pages: torch.Tensor,          # [num_blocks, page_bytes] uint8, MODEL1 layout
    indices: torch.Tensor,           # [b, s_q, topk] int32
    topk_length: Optional[torch.Tensor],
    attn_sink: Optional[torch.Tensor],
    sm_scale: float,
    page_block: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    b, s_q, h_q, d_qk = q.shape
    topk = indices.size(-1)
    assert d_qk == HEAD_DIM_QK
    assert kv_pages.dtype == torch.uint8
    expected_bytes = page_block * NOPE_ROPE_BYTES + page_block * NUM_SCALE_SLOTS
    assert kv_pages.size(-1) >= expected_bytes, (
        f"kv_pages last-dim {kv_pages.size(-1)} < required {expected_bytes}"
    )

    kv_bf16 = _pages_to_bf16(kv_pages, page_block)  # [N, 512]

    out = torch.zeros(b, s_q, h_q, HEAD_DIM_V, dtype=torch.bfloat16, device=q.device)
    lse_out = torch.empty(b, s_q, h_q, dtype=torch.float32, device=q.device)

    for bi in range(b):
        for si in range(s_q):
            idx = indices[bi, si]  # [topk]
            length = int(topk_length[bi].item()) if topk_length is not None else topk
            length = max(length, 1)
            valid = (idx >= 0) & (idx < kv_bf16.size(0))
            valid_idx = torch.where(valid, idx, torch.zeros_like(idx))
            k = kv_bf16[valid_idx.long()]  # [topk, 512]
            k = torch.where(valid.unsqueeze(-1), k, torch.zeros_like(k))
            pos = torch.arange(topk, device=q.device)
            length_mask = pos < length
            q_bs = q[bi, si].float()
            logits = (q_bs @ k.float().transpose(0, 1)) * sm_scale
            logits = torch.where(length_mask.unsqueeze(0), logits,
                                 torch.full_like(logits, float("-inf")))
            lse = torch.logsumexp(logits, dim=-1)
            p = torch.softmax(logits, dim=-1)
            v = k.float()  # MLA: V == K (full 512)
            o = p @ v
            if attn_sink is not None:
                sink = attn_sink.float()
                sink_scale = 1.0 / (1.0 + torch.exp(sink - lse))
                o = o * sink_scale.unsqueeze(-1)
                m = torch.maximum(lse, sink)
                lse = m + torch.log(torch.exp(lse - m) + torch.exp(sink - m))
            out[bi, si] = o.to(torch.bfloat16)
            lse_out[bi, si] = lse
    return out, lse_out


# =====================================================================
# Sparse PREFILL reference (flat bf16 KV).
#
# Mirrors DeepSeek FlashMLA's own torch reference `ref_sparse_attn_fwd`
# (flashmla-src/tests/ref.py) and the `flash_mla_sparse_fwd` API:
#
#   q       : [s_q, h_q, d_qk] bf16
#   kv      : [s_kv, d_qk]     bf16   (already dequantised; V == K, d_v=512)
#   indices : [s_q, topk]      int32  (-1 or >= s_kv -> invalid/masked)
#   sm_scale: float
#   attn_sink   : [h_q] f32, optional (log-domain mix into lse)
#   topk_length : [s_q] int32, optional (per-query valid prefix)
#
# Returns (out[s_q,h_q,d_v] bf16, max_logits[s_q,h_q] f32, lse[s_q,h_q] f32),
# where max_logits/lse are NATURAL-log domain. lonely queries (no valid
# token) -> out 0, max_logits -inf, lse +inf. Unlike sparse_decode_reference
# there is NO page/FP8/E8M0 unpack: KV arrives dense bf16 from sglang's
# `_forward_prefill_sparse` workspace.
# =====================================================================
def sparse_prefill_reference(
    q: torch.Tensor,                  # [s_q, h_q, d_qk] bf16
    kv: torch.Tensor,                 # [s_kv, d_qk] bf16
    indices: torch.Tensor,            # [s_q, topk] int32
    sm_scale: float,
    d_v: int = HEAD_DIM_V,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    s_q, h_q, d_qk = q.shape
    s_kv = kv.size(0)
    topk = indices.size(-1)
    assert d_qk == HEAD_DIM_QK and kv.size(-1) == HEAD_DIM_QK

    idx = indices.clone()
    if topk_length is not None:
        len_mask = (
            torch.arange(topk, device=idx.device).unsqueeze(0).broadcast_to(s_q, topk)
            >= topk_length.unsqueeze(1)
        )
        idx[len_mask] = -1
    invalid = (idx < 0) | (idx >= s_kv)             # [s_q, topk]
    idx_safe = torch.where(invalid, torch.zeros_like(idx), idx)

    qf = q.float()
    gathered = kv.index_select(0, idx_safe.flatten().long()).reshape(s_q, topk, d_qk).float()
    P = (qf @ gathered.transpose(1, 2)) * sm_scale  # [s_q, h_q, topk]
    P[invalid.unsqueeze(1).broadcast_to(P.shape)] = float("-inf")

    orig_lse = torch.logsumexp(P, dim=-1)           # [s_q, h_q]
    max_logits = P.max(dim=-1).values               # [s_q, h_q]

    if attn_sink is not None:
        lse_for_o = torch.logsumexp(
            torch.stack([orig_lse, attn_sink.float().broadcast_to(s_q, h_q)], dim=0), dim=0
        )
    else:
        lse_for_o = orig_lse.clone()
    lse_for_o[lse_for_o == float("-inf")] = float("+inf")  # -> O row becomes 0
    s_for_o = torch.exp(P - lse_for_o.unsqueeze(-1))
    out = s_for_o @ gathered[..., :d_v]             # [s_q, h_q, d_v]

    lonely = orig_lse == float("-inf")
    orig_lse = orig_lse.clone()
    orig_lse[lonely] = float("+inf")
    return out.to(torch.bfloat16), max_logits, orig_lse


def make_fake_prefill_batch(
    s_q: int = 64,
    h_q: int = 64,
    s_kv: int = 4096,
    topk: int = 256,
    with_topk_length: bool = False,
    with_attn_sink: bool = False,
    seed: int = 0,
    device: str = "cuda",
):
    """Random flat-bf16 prefill batch matching the `flash_mla_sparse_fwd`
    contract (sglang's dequantised workspace).

    Returns q[s_q,h_q,512] bf16, kv[s_kv,512] bf16, indices[s_q,topk] int32,
    attn_sink[h_q]|None, topk_length[s_q]|None.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(s_q, h_q, HEAD_DIM_QK, dtype=torch.bfloat16,
                    device=device, generator=gen) * 0.1
    kv = torch.randn(s_kv, HEAD_DIM_QK, dtype=torch.bfloat16,
                     device=device, generator=gen) * 0.5
    # Mix in some -1 (invalid) entries alongside in-range gathers.
    idx = torch.randint(-1, s_kv, (s_q, topk), dtype=torch.int32,
                        device=device, generator=gen)
    attn_sink = (
        torch.randn(h_q, dtype=torch.float32, device=device, generator=gen)
        if with_attn_sink else None
    )
    topk_length = (
        torch.randint(1, topk + 1, (s_q,), dtype=torch.int32, device=device, generator=gen)
        if with_topk_length else None
    )
    return q, kv, idx, attn_sink, topk_length


def make_fake_batch(
    b: int = 2,
    h_q: int = 64,
    num_blocks: int = 4,
    page_block: int = 64,
    topk: int = 256,
    seed: int = 0,
    device: str = "cuda",
):
    """Build a randomised batch whose packed KV arena matches the sglang
    DSv4-Flash layout (MODEL1): per page
        [page_block * 576 B nope_rope][page_block * 8 B scales]

    Returns
    -------
    q           : BF16 [b, 1, h_q, 512]
    kv_pages    : uint8 [num_blocks, page_block * (576 + 8)]
                  (raw per-page arena; pass to kernel + reference)
    kv_view     : uint8 [num_blocks, page_block, 1, 584]
                  (cosmetic sglang-style 4-D view sharing storage)
    idx         : int32 [b, 1, topk]
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(b, 1, h_q, HEAD_DIM_QK, dtype=torch.bfloat16,
                    device=device, generator=gen) * 0.1

    # NoPE bytes (FP8 e4m3fn).
    nope_bf16 = torch.randn(num_blocks, page_block, HEAD_DIM_NOPE,
                            dtype=torch.bfloat16, device=device, generator=gen) * 0.5
    nope_fp8 = nope_bf16.to(torch.float8_e4m3fn)
    nope_bytes = nope_fp8.view(dtype=torch.uint8)           # [B, P, 448]

    # RoPE bytes (BF16 -> 128 B per token).
    rope_bf16 = torch.randn(num_blocks, page_block, HEAD_DIM_ROPE,
                            dtype=torch.bfloat16, device=device, generator=gen) * 0.05
    rope_bytes = rope_bf16.view(dtype=torch.uint8).view(num_blocks, page_block,
                                                        ROPE_BYTES)

    # Concatenate per-token nope|rope -> 576 bytes, then flatten page.
    nope_rope = torch.cat([nope_bytes, rope_bytes], dim=-1)  # [B, P, 576]
    assert nope_rope.size(-1) == NOPE_ROPE_BYTES
    nope_rope_section = nope_rope.contiguous().view(num_blocks,
                                                    page_block * NOPE_ROPE_BYTES)

    # Scale section: 7 UE8M0 bytes + 1 pad, per token, laid out per page.
    scale_exp = (125 + torch.randint(0, 4,
                                     (num_blocks, page_block, NUM_ACTIVE_SCALES),
                                     device=device, generator=gen,
                                     dtype=torch.int32)).to(torch.uint8)
    pad = torch.zeros(num_blocks, page_block, 1, dtype=torch.uint8, device=device)
    scales = torch.cat([scale_exp, pad], dim=-1)             # [B, P, 8]
    assert scales.size(-1) == NUM_SCALE_SLOTS
    scale_section = scales.contiguous().view(num_blocks,
                                             page_block * NUM_SCALE_SLOTS)

    # Full per-page arena: nope_rope then scales.
    kv_pages = torch.cat([nope_rope_section, scale_section], dim=-1).contiguous()
    # Cosmetic 4-D view that matches sglang's .view(num_blocks, P, 1, 584).
    # Memory backing is NOT interleaved per-token; only the *shape* matches.
    kv_view = kv_pages.view(num_blocks, page_block, 1, BYTES_PER_TOKEN)

    total_tokens = num_blocks * page_block
    idx = torch.randint(-1, total_tokens, (b, 1, topk), dtype=torch.int32,
                        device=device, generator=gen)
    return q, kv_pages, kv_view, idx
