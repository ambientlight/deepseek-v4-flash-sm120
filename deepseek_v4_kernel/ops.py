"""Python wrappers with a `flash_mla.cuda.sparse_decode_fwd`-compatible API."""
from __future__ import annotations

from typing import Optional, Tuple

import torch

try:
    from . import cuda as _cuda  # type: ignore[attr-defined]
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "deepseek_v4_kernel.cuda not built. Run `pip install -e .` inside the "
        "deepseek-v4-kernel project directory."
    ) from exc


def sparse_decode_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    topk_length: Optional[torch.Tensor],
    attn_sink: Optional[torch.Tensor],
    tile_scheduler_metadata: Optional[torch.Tensor],
    num_splits: Optional[torch.Tensor],
    extra_kv: Optional[torch.Tensor],
    extra_indices: Optional[torch.Tensor],
    extra_topk_length: Optional[torch.Tensor],
    d_v: int,
    sm_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """See `flash_mla.cuda.sparse_decode_fwd` for semantics."""
    return _cuda.sparse_decode_fwd(
        q,
        kv,
        indices,
        topk_length,
        attn_sink,
        tile_scheduler_metadata,
        num_splits,
        extra_kv,
        extra_indices,
        extra_topk_length,
        int(d_v),
        float(sm_scale),
    )


def sparse_prefill_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Drop-in for ``sgl_kernel.flash_mla.flash_mla_sparse_fwd`` on SM_120.

    q:       [s_q, h_q, d_qk] bf16
    kv:      [s_kv, h_kv=1, d_qk] or [s_kv, d_qk] bf16 (flat, dequantised)
    indices: [s_q, h_kv=1, topk] or [s_q, topk] int32 (-1 / >= s_kv = invalid)
    Returns (out[s_q,h_q,d_v] bf16, max_logits[s_q,h_q] f32, lse[s_q,h_q] f32).
    """
    return _cuda.sparse_prefill_fwd(
        q,
        kv,
        indices,
        float(sm_scale),
        int(d_v),
        attn_sink,
        topk_length,
    )
