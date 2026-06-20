import math

import torch


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute scaled dot-product attention.

    Args:
        q: (..., L, d) query tensor.
        k: (..., S, d) key tensor.
        v: (..., S, d_v) value tensor.
        attn_mask: additive float mask broadcastable to (..., L, S), or None.

    Returns:
        (..., L, d_v) attention output.
    """
    d = q.shape[-1]
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)
    if attn_mask is not None:
        scores = scores + attn_mask
    weights = torch.softmax(scores, dim=-1)
    return weights @ v
