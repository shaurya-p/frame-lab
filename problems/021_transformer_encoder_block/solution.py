import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    d = q.shape[-1]
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)
    weights = torch.softmax(scores, dim=-1)
    return weights @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def _split_heads(self, t: torch.Tensor) -> torch.Tensor:
        N, L, _ = t.shape
        return t.reshape(N, L, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, t: torch.Tensor) -> torch.Tensor:
        N, _, L, _ = t.shape
        return t.transpose(1, 2).reshape(N, L, self.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        attn = _scaled_dot_product_attention(q, k, v)
        return self.out_proj(self._merge_heads(attn))


class EncoderBlock(nn.Module):
    """Pre-norm Vision Transformer encoder block.

    Args:
        embed_dim: embedding dimension (must be divisible by num_heads).
        num_heads: number of attention heads.
        mlp_dim: hidden dimension of the feed-forward MLP.

    Required attributes (the test suite copies reference weights into these by
    name, so they must exist with exactly these names):
        ln_1:      nn.LayerNorm(embed_dim, eps=1e-6)
        self_attn: MultiHeadAttention(embed_dim, num_heads) from problem 019,
                   exposing q_proj / k_proj / v_proj / out_proj
        ln_2:      nn.LayerNorm(embed_dim, eps=1e-6)
        mlp_fc1:   nn.Linear(embed_dim, mlp_dim)
        mlp_fc2:   nn.Linear(mlp_dim, embed_dim)
    """

    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.ln_2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp_fc1 = nn.Linear(embed_dim, mlp_dim)
        self.mlp_fc2 = nn.Linear(mlp_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the pre-norm encoder block.

        Args:
            x: (N, L, embed_dim) input sequence.

        Returns:
            (N, L, embed_dim) output sequence.
        """
        y = x + self.self_attn(self.ln_1(x))
        z = y + self.mlp_fc2(F.gelu(self.mlp_fc1(self.ln_2(y))))
        return z
