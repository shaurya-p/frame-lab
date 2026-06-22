import torch
import torch.nn as nn


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
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the pre-norm encoder block.

        Args:
            x: (N, L, embed_dim) input sequence.

        Returns:
            (N, L, embed_dim) output sequence.
        """
        raise NotImplementedError
