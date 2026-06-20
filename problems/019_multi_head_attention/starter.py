import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention.

    Args:
        embed_dim: total embedding dimension (must be divisible by num_heads).
        num_heads: number of attention heads.
    """

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-head self-attention.

        Args:
            x: (N, L, embed_dim) input sequence.

        Returns:
            (N, L, embed_dim) attended output.
        """
        raise NotImplementedError
