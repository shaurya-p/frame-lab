import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention.

    Args:
        embed_dim: total embedding dimension (must be divisible by num_heads).
        num_heads: number of attention heads.

    Required attributes (the test suite copies reference weights into these
    submodules by name, so they must exist with exactly these names):
        q_proj:   nn.Linear(embed_dim, embed_dim)
        k_proj:   nn.Linear(embed_dim, embed_dim)
        v_proj:   nn.Linear(embed_dim, embed_dim)
        out_proj: nn.Linear(embed_dim, embed_dim)
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
