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
    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.ln_2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp_fc1 = nn.Linear(embed_dim, mlp_dim)
        self.mlp_fc2 = nn.Linear(mlp_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + self.self_attn(self.ln_1(x))
        z = y + self.mlp_fc2(F.gelu(self.mlp_fc1(self.ln_2(y))))
        return z


class PatchEmbed(nn.Module):
    def __init__(self, image_size: int, patch_size: int,
                 in_channels: int, embed_dim: int):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.num_patches = (image_size // patch_size) ** 2
        self.conv_proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)
        cls = self.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        return x + self.pos_embedding


class VisionTransformer(nn.Module):
    """Vision Transformer image classifier.

    Args:
        image_size: height/width of the (square) input image.
        patch_size: height/width of each square patch.
        in_channels: number of input channels.
        embed_dim: embedding dimension per token.
        num_layers: number of encoder blocks.
        num_heads: number of attention heads per block.
        mlp_dim: hidden dimension of each block's MLP.
        num_classes: number of output classes.

    Required attributes (the test suite copies reference weights into these by
    name, so they must exist with exactly these names):
        patch_embed: PatchEmbed from 020 (exposes conv_proj, class_token,
                     pos_embedding)
        blocks:      nn.ModuleList of EncoderBlock from 021 (each exposes ln_1,
                     self_attn, ln_2, mlp_fc1, mlp_fc2)
        ln:          nn.LayerNorm(embed_dim, eps=1e-6)  (final norm)
        head:        nn.Linear(embed_dim, num_classes)
    """

    def __init__(self, image_size: int, patch_size: int, in_channels: int,
                 embed_dim: int, num_layers: int, num_heads: int,
                 mlp_dim: int, num_classes: int):
        super().__init__()
        self.patch_embed = PatchEmbed(image_size, patch_size, in_channels, embed_dim)
        self.blocks = nn.ModuleList(
            [EncoderBlock(embed_dim, num_heads, mlp_dim) for _ in range(num_layers)]
        )
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify a batch of images.

        Args:
            x: (N, in_channels, image_size, image_size) input images.

        Returns:
            (N, num_classes) class logits.
        """
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        cls = x[:, 0]
        return self.head(cls)
