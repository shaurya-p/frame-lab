import torch
import torch.nn as nn


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
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify a batch of images.

        Args:
            x: (N, in_channels, image_size, image_size) input images.

        Returns:
            (N, num_classes) class logits.
        """
        raise NotImplementedError
