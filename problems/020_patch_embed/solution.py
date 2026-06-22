import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """Vision Transformer patch embedding.

    Args:
        image_size: height/width of the (square) input image.
        patch_size: height/width of each square patch.
        in_channels: number of input channels.
        embed_dim: embedding dimension per token.

    Required attributes (the test suite copies reference weights/params into
    these by name, so they must exist with exactly these names):
        conv_proj:     nn.Conv2d(in_channels, embed_dim,
                                  kernel_size=patch_size, stride=patch_size)
        class_token:   nn.Parameter of shape (1, 1, embed_dim)
        pos_embedding: nn.Parameter of shape (1, num_patches + 1, embed_dim)
    """

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
        """Embed image patches into a token sequence.

        Args:
            x: (N, in_channels, image_size, image_size) input images.

        Returns:
            (N, num_patches + 1, embed_dim) token sequence.
        """
        x = self.conv_proj(x)            # (N, embed_dim, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)  # (N, num_patches, embed_dim)
        cls = self.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        return x + self.pos_embedding
