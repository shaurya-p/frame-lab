# 020 — patch_embed

**Difficulty:** Medium  
**Category:** Deep Learning Primitives

## Goal

Implement the input embedding stage of a Vision Transformer from scratch: project image patches with a convolution, flatten them into a token sequence, prepend a learnable class token, and add a learnable positional embedding. Do not use any `torchvision` ViT class.

## Module

```python
class PatchEmbed(nn.Module):
    def __init__(self, image_size: int, patch_size: int,
                 in_channels: int, embed_dim: int): ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (N, C, H, W) -> (N, num_patches + 1, embed_dim)
```

Images are assumed square. `num_patches = (image_size // patch_size) ** 2`, and the output sequence length is `num_patches + 1` (the extra token is the prepended class token).

## Input format

- `x`: float tensor of shape `(N, in_channels, image_size, image_size)`

## Output format

Float tensor of shape `(N, num_patches + 1, embed_dim)`.

## Required attribute names

The module **must** expose these components with exactly these names. The test suite transfers reference weights/parameters into them by name, so a module using different names will fail with an `AttributeError`:

| Attribute | Type / shape |
|---|---|
| `conv_proj` | `nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)` |
| `class_token` | `nn.Parameter` of shape `(1, 1, embed_dim)` |
| `pos_embedding` | `nn.Parameter` of shape `(1, num_patches + 1, embed_dim)` |

## Constraints

- `image_size % patch_size == 0` is required (assert in `__init__`)
- Images are square
- Do not mutate `x`

## Edge cases

- `N = 1` batch
- `patch_size == image_size` — a single patch (`num_patches = 1`, sequence length 2)
- `in_channels` other than 3

## Skills practiced

- Convolutional patch projection
- Flattening a spatial feature map into a token sequence
- Prepending a learnable class token
- Adding a learnable positional embedding
