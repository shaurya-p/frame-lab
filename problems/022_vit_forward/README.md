# 022 — vit_forward

**Difficulty:** Hard  
**Category:** Deep Learning Primitives

## Goal

Assemble a complete Vision Transformer classifier from scratch, composing the patch embedding (020) and encoder blocks (021): patch embed → `num_layers` encoder blocks → final LayerNorm → take the class token → linear head → logits. Do not use torchvision's `VisionTransformer`.

## Module

```python
class VisionTransformer(nn.Module):
    def __init__(self, image_size: int, patch_size: int, in_channels: int,
                 embed_dim: int, num_layers: int, num_heads: int,
                 mlp_dim: int, num_classes: int): ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (N, C, H, W) -> (N, num_classes)
```

## Pipeline

```
tokens = patch_embed(x)        # (N, num_patches + 1, embed_dim), pos embedding already added
for block in blocks:
    tokens = block(tokens)
tokens = ln(tokens)            # final LayerNorm, applied BEFORE selecting the class token
cls = tokens[:, 0]             # class token is at index 0
logits = head(cls)             # (N, num_classes)
```

## Input format

- `x`: float tensor of shape `(N, in_channels, image_size, image_size)`

## Output format

Float tensor of shape `(N, num_classes)`.

## Required attribute names

The module **must** expose these components with exactly these names. The test suite transfers reference weights into them by name, so a module using different names will fail with an `AttributeError`:

| Attribute | Type |
|---|---|
| `patch_embed` | `PatchEmbed` from 020 (exposes `conv_proj`, `class_token`, `pos_embedding`) |
| `blocks` | `nn.ModuleList` of `EncoderBlock` from 021 (each exposes `ln_1`, `self_attn`, `ln_2`, `mlp_fc1`, `mlp_fc2`) |
| `ln` | `nn.LayerNorm(embed_dim, eps=1e-6)` (final norm) |
| `head` | `nn.Linear(embed_dim, num_classes)` |

## Constraints

- The positional embedding is added inside `patch_embed` (after the class token is prepended, before the blocks)
- The final `ln` is applied **before** selecting the class token, not after
- The class token is at sequence index 0
- LayerNorm uses `eps=1e-6`
- `image_size % patch_size == 0` and `embed_dim % num_heads == 0`
- Do not mutate `x`

## Edge cases

- `N = 1` batch
- `num_layers = 1`
- `num_classes = 1`

## Skills practiced

- Composing validated submodules into a full model
- Ordering the final norm relative to class-token selection
- Stacking encoder blocks with `nn.ModuleList`
- Classification head on the class token
