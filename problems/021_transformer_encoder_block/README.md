# 021 — transformer_encoder_block

**Difficulty:** Medium  
**Category:** Deep Learning Primitives

## Goal

Implement a pre-norm Vision Transformer encoder block from scratch, reusing the multi-head attention logic from problem 019. Do not use torchvision's `EncoderBlock` or `nn.TransformerEncoderLayer`.

## Module

```python
class EncoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int): ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (N, L, E) -> (N, L, E)
```

The block is pre-norm:

```
y = x + MHA(LN1(x))
z = y + MLP(LN2(y))
```

where `MLP = Linear(embed_dim, mlp_dim) → GELU → Linear(mlp_dim, embed_dim)`.

## Input format

- `x`: float tensor of shape `(N, L, embed_dim)`

## Output format

Float tensor of shape `(N, L, embed_dim)`.

## Required attribute names

The module **must** expose these submodules with exactly these names. The test suite transfers reference weights into them by name, so a module using different names will fail with an `AttributeError`:

| Attribute | Type |
|---|---|
| `ln_1` | `nn.LayerNorm(embed_dim, eps=1e-6)` |
| `self_attn` | `MultiHeadAttention(embed_dim, num_heads)` from 019 (exposes `q_proj`/`k_proj`/`v_proj`/`out_proj`) |
| `ln_2` | `nn.LayerNorm(embed_dim, eps=1e-6)` |
| `mlp_fc1` | `nn.Linear(embed_dim, mlp_dim)` |
| `mlp_fc2` | `nn.Linear(mlp_dim, embed_dim)` |

## Constraints

- LayerNorm uses `eps=1e-6` (not the `nn.LayerNorm` default of `1e-5`)
- The MLP activation is the exact (erf-based) GELU — `nn.GELU()` / `F.gelu` with the default `approximate='none'`
- `embed_dim % num_heads == 0`
- Do not mutate `x`

## Edge cases

- `num_heads = 1`
- Single token (`L = 1`)
- `N = 1` batch

## Skills practiced

- Pre-norm residual structure
- Reusing a multi-head attention submodule
- Two-layer MLP with GELU
- Matching LayerNorm eps and activation conventions
