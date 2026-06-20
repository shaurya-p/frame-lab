# 019 — multi_head_attention

**Difficulty:** Medium  
**Category:** Deep Learning Primitives

## Goal

Implement multi-head self-attention as an `nn.Module` from scratch, reusing the scaled dot-product attention logic from problem 018. Do not use `nn.MultiheadAttention`.

## Module

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int): ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (N, L, embed_dim) -> (N, L, embed_dim)
```

Use separate query, key, and value projections plus an output projection, each a linear layer **with bias**. The module performs self-attention: queries, keys, and values are all derived from the same input `x`.

## Input format

- `x`: float tensor of shape `(N, L, embed_dim)` — batch of `L`-length token sequences

## Output format

Float tensor of shape `(N, L, embed_dim)`.

## Computation

```
q, k, v = q_proj(x), k_proj(x), v_proj(x)        # each (N, L, embed_dim)
split into heads → (N, num_heads, L, head_dim)   # head_dim = embed_dim // num_heads
attn = scaled_dot_product_attention(q, k, v)     # scale 1/sqrt(head_dim)
merge heads → (N, L, embed_dim)
out = out_proj(attn)
```

## Constraints

- `embed_dim % num_heads == 0` is required (each head gets `head_dim = embed_dim // num_heads`)
- Scale is `1 / sqrt(head_dim)`, not `1 / sqrt(embed_dim)`
- No dropout, no attention mask, no causal flag
- Do not mutate `x`

## Edge cases

- `num_heads = 1` — degenerates to plain single-head scaled dot-product attention
- Single token (`L = 1`)
- `N = 1` batch

## Skills practiced

- Splitting and merging attention heads via reshape + transpose
- Reusing a scaled dot-product attention primitive
- Building an `nn.Module` with multiple linear projections
- Correct per-head scaling
