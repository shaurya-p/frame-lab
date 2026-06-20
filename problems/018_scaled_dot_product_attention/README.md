# 018 — scaled_dot_product_attention

**Difficulty:** Medium  
**Category:** Deep Learning Primitives

## Goal

Compute scaled dot-product attention from scratch without using `torch.nn.functional.scaled_dot_product_attention`.

## Function signature

```python
def scaled_dot_product_attention(
    q: torch.Tensor,                         # (..., L, d)
    k: torch.Tensor,                         # (..., S, d)
    v: torch.Tensor,                         # (..., S, d_v)
    attn_mask: torch.Tensor | None = None,   # additive float mask, broadcastable to (..., L, S)
) -> torch.Tensor:                           # (..., L, d_v)
```

## Input format

- `q`: float tensor of shape `(..., L, d)` — queries (any number of leading batch dims)
- `k`: float tensor of shape `(..., S, d)` — keys, same `d` as `q`
- `v`: float tensor of shape `(..., S, d_v)` — values
- `attn_mask`: float tensor broadcastable to `(..., L, S)`, **added** to the pre-softmax scores, or `None`

## Output format

Float tensor of shape `(..., L, d_v)`.

## Computation

```
scores = (q @ kᵀ) / sqrt(d)        # d = q.shape[-1], the query/key dim
if attn_mask is not None:
    scores = scores + attn_mask
weights = softmax(scores, dim=-1)  # over the key (S) dimension
out = weights @ v
```

## Example

```python
q = torch.zeros(1, 2)          # (L=1, d=2)
k = torch.zeros(2, 2)          # (S=2, d=2)
v = torch.tensor([[1.0, 0.0],
                  [0.0, 1.0]]) # (S=2, d_v=2)

out = scaled_dot_product_attention(q, k, v)
# scores are all zero → uniform softmax → out == [[0.5, 0.5]]
```

## Constraints

- Scale is exactly `1 / sqrt(d)` where `d = q.shape[-1]` (not `d_v`)
- Softmax is taken over the last (key) dimension
- `attn_mask`, when provided, is **float and additive** — it is added to the scores. Boolean masks are not supported.
- No causal flag, no dropout
- Do not mutate `q`, `k`, `v`, or `attn_mask`

## Edge cases

- `attn_mask=None` — plain attention over all keys
- Additive mask with large negative entries that suppress specific keys (never mask every key in a row)
- Multiple leading batch dims, e.g. `(B, H, L, S)`
- `d_v ≠ d` — value dimension differs from the query/key dimension
- Single query (`L = 1`)

## Skills practiced

- Batched matrix multiply with leading dimensions
- Softmax over a chosen axis
- Additive masking before softmax
- Correct scaling by the query/key dimension
