# 017 — conv2d_forward

**Difficulty:** Medium  
**Category:** Deep Learning Primitives

## Goal

Compute the 2D convolution (cross-correlation) forward pass for a batch of inputs without using `torch.nn.functional.conv2d` or `torch.nn.Conv2d`.

## Function signature

```python
def conv2d_forward(
    x: torch.Tensor,            # (N, C_in, H, W)
    weight: torch.Tensor,       # (C_out, C_in, kH, kW)
    bias: torch.Tensor | None,  # (C_out,) or None
    stride: int = 1,
    padding: int = 0,
) -> torch.Tensor:              # (N, C_out, H_out, W_out)
```

## Input format

- `x`: float tensor of shape `(N, C_in, H, W)` — batch of input feature maps
- `weight`: float tensor of shape `(C_out, C_in, kH, kW)` — one filter per output channel
- `bias`: float tensor of shape `(C_out,)`, or `None`
- `stride`: positive integer ≥ 1
- `padding`: non-negative integer ≥ 0 — zero-pads spatial dims symmetrically before sliding

## Output format

Float tensor of shape `(N, C_out, H_out, W_out)` where:

```
H_out = floor((H + 2·padding − kH) / stride) + 1
W_out = floor((W + 2·padding − kW) / stride) + 1
```

## Example

```python
x      = torch.ones(1, 1, 3, 3)
weight = torch.ones(1, 1, 2, 2)
bias   = None

out = conv2d_forward(x, weight, bias, stride=1, padding=0)
# out.shape == (1, 1, 2, 2)
# out == [[[[4., 4.], [4., 4.]]]]
```

## Constraints

- `groups = 1` — all input channels connect to all output channels
- `dilation = 1`
- `stride ≥ 1`, `padding ≥ 0`
- Input dtype is float32 or float64
- Do not mutate `x`

## Edge cases

- `bias=None` — return output with no bias term added
- `stride=2` — output spatial size roughly halved
- `padding=1` with a 3×3 kernel — output is the same spatial size as input
- Non-square kernel (`kH ≠ kW`) — H and W dimensions shrink by different amounts
- 1×1 kernel (`kH = kW = 1`) — equivalent to per-pixel linear mixing of input channels
- Single-element spatial output (`H_out = W_out = 1`)

## Skills practiced

- Sliding window over spatial dimensions
- Tensor reshaping and batched matrix multiply
- Zero-padding spatial dimensions
- Output size calculation from kernel size, stride, and padding
