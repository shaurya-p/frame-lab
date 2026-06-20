import torch
import torch.nn.functional as F


def conv2d_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: int = 1,
    padding: int = 0,
) -> torch.Tensor:
    """Compute the 2D convolution forward pass.

    Args:
        x: (N, C_in, H, W) input feature maps.
        weight: (C_out, C_in, kH, kW) convolution filters.
        bias: (C_out,) bias vector, or None.
        stride: step size for the sliding window.
        padding: number of zeros added to each spatial side.

    Returns:
        (N, C_out, H_out, W_out) output feature maps.
    """
    N, C_in, H, W = x.shape
    C_out, _, kH, kW = weight.shape

    if padding > 0:
        x = F.pad(x, (padding, padding, padding, padding))

    _, _, H_pad, W_pad = x.shape
    H_out = (H_pad - kH) // stride + 1
    W_out = (W_pad - kW) // stride + 1

    out = x.new_zeros(N, C_out, H_out, W_out)
    w = weight.reshape(C_out, -1)  # (C_out, C_in*kH*kW)

    for h in range(H_out):
        for wi in range(W_out):
            h0, w0 = h * stride, wi * stride
            # (N, C_in*kH*kW) @ (C_in*kH*kW, C_out) → (N, C_out)
            patch = x[:, :, h0:h0 + kH, w0:w0 + kW].reshape(N, -1)
            out[:, :, h, wi] = patch @ w.T

    if bias is not None:
        out = out + bias.view(1, C_out, 1, 1)

    return out
