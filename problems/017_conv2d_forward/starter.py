import torch


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
    raise NotImplementedError
