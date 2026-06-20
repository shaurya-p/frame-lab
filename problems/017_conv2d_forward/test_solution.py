import torch
import torch.nn.functional as F
from solution import conv2d_forward


def _oracle(x, weight, bias, stride=1, padding=0):
    return F.conv2d(x, weight, bias=bias, stride=stride, padding=padding)


def _close(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_basic_stride1_pad0_no_bias():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 8, 8)
    w = torch.randn(4, 3, 3, 3)
    _close(conv2d_forward(x, w, None), _oracle(x, w, None))


def test_basic_with_bias():
    torch.manual_seed(1)
    x = torch.randn(2, 3, 8, 8)
    w = torch.randn(4, 3, 3, 3)
    b = torch.randn(4)
    _close(conv2d_forward(x, w, b), _oracle(x, w, b))


def test_stride2():
    torch.manual_seed(2)
    x = torch.randn(2, 3, 7, 7)
    w = torch.randn(4, 3, 3, 3)
    b = torch.randn(4)
    _close(conv2d_forward(x, w, b, stride=2), _oracle(x, w, b, stride=2))


def test_padding1():
    torch.manual_seed(3)
    x = torch.randn(2, 3, 5, 5)
    w = torch.randn(4, 3, 3, 3)
    b = torch.randn(4)
    _close(conv2d_forward(x, w, b, padding=1), _oracle(x, w, b, padding=1))


def test_bias_none():
    torch.manual_seed(4)
    x = torch.randn(1, 2, 6, 6)
    w = torch.randn(3, 2, 3, 3)
    _close(conv2d_forward(x, w, None), _oracle(x, w, None))


def test_non_square_kernel():
    torch.manual_seed(5)
    x = torch.randn(1, 2, 7, 9)
    w = torch.randn(3, 2, 3, 5)
    b = torch.randn(3)
    _close(conv2d_forward(x, w, b), _oracle(x, w, b))


def test_1x1_kernel():
    torch.manual_seed(6)
    x = torch.randn(2, 4, 5, 5)
    w = torch.randn(8, 4, 1, 1)
    b = torch.randn(8)
    _close(conv2d_forward(x, w, b), _oracle(x, w, b))


def test_stride2_padding1():
    torch.manual_seed(7)
    x = torch.randn(2, 3, 8, 8)
    w = torch.randn(4, 3, 3, 3)
    b = torch.randn(4)
    _close(
        conv2d_forward(x, w, b, stride=2, padding=1),
        _oracle(x, w, b, stride=2, padding=1),
    )


def test_output_shape():
    torch.manual_seed(8)
    x = torch.randn(3, 2, 10, 12)
    w = torch.randn(5, 2, 3, 4)
    out = conv2d_forward(x, w, None, stride=2, padding=1)
    # H_out = (10 + 2 - 3) // 2 + 1 = 5, W_out = (12 + 2 - 4) // 2 + 1 = 6
    assert out.shape == (3, 5, 5, 6), f"Expected (3, 5, 5, 6), got {tuple(out.shape)}"


def test_input_not_mutated():
    torch.manual_seed(9)
    x = torch.randn(2, 3, 8, 8)
    w = torch.randn(4, 3, 3, 3)
    x_copy = x.clone()
    conv2d_forward(x, w, None, stride=1, padding=1)
    assert torch.equal(x, x_copy), "Input x was mutated"
