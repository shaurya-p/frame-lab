import torch
import torch.nn.functional as F
from starter import scaled_dot_product_attention


def _close(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_basic_no_mask():
    torch.manual_seed(0)
    q = torch.randn(2, 5, 8)
    k = torch.randn(2, 7, 8)
    v = torch.randn(2, 7, 8)
    _close(
        scaled_dot_product_attention(q, k, v),
        F.scaled_dot_product_attention(q, k, v),
    )


def test_additive_mask():
    torch.manual_seed(1)
    q = torch.randn(2, 4, 8)
    k = torch.randn(2, 6, 8)
    v = torch.randn(2, 6, 8)
    # Float-additive mask; keep at least one unmasked key per row (column 0).
    mask = torch.zeros(2, 4, 6)
    mask[:, :, 1:] = torch.randn(2, 4, 5) - 5.0
    _close(
        scaled_dot_product_attention(q, k, v, mask),
        F.scaled_dot_product_attention(q, k, v, attn_mask=mask),
    )


def test_multi_leading_dims():
    torch.manual_seed(2)
    q = torch.randn(3, 4, 5, 16)  # (B, H, L, d)
    k = torch.randn(3, 4, 9, 16)
    v = torch.randn(3, 4, 9, 16)
    _close(
        scaled_dot_product_attention(q, k, v),
        F.scaled_dot_product_attention(q, k, v),
    )


def test_dv_differs_from_d():
    torch.manual_seed(3)
    q = torch.randn(2, 5, 8)
    k = torch.randn(2, 7, 8)
    v = torch.randn(2, 7, 12)  # d_v = 12 != d = 8
    _close(
        scaled_dot_product_attention(q, k, v),
        F.scaled_dot_product_attention(q, k, v),
    )


def test_single_query():
    torch.manual_seed(4)
    q = torch.randn(2, 1, 8)
    k = torch.randn(2, 6, 8)
    v = torch.randn(2, 6, 8)
    _close(
        scaled_dot_product_attention(q, k, v),
        F.scaled_dot_product_attention(q, k, v),
    )


def test_broadcast_mask():
    torch.manual_seed(5)
    q = torch.randn(3, 4, 8)
    k = torch.randn(3, 6, 8)
    v = torch.randn(3, 6, 8)
    # Mask broadcastable to (3, 4, 6); column 0 stays unmasked.
    mask = torch.zeros(1, 1, 6)
    mask[..., 1:] = -6.0
    _close(
        scaled_dot_product_attention(q, k, v, mask),
        F.scaled_dot_product_attention(q, k, v, attn_mask=mask),
    )


def test_output_shape():
    torch.manual_seed(6)
    q = torch.randn(3, 4, 5, 8)
    k = torch.randn(3, 4, 9, 8)
    v = torch.randn(3, 4, 9, 12)
    out = scaled_dot_product_attention(q, k, v)
    assert out.shape == (3, 4, 5, 12), f"Expected (3, 4, 5, 12), got {tuple(out.shape)}"


def test_inputs_not_mutated():
    torch.manual_seed(7)
    q = torch.randn(2, 5, 8)
    k = torch.randn(2, 7, 8)
    v = torch.randn(2, 7, 8)
    mask = torch.zeros(2, 5, 7)
    mask[:, :, 1:] = -4.0
    q_c, k_c, v_c, m_c = q.clone(), k.clone(), v.clone(), mask.clone()
    scaled_dot_product_attention(q, k, v, mask)
    assert torch.equal(q, q_c), "q was mutated"
    assert torch.equal(k, k_c), "k was mutated"
    assert torch.equal(v, v_c), "v was mutated"
    assert torch.equal(mask, m_c), "attn_mask was mutated"
