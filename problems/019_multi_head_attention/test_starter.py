import torch
import torch.nn as nn
from starter import MultiHeadAttention


def _make_pair(embed_dim, num_heads):
    """Build a learner module and a torch reference sharing identical weights."""
    ref = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    ref.eval()
    mha = MultiHeadAttention(embed_dim, num_heads)
    mha.eval()
    E = embed_dim
    with torch.no_grad():
        # in_proj_weight is (3*E, E), ordered q, k, v along dim 0 — direct slice, no transpose.
        mha.q_proj.weight.copy_(ref.in_proj_weight[:E])
        mha.k_proj.weight.copy_(ref.in_proj_weight[E:2 * E])
        mha.v_proj.weight.copy_(ref.in_proj_weight[2 * E:])
        mha.q_proj.bias.copy_(ref.in_proj_bias[:E])
        mha.k_proj.bias.copy_(ref.in_proj_bias[E:2 * E])
        mha.v_proj.bias.copy_(ref.in_proj_bias[2 * E:])
        mha.out_proj.weight.copy_(ref.out_proj.weight)
        mha.out_proj.bias.copy_(ref.out_proj.bias)
    return mha, ref


def _close(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_matches_oracle_16_4():
    torch.manual_seed(0)
    mha, ref = _make_pair(16, 4)
    x = torch.randn(2, 5, 16)
    ref_out, _ = ref(x, x, x, need_weights=False)
    _close(mha(x), ref_out)


def test_num_heads_1():
    torch.manual_seed(1)
    mha, ref = _make_pair(12, 1)
    x = torch.randn(3, 7, 12)
    ref_out, _ = ref(x, x, x, need_weights=False)
    _close(mha(x), ref_out)


def test_single_token():
    torch.manual_seed(2)
    mha, ref = _make_pair(16, 4)
    x = torch.randn(2, 1, 16)
    ref_out, _ = ref(x, x, x, need_weights=False)
    _close(mha(x), ref_out)


def test_output_shape():
    torch.manual_seed(3)
    mha, _ = _make_pair(32, 8)
    x = torch.randn(4, 9, 32)
    out = mha(x)
    assert out.shape == (4, 9, 32), f"Expected (4, 9, 32), got {tuple(out.shape)}"


def test_input_not_mutated():
    torch.manual_seed(4)
    mha, _ = _make_pair(16, 4)
    x = torch.randn(2, 5, 16)
    x_copy = x.clone()
    mha(x)
    assert torch.equal(x, x_copy), "Input x was mutated"


def test_embed_dim_not_divisible_raises():
    with torch.no_grad():
        try:
            MultiHeadAttention(10, 4)
        except ValueError:
            return
    raise AssertionError("Expected ValueError when embed_dim % num_heads != 0")
