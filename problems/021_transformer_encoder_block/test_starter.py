import torch
from torchvision.models.vision_transformer import EncoderBlock as RefEncoderBlock
from starter import EncoderBlock


def _make_pair(embed_dim, num_heads, mlp_dim):
    """Build a learner block and a torchvision EncoderBlock sharing weights."""
    ref = RefEncoderBlock(
        num_heads,
        hidden_dim=embed_dim,
        mlp_dim=mlp_dim,
        dropout=0.0,
        attention_dropout=0.0,
    )
    ref.eval()
    blk = EncoderBlock(embed_dim, num_heads, mlp_dim)
    blk.eval()
    E = embed_dim
    with torch.no_grad():
        blk.ln_1.weight.copy_(ref.ln_1.weight)
        blk.ln_1.bias.copy_(ref.ln_1.bias)
        blk.ln_2.weight.copy_(ref.ln_2.weight)
        blk.ln_2.bias.copy_(ref.ln_2.bias)
        # in_proj_weight is (3*E, E), ordered q, k, v along dim 0 — direct slice.
        blk.self_attn.q_proj.weight.copy_(ref.self_attention.in_proj_weight[:E])
        blk.self_attn.k_proj.weight.copy_(ref.self_attention.in_proj_weight[E:2 * E])
        blk.self_attn.v_proj.weight.copy_(ref.self_attention.in_proj_weight[2 * E:])
        blk.self_attn.q_proj.bias.copy_(ref.self_attention.in_proj_bias[:E])
        blk.self_attn.k_proj.bias.copy_(ref.self_attention.in_proj_bias[E:2 * E])
        blk.self_attn.v_proj.bias.copy_(ref.self_attention.in_proj_bias[2 * E:])
        blk.self_attn.out_proj.weight.copy_(ref.self_attention.out_proj.weight)
        blk.self_attn.out_proj.bias.copy_(ref.self_attention.out_proj.bias)
        # MLPBlock: mlp[0] = Linear(E, mlp_dim), mlp[3] = Linear(mlp_dim, E).
        blk.mlp_fc1.weight.copy_(ref.mlp[0].weight)
        blk.mlp_fc1.bias.copy_(ref.mlp[0].bias)
        blk.mlp_fc2.weight.copy_(ref.mlp[3].weight)
        blk.mlp_fc2.bias.copy_(ref.mlp[3].bias)
    return blk, ref


def _close(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_matches_oracle_16_4():
    torch.manual_seed(0)
    blk, ref = _make_pair(16, 4, 32)
    x = torch.randn(2, 5, 16)
    _close(blk(x), ref(x))


def test_num_heads_1():
    torch.manual_seed(1)
    blk, ref = _make_pair(12, 1, 24)
    x = torch.randn(3, 7, 12)
    _close(blk(x), ref(x))


def test_single_token():
    torch.manual_seed(2)
    blk, ref = _make_pair(16, 4, 32)
    x = torch.randn(2, 1, 16)
    _close(blk(x), ref(x))


def test_output_shape():
    torch.manual_seed(3)
    blk, _ = _make_pair(16, 4, 32)
    x = torch.randn(4, 9, 16)
    out = blk(x)
    assert out.shape == (4, 9, 16), f"Expected (4, 9, 16), got {tuple(out.shape)}"


def test_input_not_mutated():
    torch.manual_seed(4)
    blk, _ = _make_pair(16, 4, 32)
    x = torch.randn(2, 5, 16)
    x_copy = x.clone()
    blk(x)
    assert torch.equal(x, x_copy), "Input x was mutated"
