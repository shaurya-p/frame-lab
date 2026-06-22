import torch
from torchvision.models.vision_transformer import VisionTransformer
from starter import PatchEmbed


def _make_pair(image_size, patch_size, embed_dim, num_heads=2):
    """Build a learner module and a torchvision ViT sharing identical weights.

    torchvision's conv_proj is fixed to 3 input channels, so the oracle uses
    in_channels=3.
    """
    ref = VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=1,
        num_heads=num_heads,
        hidden_dim=embed_dim,
        mlp_dim=embed_dim * 2,
    )
    ref.eval()
    pe = PatchEmbed(image_size, patch_size, 3, embed_dim)
    pe.eval()
    with torch.no_grad():
        pe.conv_proj.weight.copy_(ref.conv_proj.weight)
        pe.conv_proj.bias.copy_(ref.conv_proj.bias)
        pe.class_token.copy_(ref.class_token)
        pe.pos_embedding.copy_(ref.encoder.pos_embedding)
    return pe, ref


def _oracle_tokens(ref, x):
    n = x.shape[0]
    tokens = ref._process_input(x)
    cls = ref.class_token.expand(n, -1, -1)
    tokens = torch.cat([cls, tokens], dim=1)
    return tokens + ref.encoder.pos_embedding


def _close(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_matches_oracle():
    torch.manual_seed(0)
    pe, ref = _make_pair(8, 4, 16)
    x = torch.randn(2, 3, 8, 8)
    _close(pe(x), _oracle_tokens(ref, x))


def test_output_shape():
    torch.manual_seed(1)
    pe, _ = _make_pair(8, 4, 16)
    x = torch.randn(3, 3, 8, 8)
    out = pe(x)
    # num_patches = (8 // 4) ** 2 = 4, sequence length = 5
    assert out.shape == (3, 5, 16), f"Expected (3, 5, 16), got {tuple(out.shape)}"


def test_num_patches():
    pe = PatchEmbed(16, 4, 3, 8)
    assert pe.num_patches == 16, f"Expected 16, got {pe.num_patches}"


def test_input_not_mutated():
    torch.manual_seed(2)
    pe, _ = _make_pair(8, 4, 16)
    x = torch.randn(2, 3, 8, 8)
    x_copy = x.clone()
    pe(x)
    assert torch.equal(x, x_copy), "Input x was mutated"


def test_image_size_not_divisible_raises():
    try:
        PatchEmbed(10, 4, 3, 16)
    except ValueError:
        return
    raise AssertionError("Expected ValueError when image_size % patch_size != 0")
