import torch
from torchvision.models.vision_transformer import VisionTransformer as RefViT
from starter import VisionTransformer


def _copy_block(dst, src, embed_dim):
    """Copy one torchvision encoder layer into a learner EncoderBlock."""
    E = embed_dim
    dst.ln_1.weight.copy_(src.ln_1.weight)
    dst.ln_1.bias.copy_(src.ln_1.bias)
    dst.ln_2.weight.copy_(src.ln_2.weight)
    dst.ln_2.bias.copy_(src.ln_2.bias)
    # in_proj_weight is (3*E, E), ordered q, k, v along dim 0 — direct slice.
    dst.self_attn.q_proj.weight.copy_(src.self_attention.in_proj_weight[:E])
    dst.self_attn.k_proj.weight.copy_(src.self_attention.in_proj_weight[E:2 * E])
    dst.self_attn.v_proj.weight.copy_(src.self_attention.in_proj_weight[2 * E:])
    dst.self_attn.q_proj.bias.copy_(src.self_attention.in_proj_bias[:E])
    dst.self_attn.k_proj.bias.copy_(src.self_attention.in_proj_bias[E:2 * E])
    dst.self_attn.v_proj.bias.copy_(src.self_attention.in_proj_bias[2 * E:])
    dst.self_attn.out_proj.weight.copy_(src.self_attention.out_proj.weight)
    dst.self_attn.out_proj.bias.copy_(src.self_attention.out_proj.bias)
    # MLPBlock: mlp[0] = Linear(E, mlp_dim), mlp[3] = Linear(mlp_dim, E).
    dst.mlp_fc1.weight.copy_(src.mlp[0].weight)
    dst.mlp_fc1.bias.copy_(src.mlp[0].bias)
    dst.mlp_fc2.weight.copy_(src.mlp[3].weight)
    dst.mlp_fc2.bias.copy_(src.mlp[3].bias)


def _make_pair(image_size=8, patch_size=4, embed_dim=16, num_layers=2,
               num_heads=2, mlp_dim=32, num_classes=5):
    """Build a learner ViT and a torchvision ViT sharing identical weights.

    torchvision's conv_proj is fixed to 3 input channels, so in_channels=3.
    """
    ref = RefViT(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=embed_dim,
        mlp_dim=mlp_dim,
        num_classes=num_classes,
    )
    ref.eval()
    vit = VisionTransformer(
        image_size, patch_size, 3, embed_dim,
        num_layers, num_heads, mlp_dim, num_classes,
    )
    vit.eval()
    with torch.no_grad():
        # Patch embed: conv_proj and class_token are top-level on the ref;
        # pos_embedding lives on the ref encoder. All land in patch_embed here.
        vit.patch_embed.conv_proj.weight.copy_(ref.conv_proj.weight)
        vit.patch_embed.conv_proj.bias.copy_(ref.conv_proj.bias)
        vit.patch_embed.class_token.copy_(ref.class_token)
        vit.patch_embed.pos_embedding.copy_(ref.encoder.pos_embedding)
        for dst, src in zip(vit.blocks, ref.encoder.layers):
            _copy_block(dst, src, embed_dim)
        vit.ln.weight.copy_(ref.encoder.ln.weight)
        vit.ln.bias.copy_(ref.encoder.ln.bias)
        vit.head.weight.copy_(ref.heads.head.weight)
        vit.head.bias.copy_(ref.heads.head.bias)
    return vit, ref


def _close(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_matches_oracle_logits():
    torch.manual_seed(0)
    vit, ref = _make_pair()
    x = torch.randn(2, 3, 8, 8)
    _close(vit(x), ref(x))


def test_single_layer():
    torch.manual_seed(1)
    vit, ref = _make_pair(num_layers=1)
    x = torch.randn(2, 3, 8, 8)
    _close(vit(x), ref(x))


def test_output_shape():
    torch.manual_seed(2)
    vit, _ = _make_pair(num_classes=5)
    x = torch.randn(3, 3, 8, 8)
    out = vit(x)
    assert out.shape == (3, 5), f"Expected (3, 5), got {tuple(out.shape)}"


def test_input_not_mutated():
    torch.manual_seed(3)
    vit, _ = _make_pair()
    x = torch.randn(2, 3, 8, 8)
    x_copy = x.clone()
    vit(x)
    assert torch.equal(x, x_copy), "Input x was mutated"
