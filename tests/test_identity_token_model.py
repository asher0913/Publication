import torch

from publication_cem.models import (
    PermutationInvariantTokenCloud,
    SpatialSlotTokenizer,
    build_split_model,
)


def test_slot_tokenizer_has_fixed_compact_shape_and_gradients() -> None:
    tokenizer = SpatialSlotTokenizer(24, token_dim=32, num_slots=4, iterations=2)
    inputs = torch.randn(3, 24, 6, 6, requires_grad=True)
    output = tokenizer(inputs)
    assert output.shape == (3, 32, 2, 2)
    output.square().mean().backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()


def test_token_cloud_is_permutation_invariant_without_positional_encoding() -> None:
    cloud = PermutationInvariantTokenCloud(
        token_dim=32, hidden_dim=48, layers=1, heads=4, dropout=0.0
    ).eval()
    tokens = torch.randn(2, 32, 2, 2)
    sequence = tokens.flatten(start_dim=2)
    permutation = torch.tensor([2, 0, 3, 1])
    shuffled = sequence[:, :, permutation].reshape_as(tokens)
    assert torch.allclose(cloud(tokens), cloud(shuffled), atol=1e-6, rtol=1e-5)


def test_mobilenet_slot_split_transmits_four_tokens() -> None:
    model = build_split_model(
        {
            "name": "mobilenet_slot_split",
            "backbone_variant": "small",
            "bottleneck_channels": 32,
            "num_slots": 4,
            "slot_iterations": 1,
            "cloud_hidden_dim": 48,
            "cloud_layers": 1,
            "cloud_heads": 4,
            "dropout": 0.0,
            "pretrained": False,
        },
        num_classes=11,
        image_size=48,
    )
    logits, clean, transmitted = model(torch.rand(2, 3, 48, 48), noise_std=0.025)
    assert logits.shape == (2, 11)
    assert clean.shape == transmitted.shape == (2, 32, 2, 2)
