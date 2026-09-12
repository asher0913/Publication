import torch

from publication_cem.server_adapter import ResidualSemanticAdapter


def test_adapter_is_identity_at_initialisation() -> None:
    adapter = ResidualSemanticAdapter(16, hidden_channels=32)
    features = torch.randn(3, 16, 8, 8)
    output = adapter(features)
    assert torch.equal(output, features)


def test_adapter_preserves_shape_and_receives_gradient() -> None:
    adapter = ResidualSemanticAdapter(16, hidden_channels=32)
    features = torch.randn(2, 16, 8, 8, requires_grad=True)
    adapter(features).square().mean().backward()
    assert adapter.residual_scale.grad is not None
    assert adapter(features).shape == features.shape


def test_adapter_rejects_non_spatial_features() -> None:
    adapter = ResidualSemanticAdapter(16)
    try:
        adapter(torch.randn(2, 16))
    except ValueError as exc:
        assert "BCHW" in str(exc)
    else:
        raise AssertionError("expected non-spatial features to be rejected")

