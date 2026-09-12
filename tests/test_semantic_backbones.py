import pytest
import torch

from publication_cem.semantic_backbones import (
    build_semantic_backbone,
    checkpoint_backbone_name,
)


@pytest.mark.parametrize(
    ("name", "channels"),
    (("mobilenet_v3_large", 960), ("resnet18", 512)),
)
def test_supported_semantic_backbones_emit_spatial_features(name, channels) -> None:
    backbone, actual_channels = build_semantic_backbone(name, pretrained=False)
    output = backbone(torch.rand(1, 3, 64, 64))
    assert output.ndim == 4
    assert output.shape[1] == channels == actual_channels


def test_legacy_checkpoint_defaults_to_mobilenet() -> None:
    assert checkpoint_backbone_name({}) == "mobilenet_v3_large"


def test_unknown_backbone_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported semantic backbone"):
        build_semantic_backbone("unknown", pretrained=False)
