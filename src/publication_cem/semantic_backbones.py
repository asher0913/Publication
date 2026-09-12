from __future__ import annotations

from torch import nn
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    ResNet18_Weights,
    mobilenet_v3_large,
    resnet18,
)


SEMANTIC_BACKBONES = ("mobilenet_v3_large", "resnet18")
DEFAULT_SEMANTIC_BACKBONE = SEMANTIC_BACKBONES[0]


def checkpoint_backbone_name(checkpoint_args: dict) -> str:
    """Resolve older checkpoints that predate the explicit backbone field."""
    return str(checkpoint_args.get("semantic_backbone", DEFAULT_SEMANTIC_BACKBONE))


def build_semantic_backbone(
    name: str,
    *,
    pretrained: bool,
) -> tuple[nn.Module, int]:
    if name == "mobilenet_v3_large":
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_large(weights=weights)
        return model.features, 960
    if name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        return features, 512
    raise ValueError(
        f"unsupported semantic backbone {name!r}; expected one of {SEMANTIC_BACKBONES}"
    )
