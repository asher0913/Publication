from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .models import ImageNetNormalise


class GlobalSemanticBottleneck(nn.Module):
    """Map an image to a compact token with no explicit spatial layout."""

    def __init__(
        self,
        backbone: nn.Module,
        backbone_channels: int,
        token_dim: int,
        num_classes: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if backbone_channels <= 0 or token_dim <= 0 or num_classes <= 0:
            raise ValueError("semantic bottleneck dimensions must be positive")
        self.normalise = ImageNetNormalise()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.project = nn.Sequential(
            nn.Linear(backbone_channels, token_dim, bias=False),
            nn.LayerNorm(token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(token_dim, num_classes)

    def encode(self, images: Tensor) -> Tensor:
        feature_map = self.backbone(self.normalise(images))
        pooled = self.pool(feature_map).flatten(start_dim=1)
        return self.project(pooled)

    def forward(
        self,
        images: Tensor,
        noise_std: float = 0.0,
        generator=None,
    ) -> tuple[Tensor, Tensor]:
        if noise_std < 0:
            raise ValueError("noise_std must be non-negative")
        token = self.encode(images)
        transmitted = token
        if noise_std:
            noise = torch.randn(
                token.shape,
                device=token.device,
                dtype=token.dtype,
                generator=generator,
            )
            transmitted = token + noise_std * noise
        return self.classifier(transmitted), transmitted


class CalibratedLogitFusion(nn.Module):
    """Learn a bounded mixture of protected-spatial and semantic logits."""

    def __init__(self, semantic_weight: float = 0.35) -> None:
        super().__init__()
        if not 0 < semantic_weight < 1:
            raise ValueError("semantic_weight must be strictly between zero and one")
        self.mix_logit = nn.Parameter(
            torch.tensor(math.log(semantic_weight / (1.0 - semantic_weight)))
        )
        self.legacy_log_temperature = nn.Parameter(torch.zeros(()))
        self.semantic_log_temperature = nn.Parameter(torch.zeros(()))

    def forward(self, legacy_logits: Tensor, semantic_logits: Tensor) -> Tensor:
        if legacy_logits.shape != semantic_logits.shape:
            raise ValueError("fusion logits must have identical shapes")
        weight = self.mix_logit.sigmoid()
        legacy_temperature = self.legacy_log_temperature.exp().clamp(0.25, 4.0)
        semantic_temperature = self.semantic_log_temperature.exp().clamp(0.25, 4.0)
        return (
            (1.0 - weight) * legacy_logits / legacy_temperature
            + weight * semantic_logits / semantic_temperature
        )

