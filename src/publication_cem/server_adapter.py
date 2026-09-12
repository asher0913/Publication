from __future__ import annotations

import torch
from torch import Tensor, nn


class ResidualSemanticAdapter(nn.Module):
    """Recover task signal at the server without changing transmitted features."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int = 64,
        reduction: int = 4,
    ) -> None:
        super().__init__()
        if channels <= 0 or hidden_channels <= 0 or reduction <= 0:
            raise ValueError("adapter dimensions must be positive")
        squeeze_channels = max(1, channels // reduction)
        groups = min(4, channels)
        while channels % groups:
            groups -= 1

        self.context = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, squeeze_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(squeeze_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        # Exact identity at initialisation; the first update opens only useful
        # residual channels before the larger adapter branch starts moving.
        self.residual_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, smashed_features: Tensor) -> Tensor:
        if smashed_features.ndim != 4:
            raise ValueError("adapter expects a BCHW smashed-feature tensor")
        residual = self.context(smashed_features)
        residual = residual * self.channel_gate(residual)
        return smashed_features + torch.tanh(self.residual_scale) * residual

