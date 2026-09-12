from __future__ import annotations

import torch
from torch import Tensor, nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
        )
        self.activation = nn.GELU()

    def forward(self, inputs: Tensor) -> Tensor:
        return self.activation(inputs + self.block(inputs))


class DualPathReconstructor(nn.Module):
    """Strong decoder with access to both transmitted representations."""

    def __init__(
        self,
        spatial_channels: int = 16,
        semantic_dim: int = 256,
        width: int = 128,
        residual_blocks: int = 4,
    ) -> None:
        super().__init__()
        if min(spatial_channels, semantic_dim, width, residual_blocks) <= 0:
            raise ValueError("reconstructor dimensions must be positive")
        self.semantic_seed = nn.Sequential(
            nn.Linear(semantic_dim, width * 4 * 4),
            nn.GELU(),
        )
        self.semantic_upsample = nn.Sequential(
            nn.ConvTranspose2d(width, width, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(width, width, 4, stride=2, padding=1),
            nn.GELU(),
        )
        self.spatial_projection = nn.Sequential(
            nn.Conv2d(spatial_channels, width, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(2 * width, width, kernel_size=1),
            nn.GELU(),
            *[ResidualBlock(width) for _ in range(residual_blocks)],
        )
        self.output = nn.Sequential(
            nn.ConvTranspose2d(width, width // 2, 4, stride=2, padding=1),
            nn.GELU(),
            ResidualBlock(width // 2),
            nn.ConvTranspose2d(width // 2, width // 4, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(width // 4, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, spatial: Tensor, semantic: Tensor) -> Tensor:
        if spatial.ndim != 4 or semantic.ndim != 2:
            raise ValueError("expected BCHW spatial features and BD semantic tokens")
        semantic_map = self.semantic_seed(semantic).view(semantic.shape[0], -1, 4, 4)
        semantic_map = self.semantic_upsample(semantic_map)
        spatial_map = self.spatial_projection(spatial)
        if semantic_map.shape[-2:] != spatial_map.shape[-2:]:
            raise ValueError("semantic and spatial paths resolve to different sizes")
        return self.output(self.fusion(torch.cat((spatial_map, semantic_map), dim=1)))


class IdentityConditionedDualPathReconstructor(DualPathReconstructor):
    """Joint reconstructor modulated by an identity posterior or one-hot prior."""

    def __init__(
        self,
        spatial_channels: int = 16,
        semantic_dim: int = 256,
        num_identities: int = 530,
        width: int = 128,
        residual_blocks: int = 4,
        condition_hidden_dim: int = 256,
    ) -> None:
        super().__init__(
            spatial_channels=spatial_channels,
            semantic_dim=semantic_dim,
            width=width,
            residual_blocks=residual_blocks,
        )
        if num_identities <= 0 or condition_hidden_dim <= 0:
            raise ValueError("identity-conditioning dimensions must be positive")
        self.num_identities = num_identities
        self.identity_modulation = nn.Sequential(
            nn.Linear(num_identities, condition_hidden_dim),
            nn.GELU(),
            nn.Linear(condition_hidden_dim, 2 * width),
        )

    def forward(
        self,
        spatial: Tensor,
        semantic: Tensor,
        identity_condition: Tensor,
    ) -> Tensor:
        if spatial.ndim != 4 or semantic.ndim != 2:
            raise ValueError("expected BCHW spatial features and BD semantic tokens")
        if identity_condition.ndim != 2:
            raise ValueError("identity condition must be a B-by-C tensor")
        if identity_condition.shape != (spatial.shape[0], self.num_identities):
            raise ValueError("identity condition has the wrong batch or class dimension")

        semantic_map = self.semantic_seed(semantic).view(semantic.shape[0], -1, 4, 4)
        semantic_map = self.semantic_upsample(semantic_map)
        spatial_map = self.spatial_projection(spatial)
        if semantic_map.shape[-2:] != spatial_map.shape[-2:]:
            raise ValueError("semantic and spatial paths resolve to different sizes")

        hidden = self.fusion(torch.cat((spatial_map, semantic_map), dim=1))
        scale, shift = self.identity_modulation(identity_condition).chunk(2, dim=1)
        scale = 0.1 * torch.tanh(scale).unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        return self.output(hidden * (1.0 + scale) + shift)


class PatchDiscriminator(nn.Module):
    def __init__(self, width: int = 64) -> None:
        super().__init__()
        layers = []
        channels = 3
        for multiplier in (1, 2, 4, 8):
            out_channels = width * multiplier
            layers.extend(
                [
                    nn.Conv2d(channels, out_channels, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            channels = out_channels
        layers.append(nn.Conv2d(channels, 1, kernel_size=4))
        self.network = nn.Sequential(*layers)

    def forward(self, images: Tensor) -> Tensor:
        return self.network(images).flatten(start_dim=1)
