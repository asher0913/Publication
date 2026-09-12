from __future__ import annotations

import torch
from torch import Tensor, nn


class FixedOrthogonalProjector(nn.Module):
    """A non-trainable orthogonal projection used only for assignments.

    Keeping the matrix as a buffer removes the scale-collapse shortcut of a
    privacy-only learned projection while preserving cheap cosine assignment.
    """

    def __init__(self, feature_dim: int, projection_dim: int, seed: int = 125):
        super().__init__()
        if not 0 < projection_dim <= feature_dim:
            raise ValueError("projection_dim must be in [1, feature_dim]")

        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        random_matrix = torch.randn(
            feature_dim, projection_dim, generator=generator, dtype=torch.float64
        )
        q, _ = torch.linalg.qr(random_matrix, mode="reduced")
        self.register_buffer("matrix", q.to(dtype=torch.float32), persistent=True)

    @property
    def feature_dim(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def projection_dim(self) -> int:
        return int(self.matrix.shape[1])

    def forward(self, features: Tensor) -> Tensor:
        if features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"expected last dimension {self.feature_dim}, got {features.shape[-1]}"
            )
        return features @ self.matrix.to(dtype=features.dtype)


class LearnedProjector(nn.Module):
    """Learned privacy-only projector retained strictly as a shortcut ablation."""

    def __init__(self, feature_dim: int, projection_dim: int, seed: int = 125):
        super().__init__()
        initial = FixedOrthogonalProjector(feature_dim, projection_dim, seed).matrix
        self.matrix = nn.Parameter(initial.clone())

    @property
    def feature_dim(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def projection_dim(self) -> int:
        return int(self.matrix.shape[1])

    def forward(self, features: Tensor) -> Tensor:
        if features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"expected last dimension {self.feature_dim}, got {features.shape[-1]}"
            )
        return features @ self.matrix.to(dtype=features.dtype)
