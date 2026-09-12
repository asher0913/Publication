from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from torch import Tensor, nn

from .regularizer import PrototypeCEMOutput


@dataclass
class ObjectiveOutput:
    total_loss: Tensor
    task_loss: Tensor
    privacy_loss: Tensor
    diagnostics: Dict[str, Tensor]


class PrivacyUtilityObjective(nn.Module):
    """Compose task and privacy losses for one ordinary backward pass."""

    def __init__(self, regularizer: nn.Module, privacy_weight: float):
        super().__init__()
        if privacy_weight < 0:
            raise ValueError("privacy_weight must be non-negative")
        self.regularizer = regularizer
        self.privacy_weight = privacy_weight

    def forward(
        self,
        task_loss: Tensor,
        smashed_features: Tensor,
        labels: Tensor,
        **regularizer_kwargs,
    ) -> ObjectiveOutput:
        privacy: PrototypeCEMOutput = self.regularizer(
            smashed_features, labels, **regularizer_kwargs
        )
        total = task_loss + self.privacy_weight * privacy.loss
        return ObjectiveOutput(
            total_loss=total,
            task_loss=task_loss,
            privacy_loss=privacy.loss,
            diagnostics=privacy.diagnostics,
        )
