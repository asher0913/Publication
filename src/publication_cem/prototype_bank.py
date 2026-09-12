from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


class ClassPrototypeBank(nn.Module):
    """Persistent full-space class prototypes updated after loss evaluation."""

    def __init__(
        self,
        num_classes: int,
        num_slots: int,
        feature_dim: int,
        momentum: float,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_slots = num_slots
        self.feature_dim = feature_dim
        self.momentum = momentum
        self.register_buffer(
            "prototypes", torch.zeros(num_classes, num_slots, feature_dim)
        )
        self.register_buffer("counts", torch.zeros(num_classes, num_slots))
        self.register_buffer(
            "valid", torch.zeros(num_classes, num_slots, dtype=torch.bool)
        )

    def get(self, class_index: int) -> Tensor:
        self._check_class(class_index)
        return self.prototypes[class_index, self.valid[class_index]]

    def valid_count(self, class_index: int) -> int:
        self._check_class(class_index)
        return int(self.valid[class_index].sum().item())

    @property
    def memory_bytes(self) -> int:
        buffers = (self.prototypes, self.counts, self.valid)
        return sum(value.numel() * value.element_size() for value in buffers)

    @torch.no_grad()
    def update_class(
        self,
        class_index: int,
        features: Tensor,
        assignments: Optional[Tensor] = None,
    ) -> None:
        """Update one class using only detached current-batch features.

        Empty slots are initialised first. Once all slots are populated, a
        detached soft assignment updates each prototype by exponential moving
        average. Calling this after the privacy loss prevents the current batch
        from appearing both as a live tensor and as a detached bank entry.
        """

        self._check_class(class_index)
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError("features must have shape [N, feature_dim]")
        if features.shape[0] == 0:
            return

        features = features.detach()
        empty_slots = torch.where(~self.valid[class_index])[0]
        initialise_count = min(int(empty_slots.numel()), int(features.shape[0]))
        if initialise_count:
            target_slots = empty_slots[:initialise_count]
            self.prototypes[class_index, target_slots] = features[:initialise_count]
            self.counts[class_index, target_slots] = 1.0
            self.valid[class_index, target_slots] = True
            features = features[initialise_count:]
            assignments = None

        if features.shape[0] == 0:
            return

        active = torch.where(self.valid[class_index])[0]
        active_prototypes = self.prototypes[class_index, active]
        if assignments is None or assignments.shape != (
            features.shape[0], active.shape[0]
        ):
            distances = torch.cdist(features, active_prototypes, p=2)
            assignments = torch.nn.functional.one_hot(
                distances.argmin(dim=1), num_classes=active.shape[0]
            ).to(dtype=features.dtype)
        else:
            assignments = assignments.detach().to(dtype=features.dtype)

        for column, slot_index in enumerate(active):
            weights = assignments[:, column]
            mass = weights.sum()
            if mass <= 0:
                continue
            batch_mean = (weights[:, None] * features).sum(dim=0) / mass
            old = self.prototypes[class_index, slot_index]
            self.prototypes[class_index, slot_index] = (
                self.momentum * old + (1.0 - self.momentum) * batch_mean
            )
            self.counts[class_index, slot_index] += mass

    def _check_class(self, class_index: int) -> None:
        if not 0 <= class_index < self.num_classes:
            raise IndexError(f"class index {class_index} is out of range")
