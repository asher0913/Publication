from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import PrototypeCEMConfig
from .projection import FixedOrthogonalProjector, LearnedProjector
from .prototype_bank import ClassPrototypeBank


@dataclass
class PrototypeCEMOutput:
    loss: Tensor
    diagnostics: Dict[str, Tensor]


class PrototypeCEMRegularizer(nn.Module):
    """Online class-conditional prototype CEM estimator.

    Soft attention assignments are computed in a fixed low-dimensional space.
    All variances used by the loss are measured against detached prototypes in
    the complete smashed-feature space seen by the attacker.
    """

    def __init__(self, config: PrototypeCEMConfig) -> None:
        super().__init__()
        self.config = config
        projector_type = (
            FixedOrthogonalProjector
            if config.projector_mode == "fixed"
            else LearnedProjector
        )
        self.projector = projector_type(
            config.feature_dim, config.projection_dim, config.seed
        )
        self.bank = ClassPrototypeBank(
            config.num_classes,
            config.num_slots,
            config.feature_dim,
            config.prototype_momentum,
        )

    def forward(
        self,
        smashed_features: Tensor,
        labels: Tensor,
        noise_variance: Tensor | None = None,
    ) -> PrototypeCEMOutput:
        features = smashed_features.flatten(start_dim=1)
        if features.shape[1] != self.config.feature_dim:
            raise ValueError(
                f"expected flattened dimension {self.config.feature_dim}, "
                f"got {features.shape[1]}"
            )
        labels = labels.reshape(-1).to(device=features.device, dtype=torch.long)
        if labels.shape[0] != features.shape[0]:
            raise ValueError("labels and features must have the same batch size")
        if labels.numel() and (labels.min() < 0 or labels.max() >= self.config.num_classes):
            raise ValueError("labels contain a class outside the configured range")
        if noise_variance is not None:
            if smashed_features.ndim < 3:
                raise ValueError("channel-wise noise requires spatial smashed features")
            noise_variance = noise_variance.reshape(-1).to(
                device=features.device, dtype=features.dtype
            )
            if noise_variance.numel() != smashed_features.shape[1]:
                raise ValueError("noise variance does not match smashed-feature channels")
            if torch.any(noise_variance <= 0):
                raise ValueError("noise variance must be strictly positive")

        class_losses: List[Tensor] = []
        class_variances: List[Tensor] = []
        class_full_variances: List[Tensor] = []
        class_bounds: List[Tensor] = []
        assignment_entropies: List[Tensor] = []
        tracking_errors: List[Tensor] = []
        occupied_slot_counts: List[Tensor] = []
        update_records = []

        for class_tensor in torch.unique(labels):
            class_index = int(class_tensor.item())
            class_features = features[labels == class_tensor]
            prototypes = self.bank.get(class_index)

            if prototypes.shape[0] >= self.config.min_prototypes_for_loss:
                # Clone because the bank is updated before backward; the loss
                # graph must retain the pre-update reference values.
                prototypes = prototypes.detach().clone()
                assignments = self._assign(class_features, prototypes)
                loss_assignments = (
                    assignments.detach()
                    if self.config.detach_assignments
                    else assignments
                )

                variance_features = (
                    class_features
                    if self.config.variance_space == "full"
                    else self.projector(class_features)
                )
                variance_prototypes = (
                    prototypes
                    if self.config.variance_space == "full"
                    else self.projector(prototypes)
                )
                squared_distances = (
                    variance_features[:, None, :] - variance_prototypes[None, :, :]
                ).square().mean(dim=-1)
                full_squared_distances = (
                    class_features[:, None, :] - prototypes[None, :, :]
                ).square().mean(dim=-1)
                mass = loss_assignments.sum(dim=0)
                occupied = mass > self.config.numerical_epsilon
                slot_variances = (
                    (loss_assignments * squared_distances).sum(dim=0)
                    / mass.clamp_min(self.config.numerical_epsilon)
                )
                slot_weights = mass / mass.sum().clamp_min(
                    self.config.numerical_epsilon
                )
                full_slot_variances = (
                    (loss_assignments * full_squared_distances).sum(dim=0)
                    / mass.clamp_min(self.config.numerical_epsilon)
                )
                slot_means = (
                    loss_assignments.transpose(0, 1) @ variance_features
                ) / mass[:, None].clamp_min(self.config.numerical_epsilon)
                tracking_error = (
                    slot_weights[occupied]
                    * (slot_means[occupied] - variance_prototypes[occupied])
                    .square()
                    .mean(dim=1)
                ).sum()

                channel_slot_variances = None
                if noise_variance is not None:
                    residuals = (
                        class_features[:, None, :] - prototypes[None, :, :]
                    ).reshape(
                        class_features.shape[0],
                        prototypes.shape[0],
                        smashed_features.shape[1],
                        -1,
                    )
                    per_example_channel_variance = residuals.square().mean(dim=-1)
                    channel_slot_variances = (
                        (
                            loss_assignments[:, :, None]
                            * per_example_channel_variance
                        ).sum(dim=0)
                        / mass[:, None].clamp_min(self.config.numerical_epsilon)
                    )

                if self.config.loss_mode == "mi_bound" and noise_variance is not None:
                    per_slot_loss = 0.5 * torch.log1p(
                        channel_slot_variances / noise_variance[None, :]
                    ).mean(dim=1)
                elif self.config.loss_mode == "mi_bound":
                    per_slot_loss = 0.5 * torch.log1p(
                        slot_variances / (self.config.noise_std**2)
                    )
                else:
                    threshold = (
                        self.config.variance_threshold
                        * self.config.noise_std**2
                        + self.config.threshold_offset
                    )
                    per_slot_loss = F.relu(
                        torch.log(slot_variances + self.config.threshold_offset)
                        - torch.log(
                            torch.as_tensor(
                                threshold,
                                device=features.device,
                                dtype=features.dtype,
                            )
                        )
                    )

                class_loss = (slot_weights[occupied] * per_slot_loss[occupied]).sum()
                weighted_variance = (
                    slot_weights[occupied] * slot_variances[occupied]
                ).sum()
                weighted_full_variance = (
                    slot_weights[occupied] * full_slot_variances[occupied]
                ).sum()
                if noise_variance is not None:
                    normalized_bound = (
                        slot_weights[occupied] * per_slot_loss[occupied]
                    ).sum()
                else:
                    normalized_bound = (
                        slot_weights[occupied]
                        * 0.5
                        * torch.log1p(
                            slot_variances[occupied] / (self.config.noise_std**2)
                        )
                    ).sum()
                entropy = -(
                    assignments
                    * torch.log(assignments.clamp_min(self.config.numerical_epsilon))
                ).sum(dim=1).mean()

                class_losses.append(class_loss)
                class_variances.append(weighted_variance.detach())
                class_full_variances.append(weighted_full_variance.detach())
                class_bounds.append(normalized_bound.detach())
                assignment_entropies.append(entropy.detach())
                tracking_errors.append(tracking_error.detach())
                occupied_slot_counts.append(
                    occupied.sum().detach().to(dtype=features.dtype)
                )
                update_records.append((class_index, class_features, assignments))
            else:
                update_records.append((class_index, class_features, None))

        if class_losses:
            loss = torch.stack(class_losses).mean()
        else:
            loss = features.sum() * 0.0

        if self.training and self.config.update_prototypes:
            with torch.no_grad():
                for class_index, class_features, assignments in update_records:
                    self.bank.update_class(class_index, class_features, assignments)

        diagnostics = {
            "active_classes": torch.as_tensor(
                len(class_losses), device=features.device, dtype=torch.long
            ),
            "mean_full_variance": self._mean_or_zero(
                class_full_variances, features
            ),
            "mean_variance_in_objective_space": self._mean_or_zero(
                class_variances, features
            ),
            "mi_bound_nats_per_dim": self._mean_or_zero(class_bounds, features),
            "mean_assignment_entropy": self._mean_or_zero(
                assignment_entropies, features
            ),
            "mean_prototype_tracking_mse": self._mean_or_zero(
                tracking_errors, features
            ),
            "mean_occupied_slots_per_active_class": self._mean_or_zero(
                occupied_slot_counts, features
            ),
            "initialized_prototypes": self.bank.valid.sum().detach().to(features.device),
            "prototype_memory_bytes": torch.as_tensor(
                self.bank.memory_bytes, device=features.device, dtype=torch.long
            ),
        }
        if noise_variance is not None:
            diagnostics.update(
                {
                    "noise_variance_min": noise_variance.min().detach(),
                    "noise_variance_max": noise_variance.max().detach(),
                    "noise_variance_mean": noise_variance.mean().detach(),
                }
            )
        return PrototypeCEMOutput(loss=loss, diagnostics=diagnostics)

    def _assign(self, features: Tensor, prototypes: Tensor) -> Tensor:
        if prototypes.shape[0] == 1:
            return torch.ones(
                features.shape[0], 1, device=features.device, dtype=features.dtype
            )
        projected_features = F.normalize(
            self.projector(features), dim=-1, eps=self.config.numerical_epsilon
        )
        projected_prototypes = F.normalize(
            self.projector(prototypes), dim=-1, eps=self.config.numerical_epsilon
        )
        logits = projected_features @ projected_prototypes.transpose(0, 1)
        return torch.softmax(logits / self.config.temperature, dim=-1)

    @staticmethod
    def _mean_or_zero(values: List[Tensor], reference: Tensor) -> Tensor:
        if not values:
            return torch.zeros((), device=reference.device, dtype=reference.dtype)
        return torch.stack(values).mean()
