from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .regularizer import PrototypeCEMOutput


@dataclass(frozen=True)
class OfficialCEMConfig:
    feature_dim: int
    num_classes: int
    dataset_size: int
    num_clusters: int = 5
    noise_std: float = 0.025
    variance_threshold: float = 0.15
    gamma: float = 0.01
    kmeans_iterations: int = 50
    batch_variance_scale: float = 10.0
    seed: int = 125


class OfficialCEMRegularizer(nn.Module):
    """Port of the public CEM scalar KMeans training-loss path.

    Full-dataset features refresh class centroids, scalar variances and weights
    after each target epoch. During the next epoch, current batch variance is
    blended with the stored variance and passed through the public thresholded
    logarithmic objective. Full covariance matrices in the public code are
    diagnostics and are not part of its target-training gradient.
    """

    def __init__(self, config: OfficialCEMConfig) -> None:
        super().__init__()
        self.config = config
        shape = (config.num_classes, config.num_clusters)
        self.register_buffer(
            "centroids",
            torch.zeros(config.num_classes, config.num_clusters, config.feature_dim),
        )
        self.register_buffer("variances", torch.zeros(*shape))
        self.register_buffer("weights", torch.zeros(*shape))
        self.register_buffer("valid", torch.zeros(*shape, dtype=torch.bool))
        self.register_buffer("refresh_count", torch.zeros((), dtype=torch.long))

    @property
    def class_length(self) -> float:
        return self.config.dataset_size / self.config.num_classes

    @torch.no_grad()
    def fit_class(self, class_index: int, features: Tensor) -> None:
        if features.ndim != 2 or features.shape[1] != self.config.feature_dim:
            raise ValueError("features must have shape [N, feature_dim]")
        if features.shape[0] <= self.config.num_clusters:
            return
        features = features.to(self.centroids.device, dtype=self.centroids.dtype)
        if self.valid[class_index].all():
            centroids = self.centroids[class_index].clone()
        else:
            centroids = self._kmeans_plus_plus(features, class_index)

        labels = torch.zeros(features.shape[0], dtype=torch.long, device=features.device)
        for _ in range(self.config.kmeans_iterations):
            new_labels = torch.cdist(features, centroids).argmin(dim=1)
            if torch.equal(labels, new_labels):
                break
            labels = new_labels
            for cluster_index in torch.unique(labels):
                index = int(cluster_index.item())
                centroids[index] = features[labels == cluster_index].mean(dim=0)

        new_variances = torch.zeros_like(self.variances[class_index])
        new_weights = torch.zeros_like(self.weights[class_index])
        new_valid = torch.zeros_like(self.valid[class_index])
        for cluster_index in range(self.config.num_clusters):
            members = features[labels == cluster_index]
            if members.shape[0] == 0:
                continue
            new_variances[cluster_index] = (
                members - centroids[cluster_index]
            ).square().mean()
            new_weights[cluster_index] = members.shape[0] / features.shape[0]
            new_valid[cluster_index] = True

        self.centroids[class_index] = centroids
        self.variances[class_index] = new_variances
        self.weights[class_index] = new_weights
        self.valid[class_index] = new_valid

    @torch.no_grad()
    def complete_refresh(self) -> None:
        self.refresh_count += 1

    def forward(self, smashed_features: Tensor, labels: Tensor) -> PrototypeCEMOutput:
        features = smashed_features.flatten(start_dim=1)
        labels = labels.reshape(-1).to(device=features.device, dtype=torch.long)
        class_losses: List[Tensor] = []
        observed_variances: List[Tensor] = []
        threshold = (
            self.config.variance_threshold * self.config.noise_std**2
            + self.config.gamma
        )

        for class_tensor in torch.unique(labels):
            class_index = int(class_tensor.item())
            valid = self.valid[class_index]
            if not valid.any():
                continue
            class_features = features[labels == class_tensor]
            centroids = self.centroids[class_index].detach().clone()
            stored_variances = self.variances[class_index].detach().clone()
            stored_weights = self.weights[class_index].detach().clone()
            assignments = torch.cdist(class_features, centroids).argmin(dim=1).detach()
            terms = []

            for cluster_tensor in torch.unique(assignments):
                cluster_index = int(cluster_tensor.item())
                if not valid[cluster_index] or stored_weights[cluster_index] <= 0:
                    continue
                members = class_features[assignments == cluster_tensor]
                current_variance = (
                    members - centroids[cluster_index]
                ).square().mean()
                expected_count = (
                    stored_weights[cluster_index] * self.class_length
                ).clamp_min(1e-12)
                blend = min(
                    0.99,
                    members.shape[0]
                    / float(expected_count.item())
                    * self.config.batch_variance_scale,
                )
                variance = (
                    blend * current_variance
                    + (1.0 - blend) * stored_variances[cluster_index]
                )
                term = (
                    F.relu(
                        torch.log(variance + self.config.gamma)
                        - torch.log(
                            torch.as_tensor(
                                threshold,
                                device=features.device,
                                dtype=features.dtype,
                            )
                        )
                    )
                    * stored_weights[cluster_index]
                )
                terms.append(term)
                observed_variances.append(variance.detach())
            if terms:
                class_losses.append(torch.stack(terms).sum())

        loss = torch.stack(class_losses).mean() if class_losses else features.sum() * 0.0
        mean_variance = (
            torch.stack(observed_variances).mean()
            if observed_variances
            else torch.zeros((), device=features.device, dtype=features.dtype)
        )
        diagnostics: Dict[str, Tensor] = {
            "active_classes": torch.as_tensor(
                len(class_losses), device=features.device, dtype=torch.long
            ),
            "mean_full_variance": mean_variance,
            "mi_bound_nats_per_dim": torch.zeros_like(mean_variance),
            "mean_assignment_entropy": torch.zeros_like(mean_variance),
            "initialized_prototypes": self.valid.sum().detach(),
            "prototype_memory_bytes": torch.as_tensor(
                sum(value.numel() * value.element_size() for value in self.buffers()),
                device=features.device,
                dtype=torch.long,
            ),
            "statistics_refresh_count": self.refresh_count.detach(),
        }
        return PrototypeCEMOutput(loss=loss, diagnostics=diagnostics)

    def _kmeans_plus_plus(self, features: Tensor, class_index: int) -> Tensor:
        generator = torch.Generator(device=features.device)
        generator.manual_seed(
            self.config.seed + class_index + 1009 * int(self.refresh_count)
        )
        selected = [
            int(
                torch.randint(
                    features.shape[0],
                    (1,),
                    generator=generator,
                    device=features.device,
                ).item()
            )
        ]
        while len(selected) < self.config.num_clusters:
            distances = torch.cdist(features, features[selected]).min(dim=1).values
            total = distances.sum()
            if total <= 0:
                remaining = [
                    index
                    for index in range(features.shape[0])
                    if index not in selected
                ]
                selected.append(remaining[0])
                continue
            probabilities = distances / total
            selected.append(
                int(
                    torch.multinomial(
                        probabilities,
                        1,
                        generator=generator,
                    ).item()
                )
            )
        return features[selected].clone()
