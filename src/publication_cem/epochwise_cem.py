from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .regularizer import PrototypeCEMOutput


@dataclass(frozen=True)
class EpochwiseCEMConfig:
    feature_dim: int
    num_classes: int
    num_clusters: int = 5
    noise_std: float = 0.025
    variance_threshold: float = 0.15
    threshold_offset: float = 0.01
    kmeans_iterations: int = 50
    kmeans_tolerance: float = 1e-4
    batch_variance_scale: float = 10.0


class EpochwiseKMeansCEMRegularizer(nn.Module):
    """Matched baseline for the public CEM scalar KMeans training path.

    This module deliberately performs an epoch-wise full-data statistic refresh.
    It does not claim to reproduce a full-covariance GMM determinant; it mirrors
    the scalar cluster-variance path exercised by the public training code.
    """

    def __init__(self, config: EpochwiseCEMConfig) -> None:
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
        self.register_buffer("class_sizes", torch.zeros(config.num_classes))

    @torch.no_grad()
    def fit_class(self, class_index: int, features: Tensor) -> None:
        if features.ndim != 2 or features.shape[1] != self.config.feature_dim:
            raise ValueError("features must have shape [N, feature_dim]")
        if features.shape[0] == 0:
            return
        device = self.centroids.device
        features = features.to(device=device, dtype=self.centroids.dtype)
        cluster_count = min(self.config.num_clusters, features.shape[0])
        centroids = self._farthest_first(features, cluster_count)

        for _ in range(self.config.kmeans_iterations):
            distances = torch.cdist(features, centroids)
            assignments = distances.argmin(dim=1)
            updated = []
            for cluster_index in range(cluster_count):
                members = features[assignments == cluster_index]
                updated.append(
                    members.mean(dim=0) if members.shape[0] else centroids[cluster_index]
                )
            updated_centroids = torch.stack(updated)
            shift = (updated_centroids - centroids).norm(dim=1).max()
            centroids = updated_centroids
            if shift <= self.config.kmeans_tolerance:
                break

        distances = torch.cdist(features, centroids)
        assignments = distances.argmin(dim=1)
        self.valid[class_index].zero_()
        self.weights[class_index].zero_()
        self.variances[class_index].zero_()
        self.centroids[class_index, :cluster_count] = centroids
        self.valid[class_index, :cluster_count] = True
        self.class_sizes[class_index] = features.shape[0]
        for cluster_index in range(cluster_count):
            members = features[assignments == cluster_index]
            if members.shape[0] == 0:
                continue
            residual = (members - centroids[cluster_index]).square().mean()
            self.variances[class_index, cluster_index] = residual
            self.weights[class_index, cluster_index] = (
                members.shape[0] / features.shape[0]
            )

    def forward(self, smashed_features: Tensor, labels: Tensor) -> PrototypeCEMOutput:
        features = smashed_features.flatten(start_dim=1)
        labels = labels.reshape(-1).to(device=features.device, dtype=torch.long)
        class_losses: List[Tensor] = []
        observed_variances: List[Tensor] = []

        for class_tensor in torch.unique(labels):
            class_index = int(class_tensor.item())
            valid = self.valid[class_index]
            if not valid.any():
                continue
            class_features = features[labels == class_tensor]
            centroids = self.centroids[class_index, valid].detach().clone()
            stored_variance = self.variances[class_index, valid].detach().clone()
            stored_weight = self.weights[class_index, valid].detach().clone()
            distances = torch.cdist(class_features, centroids)
            assignments = distances.argmin(dim=1).detach()
            terms = []
            term_weights = []

            for cluster_index in torch.unique(assignments):
                index = int(cluster_index.item())
                members = class_features[assignments == cluster_index]
                current_variance = (
                    members - centroids[index]
                ).square().mean()
                expected_count = (
                    stored_weight[index] * self.class_sizes[class_index]
                ).clamp_min(1.0)
                blend = min(
                    0.99,
                    members.shape[0] / float(expected_count.item())
                    * self.config.batch_variance_scale,
                )
                variance = (
                    blend * current_variance
                    + (1.0 - blend) * stored_variance[index]
                )
                threshold = (
                    self.config.variance_threshold * self.config.noise_std**2
                    + self.config.threshold_offset
                )
                term = F.relu(
                    torch.log(variance + self.config.threshold_offset)
                    - torch.log(
                        torch.as_tensor(
                            threshold, device=features.device, dtype=features.dtype
                        )
                    )
                )
                terms.append(term)
                term_weights.append(stored_weight[index])
                observed_variances.append(variance.detach())

            weights = torch.stack(term_weights)
            weights = weights / weights.sum().clamp_min(1e-8)
            class_losses.append((weights * torch.stack(terms)).sum())

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
                sum(x.numel() * x.element_size() for x in self.buffers()),
                device=features.device,
                dtype=torch.long,
            ),
        }
        return PrototypeCEMOutput(loss=loss, diagnostics=diagnostics)

    @staticmethod
    def _farthest_first(features: Tensor, cluster_count: int) -> Tensor:
        selected = [0]
        while len(selected) < cluster_count:
            candidates = torch.cdist(features, features[selected]).min(dim=1).values
            candidates[selected] = -1
            selected.append(int(candidates.argmax().item()))
        return features[selected].clone()
