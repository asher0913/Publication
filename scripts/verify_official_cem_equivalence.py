#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from publication_cem.official_cem import OfficialCEMConfig, OfficialCEMRegularizer


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    centroids: torch.Tensor,
    weights: torch.Tensor,
    variances: torch.Tensor,
    dataset_size: int,
    variance_threshold: float,
    noise_std: float,
    gamma: float,
    scale: float,
) -> torch.Tensor:
    class_losses = []
    class_length = dataset_size / centroids.shape[0]
    for class_tensor in torch.unique(labels):
        class_index = int(class_tensor.item())
        class_features = features[labels == class_tensor]
        assignments = torch.cdist(class_features, centroids[class_index]).argmin(
            dim=1
        )
        terms = []
        for cluster_tensor in torch.unique(assignments):
            cluster_index = int(cluster_tensor.item())
            members = class_features[assignments == cluster_tensor]
            current = (members - centroids[class_index, cluster_index]).square().mean()
            expected = weights[class_index, cluster_index] * class_length
            blend = min(0.99, members.shape[0] / float(expected) * scale)
            variance = (
                blend * current
                + (1.0 - blend) * variances[class_index, cluster_index]
            )
            threshold = variance_threshold * noise_std**2 + gamma
            terms.append(
                F.relu(torch.log(variance + gamma) - torch.log(torch.tensor(threshold)))
                * weights[class_index, cluster_index]
            )
        class_losses.append(torch.stack(terms).sum())
    return torch.stack(class_losses).mean()


def reference_kmeans(
    features: torch.Tensor,
    initial_centroids: torch.Tensor,
    iterations: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    centroids = initial_centroids.clone()
    labels = torch.zeros(features.shape[0], dtype=torch.long)
    for _ in range(iterations):
        new_labels = torch.cdist(features, centroids).argmin(dim=1)
        if torch.equal(labels, new_labels):
            break
        labels = new_labels
        for cluster_tensor in torch.unique(labels):
            cluster_index = int(cluster_tensor.item())
            centroids[cluster_index] = features[labels == cluster_tensor].mean(dim=0)
    variances = torch.stack(
        [
            (features[labels == index] - centroids[index]).square().mean()
            for index in range(centroids.shape[0])
        ]
    )
    weights = torch.stack(
        [
            (labels == index).sum().to(dtype=features.dtype) / features.shape[0]
            for index in range(centroids.shape[0])
        ]
    )
    return centroids, variances, weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "evidence" / "official_cem_equivalence.json",
    )
    args = parser.parse_args()
    torch.manual_seed(125)
    config = OfficialCEMConfig(
        feature_dim=4,
        num_classes=2,
        dataset_size=60,
        num_clusters=3,
        noise_std=0.025,
        variance_threshold=0.15,
        gamma=0.01,
        kmeans_iterations=8,
        batch_variance_scale=10.0,
        seed=125,
    )
    regularizer = OfficialCEMRegularizer(config)
    centroids = torch.tensor(
        [
            [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]],
            [[3.0, 3.0, 3.0, 3.0], [4.0, 4.0, 4.0, 4.0], [5.0, 5.0, 5.0, 5.0]],
        ]
    )
    weights = torch.tensor([[0.3, 0.4, 0.3], [0.25, 0.5, 0.25]])
    variances = torch.tensor([[0.04, 0.05, 0.06], [0.03, 0.05, 0.07]])
    regularizer.centroids.copy_(centroids)
    regularizer.weights.copy_(weights)
    regularizer.variances.copy_(variances)
    regularizer.valid.fill_(True)

    feature_values = torch.tensor(
        [
            [0.1, 0.0, 0.0, 0.0],
            [1.1, 1.0, 1.0, 1.0],
            [2.1, 2.0, 2.0, 2.0],
            [3.1, 3.0, 3.0, 3.0],
            [4.1, 4.0, 4.0, 4.0],
            [5.1, 5.0, 5.0, 5.0],
        ],
        requires_grad=True,
    )
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    ported_loss = regularizer(feature_values, labels).loss
    ported_gradient = torch.autograd.grad(ported_loss, feature_values)[0]

    reference_features = feature_values.detach().clone().requires_grad_(True)
    expected_loss = reference_loss(
        reference_features,
        labels,
        centroids,
        weights,
        variances,
        config.dataset_size,
        config.variance_threshold,
        config.noise_std,
        config.gamma,
        config.batch_variance_scale,
    )
    expected_gradient = torch.autograd.grad(expected_loss, reference_features)[0]

    kmeans_features = torch.tensor(
        [
            [0.0, 0.0],
            [0.2, 0.1],
            [2.0, 2.0],
            [2.2, 2.1],
            [4.0, 4.0],
            [4.2, 4.1],
        ]
    )
    initial = torch.tensor([[0.0, 0.0], [2.0, 2.0], [4.0, 4.0]])
    kmeans_config = OfficialCEMConfig(
        feature_dim=2,
        num_classes=1,
        dataset_size=6,
        num_clusters=3,
        kmeans_iterations=8,
    )
    kmeans_port = OfficialCEMRegularizer(kmeans_config)
    kmeans_port.centroids[0] = initial
    kmeans_port.valid[0] = True
    kmeans_port.fit_class(0, kmeans_features)
    expected_centroids, expected_variances, expected_weights = reference_kmeans(
        kmeans_features, initial, kmeans_config.kmeans_iterations
    )

    errors = {
        "loss_abs": abs(float(ported_loss - expected_loss)),
        "gradient_max_abs": float((ported_gradient - expected_gradient).abs().max()),
        "centroid_max_abs": float(
            (kmeans_port.centroids[0] - expected_centroids).abs().max()
        ),
        "variance_max_abs": float(
            (kmeans_port.variances[0] - expected_variances).abs().max()
        ),
        "weight_max_abs": float(
            (kmeans_port.weights[0] - expected_weights).abs().max()
        ),
    }
    tolerance = 1e-7
    status = "verified" if max(errors.values()) <= tolerance else "failed"
    upstream_path = ROOT / "archive" / "upstream_cem" / "model_training_paral_pruning.py"
    payload = {
        "schema_version": 1,
        "status": status,
        "scope": "Public scalar KMeans statistics and target-training CEM loss path",
        "upstream_file": str(upstream_path.relative_to(ROOT)),
        "upstream_sha256": sha256(upstream_path),
        "upstream_regions": {
            "kmeans": "lines 697-790 in archived source",
            "training_loss": "lines 890-1018 in archived source",
            "refresh_schedule": "lines 1657-1880 in archived source",
        },
        "tolerance": tolerance,
        "errors": errors,
        "limitations": [
            "Equivalence covers the scalar KMeans path used for target gradients.",
            "Full covariance log-determinants in upstream are diagnostics, not gradient terms.",
            "The clean runner intentionally supplies a controlled model/data/optimizer wrapper.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"status": status, "errors": errors}, sort_keys=True))
    if status != "verified":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
