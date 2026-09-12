import torch

from publication_cem.epochwise_cem import (
    EpochwiseCEMConfig,
    EpochwiseKMeansCEMRegularizer,
)


def test_epochwise_fit_builds_multiple_class_clusters() -> None:
    regularizer = EpochwiseKMeansCEMRegularizer(
        EpochwiseCEMConfig(
            feature_dim=2,
            num_classes=2,
            num_clusters=2,
            noise_std=0.5,
            kmeans_iterations=20,
        )
    )
    features = torch.tensor(
        [[-2.0, 0.0], [-1.9, 0.1], [2.0, 0.0], [1.9, -0.1]]
    )
    regularizer.fit_class(1, features)
    assert regularizer.valid[1].sum().item() == 2
    assert torch.allclose(regularizer.weights[1].sum(), torch.tensor(1.0))
    assert regularizer.class_sizes[1].item() == 4


def test_epochwise_loss_has_encoder_gradient() -> None:
    regularizer = EpochwiseKMeansCEMRegularizer(
        EpochwiseCEMConfig(
            feature_dim=2,
            num_classes=1,
            num_clusters=1,
            noise_std=0.5,
            variance_threshold=0.1,
        )
    )
    regularizer.fit_class(0, torch.zeros(4, 2))
    current = torch.full((2, 2), 2.0, requires_grad=True)
    output = regularizer(current, torch.zeros(2, dtype=torch.long))
    output.loss.backward()
    assert output.loss.item() > 0
    assert current.grad is not None and current.grad.norm().item() > 0
