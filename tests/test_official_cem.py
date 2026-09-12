import torch

from publication_cem.official_cem import OfficialCEMConfig, OfficialCEMRegularizer


def test_official_cem_is_inactive_until_first_full_statistics_refresh():
    regularizer = OfficialCEMRegularizer(
        OfficialCEMConfig(
            feature_dim=2,
            num_classes=1,
            dataset_size=6,
            num_clusters=2,
            kmeans_iterations=5,
        )
    )
    features = torch.tensor([[0.0, 0.0], [0.1, 0.1]], requires_grad=True)
    labels = torch.zeros(2, dtype=torch.long)
    assert regularizer(features, labels).loss.item() == 0.0

    statistics = torch.tensor(
        [[0.0, 0.0], [0.1, 0.1], [0.2, 0.2], [2.0, 2.0], [2.1, 2.1], [2.2, 2.2]]
    )
    regularizer.fit_class(0, statistics)
    regularizer.complete_refresh()
    output = regularizer(features, labels)
    assert torch.isfinite(output.loss)
    assert int(output.diagnostics["statistics_refresh_count"]) == 1


def test_official_cem_loss_backpropagates_to_full_features():
    regularizer = OfficialCEMRegularizer(
        OfficialCEMConfig(
            feature_dim=2,
            num_classes=1,
            dataset_size=10,
            num_clusters=1,
            noise_std=0.025,
        )
    )
    regularizer.centroids[0, 0] = torch.tensor([0.0, 0.0])
    regularizer.variances[0, 0] = 0.1
    regularizer.weights[0, 0] = 1.0
    regularizer.valid[0, 0] = True
    features = torch.tensor([[1.0, 1.0], [2.0, 2.0]], requires_grad=True)
    loss = regularizer(features, torch.zeros(2, dtype=torch.long)).loss
    loss.backward()
    assert features.grad is not None
    assert features.grad.abs().sum() > 0
