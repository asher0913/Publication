import torch

from publication_cem.config import PrototypeCEMConfig
from publication_cem.regularizer import PrototypeCEMRegularizer


def make_regularizer(**overrides) -> PrototypeCEMRegularizer:
    values = {
        "feature_dim": 4,
        "num_classes": 2,
        "num_slots": 1,
        "projection_dim": 2,
        "noise_std": 0.25,
        "prototype_momentum": 0.9,
        "loss_mode": "mi_bound",
        "seed": 3,
    }
    values.update(overrides)
    return PrototypeCEMRegularizer(PrototypeCEMConfig(**values))


def test_first_occurrence_initialises_without_fake_variance() -> None:
    regularizer = make_regularizer()
    sample = torch.tensor([[1.0, 0.0, 0.0, 0.0]], requires_grad=True)
    output = regularizer(sample, torch.tensor([0]))
    assert output.loss.item() == 0.0
    assert output.diagnostics["active_classes"].item() == 0
    assert regularizer.bank.valid_count(0) == 1
    assert regularizer.bank.counts[0].sum().item() == 1.0


def test_full_space_loss_reaches_encoder_features() -> None:
    regularizer = make_regularizer()
    regularizer(torch.zeros(1, 4), torch.tensor([0]))
    sample = torch.tensor([[1.0, 0.5, -0.5, 0.25]], requires_grad=True)
    output = regularizer(sample, torch.tensor([0]))
    output.loss.backward()
    assert output.loss.item() > 0
    assert sample.grad is not None
    assert sample.grad.norm().item() > 0
    assert list(regularizer.parameters()) == []


def test_projection_nullspace_is_still_penalised() -> None:
    regularizer = make_regularizer()
    regularizer(torch.zeros(1, 4), torch.tensor([0]))
    regularizer.eval()

    _, _, vh = torch.linalg.svd(regularizer.projector.matrix.T, full_matrices=True)
    null_vector = vh[2].to(dtype=torch.float32)
    assert regularizer.projector(null_vector).norm().item() < 1e-5

    sample = (2.0 * null_vector).unsqueeze(0).requires_grad_(True)
    output = regularizer(sample, torch.tensor([0]))
    output.loss.backward()
    assert output.diagnostics["mean_full_variance"].item() > 0.5
    assert sample.grad is not None and sample.grad.norm().item() > 0


def test_threshold_mode_penalises_high_not_low_variance() -> None:
    regularizer = make_regularizer(
        loss_mode="threshold",
        noise_std=0.5,
        variance_threshold=1.0,
        threshold_offset=0.01,
    )
    regularizer(torch.zeros(1, 4), torch.tensor([0]))
    regularizer.eval()
    near = regularizer(torch.full((1, 4), 0.1), torch.tensor([0])).loss
    far = regularizer(torch.full((1, 4), 2.0), torch.tensor([0])).loss
    assert near.item() == 0.0
    assert far.item() > near.item()


def test_regularizer_state_contains_projection_and_prototypes() -> None:
    source = make_regularizer()
    source(torch.ones(1, 4), torch.tensor([1]))
    state = source.state_dict()
    assert "projector.matrix" in state
    assert "bank.prototypes" in state
    target = make_regularizer()
    target.load_state_dict(state)
    assert torch.equal(source.bank.prototypes, target.bank.prototypes)


def test_projected_variance_is_blind_to_projection_nullspace_ablation() -> None:
    regularizer = make_regularizer(variance_space="projected")
    regularizer(torch.zeros(1, 4), torch.tensor([0]))
    regularizer.eval()
    _, _, vh = torch.linalg.svd(regularizer.projector.matrix.T, full_matrices=True)
    null_vector = vh[2].to(dtype=torch.float32)
    output = regularizer((2.0 * null_vector).unsqueeze(0), torch.tensor([0]))
    assert output.diagnostics["mean_variance_in_objective_space"].item() < 1e-8
    assert output.diagnostics["mean_full_variance"].item() > 0.5


def test_learned_projector_is_an_explicit_nondefault_ablation() -> None:
    regularizer = make_regularizer(projector_mode="learned")
    assert list(regularizer.parameters()) == [regularizer.projector.matrix]


def test_channelwise_bound_reaches_noise_allocation_and_features() -> None:
    regularizer = make_regularizer(feature_dim=8, projection_dim=4)
    regularizer(torch.zeros(1, 2, 2, 2), torch.tensor([0]))
    regularizer.eval()
    sample = torch.tensor(
        [[[[1.0, 0.5], [0.2, -0.1]], [[0.1, 0.3], [0.7, 1.2]]]],
        requires_grad=True,
    )
    noise_variance = torch.tensor([0.04, 0.09], requires_grad=True)
    output = regularizer(
        sample,
        torch.tensor([0]),
        noise_variance=noise_variance,
    )
    output.loss.backward()
    assert sample.grad is not None and sample.grad.norm() > 0
    assert noise_variance.grad is not None and noise_variance.grad.norm() > 0
    assert torch.allclose(
        output.diagnostics["noise_variance_mean"], torch.tensor(0.065)
    )
