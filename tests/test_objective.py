import torch

from publication_cem.config import PrototypeCEMConfig
from publication_cem.objective import PrivacyUtilityObjective
from publication_cem.regularizer import PrototypeCEMRegularizer


def test_objective_uses_explicit_privacy_weight() -> None:
    regularizer = PrototypeCEMRegularizer(
        PrototypeCEMConfig(
            feature_dim=2,
            num_classes=1,
            num_slots=1,
            projection_dim=1,
            noise_std=0.5,
        )
    )
    regularizer(torch.zeros(1, 2), torch.tensor([0]))
    task_loss = torch.tensor(3.0, requires_grad=True)
    features = torch.ones(1, 2, requires_grad=True)
    objective = PrivacyUtilityObjective(regularizer, privacy_weight=2.5)
    output = objective(task_loss, features, torch.tensor([0]))
    assert torch.allclose(
        output.total_loss, task_loss + 2.5 * output.privacy_loss
    )
    output.total_loss.backward()
    assert task_loss.grad is not None
    assert features.grad is not None and features.grad.norm().item() > 0
