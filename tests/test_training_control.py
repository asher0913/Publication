import pytest
import torch

from publication_cem.training_control import (
    scheduled_privacy_weight,
    task_priority_gradient,
)


def test_scheduled_privacy_weight_warms_up_and_ramps_linearly() -> None:
    assert scheduled_privacy_weight(0.04, 20, warmup_epochs=20, ramp_epochs=40) == 0.0
    assert scheduled_privacy_weight(0.04, 30, warmup_epochs=20, ramp_epochs=40) == pytest.approx(0.01)
    assert scheduled_privacy_weight(0.04, 60, warmup_epochs=20, ramp_epochs=40) == pytest.approx(0.04)
    assert scheduled_privacy_weight(0.04, 90, warmup_epochs=20, ramp_epochs=40) == pytest.approx(0.04)


def test_scheduled_privacy_weight_preserves_legacy_constant_weight() -> None:
    assert scheduled_privacy_weight(0.03, 1) == pytest.approx(0.03)
    assert scheduled_privacy_weight(0.03, 300) == pytest.approx(0.03)


def test_task_priority_gradient_preserves_non_conflicting_privacy_signal() -> None:
    task = torch.tensor([[1.0, 0.0]])
    privacy = torch.tensor([[0.0, 2.0]])
    result = task_priority_gradient(task, privacy, privacy_weight=0.5)
    assert torch.allclose(result.gradient, torch.tensor([[1.0, 1.0]]))
    assert result.conflict_fraction.item() == 0.0


def test_task_priority_gradient_removes_conflicting_component() -> None:
    task = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    privacy = torch.tensor([[-2.0, 3.0], [4.0, -5.0]])
    result = task_priority_gradient(task, privacy, privacy_weight=0.5)
    expected = torch.tensor([[1.0, 1.5], [2.0, 1.0]])
    assert torch.allclose(result.gradient, expected)
    assert result.conflict_fraction.item() == 1.0


@pytest.mark.parametrize("weight", [-0.1])
def test_training_control_rejects_invalid_weights(weight: float) -> None:
    with pytest.raises(ValueError):
        scheduled_privacy_weight(weight, 1)
    with pytest.raises(ValueError):
        task_priority_gradient(torch.ones(1, 2), torch.ones(1, 2), weight)
