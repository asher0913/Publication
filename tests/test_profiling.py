import torch
from torch import nn

from scripts.profile_checkpoint import parameter_count, state_bytes


def test_parameter_and_state_accounting_are_exact():
    model = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4))
    assert parameter_count(model) == 24
    expected_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in model.state_dict().values()
    )
    assert state_bytes(model) == expected_bytes
    assert state_bytes(model) > parameter_count(model) * torch.tensor(0.0).element_size()
