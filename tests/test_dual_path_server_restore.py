import torch
from torch import nn

from scripts.train_dual_path_facescrub import restore_legacy_server


class LegacyStub(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cloud = nn.Linear(2, 2)
        self.classifier = nn.Linear(2, 3)


def test_restores_cloud_and_classifier_without_touching_other_modules() -> None:
    source = LegacyStub()
    target = LegacyStub()
    checkpoint = {
        "legacy_cloud_state": source.cloud.state_dict(),
        "legacy_classifier_state": source.classifier.state_dict(),
    }
    restore_legacy_server(target, checkpoint)
    assert torch.equal(target.cloud.weight, source.cloud.weight)
    assert torch.equal(target.classifier.weight, source.classifier.weight)


def test_rejects_partial_server_state() -> None:
    target = LegacyStub()
    checkpoint = {"legacy_cloud_state": target.cloud.state_dict()}
    try:
        restore_legacy_server(target, checkpoint)
    except ValueError as error:
        assert "incomplete" in str(error)
    else:
        raise AssertionError("partial server checkpoint was accepted")
