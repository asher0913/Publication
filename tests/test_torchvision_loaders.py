import json

import pytest
import torch
from torch.utils.data import Dataset
from torchvision import datasets

from publication_cem.data import build_torchvision_loaders
from scripts.build_torchvision_protocol import build_index_manifest


class FakeCIFAR100(Dataset):
    classes = ["a", "b"]

    def __init__(self, root, train, download, transform):
        del root, download
        self.transform = transform
        self.targets = ([0] * 10 + [1] * 10) if train else ([0] * 3 + [1] * 3)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return torch.rand(3, 8, 8), self.targets[index]


def test_torchvision_protocol_loader_uses_disjoint_index_splits(monkeypatch, tmp_path):
    monkeypatch.setattr(datasets, "CIFAR100", FakeCIFAR100)
    manifest = build_index_manifest(
        "cifar100",
        [0] * 10 + [1] * 10,
        [0] * 3 + [1] * 3,
        ["a", "b"],
        seed=125,
        validation_fraction=0.2,
        auxiliary_fraction=0.2,
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    loaders = build_torchvision_loaders(
        "cifar100",
        str(tmp_path),
        image_size=8,
        batch_size=2,
        workers=0,
        seed=125,
        split_manifest=str(path),
    )
    assert len(loaders.train.dataset) == 12
    assert len(loaders.validation.dataset) == 4
    assert len(loaders.attacker_auxiliary_train.dataset) == 2
    assert len(loaders.attacker_auxiliary_validation.dataset) == 2
    assert len(loaders.test.dataset) == 6


def test_torchvision_protocol_loader_rejects_a_tampered_index_hash(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(datasets, "CIFAR100", FakeCIFAR100)
    manifest = build_index_manifest(
        "cifar100",
        [0] * 10 + [1] * 10,
        [0] * 3 + [1] * 3,
        ["a", "b"],
        seed=125,
        validation_fraction=0.2,
        auxiliary_fraction=0.2,
    )
    manifest["splits"]["target_train"]["sha256"] = "0" * 64
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="index hash"):
        build_torchvision_loaders(
            "cifar100",
            str(tmp_path),
            image_size=8,
            batch_size=2,
            workers=0,
            seed=125,
            split_manifest=str(path),
        )
