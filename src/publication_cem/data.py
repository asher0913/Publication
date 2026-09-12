from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from torch.utils.data import DataLoader, Dataset, Subset


@dataclass
class ImageFolderLoaders:
    train: DataLoader
    statistics: DataLoader
    validation: DataLoader
    test: DataLoader
    attacker_auxiliary_train: DataLoader
    attacker_auxiliary_validation: DataLoader
    class_to_index: Dict[str, int]
    split_manifest: Optional[dict] = None
    num_classes_override: Optional[int] = None
    protocol_mode: str = "manifest_or_legacy"

    @property
    def num_classes(self) -> int:
        return self.num_classes_override or len(self.class_to_index)


def _indices_for_paths(dataset: Dataset, data_root: Path, paths: list[str]) -> list[int]:
    samples = getattr(dataset, "samples", None)
    if samples is None:
        raise TypeError("manifest-backed loading requires an ImageFolder dataset")
    index_by_relative_path = {
        Path(path).resolve().relative_to(data_root.resolve()).as_posix(): index
        for index, (path, _) in enumerate(samples)
    }
    missing = [path for path in paths if path not in index_by_relative_path]
    if missing:
        raise FileNotFoundError(f"manifest paths are missing from ImageFolder: {missing[:5]}")
    return [index_by_relative_path[path] for path in paths]


def _loader(dataset, shuffle: bool, common: dict, generator=None) -> DataLoader:
    return DataLoader(
        dataset,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        drop_last=False,
        **common,
    )


def _path_list_hash(paths: list[str]) -> str:
    return hashlib.sha256("\n".join(paths).encode("utf-8")).hexdigest()


def _verify_path_splits(manifest: dict, split_names: set[str]) -> None:
    split_sets = []
    for split_name in sorted(split_names):
        record = manifest["splits"][split_name]
        paths = list(record.get("paths", []))
        if len(paths) != int(record.get("count", -1)):
            raise ValueError(f"{split_name} count differs from its path list")
        if paths != sorted(paths):
            raise ValueError(f"{split_name} paths are not in frozen order")
        if len(paths) != len(set(paths)):
            raise ValueError(f"{split_name} contains duplicate paths")
        if record.get("sha256") != _path_list_hash(paths):
            raise ValueError(f"{split_name} path hash differs from its manifest")
        split_sets.append((split_name, set(paths)))
    for position, (left_name, left) in enumerate(split_sets):
        for right_name, right in split_sets[position + 1 :]:
            if left & right:
                raise ValueError(f"protocol splits overlap: {left_name}/{right_name}")


def build_imagefolder_loaders(
    data_root: str,
    image_size: int,
    batch_size: int,
    workers: int,
    seed: int,
    split_manifest: str | None = None,
    train_augmentation: bool = False,
    protocol_mode: str = "manifest_or_legacy",
    num_classes_override: int | None = None,
) -> ImageFolderLoaders:
    try:
        import torch
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise RuntimeError("torchvision is required for ImageFolder datasets") from exc

    root = Path(data_root)
    train_root = root / "train"
    test_root = root / "val"
    if not train_root.is_dir() or not test_root.is_dir():
        raise FileNotFoundError("data_root must contain train/ and val/")

    evaluation_transform = transforms.Compose(
        [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
    )
    if train_augmentation:
        training_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.82, 1.0),
                    ratio=(0.9, 1.1),
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.12, contrast=0.12),
                transforms.ToTensor(),
            ]
        )
    else:
        training_transform = evaluation_transform
    train_source = datasets.ImageFolder(train_root, transform=training_transform)
    statistics_source = datasets.ImageFolder(train_root, transform=evaluation_transform)
    attacker_source = datasets.ImageFolder(train_root, transform=evaluation_transform)
    test_source = datasets.ImageFolder(test_root, transform=evaluation_transform)
    if train_source.class_to_idx != test_source.class_to_idx:
        raise ValueError("train and val class mappings do not match")
    if num_classes_override is not None and num_classes_override < len(
        train_source.class_to_idx
    ):
        raise ValueError("num_classes_override cannot be smaller than observed classes")

    manifest = None
    if split_manifest:
        manifest = json.loads(Path(split_manifest).read_text(encoding="utf-8"))
        if manifest.get("class_to_index") != train_source.class_to_idx:
            raise ValueError("split manifest class mapping differs from ImageFolder")
        required = {
            "target_train",
            "target_validation",
            "target_test",
            "attacker_auxiliary_train",
            "attacker_auxiliary_validation",
        }
        if not required.issubset(manifest.get("splits", {})):
            raise ValueError("split manifest is missing a required protocol split")
        _verify_path_splits(manifest, required)
        target_train = Subset(
            train_source,
            _indices_for_paths(
                train_source,
                root,
                manifest["splits"]["target_train"]["paths"],
            ),
        )
        target_validation = Subset(
            statistics_source,
            _indices_for_paths(
                train_source,
                root,
                manifest["splits"]["target_validation"]["paths"],
            ),
        )
        attacker_auxiliary_train = Subset(
            attacker_source,
            _indices_for_paths(
                train_source,
                root,
                manifest["splits"]["attacker_auxiliary_train"]["paths"],
            ),
        )
        attacker_auxiliary_validation = Subset(
            attacker_source,
            _indices_for_paths(
                train_source,
                root,
                manifest["splits"]["attacker_auxiliary_validation"]["paths"],
            ),
        )
        target_test = Subset(
            test_source,
            _indices_for_paths(
                test_source,
                root,
                manifest["splits"]["target_test"]["paths"],
            ),
        )
    elif protocol_mode == "cem_facescrub_paper":
        # The public CEM implementation trains the inversion model on the first
        # 90% of FaceScrub validation and reports inference on the final 10%.
        split_index = len(test_source) - len(test_source) // 10
        target_train = train_source
        target_validation = test_source
        target_test = Subset(test_source, list(range(split_index, len(test_source))))
        attacker_auxiliary_train = Subset(
            test_source, list(range(0, split_index))
        )
        attacker_auxiliary_validation = Subset(
            test_source, list(range(split_index, len(test_source)))
        )
    elif protocol_mode == "manifest_or_legacy":
        # Backward-compatible path for unit and synthetic smoke tests. Final
        # publication configs are required to provide a split manifest.
        target_train = train_source
        target_validation = test_source
        target_test = test_source
        attacker_auxiliary_train = train_source
        attacker_auxiliary_validation = test_source
    else:
        raise ValueError(f"unsupported ImageFolder protocol_mode: {protocol_mode}")

    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    attacker_generator = torch.Generator()
    attacker_generator.manual_seed(seed + 1)
    common = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": workers > 0,
    }
    return ImageFolderLoaders(
        train=_loader(target_train, True, common, train_generator),
        statistics=_loader(
            Subset(statistics_source, target_train.indices)
            if isinstance(target_train, Subset)
            else statistics_source,
            False,
            common,
        ),
        validation=_loader(target_validation, False, common),
        test=_loader(target_test, False, common),
        attacker_auxiliary_train=_loader(
            attacker_auxiliary_train, True, common, attacker_generator
        ),
        attacker_auxiliary_validation=_loader(
            attacker_auxiliary_validation, False, common
        ),
        class_to_index=train_source.class_to_idx,
        split_manifest=manifest,
        num_classes_override=num_classes_override,
        protocol_mode=protocol_mode,
    )


def _indices_from_manifest(
    manifest: dict,
    split_name: str,
    dataset_size: int,
) -> list[int]:
    record = manifest["splits"][split_name]
    indices = [int(index) for index in record.get("indices", [])]
    if len(indices) != int(record.get("count", -1)):
        raise ValueError(f"{split_name} count differs from its index list")
    if len(indices) != len(set(indices)):
        raise ValueError(f"{split_name} contains duplicate indices")
    if any(index < 0 or index >= dataset_size for index in indices):
        raise IndexError(f"{split_name} contains an out-of-range index")
    source = str(record.get("source", ""))
    payload = "\n".join(f"{source}:{index}" for index in indices)
    expected_hash = hashlib.sha256(payload.encode("ascii")).hexdigest()
    if record.get("sha256") != expected_hash:
        raise ValueError(f"{split_name} index hash differs from its manifest")
    return indices


def build_torchvision_loaders(
    dataset_name: str,
    data_root: str,
    image_size: int,
    batch_size: int,
    workers: int,
    seed: int,
    split_manifest: str,
    download: bool = False,
) -> ImageFolderLoaders:
    try:
        import torch
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise RuntimeError("torchvision is required for benchmark datasets") from exc
    dataset_types = {"cifar10": datasets.CIFAR10, "cifar100": datasets.CIFAR100}
    if dataset_name not in dataset_types:
        raise ValueError(f"unsupported torchvision dataset: {dataset_name}")
    if not split_manifest:
        raise ValueError("torchvision datasets require a frozen index split manifest")
    transform = transforms.Compose(
        [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
    )
    dataset_type = dataset_types[dataset_name]
    train_source = dataset_type(
        root=data_root, train=True, download=download, transform=transform
    )
    test_source = dataset_type(
        root=data_root, train=False, download=download, transform=transform
    )
    manifest = json.loads(Path(split_manifest).read_text(encoding="utf-8"))
    if manifest.get("dataset") != dataset_name:
        raise ValueError("index manifest dataset differs from configuration")
    class_to_index = {name: index for index, name in enumerate(train_source.classes)}
    if manifest.get("class_to_index") != class_to_index:
        raise ValueError("index manifest class mapping differs from dataset")
    required = {
        "target_train",
        "target_validation",
        "target_test",
        "attacker_auxiliary_train",
        "attacker_auxiliary_validation",
    }
    if not required.issubset(manifest.get("splits", {})):
        raise ValueError("index manifest is missing a required protocol split")
    datasets_by_split = {
        "target_train": Subset(
            train_source,
            _indices_from_manifest(manifest, "target_train", len(train_source)),
        ),
        "target_validation": Subset(
            train_source,
            _indices_from_manifest(manifest, "target_validation", len(train_source)),
        ),
        "target_test": Subset(
            test_source,
            _indices_from_manifest(manifest, "target_test", len(test_source)),
        ),
        "attacker_auxiliary_train": Subset(
            train_source,
            _indices_from_manifest(
                manifest, "attacker_auxiliary_train", len(train_source)
            ),
        ),
        "attacker_auxiliary_validation": Subset(
            train_source,
            _indices_from_manifest(
                manifest, "attacker_auxiliary_validation", len(train_source)
            ),
        ),
    }
    target_train_indices = set(datasets_by_split["target_train"].indices)
    target_validation_indices = set(datasets_by_split["target_validation"].indices)
    attacker_train_indices = set(datasets_by_split["attacker_auxiliary_train"].indices)
    attacker_validation_indices = set(
        datasets_by_split["attacker_auxiliary_validation"].indices
    )
    train_sets = (
        target_train_indices,
        target_validation_indices,
        attacker_train_indices,
        attacker_validation_indices,
    )
    if any(
        left & right
        for position, left in enumerate(train_sets)
        for right in train_sets[position + 1 :]
    ):
        raise ValueError("training-derived protocol splits overlap")
    train_generator = torch.Generator().manual_seed(seed)
    attacker_generator = torch.Generator().manual_seed(seed + 1)
    common = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": workers > 0,
    }
    return ImageFolderLoaders(
        train=_loader(
            datasets_by_split["target_train"], True, common, train_generator
        ),
        statistics=_loader(datasets_by_split["target_train"], False, common),
        validation=_loader(datasets_by_split["target_validation"], False, common),
        test=_loader(datasets_by_split["target_test"], False, common),
        attacker_auxiliary_train=_loader(
            datasets_by_split["attacker_auxiliary_train"],
            True,
            common,
            attacker_generator,
        ),
        attacker_auxiliary_validation=_loader(
            datasets_by_split["attacker_auxiliary_validation"], False, common
        ),
        class_to_index=class_to_index,
        split_manifest=manifest,
        protocol_mode="manifest",
    )


def build_protocol_loaders(
    data_config: Mapping[str, Any],
    seed: int,
    batch_size: int | None = None,
) -> ImageFolderLoaders:
    common = {
        "data_root": str(data_config["root"]),
        "image_size": int(data_config["image_size"]),
        "batch_size": int(batch_size or data_config["batch_size"]),
        "workers": int(data_config.get("workers", 4)),
        "seed": seed,
        "split_manifest": data_config.get("split_manifest"),
        "train_augmentation": bool(data_config.get("train_augmentation", False)),
        "protocol_mode": str(data_config.get("protocol_mode", "manifest_or_legacy")),
    }
    dataset_name = str(data_config.get("name", "imagefolder")).lower()
    if dataset_name in {"facescrub", "imagefolder"}:
        return build_imagefolder_loaders(
            num_classes_override=(
                int(data_config["num_classes_override"])
                if data_config.get("num_classes_override") is not None
                else None
            ),
            **common,
        )
    if dataset_name in {"cifar10", "cifar100"}:
        common.pop("train_augmentation")
        common.pop("protocol_mode")
        return build_torchvision_loaders(
            dataset_name=dataset_name,
            download=bool(data_config.get("download", False)),
            **common,
        )
    raise ValueError(f"unsupported dataset: {dataset_name}")
