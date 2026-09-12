#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def index_hash(source: str, indices: list[int]) -> str:
    payload = "\n".join(f"{source}:{index}" for index in indices)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def allocate_indices(
    indices: list[int],
    class_name: str,
    seed: int,
    validation_fraction: float,
    auxiliary_fraction: float,
    auxiliary_validation_fraction: float,
) -> tuple[list[int], list[int], list[int], list[int]]:
    if len(indices) < 4:
        raise ValueError(f"class {class_name} needs at least four training images")
    shuffled = list(indices)
    random.Random(f"{seed}:{class_name}").shuffle(shuffled)
    validation_count = max(1, round(len(shuffled) * validation_fraction))
    auxiliary_count = max(2, round(len(shuffled) * auxiliary_fraction))
    if validation_count + auxiliary_count >= len(shuffled):
        raise ValueError(f"split fractions leave no target-train image for {class_name}")
    validation = shuffled[:validation_count]
    auxiliary = shuffled[validation_count : validation_count + auxiliary_count]
    target_train = shuffled[validation_count + auxiliary_count :]
    auxiliary_validation_count = max(
        1, round(len(auxiliary) * auxiliary_validation_fraction)
    )
    if auxiliary_validation_count >= len(auxiliary):
        raise ValueError(f"attacker validation consumes class {class_name} auxiliary data")
    return (
        sorted(target_train),
        sorted(validation),
        sorted(auxiliary[auxiliary_validation_count:]),
        sorted(auxiliary[:auxiliary_validation_count]),
    )


def build_index_manifest(
    dataset: str,
    train_labels: list[int],
    test_labels: list[int],
    classes: list[str],
    seed: int,
    validation_fraction: float,
    auxiliary_fraction: float,
    auxiliary_validation_fraction: float = 0.1,
) -> dict:
    grouped = defaultdict(list)
    for index, label in enumerate(train_labels):
        grouped[int(label)].append(index)
    if set(grouped) != set(range(len(classes))):
        raise ValueError("training labels do not cover the declared classes")
    splits = {
        "target_train": [],
        "target_validation": [],
        "attacker_auxiliary_train": [],
        "attacker_auxiliary_validation": [],
    }
    class_counts = {name: {} for name in splits}
    for class_index, class_name in enumerate(classes):
        allocated = allocate_indices(
            grouped[class_index],
            class_name,
            seed,
            validation_fraction,
            auxiliary_fraction,
            auxiliary_validation_fraction,
        )
        for split_name, indices in zip(splits, allocated, strict=True):
            splits[split_name].extend(indices)
            class_counts[split_name][class_name] = len(indices)
    train_sets = [set(indices) for indices in splits.values()]
    if any(
        left & right
        for position, left in enumerate(train_sets)
        for right in train_sets[position + 1 :]
    ):
        raise AssertionError("training-derived protocol splits overlap")
    if set().union(*train_sets) != set(range(len(train_labels))):
        raise AssertionError("training-derived splits do not partition source train")
    records = {}
    for split_name, indices in splits.items():
        indices.sort()
        records[split_name] = {
            "source": "train",
            "count": len(indices),
            "sha256": index_hash("train", indices),
            "class_counts": class_counts[split_name],
            "indices": indices,
        }
    test_indices = list(range(len(test_labels)))
    test_counts = {class_name: 0 for class_name in classes}
    for label in test_labels:
        test_counts[classes[int(label)]] += 1
    records["target_test"] = {
        "source": "test",
        "count": len(test_indices),
        "sha256": index_hash("test", test_indices),
        "class_counts": test_counts,
        "indices": test_indices,
    }
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "seed": seed,
        "validation_fraction": validation_fraction,
        "auxiliary_fraction": auxiliary_fraction,
        "auxiliary_validation_fraction": auxiliary_validation_fraction,
        "class_to_index": {name: index for index, name in enumerate(classes)},
        "source_policy": {
            "target_train": "stratified source-train subset; target fitting only",
            "target_validation": "disjoint source-train subset; target selection only",
            "attacker_auxiliary_train": "disjoint source-train subset; attack fitting only",
            "attacker_auxiliary_validation": "disjoint source-train subset; attack selection only",
            "target_test": "complete official test set; final evaluation only",
        },
        "splits": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("cifar10", "cifar100"), required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/torchvision")
    parser.add_argument("--seed", type=int, default=20_260_716)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--auxiliary-fraction", type=float, default=0.2)
    parser.add_argument("--auxiliary-validation-fraction", type=float, default=0.1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    fractions = (
        args.validation_fraction,
        args.auxiliary_fraction,
        args.auxiliary_validation_fraction,
    )
    if any(not 0 < fraction < 1 for fraction in fractions):
        raise ValueError("all split fractions must be in (0, 1)")
    from torchvision import datasets

    dataset_type = datasets.CIFAR10 if args.dataset == "cifar10" else datasets.CIFAR100
    train = dataset_type(root=args.data_root, train=True, download=True)
    test = dataset_type(root=args.data_root, train=False, download=True)
    manifest = build_index_manifest(
        args.dataset,
        [int(label) for label in train.targets],
        [int(label) for label in test.targets],
        list(train.classes),
        args.seed,
        args.validation_fraction,
        args.auxiliary_fraction,
        args.auxiliary_validation_fraction,
    )
    output = args.output or ROOT / "evidence" / f"{args.dataset}_split_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "splits": {
                    name: {"count": record["count"], "sha256": record["sha256"]}
                    for name, record in manifest["splits"].items()
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
