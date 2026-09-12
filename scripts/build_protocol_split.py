#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def split_hash(paths: list[str]) -> str:
    return hashlib.sha256("\n".join(paths).encode("utf-8")).hexdigest()


def class_files(root: Path, source: str) -> dict[str, list[str]]:
    source_root = root / source
    if not source_root.is_dir():
        raise FileNotFoundError(f"missing source directory: {source_root}")
    grouped = {}
    for class_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        grouped[class_dir.name] = [
            path.relative_to(root).as_posix()
            for path in sorted(class_dir.rglob("*"))
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
    return grouped


def allocate_class(
    paths: list[str],
    class_name: str,
    seed: int,
    validation_fraction: float,
    auxiliary_fraction: float,
) -> tuple[list[str], list[str], list[str], list[str]]:
    if len(paths) < 3:
        raise ValueError(f"class {class_name} needs at least three training images")
    shuffled = list(paths)
    random.Random(f"{seed}:{class_name}").shuffle(shuffled)
    validation_count = max(1, round(len(shuffled) * validation_fraction))
    auxiliary_count = max(1, round(len(shuffled) * auxiliary_fraction))
    if validation_count + auxiliary_count >= len(shuffled):
        raise ValueError(f"split fractions leave no target-train image for {class_name}")
    validation = shuffled[:validation_count]
    auxiliary = shuffled[validation_count : validation_count + auxiliary_count]
    target_train = shuffled[validation_count + auxiliary_count :]
    auxiliary_validation_count = max(1, round(len(auxiliary) * 0.1))
    if auxiliary_validation_count >= len(auxiliary):
        raise ValueError(
            f"attacker auxiliary allocation needs at least two images for {class_name}"
        )
    auxiliary_validation = auxiliary[:auxiliary_validation_count]
    auxiliary_train = auxiliary[auxiliary_validation_count:]
    return (
        sorted(target_train),
        sorted(validation),
        sorted(auxiliary_train),
        sorted(auxiliary_validation),
    )


def build_manifest(
    data_root: Path,
    seed: int,
    validation_fraction: float,
    auxiliary_fraction: float,
) -> dict:
    training = class_files(data_root, "train")
    target_test_by_class = class_files(data_root, "val")
    if set(training) != set(target_test_by_class):
        raise ValueError("train and val class names differ")

    splits = {
        "target_train": [],
        "target_validation": [],
        "target_test": [],
        "attacker_auxiliary_train": [],
        "attacker_auxiliary_validation": [],
    }
    class_counts = {name: {} for name in splits}
    for class_name in sorted(training):
        target_train, validation, auxiliary_train, auxiliary_validation = allocate_class(
            training[class_name],
            class_name,
            seed,
            validation_fraction,
            auxiliary_fraction,
        )
        values = {
            "target_train": target_train,
            "target_validation": validation,
            "target_test": sorted(target_test_by_class[class_name]),
            "attacker_auxiliary_train": auxiliary_train,
            "attacker_auxiliary_validation": auxiliary_validation,
        }
        for split_name, paths in values.items():
            splits[split_name].extend(paths)
            class_counts[split_name][class_name] = len(paths)

    records = {}
    for split_name, paths in splits.items():
        paths.sort()
        records[split_name] = {
            "count": len(paths),
            "sha256": split_hash(paths),
            "class_counts": class_counts[split_name],
            "paths": paths,
        }

    train_derived = (
        set(records["target_train"]["paths"])
        | set(records["target_validation"]["paths"])
        | set(records["attacker_auxiliary_train"]["paths"])
        | set(records["attacker_auxiliary_validation"]["paths"])
    )
    if len(train_derived) != len(set(sum(training.values(), []))):
        raise AssertionError("derived training splits do not partition source train")
    if any(
        set(records[left]["paths"]) & set(records[right]["paths"])
        for index, left in enumerate(records)
        for right in list(records)[index + 1 :]
    ):
        raise AssertionError("protocol splits overlap")

    class_names = sorted(training)
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root_at_generation": str(data_root.resolve()),
        "seed": seed,
        "validation_fraction": validation_fraction,
        "auxiliary_fraction": auxiliary_fraction,
        "source_policy": {
            "target_train": "stratified subset of source train",
            "target_validation": "stratified subset of source train; model selection only",
            "attacker_auxiliary_train": "stratified subset of source train; disjoint attack training images",
            "attacker_auxiliary_validation": "stratified subset of source train; attacker selection only",
            "target_test": "all source val images; final evaluation only",
        },
        "class_to_index": {name: index for index, name in enumerate(class_names)},
        "class_name_sha256": hashlib.sha256(
            "\n".join(class_names).encode("utf-8")
        ).hexdigest(),
        "splits": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20_260_716)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--auxiliary-fraction", type=float, default=0.2)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "evidence"
        / "data_split_manifest.json",
    )
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 1:
        raise ValueError("validation fraction must be in (0, 1)")
    if not 0 < args.auxiliary_fraction < 1:
        raise ValueError("auxiliary fraction must be in (0, 1)")
    manifest = build_manifest(
        args.data_root,
        args.seed,
        args.validation_fraction,
        args.auxiliary_fraction,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                name: {"count": value["count"], "sha256": value["sha256"]}
                for name, value in manifest["splits"].items()
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
