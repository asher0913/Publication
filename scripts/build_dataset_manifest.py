#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def split_record(path: Path):
    class_dirs = sorted(item for item in path.iterdir() if item.is_dir())
    classes = {}
    for class_dir in class_dirs:
        classes[class_dir.name] = sum(
            1
            for item in class_dir.rglob("*")
            if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
        )
    return {
        "class_count": len(classes),
        "image_count": sum(classes.values()),
        "classes": classes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "evidence" / "dataset_manifest.json",
    )
    args = parser.parse_args()
    train = split_record(args.data_root / "train")
    validation = split_record(args.data_root / "val")
    if set(train["classes"]) != set(validation["classes"]):
        raise ValueError("train and validation identities differ")
    class_mapping = "\n".join(sorted(train["classes"]))
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(args.data_root.resolve()),
        "class_name_sha256": hashlib.sha256(class_mapping.encode()).hexdigest(),
        "train": train,
        "validation": validation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"train": train["image_count"], "val": validation["image_count"], "classes": train["class_count"]}))


if __name__ == "__main__":
    main()
