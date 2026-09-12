#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--size", type=int, default=48)
    return parser.parse_args()


def source_revision(source: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def prepare(source: Path, output: Path, size: int) -> dict:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "opencv-python-headless==4.8.1.78 is required for CEM-compatible resizing"
        ) from exc
    if size <= 0:
        raise ValueError("size must be positive")
    raw_root = source / "raw"
    splits = {"train": raw_root / "train", "val": raw_root / "validate"}
    if not all(path.is_dir() for path in splits.values()):
        raise FileNotFoundError("source must contain raw/train and raw/validate")
    manifest_path = output / "dataset_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("size") == size and existing.get("source_revision") == source_revision(source):
            return existing
        raise FileExistsError("existing prepared dataset has a different source or size")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite incomplete output: {output}")

    temporary = output.with_name(f".{output.name}.preparing")
    if temporary.exists():
        shutil.rmtree(temporary)
    path_records: list[str] = []
    split_counts = {}
    class_names = None
    try:
        for output_split, input_root in splits.items():
            classes = sorted(path.name for path in input_root.iterdir() if path.is_dir())
            if class_names is None:
                class_names = classes
            elif classes != class_names:
                raise ValueError("FaceScrub train and validation classes differ")
            count = 0
            for class_name in classes:
                destination = temporary / output_split / class_name
                destination.mkdir(parents=True, exist_ok=True)
                for image_path in sorted((input_root / class_name).glob("*.jpg")):
                    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                    if image is None:
                        raise ValueError(f"OpenCV could not read {image_path}")
                    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
                    destination_path = destination / image_path.name
                    if not cv2.imwrite(str(destination_path), resized):
                        raise OSError(f"OpenCV could not write {destination_path}")
                    path_records.append(
                        destination_path.relative_to(temporary).as_posix()
                    )
                    count += 1
            split_counts[output_split] = count
        manifest = {
            "dataset": "FaceScrub",
            "preparation": "CEM public-code compatible OpenCV resize",
            "source": str(source.resolve()),
            "source_revision": source_revision(source),
            "size": size,
            "classes": len(class_names or []),
            "split_counts": split_counts,
            "path_list_sha256": hashlib.sha256(
                "\n".join(path_records).encode("utf-8")
            ).hexdigest(),
        }
        (temporary / "dataset_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.rename(output)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    args = parse_args()
    result = prepare(args.source, args.output, args.size)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
