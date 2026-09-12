#!/usr/bin/env python3
from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import torch
import torchvision


ROOT = Path(__file__).resolve().parents[1]


EXPECTED = {
    "python_major_minor": "3.11",
    "torch": "2.2.2",
    "torchvision": "0.17.2",
    "cuda": "12.1",
    "numpy": "1.26.4",
    "Pillow": "10.2.0",
    "scipy": "1.12.0",
    "facenet-pytorch": "2.6.0",
    "lpips": "0.1.4",
}


def base_version(value: str) -> str:
    return value.split("+")[0]


def validate_versions(observed: dict) -> list[str]:
    mismatches = []
    for name, expected in EXPECTED.items():
        if observed.get(name) != expected:
            mismatches.append(
                f"{name}: expected {expected}, observed {observed.get(name)}"
            )
    if observed.get("cuda_available") is not True:
        mismatches.append("cuda_available: expected True")
    if not observed.get("gpus"):
        mismatches.append("gpus: no CUDA device detected")
    if observed.get("cudnn") is None:
        mismatches.append("cudnn: no cuDNN runtime detected")
    return mismatches


def main() -> None:
    observed = {
        "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
        "python_full": platform.python_version(),
        "torch": base_version(torch.__version__),
        "torchvision": base_version(torchvision.__version__),
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cudnn": torch.backends.cudnn.version(),
        "gpus": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
    }
    for package in ("numpy", "Pillow", "scipy", "facenet-pytorch", "lpips"):
        try:
            observed[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            observed[package] = None
    mismatches = validate_versions(observed)
    payload = {
        "schema_version": 1,
        "status": "PASS" if not mismatches else "FAIL",
        "expected": EXPECTED,
        "observed": observed,
        "mismatches": mismatches,
    }
    output = ROOT / "evidence/environment/formal_environment_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
