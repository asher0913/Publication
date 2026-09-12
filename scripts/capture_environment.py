#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def command_output(command: list[str]) -> dict:
    try:
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except FileNotFoundError:
        return {"returncode": None, "output": "command not found"}
    return {"returncode": completed.returncode, "output": completed.stdout.strip()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "evidence/environment"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pip_freeze = command_output([sys.executable, "-m", "pip", "freeze"])
    (args.output_dir / "pip-freeze.txt").write_text(
        pip_freeze["output"] + "\n", encoding="utf-8"
    )
    nvidia_smi = command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader",
        ]
    )
    (args.output_dir / "nvidia-smi.txt").write_text(
        nvidia_smi["output"] + "\n", encoding="utf-8"
    )
    gpus = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            gpus.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                    "compute_capability": f"{properties.major}.{properties.minor}",
                }
            )
    payload = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "torchvision": command_output(
            [sys.executable, "-c", "import torchvision; print(torchvision.__version__)"]
        )["output"],
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpus": gpus,
        "nvidia_smi_returncode": nvidia_smi["returncode"],
        "pip_freeze_returncode": pip_freeze["returncode"],
    }
    (args.output_dir / "environment.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
