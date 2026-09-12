#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-knowledge", required=True, type=Path)
    parser.add_argument("--inference-knowledge", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def combine(training: dict, inference: dict) -> dict:
    if training["attack_type"] != inference["attack_type"]:
        raise ValueError("cannot combine different attack types")
    if training["attack_knowledge"] != "training":
        raise ValueError("training input does not use training knowledge")
    if inference["attack_knowledge"] != "inference":
        raise ValueError("inference input does not use inference knowledge")
    if training["dual_path_checkpoint"] != inference["dual_path_checkpoint"]:
        raise ValueError("attack runs target different checkpoints")
    for name in ("effective_legacy_noise_std", "effective_semantic_noise_std"):
        if training[name] != inference[name]:
            raise ValueError(f"attack runs use different {name}")
    return {
        "attack_type": training["attack_type"],
        "training": training["evaluation"],
        "inference": inference["evaluation"],
        "training_knowledge_run": training,
        "inference_knowledge_run": inference,
    }


def main() -> None:
    args = parse_args()
    result = combine(
        load_json(args.training_knowledge),
        load_json(args.inference_knowledge),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
