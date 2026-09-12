#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_positive(values: list[float]) -> bool:
    return bool(values) and all(math.isfinite(value) and value > 0 for value in values)


def summarise(results_root: Path, environment_path: Path) -> dict:
    environment = read_json(environment_path)
    if environment.get("cuda_available") is not True or not environment.get("gpus"):
        raise ValueError("formal pilot must be summarised from a CUDA environment")
    expected = {
        "seed119_no_defense": ("none", ("conv_decoder",)),
        "seed119_official_cem": ("official_cem", ("conv_decoder",)),
        "seed119_proposed": (
            "prototype_cem",
            ("conv_decoder", "residual_decoder", "gan", "adaptive"),
        ),
    }
    records = {}
    for run_name, (expected_defense, attack_types) in expected.items():
        run_dir = results_root / run_name
        required = {
            "config": run_dir / "resolved_config.json",
            "target_log": run_dir / "training.jsonl",
            "target_selection": run_dir / "target_selection_metrics.json",
            "checkpoint": run_dir / "checkpoint_best.pt",
        }
        missing = [name for name, path in required.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{run_name} pilot outputs missing: {missing}")
        config = read_json(required["config"])
        target_selection = read_json(required["target_selection"])
        target_log = read_jsonl(required["target_log"])
        if config["defense"]["name"] != expected_defense:
            raise ValueError(f"{run_name} has an unexpected defence")
        if len(target_log) != 2:
            raise ValueError(f"{run_name} target pilot must contain two epochs")
        if target_selection.get("target_test_accessed") is not False:
            raise ValueError(f"{run_name} pilot accessed the target test split")
        target_seconds = [float(row["total_epoch_seconds"]) for row in target_log]
        peak_memory = max(int(row["peak_gpu_memory_bytes"]) for row in target_log)
        if not finite_positive(target_seconds) or peak_memory <= 0:
            raise ValueError(f"{run_name} has invalid timing or CUDA-memory records")
        attack_records = {}
        for attack_type in attack_types:
            attack_dir = run_dir / "pilot_attacks" / attack_type
            attack_log_path = attack_dir / "attack_training.jsonl"
            attack_selection_path = attack_dir / "attack_selection_metrics.json"
            decoder_path = attack_dir / "decoder.pt"
            missing_attack = [
                path.name
                for path in (attack_log_path, attack_selection_path, decoder_path)
                if not path.is_file()
            ]
            if missing_attack:
                raise FileNotFoundError(
                    f"{run_name}/{attack_type} missing: {missing_attack}"
                )
            attack_log = read_jsonl(attack_log_path)
            attack_selection = read_json(attack_selection_path)
            if len(attack_log) != 2:
                raise ValueError(
                    f"{run_name}/{attack_type} pilot must contain two epochs"
                )
            if attack_selection.get("target_test_accessed") is not False:
                raise ValueError(
                    f"{run_name}/{attack_type} accessed the target test split"
                )
            attack_seconds = [float(row["epoch_seconds"]) for row in attack_log]
            if not finite_positive(attack_seconds):
                raise ValueError(f"{run_name}/{attack_type} timing is invalid")
            attack_records[attack_type] = {
                "attack_epoch_seconds_mean": mean(attack_seconds),
                "estimated_attack_hours_100_epochs": (
                    mean(attack_seconds) * 100 / 3600
                ),
                "best_auxiliary_validation_mse": float(
                    attack_selection["best_auxiliary_validation_mse"]
                ),
                "decoder_sha256": sha256(decoder_path),
                "target_test_accessed": False,
            }
        records[run_name] = {
            "defense": expected_defense,
            "target_epoch_seconds_mean": mean(target_seconds),
            "estimated_target_hours_300_epochs": mean(target_seconds) * 300 / 3600,
            "peak_gpu_memory_bytes": peak_memory,
            "best_validation_accuracy": float(
                target_selection["best_validation_accuracy"]
            ),
            "checkpoint_sha256": sha256(required["checkpoint"]),
            "attacks": attack_records,
            "target_test_accessed": False,
        }
    return {
        "schema_version": 1,
        "status": "PASS",
        "purpose": "Full-data engineering pilot; not scientific evidence",
        "formal_cuda_environment": True,
        "environment": str(environment_path),
        "environment_sha256": sha256(environment_path),
        "methods": records,
        "target_test_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=ROOT / "results/facescrub/formal_gpu_pilot",
    )
    parser.add_argument(
        "--environment",
        type=Path,
        default=ROOT / "evidence/environment/environment.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "evidence/formal_gpu_pilot.json",
    )
    args = parser.parse_args()
    payload = summarise(args.results_root, args.environment)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"status": payload["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
