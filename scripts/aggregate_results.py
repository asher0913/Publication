#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


PRIVACY_METRICS = (
    "mse",
    "ssim",
    "psnr",
    "lpips",
    "identity_top1_success",
    "face_cosine_similarity",
    "verification_tar_at_far",
)


def read_last_jsonl(path: Path):
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return records[-1] if records else {}


def method_name(config):
    defense = config["defense"]
    variant = defense.get("variant")
    if variant == "utility_aligned_anisotropic_cem":
        calibration = defense["calibration"]
        return (
            f"uacem_{calibration['score']}_a{calibration['alpha']:g}_"
            f"n{defense['noise_std']:g}"
        )
    if variant:
        return str(variant)
    if defense["name"] == "prototype_cem":
        slots = defense["regularizer"]["num_slots"]
        weight = defense["privacy_weight"]
        noise = defense["noise_std"]
        return f"prototype_cem_k{slots}_w{weight:g}_n{noise:g}"
    if defense["name"] == "epochwise_cem":
        weight = defense["privacy_weight"]
        noise = defense["noise_std"]
        return f"epochwise_cem_w{weight:g}_n{noise:g}"
    if defense["name"] == "official_cem":
        weight = defense["privacy_weight"]
        noise = defense["noise_std"]
        return f"official_cem_w{weight:g}_n{noise:g}"
    if defense["name"] == "gaussian":
        return f"gaussian_n{defense['noise_std']:g}"
    if defense["name"] == "none":
        return defense["name"]
    raise ValueError(f"unknown defense: {defense['name']}")


def bootstrap_interval(values, iterations=10_000, seed=20_260_716):
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], values[0]
    generator = random.Random(seed)
    estimates = sorted(
        mean(generator.choices(values, k=len(values))) for _ in range(iterations)
    )
    return estimates[int(0.025 * (iterations - 1))], estimates[
        int(0.975 * (iterations - 1))
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    args = parser.parse_args()
    rows = []
    for config_path in sorted(args.results_root.rglob("resolved_config.json")):
        run_dir = config_path.parent
        train_log = run_dir / "training.jsonl"
        target_test_path = run_dir / "target_test_metrics.json"
        attack_paths = sorted(run_dir.rglob("attack_metrics.json"))
        selection_path = run_dir / "target_selection_metrics.json"
        if not target_test_path.exists() or not attack_paths:
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        target = read_last_jsonl(train_log) if train_log.exists() else {}
        selection = (
            json.loads(selection_path.read_text(encoding="utf-8"))
            if selection_path.exists()
            else {}
        )
        if not target and (run_dir / "channel_calibration.json").is_file():
            calibration = json.loads(
                (run_dir / "channel_calibration.json").read_text(encoding="utf-8")
            )
            source_checkpoint = Path(calibration["source_checkpoint"])
            source_log = source_checkpoint.parent / "training.jsonl"
            if source_log.is_file():
                target = read_last_jsonl(source_log)
        target_test = json.loads(target_test_path.read_text(encoding="utf-8"))
        selection_accuracy = selection.get(
            "best_validation_accuracy", target.get("validation_accuracy")
        )
        if selection_accuracy is None:
            raise ValueError(f"selection accuracy is missing for {run_dir}")
        for attack_path in attack_paths:
            attack = json.loads(attack_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "run_dir": str(run_dir),
                    "attack_dir": str(attack_path.parent),
                    "method": method_name(config),
                    "target_seed": config["seed"],
                    "attacker_seed": attack["attack_seed"],
                    "attack_type": attack["attack_type"],
                    "accuracy": target_test["test_accuracy"],
                    "selection_accuracy": selection_accuracy,
                    **{metric: attack.get(metric) for metric in PRIVACY_METRICS},
                    "training_epoch_seconds": target.get("training_epoch_seconds"),
                    "statistics_refresh_seconds": target.get(
                        "statistics_refresh_seconds"
                    ),
                    "peak_gpu_memory_bytes": target.get("peak_gpu_memory_bytes"),
                }
            )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_prefix.with_suffix(".csv")
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    target_groups = defaultdict(list)
    for row in rows:
        key = (
            row["method"],
            row["attack_type"],
            row["target_seed"],
            row["run_dir"],
        )
        target_groups[key].append(row)

    target_rows = []
    for (method, attack_type, target_seed, run_dir), attack_rows in target_groups.items():
        target_row = {
            "run_dir": run_dir,
            "method": method,
            "attack_type": attack_type,
            "target_seed": target_seed,
            "attacker_runs": len(attack_rows),
            "accuracy": float(attack_rows[0]["accuracy"]),
        }
        for metric in PRIVACY_METRICS:
            values = [
                float(row[metric]) for row in attack_rows if row[metric] is not None
            ]
            target_row[metric] = mean(values) if values else None
        target_rows.append(target_row)

    target_csv_path = args.output_prefix.with_name(
        args.output_prefix.name + "_target_level"
    ).with_suffix(".csv")
    if target_rows:
        with target_csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(target_rows[0]))
            writer.writeheader()
            writer.writerows(target_rows)

    grouped = defaultdict(list)
    for row in target_rows:
        grouped[(row["method"], row["attack_type"])].append(row)
    summary = {}
    for (method, attack_type), method_rows in grouped.items():
        summary_key = f"{method}__{attack_type}"
        summary[summary_key] = {
            "method": method,
            "attack_type": attack_type,
            "target_runs": len(method_rows),
            "attack_runs": sum(int(row["attacker_runs"]) for row in method_rows),
        }
        for metric in ("accuracy", *PRIVACY_METRICS):
            values = [
                float(row[metric])
                for row in method_rows
                if row.get(metric) is not None
            ]
            if not values:
                continue
            ci_low, ci_high = bootstrap_interval(values)
            summary[summary_key][f"{metric}_mean"] = mean(values)
            summary[summary_key][f"{metric}_std"] = (
                stdev(values) if len(values) > 1 else None
            )
            summary[summary_key][f"{metric}_ci95_low"] = ci_low
            summary[summary_key][f"{metric}_ci95_high"] = ci_high
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "attack_runs": len(rows),
                "target_runs": len(target_rows),
                "methods": len(summary),
            }
        )
    )


if __name__ == "__main__":
    main()
