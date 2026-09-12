#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[1]


def bootstrap_mean_interval(
    values: list[float],
    iterations: int,
    confidence_level: float,
    seed: int = 20_260_716,
) -> tuple[float, float]:
    if not values:
        raise ValueError("cannot bootstrap an empty sample")
    if len(values) == 1:
        return values[0], values[0]
    generator = random.Random(seed)
    estimates = sorted(
        mean(generator.choices(values, k=len(values))) for _ in range(iterations)
    )
    tail = (1.0 - confidence_level) / 2.0
    low = estimates[int(tail * (iterations - 1))]
    high = estimates[int((1.0 - tail) * (iterations - 1))]
    return low, high


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def finite_value(row: dict, name: str) -> float | None:
    raw = row.get(name)
    if raw in (None, "", "None", "nan"):
        return None
    value = float(raw)
    return value if math.isfinite(value) else None


def paired_analysis(
    rows: list[dict],
    proposed_pattern: str,
    baseline_pattern: str,
    rules: dict,
) -> dict:
    proposed_matcher = re.compile(proposed_pattern)
    baseline_matcher = re.compile(baseline_pattern)
    grouped = defaultdict(dict)
    for row in rows:
        method = row["method"]
        role = (
            "proposed"
            if proposed_matcher.fullmatch(method)
            else "baseline"
            if baseline_matcher.fullmatch(method)
            else None
        )
        if role is not None:
            grouped[(row["attack_type"], int(row["target_seed"]))][role] = row
    by_attack = defaultdict(list)
    for (attack_type, target_seed), pair in grouped.items():
        if set(pair) == {"proposed", "baseline"}:
            by_attack[attack_type].append((target_seed, pair))

    output = {}
    for attack_type, pairs in sorted(by_attack.items()):
        pairs.sort(key=lambda value: value[0])
        accuracy_differences = [
            float(pair["proposed"]["accuracy"])
            - float(pair["baseline"]["accuracy"])
            for _, pair in pairs
        ]
        utility_difference = mean(accuracy_differences)
        utility_ok = utility_difference >= -float(rules["utility_drop_limit"])
        metrics = {}
        supported_metrics = []
        primary_metrics = set(
            rules.get(
                "primary_privacy_metrics",
                rules["privacy_benefit_direction"],
            )
        )
        for metric, direction in rules["privacy_benefit_direction"].items():
            benefits = []
            for _, pair in pairs:
                proposed = finite_value(pair["proposed"], metric)
                baseline = finite_value(pair["baseline"], metric)
                if proposed is not None and baseline is not None:
                    benefits.append(float(direction) * (proposed - baseline))
            if not benefits:
                continue
            low, high = bootstrap_mean_interval(
                benefits,
                int(rules["bootstrap_iterations"]),
                float(rules["confidence_level"]),
            )
            margin = float(rules["minimum_practical_effect"][metric])
            supported = (
                len(benefits) >= int(rules["minimum_target_pairs"])
                and low > margin
            )
            if supported:
                supported_metrics.append(metric)
            metrics[metric] = {
                "paired_target_runs": len(benefits),
                "oriented_benefit_mean": mean(benefits),
                "oriented_benefit_std": stdev(benefits) if len(benefits) > 1 else None,
                "confidence_level": rules["confidence_level"],
                "ci_low": low,
                "ci_high": high,
                "minimum_practical_effect": margin,
                "supported": supported,
            }
        enough_pairs = len(pairs) >= int(rules["minimum_target_pairs"])
        supported_primary_metrics = [
            metric for metric in supported_metrics if metric in primary_metrics
        ]
        output[attack_type] = {
            "target_seeds": [seed for seed, _ in pairs],
            "paired_target_runs": len(pairs),
            "accuracy_difference_proposed_minus_baseline": utility_difference,
            "utility_drop_limit": rules["utility_drop_limit"],
            "utility_ok": utility_ok,
            "enough_pairs": enough_pairs,
            "supported_privacy_metrics": supported_metrics,
            "supported_primary_privacy_metrics": supported_primary_metrics,
            "h1_status": (
                "SUPPORTED"
                if enough_pairs and utility_ok and supported_primary_metrics
                else "NOT_SUPPORTED"
            ),
            "metrics": metrics,
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-level-csv", required=True, type=Path)
    parser.add_argument(
        "--rules", type=Path, default=ROOT / "configs/decision_rules.json"
    )
    parser.add_argument(
        "--proposed-pattern", default=r"prototype_cem_k8_.*"
    )
    parser.add_argument(
        "--baseline-pattern", default=r"official_cem_.*"
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rules = json.loads(args.rules.read_text(encoding="utf-8"))
    result = paired_analysis(
        read_rows(args.target_level_csv),
        args.proposed_pattern,
        args.baseline_pattern,
        rules,
    )
    primary_attack_type = rules.get("primary_attack_type", "conv_decoder")
    payload = {
        "schema_version": 1,
        "status": "ANALYSED",
        "input": str(args.target_level_csv),
        "rules": rules,
        "primary_attack_type": primary_attack_type,
        "global_h1_status": result.get(primary_attack_type, {}).get(
            "h1_status", "NOT_SUPPORTED"
        ),
        "comparisons": result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "comparisons": len(result)}))


if __name__ == "__main__":
    main()
