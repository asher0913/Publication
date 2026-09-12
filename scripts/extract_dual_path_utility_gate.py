#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--legacy-noise-std", required=True, type=float)
    parser.add_argument("--semantic-noise-std", required=True, type=float)
    parser.add_argument("--utility-threshold", type=float, default=0.8033)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def extract_candidate(probe: dict, legacy_noise: float, semantic_noise: float) -> dict:
    matches = [
        candidate
        for candidate in probe["candidates"]
        if float(candidate["legacy_noise_std"]) == legacy_noise
        and float(candidate["semantic_noise_std"]) == semantic_noise
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one probe candidate for ({legacy_noise}, {semantic_noise}); "
            f"found {len(matches)}"
        )
    return matches[0]


def build_gate(candidate: dict, threshold: float, probe_path: Path) -> dict:
    conservative_accuracy = float(candidate["minimum_fused_accuracy"])
    passed = conservative_accuracy > threshold
    return {
        "status": "PASS" if passed else "FAIL",
        "best_validation_accuracy": conservative_accuracy,
        "conservative_repeated_validation_accuracy": conservative_accuracy,
        "mean_repeated_validation_accuracy": float(candidate["mean_fused_accuracy"]),
        "maximum_repeated_validation_accuracy": float(
            candidate["maximum_fused_accuracy"]
        ),
        "published_accuracy_threshold": threshold,
        "strictly_exceeds_published_accuracy": passed,
        "effective_legacy_noise_std": float(candidate["legacy_noise_std"]),
        "effective_semantic_noise_std": float(candidate["semantic_noise_std"]),
        "utility_runs": candidate["utility_runs"],
        "source_probe": str(probe_path),
        "selection_uses_minimum_over_noise_seeds": True,
    }


def main() -> None:
    args = parse_args()
    probe = json.loads(args.probe.read_text(encoding="utf-8"))
    candidate = extract_candidate(
        probe, args.legacy_noise_std, args.semantic_noise_std
    )
    result = build_gate(candidate, args.utility_threshold, args.probe)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
