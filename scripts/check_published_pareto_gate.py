#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-summary", required=True, type=Path)
    parser.add_argument("--decoder-metrics", required=True, type=Path)
    parser.add_argument("--gan-metrics", required=True, type=Path)
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=Path("configs/published_cem_facescrub_thresholds.json"),
    )
    parser.add_argument("--baseline", default="noise_arl_cem")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def utility_accuracy(summary: dict) -> float:
    for key in ("best_validation_accuracy", "validation_accuracy", "accuracy"):
        if key in summary:
            return float(summary[key])
    validation = summary.get("validation")
    if isinstance(validation, dict) and "accuracy" in validation:
        return float(validation["accuracy"])
    raise KeyError("target summary has no validation accuracy")


def attack_mse(metrics: dict) -> float:
    for key in ("mse", "inference_mse", "attacker_validation_mse"):
        if key in metrics:
            return float(metrics[key])
    raise KeyError("attack metrics have no inference MSE")


def evaluate_gate(
    accuracy: float,
    decoder_mse: float,
    gan_mse: float,
    baseline: dict,
) -> dict:
    observed = {
        "accuracy": accuracy,
        "decoder_inference_mse": decoder_mse,
        "gan_inference_mse": gan_mse,
    }
    checks = {name: value > float(baseline[name]) for name, value in observed.items()}
    margins = {name: value - float(baseline[name]) for name, value in observed.items()}
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "strictly_dominates": all(checks.values()),
        "observed": observed,
        "baseline": baseline,
        "checks": checks,
        "absolute_margins": margins,
    }


def main() -> None:
    args = parse_args()
    thresholds = load_json(args.thresholds)
    if args.baseline not in thresholds["baselines"]:
        raise KeyError(f"unknown published baseline: {args.baseline}")
    result = evaluate_gate(
        utility_accuracy(load_json(args.target_summary)),
        attack_mse(load_json(args.decoder_metrics)),
        attack_mse(load_json(args.gan_metrics)),
        thresholds["baselines"][args.baseline],
    )
    result.update(
        source=thresholds["source"],
        compared_baseline=args.baseline,
        target_summary=str(args.target_summary),
        decoder_metrics=str(args.decoder_metrics),
        gan_metrics=str(args.gan_metrics),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
