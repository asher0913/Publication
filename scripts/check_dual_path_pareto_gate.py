#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--utility-gate", required=True, type=Path)
    parser.add_argument("--decoder-metrics", required=True, type=Path)
    parser.add_argument("--gan-metrics", required=True, type=Path)
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=Path("configs/published_cem_facescrub_thresholds.json"),
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_gate(utility: dict, decoder: dict, gan: dict, baseline: dict) -> dict:
    observed = {
        "accuracy": float(utility["best_validation_accuracy"]),
        "decoder_training_mse": float(decoder["training"]["mse"]),
        "decoder_inference_mse": float(decoder["inference"]["mse"]),
        "gan_training_mse": float(gan["training"]["mse"]),
        "gan_inference_mse": float(gan["inference"]["mse"]),
    }
    checks = {name: value > float(baseline[name]) for name, value in observed.items()}
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "strictly_dominates_published_noise_arl_cem": all(checks.values()),
        "observed": observed,
        "baseline": baseline,
        "checks": checks,
        "absolute_margins": {
            name: value - float(baseline[name]) for name, value in observed.items()
        },
    }


def main() -> None:
    args = parse_args()
    thresholds = load_json(args.thresholds)
    result = evaluate_gate(
        load_json(args.utility_gate),
        load_json(args.decoder_metrics),
        load_json(args.gan_metrics),
        thresholds["baselines"]["noise_arl_cem"],
    )
    result["source"] = thresholds["source"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
