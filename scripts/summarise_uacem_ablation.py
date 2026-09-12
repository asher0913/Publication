#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/uacem_formal/ablation"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/uacem_formal/ablation_summary.json"),
    )
    args = parser.parse_args()
    rows = []
    for calibration_path in sorted(args.results_root.glob("*/channel_calibration.json")):
        run_dir = calibration_path.parent
        selection = json.loads(
            (run_dir / "target_selection_metrics.json").read_text(encoding="utf-8")
        )
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        mse_values = []
        for metrics_path in sorted(
            (run_dir / "attacks/conv_decoder").glob("seed*/attack_selection_metrics.json")
        ):
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            mse_values.append(float(metrics["best_auxiliary_validation_mse"]))
        if len(mse_values) != 3:
            raise ValueError(f"{run_dir} requires three ablation attacks")
        rows.append(
            {
                "variant": run_dir.name,
                "score": calibration["score"],
                "alpha": float(calibration["alpha"]),
                "validation_accuracy": float(selection["best_validation_accuracy"]),
                "attacker_validation_mse_mean": mean(mse_values),
                "attacker_validation_mse_std": stdev(mse_values),
                "attacker_runs": len(mse_values),
                "fixed_total_noise_power": calibration["fixed_total_noise_power"],
            }
        )
    if len(rows) != 5:
        raise ValueError(f"expected five ablation variants, found {len(rows)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"schema_version": 1, "variants": rows}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"variants": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
