#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.aggregate_dual_path_publication import bootstrap_mean_ci  # noqa: E402


ATTACKS = ("decoder", "gan")
KNOWLEDGE = ("training", "inference")
PROFILE_MATCH_KEYS = (
    "client_parameters",
    "server_parameters",
    "total_payload_elements",
    "transmitted_bytes_per_sample_fp32",
)


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def collect_control(
    method: str,
    root: Path,
    target_seeds: list[int],
    attacker_seeds: list[int],
) -> dict:
    utility_rows = []
    attack_rows = []
    profile_rows = []
    target_attack_means: dict[str, list[float]] = {}
    for target_seed in target_seeds:
        utility_path = root / "targets" / f"seed{target_seed}" / "utility_l031_s010.json"
        utility = load_json(utility_path)
        utility_rows.append(
            {
                "method": method,
                "target_seed": target_seed,
                "accuracy": float(utility["mean_repeated_validation_accuracy"]),
                "minimum_accuracy": float(utility["minimum_repeated_validation_accuracy"]),
                "path": str(utility_path),
            }
        )
        profile_path = root / "targets" / f"seed{target_seed}" / "efficiency_profile.json"
        profile_rows.append(
            {
                "method": method,
                "target_seed": target_seed,
                "path": str(profile_path),
                **load_json(profile_path),
            }
        )
        for attack in ATTACKS:
            for knowledge in KNOWLEDGE:
                target_values = []
                for attacker_seed in attacker_seeds:
                    metrics_path = (
                        root
                        / "attacks"
                        / f"target_seed{target_seed}"
                        / f"attacker_seed{attacker_seed}"
                        / f"{attack}_{knowledge}"
                        / "attack_metrics.json"
                    )
                    payload = load_json(metrics_path)
                    if not payload.get("attacker_observes_spatial_and_semantic_paths"):
                        raise ValueError(f"attack did not observe both paths: {metrics_path}")
                    mse = float(payload["evaluation"]["mse"])
                    target_values.append(mse)
                    attack_rows.append(
                        {
                            "method": method,
                            "target_seed": target_seed,
                            "attacker_seed": attacker_seed,
                            "attack": attack,
                            "knowledge": knowledge,
                            "mse": mse,
                            "path": str(metrics_path),
                        }
                    )
                key = f"{attack}_{knowledge}_mse"
                target_attack_means.setdefault(key, []).append(
                    float(np.mean(target_values))
                )

    intervals = {
        "accuracy": bootstrap_mean_ci([row["accuracy"] for row in utility_rows])
    }
    intervals.update(
        {
            key: bootstrap_mean_ci(values)
            for key, values in target_attack_means.items()
        }
    )
    return {
        "method": method,
        "target_seeds": target_seeds,
        "attacker_seeds": attacker_seeds,
        "intervals": intervals,
        "utility_rows": utility_rows,
        "attack_rows": attack_rows,
        "profile_rows": profile_rows,
    }


def main_method(main_summary: dict) -> dict:
    return {
        "method": "dual_path_cem",
        "target_seeds": list(main_summary["target_seeds"]),
        "attacker_seeds": list(main_summary["attacker_seeds"]),
        "intervals": {
            key: main_summary["intervals"][key]
            for key in (
                "accuracy",
                "decoder_training_mse",
                "decoder_inference_mse",
                "gan_training_mse",
                "gan_inference_mse",
            )
        },
        "profile_rows": list(main_summary["efficiency_records"]),
    }


def mean_profile(method: dict, key: str) -> float:
    return float(np.mean([float(row[key]) for row in method["profile_rows"]]))


def validate_capacity_match(matched: dict, main: dict) -> dict:
    by_seed = {int(row["target_seed"]): row for row in main["profile_rows"]}
    checks = {}
    for row in matched["profile_rows"]:
        seed = int(row["target_seed"])
        reference = by_seed[seed]
        for key in PROFILE_MATCH_KEYS:
            name = f"seed{seed}_{key}"
            checks[name] = int(row[key]) == int(reference[key])
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"capacity match failed: {failed}")
    return {
        "status": "PASS",
        "checked_fields": list(PROFILE_MATCH_KEYS),
        "checks": checks,
    }


def format_interval(interval: dict, scale: float = 1.0) -> str:
    return (
        f"{interval['mean'] * scale:.4f} "
        f"[{interval['lower'] * scale:.4f}, {interval['upper'] * scale:.4f}]"
    )


def write_table(methods: list[dict], output: Path) -> None:
    labels = {
        "dual_path_cem": ("DualPath-CEM", "MobileNetV3-L"),
        "capacity_matched": ("One-stage control", "MobileNetV3-L"),
        "resnet18": ("Backbone control", "ResNet-18"),
    }
    rows = [
        r"\begin{tabular}{llrcccccr}",
        r"\toprule",
        (
            r"Method & Semantic backbone & $n$ & Accuracy & Dec.-train & "
            r"Dec.-infer & GAN-train & GAN-infer & Client params \\"
        ),
        r"\midrule",
    ]
    for method in methods:
        label, backbone = labels[method["method"]]
        intervals = method["intervals"]
        values = [
            format_interval(intervals[key])
            for key in (
                "accuracy",
                "decoder_training_mse",
                "decoder_inference_mse",
                "gan_training_mse",
                "gan_inference_mse",
            )
        ]
        rows.append(
            f"{label} & {backbone} & {len(method['target_seeds'])} & "
            + " & ".join(values)
            + f" & {mean_profile(method, 'client_parameters') / 1e6:.2f}M \\\\"
        )
    rows.extend((r"\bottomrule", r"\end{tabular}"))
    output.write_text("\n".join(rows) + "\n", encoding="utf-8")


def relative_change(value: float, reference: float) -> float:
    return 100.0 * (value - reference) / reference


def write_narrative(methods: list[dict], capacity_check: dict, output: Path) -> None:
    by_name = {method["method"]: method for method in methods}
    main = by_name["dual_path_cem"]["intervals"]
    matched = by_name["capacity_matched"]["intervals"]
    resnet = by_name["resnet18"]["intervals"]
    match_accuracy_gap = 100.0 * (
        main["accuracy"]["mean"] - matched["accuracy"]["mean"]
    )
    resnet_accuracy_gap = 100.0 * (
        resnet["accuracy"]["mean"] - main["accuracy"]["mean"]
    )
    matched_privacy = ", ".join(
        f"{relative_change(matched[key]['mean'], main[key]['mean']):+.1f}\\%"
        for key in (
            "decoder_training_mse",
            "decoder_inference_mse",
            "gan_training_mse",
            "gan_inference_mse",
        )
    )
    text = (
        "The one-stage control matches the complete model in parameters, payload, "
        "and training budget. At deployment noise, two-stage training increased "
        f"mean accuracy by {match_accuracy_gap:.2f} percentage points. Its four "
        "reconstruction-MSE differences from the complete model were "
        f"{matched_privacy}, in table order. Since capacity and training length "
        "are held fixed, the accuracy gap is attributable to adaptation at the "
        "deployment noise.\n\n"
        f"Replacing MobileNetV3-Large with ResNet-18 changed mean accuracy by "
        f"{resnet_accuracy_gap:+.2f} percentage points. Table~"
        "\\ref{tab:supplementary-controls} reports the corresponding decoder and "
        "GAN results. This control uses three target seeds, so we treat it as a "
        "backbone transfer check rather than a replacement for the five-seed main "
        "experiment.\n"
    )
    if capacity_check["status"] != "PASS":
        raise ValueError("refusing to write narrative without a capacity match")
    output.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(paths: list[Path], output: Path) -> None:
    files = {
        str(path): {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in paths
        if path.is_file()
    }
    save_json(output, {"schema_version": 1, "files": files})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--main-results-root", required=True, type=Path)
    parser.add_argument("--paper-output", required=True, type=Path)
    parser.add_argument(
        "--matched-target-seeds", nargs="+", type=int, default=list(range(126, 131))
    )
    parser.add_argument(
        "--backbone-target-seeds", nargs="+", type=int, default=list(range(126, 129))
    )
    parser.add_argument(
        "--attacker-seeds", nargs="+", type=int, default=[10125, 20125, 30125]
    )
    args = parser.parse_args()

    main_summary = load_json(args.main_results_root / "formal_summary.json")
    if main_summary.get("status") != "PASS":
        raise ValueError("main formal campaign has not passed")
    methods = [
        main_method(main_summary),
        collect_control(
            "capacity_matched",
            args.results_root / "capacity_matched",
            args.matched_target_seeds,
            args.attacker_seeds,
        ),
        collect_control(
            "resnet18",
            args.results_root / "resnet18",
            args.backbone_target_seeds,
            args.attacker_seeds,
        ),
    ]
    capacity_check = validate_capacity_match(methods[1], methods[0])
    summary = {
        "status": "PASS",
        "methods": methods,
        "capacity_match": capacity_check,
        "protocol": {
            "matched_total_epochs": 120,
            "deployment_legacy_noise_std": 0.31,
            "deployment_semantic_noise_std": 0.10,
            "primary_attacks": list(ATTACKS),
            "knowledge_settings": list(KNOWLEDGE),
        },
    }
    args.paper_output.mkdir(parents=True, exist_ok=True)
    summary_path = args.results_root / "supplementary_summary.json"
    table_path = args.paper_output / "controls_table.tex"
    narrative_path = args.paper_output / "controls_narrative.tex"
    utility_path = args.results_root / "utility_target_level.csv"
    attacks_path = args.results_root / "attack_runs.csv"
    save_json(summary_path, summary)
    write_table(methods, table_path)
    write_narrative(methods, capacity_check, narrative_path)
    write_csv(
        utility_path,
        methods[1]["utility_rows"] + methods[2]["utility_rows"],
    )
    write_csv(
        attacks_path,
        methods[1]["attack_rows"] + methods[2]["attack_rows"],
    )
    write_manifest(
        [summary_path, table_path, narrative_path, utility_path, attacks_path],
        args.results_root / "supplementary_materials_manifest.json",
    )
    print(
        json.dumps(
            {
                "status": "SUPPLEMENTARY_EVIDENCE_READY",
                "summary": str(summary_path),
                "paper_output": str(args.paper_output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
